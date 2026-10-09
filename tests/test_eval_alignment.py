"""Tests for placing retrieved chunks into a dataset's passage id space."""
import json
from pathlib import Path

import pytest

from eval.alignment import PassageResolver, load_passages, normalize_text, text_digest

REPO = Path(__file__).resolve().parents[1]
REPO_TEXT = '第一条　企业年金基金，是指根据依法制定的企业年金计划筹集的资金。'


def hits(*items):
    return [{'id': f'c{index}', 'content': text} for index, text in enumerate(items, 1)]


def test_exact_text_match_resolves_to_passage_id():
    resolver = PassageResolver({'P1': REPO_TEXT})
    result = resolver.resolve(hits(REPO_TEXT, '完全无关的内容'))
    assert result.resolved == [{'id': 'c1', 'rank': 1, 'passage_ids': ['P1'],
                                'match': 'exact_text'}]
    assert result.ranked_passage_ids == ['P1']
    assert result.unresolved == [{'id': 'c2', 'rank': 2, 'reason': 'no_matching_passage'}]
    assert result.coverage == 0.5


def test_a_passage_inside_a_longer_chunk_resolves():
    """The real case: a chunk spans a heading plus several labelled passages."""
    resolver = PassageResolver({'P1': '第一条　正文甲。', 'P2': '第二条　正文乙。'})
    chunk = '**标题**\n第一条　正文甲。\n第二条　正文乙。\n结尾说明。'
    result = resolver.resolve([{'id': 'c1', 'content': chunk}], ['P1'])
    assert result.ranked_passage_ids == ['P1']
    assert result.resolved[0]['match'] == 'contained_text'


def test_one_chunk_carrying_several_labels_credits_all_of_them():
    """A FAQ chunk covers a heading plus two answers; both labels must count."""
    resolver = PassageResolver({'P1': '正文甲', 'P2': '正文乙'})
    chunk = '引言 正文甲 转折 正文乙 结束'
    result = resolver.resolve([{'id': 'c1', 'content': chunk}], ['P1', 'P2'])
    assert sorted(result.ranked_passage_ids) == ['P1', 'P2']
    assert result.resolved[0]['passage_ids'] == ['P2', 'P1'] or \
        sorted(result.resolved[0]['passage_ids']) == ['P1', 'P2']
    assert result.unresolved == []


def test_repeated_labels_do_not_add_rank_positions():
    """Seeing the same passage twice must not push other results down."""
    resolver = PassageResolver({'P1': '正文甲'})
    result = resolver.resolve(hits('正文甲', '正文甲 补充', '其他'), ['P1'])
    assert result.ranked_passage_ids == ['P1']
    assert len(result.labels_per_hit) == 3


def test_whitespace_and_markup_do_not_matter():
    resolver = PassageResolver({'P1': '**第一条**\n\n正文  第二行'})
    variant = '第一条 正文 第二行'
    assert normalize_text(variant) == normalize_text('**第一条**\n\n正文  第二行')
    assert resolver.resolve(hits(variant)).resolved[0]['match'] == 'exact_text'


def test_explicit_passage_id_wins_without_text_matching():
    resolver = PassageResolver({'P1': '一些文本'})
    result = resolver.resolve([{'id': 'c1', 'content': '完全不同的内容', 'passage_id': 'P1'}])
    assert result.resolved[0] == {'id': 'c1', 'rank': 1, 'passage_ids': ['P1'],
                                  'match': 'explicit_id'}


def test_labels_outside_the_question_are_not_counted():
    """A chunk matching another question's passage must not count as a hit."""
    resolver = PassageResolver({'P1': '文本一', 'P2': '文本二'})
    result = resolver.resolve(hits('文本二'), passage_ids=['P1'])
    assert result.resolved == []
    assert result.ranked_passage_ids == []
    assert result.unresolved == [{'id': 'c1', 'rank': 1,
                                  'reason': 'labelled_for_another_question'}]
    # The same hit does resolve when the question really is about P2.
    assert resolver.resolve(hits('文本二'), passage_ids=['P2']).ranked_passage_ids == ['P2']


def test_ambiguous_passage_text_is_reported_not_guessed():
    resolver = PassageResolver({'P1': '相同文本', 'P2': '相同文本'})
    assert len(resolver.duplicates) == 1
    assert resolver.unique == {}
    # Ambiguous text cannot be credited to either label by content...
    result = resolver.resolve(hits('相同文本'), ['P1', 'P2'])
    assert result.ranked_passage_ids == []
    assert result.unresolved == [{'id': 'c1', 'rank': 1, 'reason': 'ambiguous_passage_text'}]
    # ...but an explicit id resolves it without guessing.
    assert resolver.resolve([{'id': 'c1', 'content': '相同文本', 'passage_id': 'P2'}],
                            ['P2']).ranked_passage_ids == ['P2']


def test_a_passage_no_chunk_can_isolate_is_reported_as_unmatched():
    """P1 is a substring of P3, so a chunk containing it matches P3 first.

    The chunk credits both labels rather than dropping one, and the question
    that only wants P1 still sees it resolved through containment.
    """
    resolver = PassageResolver({'P1': '第三条 正文', 'P3': '第三条 正文并附说明'})
    chunk = '前言 第三条 正文并附说明 结尾'
    both = resolver.resolve([{'id': 'c1', 'content': chunk}], ['P1', 'P3'])
    assert sorted(both.ranked_passage_ids) == ['P1', 'P3']
    only_p1 = resolver.resolve([{'id': 'c1', 'content': chunk}], ['P1'])
    assert only_p1.ranked_passage_ids == ['P1']


def test_empty_hits_yield_no_coverage_claim():
    resolver = PassageResolver({'P1': '文本'})
    result = resolver.resolve([])
    assert result.ranked_passage_ids == []
    assert result.unresolved == [] and result.coverage is None


def test_load_passages_reads_both_corpus_index_layouts(tmp_path):
    path = tmp_path / 'index.jsonl'
    path.write_text('\n'.join(json.dumps(record, ensure_ascii=False) for record in [
        {'id': 'D1-S01', 'text': '证据一'},
        {'id': 'CAS-01-A002', 'text': '条文二'},
    ]) + '\n', encoding='utf-8')
    passages = load_passages(path)
    assert passages == {'D1-S01': '证据一', 'CAS-01-A002': '条文二'}
    with pytest.raises(KeyError):
        load_passages(tmp_path / 'index.jsonl', id_field='missing')


def test_real_corpus_indexes_resolve_against_real_question_labels():
    """The alignment must hold for the shipped fixtures, both knowledge bases."""
    baseline = REPO / 'experiments' / 'rag-baseline'
    smart = load_passages(baseline / 'evidence.jsonl')
    accounting = load_passages(baseline / 'accounting' / 'articles.jsonl')
    assert 'D1-S01' in smart and 'CAS-10-2006-A002' in accounting
    # A passage resolved by its own text keeps its id: retrieval of that text
    # must be scored as a hit rather than as an alignment failure.
    resolver = PassageResolver(smart)
    result = resolver.resolve([{'id': 'chunk-1', 'content': smart['D1-S01']}], ['D1-S01'])
    assert result.ranked_passage_ids == ['D1-S01']
    assert result.resolved[0]['match'] == 'exact_text'
    assert text_digest(smart['D1-S01']) in resolver.unique


def test_real_retrieval_shape_resolves_by_containment():
    """A corpus chunk that spans a heading plus a labelled passage resolves.

    The corpus carries markdown bold headings that the evidence file repeats
    inside its text, so exact equality fails and containment is the rule that
    actually applies. Getting this wrong scored a full retrieval run as 0.03.
    """
    baseline = REPO / 'experiments' / 'rag-baseline'
    smart = load_passages(baseline / 'evidence.jsonl')
    resolver = PassageResolver(smart)
    passage_id = 'D1-S02'
    chunk = f"**设置键按了没反应怎么办？**\n{smart[passage_id]}\n**另外一个问题？**\n后续正文。"
    result = resolver.resolve([{'id': 'chunk-1', 'content': chunk}], [passage_id])
    assert result.ranked_passage_ids == [passage_id]
    assert result.resolved[0]['match'] == 'contained_text'
