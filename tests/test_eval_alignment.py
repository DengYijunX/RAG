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
    assert result.resolved == [('c1', 'P1')]
    # One slot per hit: the unresolved second hit keeps its rank.
    assert result.ranked_passage_ids == ['P1', None]
    assert result.unresolved == [{'id': 'c2', 'rank': 2, 'reason': 'no_matching_passage'}]
    assert result.coverage == 0.5


def test_whitespace_and_line_endings_do_not_matter():
    resolver = PassageResolver({'P1': '第一条　正文\n第二行'})
    variant = '第一条  正文\r\n  第二行\n'
    assert normalize_text(variant) == normalize_text('第一条　正文\n第二行')
    assert resolver.resolve(hits(variant)).resolved == [('c1', 'P1')]


def test_explicit_passage_id_wins_without_text_matching():
    resolver = PassageResolver({'P1': '一些文本'})
    result = resolver.resolve([{'id': 'c1', 'content': '完全不同的内容', 'passage_id': 'P1'}])
    assert result.resolved == [('c1', 'P1')]


def test_labels_outside_the_question_are_not_counted():
    """A chunk matching another question's passage must not count as a hit."""
    resolver = PassageResolver({'P1': '文本一', 'P2': '文本二'})
    result = resolver.resolve(hits('文本二'), passage_ids=['P1'])
    assert result.resolved == []
    assert result.ranked_passage_ids == [None]
    assert result.unresolved == [{'id': 'c1', 'rank': 1,
                                  'reason': 'labelled_for_another_question'}]
    # The same hit does resolve when the question really is about P2.
    assert resolver.resolve(hits('文本二'), passage_ids=['P2']).resolved == [('c1', 'P2')]


def test_ambiguous_passage_text_is_reported_not_guessed():
    resolver = PassageResolver({'P1': '相同文本', 'P2': '相同文本'})
    assert len(resolver.duplicates) == 1
    assert resolver.unique == {}
    result = resolver.resolve(hits('相同文本'))
    assert result.unresolved == [{'id': 'c1', 'rank': 1, 'reason': 'ambiguous_passage_text'}]
    assert result.ranked_passage_ids == [None]
    # With an explicit id the ambiguity disappears.
    assert resolver.resolve([{'id': 'c1', 'content': '相同文本', 'passage_id': 'P2'}],
                            ['P2']).resolved == [('c1', 'P2')]


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
    assert result.resolved == [('chunk-1', 'D1-S01')]
    assert text_digest(smart['D1-S01']) in resolver.unique
