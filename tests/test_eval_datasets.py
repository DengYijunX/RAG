"""Dataset reader tests.

Includes a check that the real splits in ``experiments/rag-baseline`` load
with the counts the handoff documents, so a schema drift in those files is
reported here rather than silently shrinking an evaluation set.
"""
import json
from pathlib import Path

import pytest

from eval.datasets import (DatasetError, load_all, load_dataset, load_knowledge_base_dataset,
                           passage_ids_from_excerpts, parse_question)

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / 'experiments' / 'rag-baseline'


def write_jsonl(path, records):
    path.write_text('\n'.join(json.dumps(record, ensure_ascii=False) for record in records) + '\n',
                    encoding='utf-8')
    return path


def test_loads_a_minimal_file(tmp_path):
    path = write_jsonl(tmp_path / 'q.jsonl', [
        {'id': 'q1', 'question': '问题一', 'evidence_ids': ['p1', 'p2'], 'split': 'dev',
         'answer': '答案一'},
        {'id': 'q2', 'question': '问题二', 'split': 'test'},
    ])
    dataset = load_dataset(path)
    assert [question.id for question in dataset.questions] == ['q1', 'q2']
    assert dataset.questions[0].relevant_passage_ids == ['p1', 'p2']
    assert dataset.questions[0].answer == '答案一'
    assert dataset.questions[1].relevant_passage_ids == []
    assert dataset.graded == [dataset.questions[0]]
    assert dataset.splits() == ['dev', 'test']
    assert dataset.summary() == {'name': 'q', 'questions': 2, 'graded': 1, 'splits': ['dev', 'test']}


def test_split_filter_and_name(tmp_path):
    path = write_jsonl(tmp_path / 'q.jsonl', [
        {'id': 'q1', 'question': 'a', 'split': 'dev'},
        {'id': 'q2', 'question': 'b', 'split': 'test'},
    ])
    dev = load_dataset(path, split='dev', name='自定义')
    assert dev.name == '自定义' and [question.id for question in dev.questions] == ['q1']


def test_alternate_field_names_are_accepted(tmp_path):
    path = write_jsonl(tmp_path / 'q.jsonl', [
        {'query_id': 'x1', 'query': '问题', 'relevant_passage_ids': ['p1'],
         'reference_answer': '答案'},
    ])
    question = load_dataset(path).questions[0]
    assert question.id == 'x1' and question.question == '问题'
    assert question.relevant_passage_ids == ['p1'] and question.answer == '答案'


def test_reference_excerpts_give_labels_and_kept_metadata(tmp_path):
    path = write_jsonl(tmp_path / 'q.jsonl', [
        {'id': 'a1', 'question': '问题', 'knowledge_base': 'accounting',
         'reference_excerpts': [{'article_id': 'CAS-01-A002', 'text': '正文'},
                                {'text': '没有 id 的摘录'}],
         'category': 'article_lookup', 'annotation_status': 'draft_needs_domain_review'},
    ])
    question = load_dataset(path).questions[0]
    assert question.relevant_passage_ids == ['CAS-01-A002']
    assert question.knowledge_base == 'accounting'
    # Extra columns survive in metadata instead of being dropped silently.
    assert question.metadata['category'] == 'article_lookup'
    assert question.metadata['annotation_status'] == 'draft_needs_domain_review'
    # Only fields the loader already turned into attributes are consumed.
    assert 'reference_excerpts' not in question.metadata


def test_labels_are_deduplicated_and_ordered(tmp_path):
    path = write_jsonl(tmp_path / 'q.jsonl', [
        {'id': 'q1', 'question': 'a', 'evidence_ids': ['p2', 'p1', 'p2']},
    ])
    assert load_dataset(path).questions[0].relevant_passage_ids == ['p2', 'p1']


@pytest.mark.parametrize('record,needle', [
    ({'question': '没有问题 ID'}, '缺少题目 ID'),
    ({'id': 'q1'}, '缺少题目文本'),
    ({'id': 'q1', 'question': '   '}, '缺少题目文本'),
])
def test_invalid_records_report_the_reason(tmp_path, record, needle):
    path = write_jsonl(tmp_path / 'q.jsonl', [record])
    with pytest.raises(DatasetError) as error:
        load_dataset(path)
    assert needle in str(error.value)


def test_blank_lines_are_ignored_but_bad_json_is_not(tmp_path):
    path = tmp_path / 'q.jsonl'
    path.write_text('\n{"id": "q1", "question": "a"}\n\n', encoding='utf-8')
    assert len(load_dataset(path)) == 1
    path.write_text('{"id": "q1", "question": "a"}\n{"id": "q2"\n', encoding='utf-8')
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_parse_question_reports_line_number(tmp_path):
    with pytest.raises(DatasetError) as error:
        parse_question({'id': 'q1'}, Path('x.jsonl'), 7)
    assert 'x.jsonl:7' in str(error.value)


def test_load_knowledge_base_dataset_prefers_the_base_directory(tmp_path):
    smart = tmp_path / 'smart-home'
    smart.mkdir()
    (tmp_path / 'questions.jsonl').write_text('{"id":"s1","question":"a"}\n', encoding='utf-8')
    accounting = tmp_path / 'accounting'
    accounting.mkdir()
    (accounting / 'questions.jsonl').write_text('{"id":"a1","question":"b"}\n', encoding='utf-8')
    assert load_knowledge_base_dataset(tmp_path, 'accounting').questions[0].id == 'a1'
    # smart-home has no directory of its own, so the root file is the answer.
    assert load_knowledge_base_dataset(tmp_path, 'smart-home').questions[0].id == 's1'
    with pytest.raises(DatasetError):
        load_knowledge_base_dataset(tmp_path / 'empty', 'accounting')


def test_real_splits_load_with_the_counts_actually_on_disk():
    """v1 delivery counts, verified on 2026-10-09.

    This replaces the stale "dev128 / test97" note: the delivered v1 set is
    201 questions = smart-home 60 (40 dev / 20 test) + accounting 141
    (82 dev / 59 test), i.e. 122 dev and 79 test. The 225-question review
    candidate set is v2 and is loaded separately below.
    """
    sets = load_all(BASELINE)
    accounting_dev = load_knowledge_base_dataset(BASELINE, 'accounting', split='dev')
    accounting_test = load_knowledge_base_dataset(BASELINE, 'accounting', split='test')
    smart_dev = load_knowledge_base_dataset(BASELINE, 'smart-home', split='dev')
    smart_test = load_knowledge_base_dataset(BASELINE, 'smart-home', split='test')
    assert (len(accounting_dev), len(accounting_test)) == (82, 59)
    assert (len(smart_dev), len(smart_test)) == (40, 20)
    assert len(accounting_dev) + len(smart_dev) == 122
    assert len(accounting_test) + len(smart_test) == 79
    assert sum(len(dataset) for dataset in sets.values()) == 201
    assert set(sets) == {'smart-home', 'accounting'}
    # The loader reports what is graded instead of pretending every question
    # is scorable: smart-home has questions without evidence ids.
    assert len(sets['accounting'].graded) >= 120
    assert len(sets['smart-home'].graded) >= 50
    assert len(sets['smart-home'].graded) < len(sets['smart-home'])
    # Excerpt labels must line up with the retrieval index's article ids.
    articles = {json.loads(line)['id'] for line in
                (BASELINE / 'accounting' / 'articles.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()}
    labelled = {label for question in sets['accounting'].questions
                for label in question.relevant_passage_ids}
    assert labelled and labelled <= articles


def test_v2_review_candidate_set_loads_with_225_questions():
    """The v2 set is the 225-question review candidate, not signed-off gold."""
    sets = load_all(BASELINE, version='v2')
    assert sum(len(dataset) for dataset in sets.values()) == 225
    assert len(sets['smart-home']) == 72
    assert len(sets['accounting']) == 153
    assert len(load_knowledge_base_dataset(BASELINE, 'smart-home', split='dev', version='v2')) == 43
    assert len(load_knowledge_base_dataset(BASELINE, 'smart-home', split='test', version='v2')) == 29
    assert len(load_knowledge_base_dataset(BASELINE, 'accounting', split='dev', version='v2')) == 85
    assert len(load_knowledge_base_dataset(BASELINE, 'accounting', split='test', version='v2')) == 68
    # Annotations are still drafts: the reader must not present them as gold.
    statuses = {question.metadata.get('annotation_status') for question in sets['accounting'].questions}
    assert all(status and status.startswith('draft') for status in statuses)
    with pytest.raises(DatasetError):
        load_knowledge_base_dataset(BASELINE, 'accounting', version='v3')

