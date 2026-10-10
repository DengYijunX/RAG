"""Tests for recording a human review and checking it against the assistant.

The point of this path is that a reviewer's scores must be stored separately from
the assistant's, must carry the provenance of the review itself, and must be
validated: a typo has to fail loudly rather than become a published number.
"""
import json

import pytest

from human_review import apply_review, load_scores, summarise

PROVENANCE = {'reviewer': 'project owner', 'reviewer_is_independent': False}


def entry(identifier, **overrides):
    payload = {'question_id': identifier, 'key_points': 2, 'citation_faithfulness': 2,
               'behavior_match': 2, 'error_labels': []}
    payload.update(overrides)
    return payload


def write_scores(tmp_path, entries):
    path = tmp_path / 'scores.jsonl'
    path.write_text('\n'.join(json.dumps(item, ensure_ascii=False) for item in entries) + '\n',
                    encoding='utf-8')
    return path


def write_run(tmp_path, question_ids, assistant_scores=None):
    run = tmp_path / 'run'
    run.mkdir()
    records = []
    for identifier in question_ids:
        records.append({
            'question_id': identifier,
            'assistant_scores': (assistant_scores or {}).get(
                identifier, {'key_points': 2, 'citation_faithfulness': 2, 'behavior_match': 2}),
            'assistant_error_labels': [],
            'human_scores': {'key_points': None, 'citation_faithfulness': None,
                             'behavior_match': None},
            'human_error_labels': [], 'human_reviewed': False,
        })
    with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return run


def load_records(run):
    return [json.loads(line) for line in
            (run / 'records.jsonl').read_text(encoding='utf-8').splitlines()]


def test_valid_scores_load(tmp_path):
    loaded = load_scores(write_scores(tmp_path, [entry('Q1'), entry('Q2', key_points='na')]))
    assert loaded['Q2']['key_points'] == 'na'
    assert loaded['Q1']['error_labels'] == []


@pytest.mark.parametrize('value', [3, -1, '2', None, 'good'])
def test_invalid_values_are_refused(tmp_path, value):
    with pytest.raises(SystemExit):
        load_scores(write_scores(tmp_path, [entry('Q1', key_points=value)]))


def test_unknown_error_labels_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        load_scores(write_scores(tmp_path, [entry('Q1', error_labels=['made_up'])]))


def test_duplicates_and_empty_files_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        load_scores(write_scores(tmp_path, [entry('Q1'), entry('Q1')]))
    with pytest.raises(SystemExit):
        load_scores(write_scores(tmp_path, []))


def test_a_score_for_a_question_outside_the_run_is_refused(tmp_path):
    run = write_run(tmp_path, ('Q1',))
    with pytest.raises(SystemExit):
        apply_review(run, load_scores(write_scores(tmp_path, [entry('Q9')])), PROVENANCE)


def test_review_is_written_into_human_fields_with_provenance(tmp_path):
    run = write_run(tmp_path, ('Q1', 'Q2'))
    scores = load_scores(write_scores(tmp_path, [entry('Q1', key_points=1, note='漏了一条')]))
    records, applied = apply_review(run, scores, PROVENANCE)
    assert applied == 1
    by_id = {record['question_id']: record for record in load_records(run)}
    assert by_id['Q1']['human_scores']['key_points'] == 1
    assert by_id['Q1']['human_note'] == '漏了一条'
    assert by_id['Q1']['human_reviewed'] is True
    assert by_id['Q1']['human_review_meta'] == PROVENANCE
    # The unreviewed question must not be marked as reviewed.
    assert by_id['Q2']['human_reviewed'] is False


def test_assistant_scores_are_never_overwritten(tmp_path):
    """The two passes must stay separate: conflating them loses the comparison."""
    run = write_run(tmp_path, ('Q1',))
    apply_review(run, load_scores(write_scores(tmp_path, [entry('Q1', key_points=0)])),
                 PROVENANCE)
    record = load_records(run)[0]
    assert record['assistant_scores']['key_points'] == 2
    assert record['human_scores']['key_points'] == 0


def test_summary_rates_and_agreement(tmp_path):
    run = write_run(tmp_path, ('Q1', 'Q2'),
                    assistant_scores={'Q2': {'key_points': 1, 'citation_faithfulness': 2,
                                             'behavior_match': 2}})
    scores = load_scores(write_scores(tmp_path, [entry('Q1'), entry('Q2')]))
    records, _ = apply_review(run, scores, PROVENANCE)
    summary = summarise(records, PROVENANCE)
    assert summary['human_full_marks']['key_points'] == {'numerator': 2, 'denominator': 2,
                                                         'rate': 1.0}
    assert summary['agreement_with_assistant']['key_points'] == {
        'numerator': 1, 'denominator': 2, 'rate': 0.5}
    assert summary['disagreements'] == [{
        'question_id': 'Q2', 'dimension': 'key_points', 'human': 2, 'assistant': 1,
        'direction': 'assistant_stricter'}]


def test_na_is_excluded_from_both_rates(tmp_path):
    run = write_run(tmp_path, ('Q1', 'Q2'))
    scores = load_scores(write_scores(tmp_path, [entry('Q1', key_points='na'), entry('Q2')]))
    records, _ = apply_review(run, scores, PROVENANCE)
    summary = summarise(records, PROVENANCE)
    assert summary['human_full_marks']['key_points'] == {'numerator': 1, 'denominator': 1,
                                                         'rate': 1.0}
    assert summary['agreement_with_assistant']['key_points']['denominator'] == 1


def test_error_labels_are_counted(tmp_path):
    run = write_run(tmp_path, ('Q1', 'Q2'))
    scores = load_scores(write_scores(tmp_path, [
        entry('Q1', error_labels=['retrieval_missing']),
        entry('Q2', error_labels=['retrieval_missing', 'missing_clarification'])]))
    records, _ = apply_review(run, scores, PROVENANCE)
    summary = summarise(records, PROVENANCE)
    assert summary['human_error_labels'] == {'retrieval_missing': 2, 'missing_clarification': 1}


def test_the_summary_records_that_the_reviewer_is_not_independent(tmp_path):
    """An owner confirming an AI's scores must not read as independent review."""
    run = write_run(tmp_path, ('Q1',))
    records, _ = apply_review(run, load_scores(write_scores(tmp_path, [entry('Q1')])),
                              PROVENANCE)
    summary = summarise(records, PROVENANCE)
    assert summary['provenance']['reviewer_is_independent'] is False
    assert 'reviewer' in summary['provenance']
