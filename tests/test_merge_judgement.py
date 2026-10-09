"""Tests for the judgement merge and the agreement check.

The judgement is written into ``lead_*`` fields, never ``human_*``: an AI
judgement is not a human sign-off, and the rubric requires the two to stay
separate. A malformed judgement must fail loudly rather than become a number.
"""
import json

import pytest

from merge_judgement import agreement, load_judgement


def write(path, entries):
    path.write_text('\n'.join(json.dumps(entry, ensure_ascii=False) for entry in entries) + '\n',
                    encoding='utf-8')
    return path


def entry(identifier, **overrides):
    payload = {'question_id': identifier, 'key_points': 2, 'citation_faithfulness': 2,
               'behavior_match': 2, 'error_labels': [], 'note': 'ok'}
    payload.update(overrides)
    return payload


def test_a_valid_judgement_loads(tmp_path):
    path = write(tmp_path / 'j.jsonl', [entry('Q1'), entry('Q2', key_points='na')])
    loaded = load_judgement(path)
    assert set(loaded) == {'Q1', 'Q2'}
    assert loaded['Q2']['key_points'] == 'na'
    assert loaded['Q1']['error_labels'] == []


@pytest.mark.parametrize('value', [3, -1, '2', None, 'good'])
def test_invalid_scores_are_refused(tmp_path, value):
    path = write(tmp_path / 'j.jsonl', [entry('Q1', key_points=value)])
    with pytest.raises(SystemExit):
        load_judgement(path)


def test_unknown_labels_are_refused(tmp_path):
    path = write(tmp_path / 'j.jsonl', [entry('Q1', error_labels=['nonsense'])])
    with pytest.raises(SystemExit):
        load_judgement(path)


def test_duplicate_questions_are_refused(tmp_path):
    path = write(tmp_path / 'j.jsonl', [entry('Q1'), entry('Q1')])
    with pytest.raises(SystemExit):
        load_judgement(path)


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        load_judgement(tmp_path / 'absent.jsonl')


def test_labels_are_deduplicated(tmp_path):
    path = write(tmp_path / 'j.jsonl',
                 [entry('Q1', error_labels=['retrieval_missing', 'retrieval_missing'])])
    assert load_judgement(path)['Q1']['error_labels'] == ['retrieval_missing']


def record(identifier, lead, assistant, lead_labels=(), assistant_labels=()):
    return {'question_id': identifier,
            'lead_scores': dict(zip(('key_points', 'citation_faithfulness', 'behavior_match'),
                                    lead)),
            'assistant_scores': dict(zip(('key_points', 'citation_faithfulness', 'behavior_match'),
                                         assistant)),
            'lead_error_labels': list(lead_labels),
            'assistant_error_labels': list(assistant_labels)}


def test_agreement_counts_exact_matches_per_dimension():
    records = [
        record('Q1', (2, 2, 2), (2, 2, 2)),          # all three match
        record('Q2', (1, 2, 1), (1, 2, 2)),          # behaviour differs
        record('Q3', (2, 2, 2), (2, 2, 2), lead_labels=['retrieval_missing']),
    ]
    report = agreement(records, 'kb')
    assert report['questions'] == 3
    assert report['all_three_match'] == 2
    assert report['per_dimension_match']['behavior_match'] == {
        'numerator': 2, 'denominator': 3, 'rate': pytest.approx(0.6667, abs=1e-3)}
    assert report['per_dimension_match']['key_points']['rate'] == 1.0


def test_na_rows_leave_a_dimension_out_of_the_comparison():
    """A dimension marked not-applicable on either side is not a disagreement."""
    records = [
        record('Q1', ('na', 'na', 2), (2, 1, 2)),
        record('Q2', (2, 2, 2), (2, 2, 2)),
    ]
    report = agreement(records, 'kb')
    assert report['per_dimension_match']['key_points'] == {'numerator': 1, 'denominator': 1,
                                                           'rate': 1.0}
    # only Q2 has all three dimensions scored on both sides
    assert report['fully_comparable'] == 1


def test_label_disagreements_are_listed_both_ways():
    records = [record('Q1', (2, 2, 2), (2, 2, 2),
                      lead_labels=['retrieval_missing'],
                      assistant_labels=['retrieval_missing', 'wrong_scope'])]
    report = agreement(records, 'kb')
    assert len(report['label_disagreements']) == 1
    disagreement = report['label_disagreements'][0]
    assert disagreement['labels_only_assistant'] == ['wrong_scope']
    assert disagreement['labels_only_lead'] == []


def test_records_without_a_judgement_are_skipped():
    records = [record('Q1', (2, 2, 2), (2, 2, 2)),
               {'question_id': 'Q2', 'lead_scores': {}, 'assistant_scores': {}}]
    report = agreement(records, 'kb')
    assert report['questions'] == 1
