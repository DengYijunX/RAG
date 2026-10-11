"""Tests for exporting and importing the human review sheet.

The import path is the dangerous one: a typo in a score or an error label must be
refused loudly, because a silently accepted value becomes a published number.
"""
import csv
import json

import pytest

from review_sheet import (CSV_COLUMNS, EXAMPLE_ROW, FILL_COLUMNS, evidence_status, export,
                          import_sheet, parse_score, passage_sources, rows_for)


class StubQuestion:
    def __init__(self, identifier, split='dev'):
        self.id = identifier
        self.split = split
        self.relevant_passage_ids = ['P1']
        self.metadata = {'category': 'single', 'expected_action': 'answer',
                         'expected_points': ['要点一', '要点二']}


def write_run(tmp_path, question_ids=('Q1', 'Q2')):
    run = tmp_path / 'run'
    run.mkdir()
    records = []
    for identifier in question_ids:
        records.append({
            'question_id': identifier, 'question': f'{identifier} 的问题',
            'expected_action': 'answer',
            'answer': f'{identifier} 的回答[1]',
            'citations': [{'citation': 1, 'source': 'doc.md', 'start': 0, 'end': 10,
                           'content': '片段正文'}],
            'assistant_scores': {'key_points': 2, 'citation_faithfulness': 1,
                                 'behavior_match': 2},
            'assistant_error_labels': ['retrieval_missing'],
            'assistant_justification': '理由',
            'human_scores': {'key_points': None, 'citation_faithfulness': None,
                             'behavior_match': None},
            'human_error_labels': [], 'human_reviewed': False,
        })
    with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return run


def stub_context(question_ids):
    return {identifier: StubQuestion(identifier) for identifier in question_ids}


def stub_passages(_path):
    """Stand in for the dataset's labelled passages.

    ``rows_for`` needs them to report whether the evidence the answer should have
    used was actually in front of it; without the stub the tests read the corpus.
    """
    return {'P1': '片段正文'}


def test_export_writes_both_views(tmp_path, monkeypatch):
    run = write_run(tmp_path)
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1', 'Q2')))
    export(run, 'smart-home', 'v2')
    assert (run / 'review.csv').is_file() and (run / 'review.md').is_file()

    with open(run / 'review.csv', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0]) == CSV_COLUMNS
    assert rows[0]['assistant_key_points'] == '2'
    assert rows[0]['question'] == 'Q1 的问题'
    assert '要点一' in rows[0]['reference_points']
    assert all(rows[0][f'human_{dimension}'] == ''
               for dimension in ('key_points', 'citation_faithfulness', 'behavior_match'))

    markdown = (run / 'review.md').read_text(encoding='utf-8')
    assert '# 人工审阅表' in markdown
    assert 'Q1 的回答[1]' in markdown
    assert '人工评分' in markdown


def fill_sheet(run, values, filename='review.csv'):
    """Write human columns into an exported sheet."""
    path = run / filename
    with open(path, encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0])
    for row in rows:
        for key, value in values.get(row['question_id'], {}).items():
            row[key] = value
    with open(path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_writes_a_short_fill_in_sheet(tmp_path, monkeypatch):
    """The short sheet is what a person edits: five cells, plus an example row."""
    run = write_run(tmp_path)
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1', 'Q2')))
    export(run, 'smart-home', 'v2')
    path = run / '待填写.csv'
    assert path.is_file()
    with open(path, encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == FILL_COLUMNS
    assert len(rows) == 3                      # 1 example + 2 questions
    assert rows[0]['question_id'].startswith('（示例')
    assert rows[0]['human_key_points'] == '2'  # the example shows a filled value
    assert rows[1]['human_key_points'] == ''   # real rows are empty
    assert '助手初审' in rows[1]['assistant_summary']


def test_the_short_sheet_round_trips_and_skips_the_example_row(tmp_path, monkeypatch):
    run = write_run(tmp_path)
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1', 'Q2')))
    export(run, 'smart-home', 'v2')
    fill_sheet(run, {'Q1': {'human_key_points': '2', 'human_citation_faithfulness': '2',
                            'human_behavior_match': '2'},
                     'Q2': {'human_key_points': '1', 'human_citation_faithfulness': '0',
                            'human_behavior_match': '2'}},
               filename='待填写.csv')
    import_sheet(run, 'smart-home', 'v2')
    summary = json.loads((run / 'human_summary.json').read_text(encoding='utf-8'))
    assert summary['human_reviewed'] == 2
    assert summary['incomplete'] == []
    assert summary['skipped_rows'] == [EXAMPLE_ROW['question_id']]
    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {record['question_id'] for record in records
            if record['human_reviewed']} == {'Q1', 'Q2'}


def test_import_records_human_scores_and_reports_rates(tmp_path, monkeypatch):
    run = write_run(tmp_path)
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1', 'Q2')))
    export(run, 'smart-home', 'v2')
    fill_sheet(run, {
        'Q1': {'human_key_points': '2', 'human_citation_faithfulness': '2',
               'human_behavior_match': '2'},
        'Q2': {'human_key_points': '0', 'human_citation_faithfulness': '1',
               'human_behavior_match': 'na', 'human_error_labels': 'unsupported_claim、wrong_scope',
               'human_note': '版本用错'},
    })
    import_sheet(run, 'smart-home', 'v2', run / 'review.csv')

    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {record['question_id']: record for record in records}
    assert by_id['Q1']['human_scores'] == {'key_points': 2, 'citation_faithfulness': 2,
                                           'behavior_match': 2}
    assert by_id['Q1']['human_reviewed'] is True
    assert by_id['Q2']['human_scores']['behavior_match'] == 'na'
    assert by_id['Q2']['human_error_labels'] == ['unsupported_claim', 'wrong_scope']
    assert by_id['Q2']['human_note'] == '版本用错'

    summary = json.loads((run / 'human_summary.json').read_text(encoding='utf-8'))
    assert summary['human_reviewed'] == 2
    assert summary['human_full_marks']['key_points'] == {'numerator': 1, 'denominator': 2,
                                                         'rate': 0.5}
    # behaviour_match excludes the 'na' row from its denominator
    assert summary['human_full_marks']['behavior_match'] == {'numerator': 1, 'denominator': 1,
                                                             'rate': 1.0}
    assert summary['human_error_labels'] == {'unsupported_claim': 1, 'wrong_scope': 1}


def test_unfilled_rows_are_reported_and_excluded(tmp_path, monkeypatch):
    run = write_run(tmp_path, question_ids=('Q1', 'Q2'))
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1', 'Q2')))
    export(run, 'smart-home', 'v2')
    fill_sheet(run, {'Q1': {'human_key_points': '1'}})
    import_sheet(run, 'smart-home', 'v2', run / 'review.csv')
    summary = json.loads((run / 'human_summary.json').read_text(encoding='utf-8'))
    assert summary['human_reviewed'] == 1
    assert summary['incomplete'] == ['Q2']


@pytest.mark.parametrize('value', ['3', '-1', '很好', 'two'])
def test_invalid_scores_are_refused(tmp_path, monkeypatch, value):
    run = write_run(tmp_path, question_ids=('Q1',))
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1',)))
    export(run, 'smart-home', 'v2')
    fill_sheet(run, {'Q1': {'human_key_points': value}})
    with pytest.raises(SystemExit):
        import_sheet(run, 'smart-home', 'v2', run / 'review.csv')


def test_unknown_error_labels_are_refused(tmp_path, monkeypatch):
    run = write_run(tmp_path, question_ids=('Q1',))
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1',)))
    export(run, 'smart-home', 'v2')
    fill_sheet(run, {'Q1': {'human_key_points': '2', 'human_error_labels': 'made_up'}})
    with pytest.raises(SystemExit):
        import_sheet(run, 'smart-home', 'v2', run / 'review.csv')


def test_parse_score_accepts_the_documented_forms():
    assert parse_score('0', 'x') == 0
    assert parse_score(' 2 ', 'x') == 2
    assert parse_score('na', 'x') == 'na'
    assert parse_score('N/A', 'x') == 'na'
    assert parse_score('不适用', 'x') == 'na'
    assert parse_score('', 'x') is None


def test_rows_for_marks_invalid_assistant_scores(tmp_path, monkeypatch):
    run = write_run(tmp_path, question_ids=('Q1',))
    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    records[0]['assistant_scores'] = {}
    with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(records[0], ensure_ascii=False) + '\n')
    monkeypatch.setattr('review_sheet.load_passages', stub_passages)
    monkeypatch.setattr('review_sheet.load_context',
                        lambda kb, version: stub_context(('Q1',)))
    rows = rows_for(run, 'smart-home', 'v2')
    assert rows[0]['assistant_key_points'] == 'invalid'


# --- evidence status: the text alone is not enough, the file must match too ----

PASSAGES = {'CAS-37-2017-A008': '第八条 金融负债，是指企业符合下列条件之一的负债。',
            'CAS-09-2014-A002': '第二条 职工薪酬，是指企业为获得职工提供的服务而给予的报酬。'}


def cited(source, content):
    return {'citations': [{'citation': 1, 'source': source, 'content': content}]}


def test_evidence_present_when_text_and_source_agree():
    status = evidence_status(cited('CAS-37-2017.md', PASSAGES['CAS-37-2017-A008']),
                             {'CAS-37-2017-A008': PASSAGES['CAS-37-2017-A008']},
                             PASSAGES, passage_sources(PASSAGES))
    assert status.startswith('上下文已含全部')


def test_identical_text_from_the_wrong_version_is_not_evidence():
    """CAS-37 第八条 is word-for-word identical in the 2014 and 2017 versions.

    Matching text alone reports the evidence as present when the retrieved chunk
    came from the wrong year, which is exactly the mistake these questions test
    for. Measured case: ACC-Q045 was scored "evidence present" while the citation
    was CAS-37-2014.md and the label was CAS-37-2017-A008.
    """
    status = evidence_status(cited('CAS-37-2014.md', PASSAGES['CAS-37-2017-A008']),
                             {'CAS-37-2017-A008': PASSAGES['CAS-37-2017-A008']},
                             PASSAGES, passage_sources(PASSAGES))
    assert '错误版本' in status and 'CAS-37-2014.md' in status


def test_without_provenance_only_the_text_is_checked():
    """Callers that cannot supply sources still get the weaker check, not a crash."""
    status = evidence_status(cited('anything.md', PASSAGES['CAS-09-2014-A002']),
                             {'CAS-09-2014-A002': PASSAGES['CAS-09-2014-A002']})
    assert status.startswith('上下文已含全部')


def test_missing_text_is_reported_as_a_retrieval_problem():
    status = evidence_status(cited('CAS-37-2017.md', '别的条文'),
                             {'CAS-37-2017-A008': PASSAGES['CAS-37-2017-A008']},
                             PASSAGES, passage_sources(PASSAGES))
    assert '检索没给到' in status


def test_questions_without_labels_say_so():
    assert evidence_status(cited('a.md', 'x'), {}).startswith('该题无标注要点')


def test_passage_sources_maps_the_id_prefix_to_the_file():
    assert passage_sources({'CAS-37-2017-A008': 'x'}) == {'CAS-37-2017-A008': 'CAS-37-2017.md'}
    assert passage_sources({'D3-S03': 'x'}) == {'D3-S03': 'D3-S03.md'}
