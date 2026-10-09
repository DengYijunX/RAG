"""Tests for reading chunk records from a published generation.

The application and the evaluation baseline publish different files. A reader
that assumed one would work in one place and fail in the other — which is
exactly what happened when expansion was first wired in: the application's
``manifest.json`` exists, the baseline's generation only has ``records.jsonl``,
and expansion died with FileNotFoundError on every evaluated question.
"""
import json

import pytest

from rag.generation import load_chunk_records


def write_manifest(directory, records):
    path = directory / 'manifest.json'
    path.write_text(json.dumps({'chunk_records': records}, ensure_ascii=False), encoding='utf-8')
    return path


def write_records(directory, records):
    path = directory / 'records.jsonl'
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


def test_the_application_manifest_shape_is_read(tmp_path):
    write_manifest(tmp_path, [{'id': 'c1', 'source': 'a.md', 'content': '正文'}])
    records = load_chunk_records(tmp_path)
    assert records[0]['id'] == 'c1'
    assert records[0]['content'] == '正文'
    assert records[0]['source'] == 'a.md'


def test_the_baseline_records_shape_is_read(tmp_path):
    """The baseline names the text field ``document`` and nests ``source``."""
    write_records(tmp_path, [{'id': 'c1', 'document': '正文', 'metadata': {'source': 'a.md'}}])
    records = load_chunk_records(tmp_path)
    assert records[0]['content'] == '正文'
    assert records[0]['source'] == 'a.md'


def test_manifest_wins_when_both_files_exist(tmp_path):
    write_manifest(tmp_path, [{'id': 'manifest', 'source': 'a.md', 'content': '来自 manifest'}])
    write_records(tmp_path, [{'id': 'records', 'document': '来自 records',
                              'metadata': {'source': 'b.md'}}])
    assert load_chunk_records(tmp_path)[0]['id'] == 'manifest'


def test_a_missing_generation_raises_rather_than_looking_empty(tmp_path):
    """An empty list would make every hit look unexpandable, hiding the problem."""
    with pytest.raises(FileNotFoundError):
        load_chunk_records(tmp_path)


def test_a_manifest_without_chunk_records_raises(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'chunks': [{'id': 'c1'}]}),
                                           encoding='utf-8')
    assert load_chunk_records(tmp_path)[0]['id'] == 'c1'
    (tmp_path / 'manifest.json').write_text(json.dumps({'other': 1}), encoding='utf-8')
    with pytest.raises(KeyError):
        load_chunk_records(tmp_path)


def test_extra_fields_are_preserved(tmp_path):
    write_manifest(tmp_path, [{'id': 'c1', 'source': 'a.md', 'content': '正文',
                               'start': 10, 'end': 20}])
    record = load_chunk_records(tmp_path)[0]
    assert record['start'] == 10 and record['end'] == 20


def test_expansion_works_on_a_baseline_shaped_generation(tmp_path):
    """The end-to-end reason this module exists."""
    from rag.context import expand_hits

    write_records(tmp_path, [
        {'id': 'c1', 'document': '第一段', 'metadata': {'source': 'a.md'}},
        {'id': 'c2', 'document': '第二段', 'metadata': {'source': 'a.md'}},
    ])
    hits = [{'id': 'c2', 'source': 'a.md', 'content': '第二段'}]
    output = expand_hits(hits, load_chunk_records(tmp_path), radius=1)
    assert output[0]['content'] == '第一段\n第二段'
    assert output[0]['expanded_span'] == ['c1', 'c2']
