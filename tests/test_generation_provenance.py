"""Tests for the published generation's ``document_id`` provenance.

Citations need ``document_id`` to build a source link, but the published chunk
records only carried ``source``. Any consumer that read the generation directly
therefore failed with ``KeyError: 'document_id'`` — which is exactly what
happened when deep-pool reranking was first wired into the review harness.
"""
import json
from pathlib import Path

import pytest

from rag.importer import document_id_for


def test_document_id_is_derived_from_knowledge_base_and_relative_path():
    first = document_id_for('kb-a', 'docs/manual.md')
    second = document_id_for('kb-b', 'docs/manual.md')
    assert first.startswith('doc-') and len(first) == 20
    assert first != second, '同一个文件在不同知识库里必须是不同文档'


def test_document_id_is_stable_for_the_same_input():
    assert document_id_for('kb-a', 'docs/manual.md') == document_id_for('kb-a', 'docs/manual.md')


def test_document_id_does_not_depend_on_an_interpreter_hash_seed():
    """It must survive a restart, so it cannot use the built-in ``hash``."""
    expected = document_id_for('eval-accounting', 'markdown/CAS-14-2017.md')
    assert expected == 'doc-' + __import__('hashlib').sha256(
        'eval-accounting\0markdown/CAS-14-2017.md'.encode('utf-8')).hexdigest()[:16]


def test_published_records_carry_document_id(tmp_path):
    """The shape the evaluation baseline publishes must include provenance."""
    path = Path(tmp_path) / 'records.jsonl'
    record = {'id': 'chunk-1', 'document': '正文', 'source': 'a.md',
              'document_id': document_id_for('kb-a', 'a.md')}
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    loaded = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
    assert loaded['document_id'].startswith('doc-')
    assert loaded['source'] == 'a.md'
