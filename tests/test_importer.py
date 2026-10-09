import json
from pathlib import Path

import pytest

from rag.importer import (ImportReport, enumerate_sources, import_documents, media_type_for,
                          read_manifest, write_manifest)

MARKDOWN = """# 运营手册

## 进站

乘客持票进站，闸机开启。

## 出站

出站时刷卡扣费。
"""


def build_corpus(root, name='source'):
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'guide.md').write_text(MARKDOWN, encoding='utf-8')
    (directory / 'notes.txt').write_text('第一条 运营时间\n第二条 安全须知\n', encoding='utf-8')
    return directory


def test_media_types_and_unsupported_files(tmp_path):
    assert media_type_for('a.md') == 'text/markdown'
    assert media_type_for('a.TXT') == 'text/plain'
    assert media_type_for('a.pdf') == 'application/pdf'
    assert media_type_for('a.docx') is None
    directory = build_corpus(tmp_path)
    (directory / 'raw.docx').write_bytes(b'not really a docx')
    sources = enumerate_sources(directory)
    assert [s.relative for s in sources] == ['guide.md', 'notes.txt']
    assert all(len(s.sha256) == 64 and s.size > 0 for s in sources)


def test_import_produces_located_chunks(tmp_path):
    directory = build_corpus(tmp_path)
    report = import_documents('kb-1', directory)
    assert report.as_dict()['manifest_version'] == 1
    assert sorted(d['source'] for d in report.documents) == ['guide.md', 'notes.txt']
    guide_chunks = [chunk for chunk in report.chunks if chunk['source'] == 'guide.md']
    assert guide_chunks, 'guide.md must produce at least one chunk'
    text = MARKDOWN
    for chunk in guide_chunks:
        assert chunk['start'] is not None and chunk['start'] < chunk['end']
        assert text[chunk['start']:chunk['end']] == chunk['content']
    assert [c['chunk_index'] for c in guide_chunks] == list(range(len(guide_chunks)))


def test_document_identity_is_stable_and_distinct(tmp_path):
    directory = build_corpus(tmp_path)
    first = import_documents('kb-1', directory)
    second = import_documents('kb-1', directory)
    assert [d['id'] for d in first.documents] == [d['id'] for d in second.documents]
    assert all(d['id'].startswith('doc-') for d in first.documents)
    other = import_documents('kb-2', directory)
    assert {d['id'] for d in other.documents}.isdisjoint({d['id'] for d in first.documents})
    assert not {c['id'] for c in other.chunks} & {c['id'] for c in first.chunks}


def test_content_hash_detects_change(tmp_path):
    directory = build_corpus(tmp_path)
    before = import_documents('kb', directory).documents[0]
    (directory / 'guide.md').write_text(MARKDOWN + '\n## 续段\n\n新增内容。\n', encoding='utf-8')
    after = [d for d in import_documents('kb', directory).documents if d['source'] == 'guide.md'][0]
    assert before['id'] == after['id']
    assert before['content_sha256'] != after['content_sha256']
    assert after['sha256'] != before['sha256']


def test_markdown_sections_are_recorded(tmp_path):
    directory = build_corpus(tmp_path)
    record = [d for d in import_documents('kb', directory).documents if d['source'] == 'guide.md'][0]
    titles = [s['title'] for s in record['sections']]
    assert titles == ['运营手册', '进站', '出站']
    assert record['sections'][1]['path'][0] == record['sections'][0]['id']
    assert record['positions'] is True


def test_plain_text_has_no_markdown_structure(tmp_path):
    directory = build_corpus(tmp_path)
    record = [d for d in import_documents('kb', directory).documents if d['source'] == 'notes.txt'][0]
    assert record['parser'] == 'text'
    assert record['sections'] == []
    assert record['positions'] is True
    assert record['chunks'] >= 1


def test_empty_document_is_marked_not_positioned(tmp_path):
    directory = build_corpus(tmp_path)
    (directory / 'blank.md').write_text('   \n\n', encoding='utf-8')
    record = [d for d in import_documents('kb', directory).documents if d['source'] == 'blank.md'][0]
    assert record['empty'] is True
    assert record['positions'] is False
    assert record['chunks'] == 0


def test_pdf_reports_pages_without_inventing_positions(tmp_path):
    from PyPDF2 import PdfWriter

    directory = build_corpus(tmp_path)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(directory / 'manual.pdf', 'wb') as handle:
        writer.write(handle)
    record = [d for d in import_documents('kb', directory).documents if d['source'] == 'manual.pdf'][0]
    assert record['media_type'] == 'application/pdf'
    assert record['parser'] == 'pdf'
    assert record['positions'] is False
    assert record['pages'] == 2
    assert record['sections'] == []


def test_line_count_ignores_trailing_newline(tmp_path):
    directory = tmp_path / 'lines'
    directory.mkdir()
    (directory / 'a.md').write_text('# 标题\n\n正文\n', encoding='utf-8')
    record = import_documents('kb', directory).documents[0]
    assert record['lines'] == 3


def test_unsupported_files_are_reported_not_silently_dropped(tmp_path):
    directory = build_corpus(tmp_path)
    (directory / 'sheet.xlsx').write_bytes(b'not a real workbook')
    (directory / 'photo.png').write_bytes(b'\x89PNG')
    report = import_documents('kb', directory)
    skipped = {entry['source']: entry['reason'] for entry in report.skipped}
    assert skipped == {'photo.png': 'unsupported_media_type', 'sheet.xlsx': 'unsupported_media_type'}
    assert {d['source'] for d in report.documents} == {'guide.md', 'notes.txt'}


def test_unparsable_file_is_skipped_without_blocking(tmp_path):
    directory = build_corpus(tmp_path)
    (directory / 'broken.pdf').write_bytes(b'%PDF-1.4 this is not a real pdf')
    report = import_documents('kb', directory)
    assert [s['source'] for s in report.skipped] == ['broken.pdf']
    assert 'broken.pdf' not in [d['source'] for d in report.documents]
    assert report.documents


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        import_documents('kb', tmp_path / 'nope')


def test_manifest_round_trip_and_no_temp_left(tmp_path):
    directory = build_corpus(tmp_path)
    report = import_documents('kb', directory)
    target = write_manifest(report, tmp_path / 'out' / 'manifest.json')
    payload = read_manifest(target)
    assert payload['document_count'] == len(report.documents)
    assert payload['chunk_count'] == len(report.chunks)
    assert not list((tmp_path / 'out').glob('*.tmp'))
    assert json.loads(target.read_text(encoding='utf-8'))['documents'][0]['kb_id'] == 'kb'


def test_report_defaults():
    report = ImportReport()
    assert report.as_dict()['document_count'] == 0
