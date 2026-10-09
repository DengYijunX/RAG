"""HTTP contract tests for the generic knowledge-base lifecycle.

The full journey is driven through the API with an injected in-memory vector
store and a stubbed generator, so it proves the interface, not model quality.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.kb_service import InMemoryVectors, KnowledgeBaseService
from tests.test_kb_service import CORPUS, fake_vectors
from tests.test_store import get_test_config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import rag.api as api

    vectors = {}
    service = KnowledgeBaseService(
        get_test_config(str(tmp_path / 'store')), storage_root=tmp_path / 'knowledge',
        vector_backend=lambda directory: InMemoryVectors(directory, vectors))
    service.vectors = fake_vectors
    service.generate = lambda question, citations: '依据片段回答 [1]'
    monkeypatch.setattr(api, 'bases', service)
    return TestClient(api.app), service


def upload(client, kb, corpus=CORPUS):
    files = [('files', (name.split('/')[-1], text.encode('utf-8'), 'text/markdown'))
             for name, text in corpus.items()]
    return client.post(f'/knowledge-bases/{kb}/sources', files=files)


def test_create_list_import_index_query_source(client):
    http, _ = client
    created = http.post('/knowledge-bases', json={'name': '城市轨道交通运营',
                                                  'description': '第三语料'})
    assert created.status_code == 201
    kb = created.json()['id']

    listing = http.get('/knowledge-bases').json()
    entry = next(row for row in listing if row['id'] == kb)
    assert entry['name'] == '城市轨道交通运营' and entry['ready'] is False

    imported = upload(http, kb)
    assert imported.status_code == 200
    assert imported.json()['documents'] == len(CORPUS)
    assert imported.json()['chunks'] > 0

    assert http.post(f'/knowledge-bases/{kb}/index').status_code == 200
    ready = http.get(f'/knowledge-bases/{kb}').json()
    assert ready['ready'] is True and ready['chunk_count'] > 0

    answer = http.post(f'/knowledge-bases/{kb}/query', json={'question': '2号线首班车几点'})
    assert answer.status_code == 200
    payload = answer.json()
    assert payload['status'] == 'answer' and payload['citations']

    source = http.get(payload['citations'][0]['url'])
    assert source.status_code == 200
    assert source.json()['content']


def test_import_from_server_directory(client, tmp_path):
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '服务端目录导入'}).json()['id']
    directory = tmp_path / 'corpus'
    directory.mkdir()
    (directory / 'a.md').write_text('# 标题\n\n正文内容。\n', encoding='utf-8')
    response = http.post(f'/knowledge-bases/{kb}/sources',
                         data={'path': str(directory), 'glob': '**/*'})
    assert response.status_code == 200 and response.json()['documents'] == 1


def test_duplicate_name_conflicts(client):
    http, _ = client
    assert http.post('/knowledge-bases', json={'name': '同名库'}).status_code == 201
    duplicate = http.post('/knowledge-bases', json={'name': '同名库'})
    assert duplicate.status_code == 409
    assert http.post('/knowledge-bases', json={'name': '   '}).status_code == 422
    assert http.post('/knowledge-bases', json={'name': 'x' * 81}).status_code == 422


def test_index_without_import_is_reported(client):
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '空库'}).json()['id']
    response = http.post(f'/knowledge-bases/{kb}/index')
    assert response.status_code == 409
    assert '导入' in response.json()['detail']


def test_query_before_index_returns_503(client):
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '未建索引'}).json()['id']
    upload(http, kb)
    response = http.post(f'/knowledge-bases/{kb}/query', json={'question': '任意问题'})
    assert response.status_code == 503


def test_unknown_base_and_bad_input(client):
    http, _ = client
    assert http.get('/knowledge-bases/missing').status_code == 404
    assert http.post('/knowledge-bases/missing/sources', data={'path': '.'}).status_code == 404
    assert http.post('/knowledge-bases/missing/index').status_code == 404
    assert http.post('/knowledge-bases/missing/query', json={'question': 'hi'}).status_code == 404
    assert http.get('/knowledge-bases/missing/sources/abc').status_code == 404
    kb = http.post('/knowledge-bases', json={'name': '输入校验'}).json()['id']
    assert http.post(f'/knowledge-bases/{kb}/sources', data={}).status_code == 422
    assert http.post(f'/knowledge-bases/{kb}/sources', data={'path': 'Z:/nope'}).status_code == 422
    assert http.post(f'/knowledge-bases/{kb}/query', json={'question': '   '}).status_code == 422


def test_import_rejects_hidden_files_and_reports_unsupported(client):
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '文件校验'}).json()['id']
    hidden = http.post(f'/knowledge-bases/{kb}/sources',
                       files=[('files', ('.env', b'secret', 'text/plain'))])
    assert hidden.status_code == 422
    mixed = http.post(f'/knowledge-bases/{kb}/sources',
                      files=[('files', ('keep.md', '# 标题\n\n正文。\n'.encode('utf-8'), 'text/markdown')),
                             ('files', ('raw.docx', b'zzz', 'application/octet-stream'))])
    assert mixed.status_code == 200
    payload = mixed.json()
    assert payload['documents'] == 1
    assert payload['skipped'] == [{'source': 'raw.docx', 'reason': 'unsupported_media_type'}]


def test_import_rejects_duplicate_upload_names(client):
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '重名'}).json()['id']
    response = http.post(f'/knowledge-bases/{kb}/sources',
                         files=[('files', ('a.md', b'# A\n', 'text/markdown')),
                                ('files', ('a.md', b'# B\n', 'text/markdown'))])
    assert response.status_code == 422
    assert '重名' in response.json()['detail']
    assert http.get(f'/knowledge-bases/{kb}/sources').json() == []


def test_registered_bases_are_listed_without_fixture_data(client, tmp_path, monkeypatch):
    """The generic product path must not depend on the evaluation fixtures."""
    http, _ = client
    kb = http.post('/knowledge-bases', json={'name': '无夹具也要可见'}).json()['id']
    import rag.knowledge as knowledge_module
    monkeypatch.setattr(knowledge_module, 'DATA', tmp_path / 'no-such-fixtures')
    listing = http.get('/knowledge-bases')
    assert listing.status_code == 200
    assert [row['id'] for row in listing.json()] == [kb]


def test_sources_listing_unknown_base_is_404(client):
    http, _ = client
    assert http.get('/knowledge-bases/missing/sources').status_code == 404
    kb = http.post('/knowledge-bases', json={'name': '空来源列表'}).json()['id']
    assert http.get(f'/knowledge-bases/{kb}/sources').json() == []


def test_provider_errors_are_sanitized(client):
    http, service = client
    kb = http.post('/knowledge-bases', json={'name': '错误脱敏'}).json()['id']
    upload(http, kb)

    def explode(texts):
        raise RuntimeError('sk-private-provider-detail')

    service.vectors = explode
    response = http.post(f'/knowledge-bases/{kb}/index')
    assert response.status_code == 502
    assert 'sk-private-provider-detail' not in response.text


def test_builtin_fixtures_still_served(client):
    """The two evaluation fixtures keep their existing endpoints."""
    http, _ = client
    listing = http.get('/knowledge-bases').json()
    assert {row['id'] for row in listing} >= {'smart-home', 'accounting'}
