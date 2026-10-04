from dataclasses import replace
import json

import pytest
from fastapi.testclient import TestClient

from rag.knowledge import KnowledgeService, IndexUnavailable, DATA, normalize
from rag.knowledge_routing import route_accounting
from tests.test_store import get_test_config


@pytest.fixture
def service(tmp_path, monkeypatch):
    instance = KnowledgeService(get_test_config(str(tmp_path)))
    monkeypatch.setattr(instance, 'vectors', lambda texts: [[1., float(len(t) % 7), 0.] for t in texts])
    monkeypatch.setattr(instance, 'generate', lambda question, citations: '测试回答 [1]')
    return instance


def test_two_real_indexes_are_isolated_and_rebuild_is_atomic(service, monkeypatch):
    first = service.build('smart-home')
    service.build('accounting')
    for kb, expected in [('smart-home', 26), ('accounting', 749)]:
        col = service.collection(kb, service.active(kb))
        assert col.count() == expected
        assert all(key.startswith(kb+':') for key in col.get()['ids'])
    monkeypatch.setattr(service, 'vectors', lambda texts: (_ for _ in ()).throw(RuntimeError('batch failed')))
    with pytest.raises(RuntimeError):
        service.build('smart-home')
    assert service.active('smart-home')['generation'] == first['generation']
    assert service.collection('accounting', service.active('accounting')).count() == 749


def test_model_change_invalidates_index(service):
    service.build('smart-home')
    changed = KnowledgeService(replace(service.config, embedding_model='different-model'))
    with pytest.raises(IndexUnavailable):
        changed.active('smart-home')


def test_explicit_article_and_citations_use_only_correct_source(service):
    service.build('accounting')
    response = service.query('accounting', '请引用2014年第9号职工薪酬准则第二条')
    citation = response['citations'][0]
    assert citation['source'] == 'CAS-09-2014.md'
    assert citation['article_labels'] == ['第二条']
    assert citation['version_year'] == 2014
    assert '其他长期职工福利' in citation['content']
    assert normalize(citation['content']) in normalize(service.source('accounting', citation['source_id'])['content'])
    with pytest.raises(KeyError):
        service.source('smart-home', citation['source_id'])


@pytest.mark.parametrize('kb,question,status', [
    ('accounting', '收入准则第二条', 'clarify'),
    ('accounting', '2025年收入准则第四条', 'insufficient_evidence'),
    ('accounting', '2017年收入准则第999条', 'insufficient_evidence'),
    ('accounting', '2017年与2025年收入准则第四条', 'insufficient_evidence'),
    ('smart-home', '门锁滴滴响是不是没电了', 'clarify'),
    ('smart-home', 'X3如何重置', 'insufficient_evidence'),
])
def test_guards_do_not_call_models(service, monkeypatch, kb, question, status):
    def forbidden(*args):
        pytest.fail('Guard must not call model')
    monkeypatch.setattr(service, 'vectors', forbidden)
    monkeypatch.setattr(service, 'generate', forbidden)
    result = service.query(kb, question)
    assert result['status'] == status
    assert result['citations'] == []


def test_promoted_router_matches_recorded_dev_routes(service):
    path = DATA/'evaluation/hybrid_runs/accounting/20261004T155449Z-35487740/results.jsonl'
    documents = service.corpus('accounting').documents
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert route_accounting(record['question'], documents) == record['routing']


def test_http_contracts_missing_index_and_safe_provider_errors(service, monkeypatch):
    import rag.api as api
    monkeypatch.setattr(api, 'knowledge', service)
    with TestClient(api.app) as client:
        assert len(client.get('/knowledge-bases').json()) == 2
        assert client.post('/knowledge-bases/missing/query', json={'question': 'hello'}).status_code == 404
        assert client.post('/knowledge-bases/accounting/query', json={'question': '  '}).status_code == 422
        assert client.post('/knowledge-bases/accounting/query', json={'question': 'a'*2001}).status_code == 422
        assert client.get('/knowledge-bases/accounting/sources/not-a-source').status_code == 404
        assert client.post('/knowledge-bases/smart-home/query', json={'question': 'X1换手机'}).status_code == 503
        monkeypatch.setattr(service, 'query', lambda *args: (_ for _ in ()).throw(RuntimeError('private-provider-detail')))
        response = client.post('/knowledge-bases/accounting/query', json={'question': 'hello'})
        assert response.status_code == 502 and 'private-provider-detail' not in response.text
