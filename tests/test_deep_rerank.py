"""Tests for deep-pool reranking on the product path.

The switch must be inert when off, and the ordering must not change what
retrieval metrics see. Reranking only reorders the head; the tail follows
untouched so a small pool cannot cap recall.
"""
from dataclasses import replace

import pytest

from rag.config import load_config
from rag.kb_service import KnowledgeBaseService
from rag.rerank import SpecializedReranker, rerank_hits
from tests.test_store import get_test_config


def hits(count):
    return [{'id': f'c{index}', 'content': f'正文{index}', 'source': 'a.md'}
            for index in range(1, count + 1)]


class ReversingReranker:
    """Deterministic stand-in: reverses the candidate list."""

    def __init__(self):
        self.calls = 0
        self.seen = []

    def rerank(self, question, candidates):
        self.calls += 1
        self.seen.append([candidate['id'] for candidate in candidates])
        from rag.rerank import RerankOutcome

        reordered = list(reversed(candidates))
        return reordered, RerankOutcome(order=[c['id'] for c in reordered], applied=True)


def config(**overrides):
    base = get_test_config('.')
    return replace(base, **overrides)


def test_a_pool_of_zero_leaves_the_order_untouched():
    original = hits(5)
    assert [h['id'] for h in rerank_hits('问题', original, 0, config())] == [
        'c1', 'c2', 'c3', 'c4', 'c5']


def test_a_single_hit_is_not_sent_to_the_reranker():
    class Exploding:
        def rerank(self, question, candidates):
            raise AssertionError('不应被调用')

    assert rerank_hits('问题', hits(1), 20, config(), reranker=Exploding())[0]['id'] == 'c1'


def test_only_the_head_is_reranked_and_the_tail_follows():
    instance = ReversingReranker()
    output = rerank_hits('问题', hits(5), 3, config(), reranker=instance)
    assert instance.seen == [['c1', 'c2', 'c3']]
    # head reversed, tail untouched
    assert [hit['id'] for hit in output] == ['c3', 'c2', 'c1', 'c4', 'c5']
    assert len(output) == 5


def test_no_hit_is_lost_or_duplicated():
    output = rerank_hits('问题', hits(7), 4, config(), reranker=ReversingReranker())
    identifiers = [hit['id'] for hit in output]
    assert sorted(identifiers) == sorted(f'c{index}' for index in range(1, 8))


def test_the_switch_is_off_unless_a_pool_is_configured():
    assert config().deep_rerank_enabled is False
    assert config(rerank_pool=20, rerank_base_url='https://x.test/r',
                  rerank_api_key='sk-x', rerank_model='m').deep_rerank_enabled is True


def test_a_pool_without_credentials_is_not_enabled():
    """Switching a pool on must not fail every query when the key is missing."""
    assert config(rerank_pool=20).deep_rerank_enabled is False
    assert config(rerank_pool=20, rerank_api_key='sk-x').deep_rerank_enabled is False


def test_query_leaves_retrieval_alone_when_the_switch_is_off(monkeypatch):
    """With the pool at 0 the service must not construct a reranker at all."""
    calls = []
    monkeypatch.setattr('rag.rerank.SpecializedReranker',
                        lambda *a, **k: calls.append(1) or pytest.fail('不应构造重排器'))
    service = KnowledgeBaseService(config(), storage_root='.')
    monkeypatch.setattr(service, 'retrieve', lambda *a, **k: hits(6))
    monkeypatch.setattr(service, 'citations', lambda *a, **k: [])
    service.query('kb', '问题')
    assert calls == []


def test_query_asks_for_the_deep_pool_when_enabled(monkeypatch):
    requested = {}
    service = KnowledgeBaseService(
        config(rerank_pool=20, retrieval_depth=50, rerank_base_url='https://x.test/r',
               rerank_api_key='sk-x', rerank_model='m'), storage_root='.')

    def fake_retrieve(kb_id, question, top_k=4):
        requested['top_k'] = top_k
        return hits(50)

    monkeypatch.setattr(service, 'retrieve', fake_retrieve)
    monkeypatch.setattr('rag.kb_service.rerank_hits',
                        lambda question, hits_, pool, cfg: hits_[:4])
    monkeypatch.setattr(service, 'citations', lambda *a, **k: [])
    service.query('kb', '问题')
    assert requested['top_k'] == 50


def test_rerank_then_expand_is_the_documented_order(monkeypatch):
    """Expansion must run on the surviving top_k, not on the whole deep pool."""
    order = []
    service = KnowledgeBaseService(
        config(rerank_pool=20, rerank_base_url='https://x.test/r',
               rerank_api_key='sk-x', rerank_model='m'), storage_root='.')
    monkeypatch.setattr(service, 'retrieve', lambda *a, **k: hits(30))
    monkeypatch.setattr('rag.kb_service.rerank_hits',
                        lambda question, hits_, pool, cfg: order.append(
                            ('rerank', pool, len(hits_))) or hits_)
    monkeypatch.setattr(service, 'context_hits',
                        lambda kb, hits_, radius=None: order.append(
                            ('expand', len(hits_))) or hits_)
    monkeypatch.setattr(service, 'citations', lambda *a, **k: [])
    service.query('kb', '问题')
    assert order == [('rerank', 20, 30), ('expand', 4)]


def test_env_variables_are_read(monkeypatch):
    monkeypatch.setenv('RAG_RETRIEVAL_DEPTH', '50')
    monkeypatch.setenv('RAG_RERANK_POOL', '20')
    loaded = load_config()
    assert loaded.retrieval_depth == 50 and loaded.rerank_pool == 20
    assert loaded.deep_rerank_enabled == loaded.rerank_configured


def test_code_defaults_are_off(monkeypatch):
    """The switches are off unless something enables them.

    ``load_config`` reads ``.env``, so clearing the environment is not enough: a
    developer whose ``.env`` enables the pool would see this pass for the wrong
    reason, or fail for one. ``load_dotenv`` is stubbed out so the assertions are
    about the code's defaults, not about the local file.
    """
    for name in ('RAG_RETRIEVAL_DEPTH', 'RAG_RERANK_POOL', 'RAG_CONTEXT_RADIUS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr('rag.config.load_dotenv', lambda *args, **kwargs: False)
    loaded = load_config()
    assert loaded.retrieval_depth == 0 and loaded.rerank_pool == 0
    assert loaded.deep_rerank_enabled is False
    assert loaded.context_radius == 0


def test_the_pool_is_enabled_by_configuration(monkeypatch):
    """The switches must actually turn the feature on; off-by-default alone is not enough."""
    monkeypatch.setattr('rag.config.load_dotenv', lambda *args, **kwargs: False)
    monkeypatch.setenv('RAG_RETRIEVAL_DEPTH', '50')
    monkeypatch.setenv('RAG_RERANK_POOL', '20')
    loaded = load_config()
    assert loaded.retrieval_depth == 50
    assert loaded.rerank_pool == 20
    assert loaded.deep_rerank_enabled is loaded.rerank_configured


def test_the_backend_default_is_faiss(monkeypatch):
    """See docs/ARCHITECTURE_REFERENCES.md ADR-010 for why."""
    monkeypatch.delenv('RAG_VECTOR_BACKEND', raising=False)
    monkeypatch.setattr('rag.config.load_dotenv', lambda *args, **kwargs: False)
    assert load_config().vector_backend == 'faiss'
