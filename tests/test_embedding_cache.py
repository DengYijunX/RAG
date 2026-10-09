"""Tests that an embedding cache cannot survive a model change.

Mixing vectors from two embedding models in one index produces a silently
meaningless result — retrieval still returns something, so nothing looks
broken. These tests pin the guard.
"""
import json
from types import SimpleNamespace

import pytest

from retrieval_baseline import EmbeddingCache, embedding_fingerprint
from tests.test_store import get_test_config

FINGERPRINT = {'model': 'model-a', 'base_url': 'https://a.test/v1', 'dimensions': None}


class FakeService:
    """Returns a distinct vector per text and counts provider calls."""

    def __init__(self, batch_size=None):
        self.calls = []
        self.batch_size = batch_size

    def embed_batched(self, texts, batch_size=None):
        size = batch_size or self.batch_size or len(texts)
        for offset in range(0, len(texts), size):
            self.calls.append(list(texts[offset:offset + size]))
        return [[float(len(text)), 1.0] for text in texts]


def test_vectors_are_computed_once_and_reused(tmp_path):
    cache = EmbeddingCache(tmp_path / 'c.json', FINGERPRINT)
    service = FakeService()
    first = cache.vectors(service, ['甲', '乙'])
    cache.save()
    assert service.calls == [['甲', '乙']]

    reopened = EmbeddingCache(tmp_path / 'c.json', FINGERPRINT)
    again = reopened.vectors(service, ['甲', '乙'])
    assert again == first
    assert service.calls == [['甲', '乙']]        # no second provider call
    assert len(reopened) == 2


def test_a_different_model_discards_the_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / 'c.json', FINGERPRINT)
    service = FakeService()
    cache.vectors(service, ['甲'])
    cache.save()

    changed = EmbeddingCache(tmp_path / 'c.json',
                             {**FINGERPRINT, 'model': 'model-b'})
    assert len(changed) == 0
    assert 'model_or_url_changed' in changed.discarded_reason
    changed.vectors(service, ['甲'])
    assert service.calls == [['甲'], ['甲']]      # recomputed, not reused


def test_a_different_base_url_discards_the_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / 'c.json', FINGERPRINT)
    cache.vectors(FakeService(), ['甲'])
    cache.save()
    changed = EmbeddingCache(tmp_path / 'c.json',
                             {**FINGERPRINT, 'base_url': 'https://b.test/v1'})
    assert len(changed) == 0


def test_a_legacy_text_only_cache_is_refused(tmp_path):
    """The old format was a bare text->vector map; it must not be adopted."""
    path = tmp_path / 'c.json'
    path.write_text(json.dumps({'甲': [1.0, 2.0]}), encoding='utf-8')
    cache = EmbeddingCache(path, FINGERPRINT)
    assert len(cache) == 0 and cache.discarded_reason == 'cache_has_no_fingerprint'


def test_an_unreadable_cache_is_refused_not_fatal(tmp_path):
    path = tmp_path / 'c.json'
    path.write_text('{not json', encoding='utf-8')
    cache = EmbeddingCache(path, FINGERPRINT)
    assert len(cache) == 0 and cache.discarded_reason == 'cache_file_unreadable'


def test_saved_payload_records_the_fingerprint(tmp_path):
    path = tmp_path / 'c.json'
    cache = EmbeddingCache(path, FINGERPRINT)
    cache.vectors(FakeService(), ['甲'])
    cache.save()
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['fingerprint'] == FINGERPRINT
    assert set(payload) == {'fingerprint', 'entries'}


def test_fingerprint_uses_the_configured_model_and_endpoint():
    config = get_test_config('.')
    fingerprint = embedding_fingerprint(config)
    assert fingerprint['model'] == config.embedding_model
    assert fingerprint['base_url'] == config.embedding_base_url


def test_a_short_response_is_an_error_not_a_silent_gap():
    class ShortService:
        def embed_batched(self, texts, batch_size=None):
            return [[1.0]]

    cache = EmbeddingCache('unused.json', FINGERPRINT)
    with pytest.raises(RuntimeError):
        cache.vectors(ShortService(), ['甲', '乙'])


def test_requests_are_split_into_batches(tmp_path):
    """Providers cap the inputs per request; a rejected batch fails the build."""
    cache = EmbeddingCache(tmp_path / 'c.json', FINGERPRINT)
    service = FakeService(batch_size=3)
    vectors = cache.vectors(service, ['甲', '乙', '丙', '丁', '戊'])
    assert len(vectors) == 5
    assert [len(call) for call in service.calls] == [3, 2]


def test_embed_batched_splits_and_preserves_order():
    from dataclasses import replace

    from rag.kb_service import KnowledgeBaseService
    from tests.test_store import get_test_config

    seen = []

    def fake_vectors(self, texts):
        seen.append(len(texts))
        return [[float(text)] for text in texts]

    service = KnowledgeBaseService(get_test_config('.'), storage_root='.')
    service.vectors = fake_vectors.__get__(service, KnowledgeBaseService)
    output = service.embed_batched([1, 2, 3, 4, 5], batch_size=2)
    assert seen == [2, 2, 1]
    assert output == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert replace(service.config).embedding_model == service.config.embedding_model


def test_embed_batched_rejects_a_nonsense_batch_size():
    from rag.kb_service import KnowledgeBaseService
    from tests.test_store import get_test_config

    service = KnowledgeBaseService(get_test_config('.'), storage_root='.')
    with pytest.raises(ValueError):
        service.embed_batched(['甲'], batch_size=0)
