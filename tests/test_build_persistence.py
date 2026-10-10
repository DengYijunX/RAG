"""Tests for the build step that publishes an index.

The bug this guards: ``build()`` never called ``backend.flush()``. Chroma writes
on every ``add()`` so its flush is a no-op and nothing showed up, but the FAISS
backend keeps the index in memory until flush — so the published generation held
only its manifest, and every query answered "insufficient evidence" instead of
failing. Making the backend swappable is what exposed it.

Vectors are injected so these run without a network call and without a native
index: the behaviour under test is persistence and atomic publication, not
embedding quality.
"""
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from rag.backends import FaissVectors, build_backend
from rag.kb_service import KnowledgeBaseService
from tests.test_store import get_test_config

CORPUS = Path(__file__).resolve().parents[1] / 'test_docs' / 'third-corpus'
DIMENSIONS = 8


def fake_vectors(texts):
    """Deterministic unit-ish vectors: same text, same vector, every run."""
    output = []
    for text in texts:
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        output.append([1.0 + digest[index % len(digest)] for index in range(DIMENSIONS)])
    return output


@pytest.fixture
def service(tmp_path, monkeypatch):
    instance = KnowledgeBaseService(replace(get_test_config('.'), vector_backend='faiss'),
                                    storage_root=tmp_path)
    monkeypatch.setattr(instance, 'vectors', fake_vectors)
    return instance


def build(service):
    base = service.register('kb')
    service.import_sources(base.id, CORPUS)
    return base, service.build(base.id)


def test_build_persists_the_index_so_it_survives_a_new_backend(service):
    """A second backend object must see the vectors, not an empty generation."""
    base, info = build(service)
    directory = service.base_dir(base.id) / info['generation']
    assert (directory / 'index.faiss').is_file()
    assert (directory / 'records.jsonl').is_file()
    assert (directory / 'backend.json').is_file()
    assert build_backend('faiss', directory).count() == info['chunks']


def test_a_built_index_can_actually_be_retrieved_from(service):
    """The end-to-end symptom: the build succeeded but every query found nothing."""
    base, info = build(service)
    hits = service.retrieve(base.id, '怎么给植物浇水？', 3)
    assert hits, '建索引后必须能检索到内容'
    assert len(hits) <= 3
    assert all(hit['content'] for hit in hits)


def test_a_query_matches_the_chunk_holding_its_text(service):
    """Sanity: with one-hot-ish vectors the nearest chunk is the related one."""
    base, _ = build(service)
    hits = service.retrieve(base.id, '怎么给植物浇水？', 1)
    assert '浇水' in hits[0]['content'] or 'water' in hits[0]['content'].lower()


def test_the_published_pointer_is_written_only_after_a_successful_flush(service):
    """A failed publish must leave the previous index serving."""
    base, first = build(service)

    class BrokenFlush(FaissVectors):
        def flush(self):
            raise RuntimeError('磁盘写失败')

    service.vector_backend = BrokenFlush
    with pytest.raises(Exception):
        service.build(base.id)
    assert service.active(base.id)['generation'] == first['generation']


def test_a_failed_publish_leaves_no_generation_directory(service):
    base, _ = build(service)
    before = {path.name for path in service.base_dir(base.id).iterdir()}

    class BrokenFlush(FaissVectors):
        def flush(self):
            raise RuntimeError('磁盘写失败')

    service.vector_backend = BrokenFlush
    with pytest.raises(Exception):
        service.build(base.id)
    assert {path.name for path in service.base_dir(base.id).iterdir()} == before


def test_an_add_failure_also_leaves_the_pointer_alone(service):
    base, first = build(service)

    class BrokenAdd(FaissVectors):
        def add(self, **kwargs):
            raise RuntimeError('写入失败')

    service.vector_backend = BrokenAdd
    with pytest.raises(Exception):
        service.build(base.id)
    assert service.active(base.id)['generation'] == first['generation']


def test_the_chroma_backend_still_satisfies_the_interface():
    """It is the alternative deployment, so it must keep the same four methods."""
    from rag.chroma_backend import ChromaVectors

    for name in ('add', 'search', 'count', 'flush'):
        assert callable(getattr(ChromaVectors, name)), name


def test_the_search_after_reopen_matches_the_search_before(service):
    """What the product serves must not depend on whether the process restarted."""
    base, info = build(service)
    before = service.retrieve(base.id, '怎么给植物浇水？', 3)
    directory = service.base_dir(base.id) / info['generation']
    reopened = build_backend('faiss', directory)
    vector = fake_vectors(['怎么给植物浇水？'])[0]
    after = reopened.search(vector, 3)
    assert [hit['id'] for hit in before] == [hit['id'] for hit in after]
