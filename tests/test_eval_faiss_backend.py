"""Tests for the FAISS evaluation backend.

FAISS is used because it works on this machine where the product's HNSW index
does not; these tests check that it behaves like a faithful vector store, and
that it refuses input that would make cosine similarity wrong.
"""
import math

import pytest

faiss = pytest.importorskip('faiss')
np = pytest.importorskip('numpy')

from eval.backends.faiss_store import FaissVectors, NormalizationError


def unit(*values):
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def make(tmp_path, count=3):
    store = FaissVectors(tmp_path)
    ids = [f'c{index}' for index in range(count)]
    vectors = [unit(1.0, index, 0.0) for index in range(count)]
    store.add(ids=ids, vectors=vectors, documents=[f'doc {index}' for index in range(count)],
              metadatas=[{'source': f's{index}.md'} for index in range(count)])
    return store, ids


def test_add_and_search_returns_nearest_first(tmp_path):
    store, ids = make(tmp_path, 4)
    assert store.count() == 4
    hits = store.search(unit(1.0, 2.0, 0.0), limit=2)
    assert [hit['id'] for hit in hits] == ['c2', ids[3]] or [hit['id'] for hit in hits] == ['c2', 'c3']
    assert hits[0]['content'] == 'doc 2'
    assert hits[0]['source'] == 's2.md'
    # distance is 1 - cosine, so the closest hit is the smallest
    assert hits[0]['distance'] < hits[1]['distance']


def test_exact_search_matches_brute_force_cosine(tmp_path):
    store, ids = make(tmp_path, 6)
    query = unit(2.0, 1.0, 3.0)
    hits = store.search(query, limit=6)
    manual = sorted(
        ((identifier, sum(a * b for a, b in zip(query, np.asarray(record).tolist())))
         for identifier, record in zip(ids, [unit(1.0, i, 0.0) for i in range(6)])),
        key=lambda item: -item[1])
    assert [hit['id'] for hit in hits] == [identifier for identifier, _ in manual]


def test_raw_vectors_are_normalised_internally(tmp_path):
    """Unnormalised input must still rank by cosine, not by magnitude."""
    store = FaissVectors(tmp_path)
    # Same direction, very different lengths: both must score as identical.
    store.add(ids=['a', 'b'], vectors=[[3.0, 4.0, 0.0], [30.0, 40.0, 0.0]],
              documents=['da', 'db'], metadatas=[{'source': 's'}, {'source': 's'}])
    hits = store.search([6.0, 8.0, 0.0], limit=2)
    assert {hit['id'] for hit in hits} == {'a', 'b'}
    assert all(hit['distance'] == pytest.approx(0.0, abs=1e-6) for hit in hits)


def test_degenerate_vectors_are_refused(tmp_path):
    store = FaissVectors(tmp_path)
    with pytest.raises(NormalizationError):
        store.add(ids=['c1'], vectors=[[0.0, 0.0, 0.0]], documents=['x'],
                  metadatas=[{'source': 's'}])
    with pytest.raises(NormalizationError):
        store.add(ids=['c1'], vectors=[[float('nan'), 1.0, 0.0]], documents=['x'],
                  metadatas=[{'source': 's'}])


def test_flush_persists_and_reloads(tmp_path):
    store, ids = make(tmp_path, 3)
    store.flush()
    assert (tmp_path / 'index.faiss').exists()
    assert (tmp_path / 'records.jsonl').exists()
    import json
    meta = json.loads((tmp_path / 'backend.json').read_text(encoding='utf-8'))
    assert meta == {'backend': 'faiss-flat-ip', 'dimensions': 3, 'vectors': 3,
                    'metric': 'cosine', 'exact': True}

    reopened = FaissVectors(tmp_path)
    assert reopened.count() == 3
    hits = reopened.search(unit(1.0, 1.0, 0.0), limit=1)
    assert hits[0]['id'] == 'c1' and hits[0]['content'] == 'doc 1'


def test_search_on_an_unwritten_index_reports_clearly(tmp_path):
    store = FaissVectors(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.search(unit(1.0, 0.0), limit=1)


def test_search_with_zero_limit_returns_nothing(tmp_path):
    store, _ = make(tmp_path, 2)
    assert store.search(unit(1.0, 0.0, 0.0), limit=0) == []


def test_dimension_mismatch_is_refused(tmp_path):
    store, _ = make(tmp_path, 2)
    with pytest.raises(ValueError):
        store.add(ids=['x'], vectors=[unit(1.0, 0.0)], documents=['x'],
                  metadatas=[{'source': 's'}])


def test_bigger_than_the_product_threshold(tmp_path):
    """The product's index dies around 96-128 vectors; this one must not.

    This is the whole reason the backend exists, so it is asserted directly.
    """
    count, dimension = 2000, 1024
    rng = np.random.RandomState(0)
    vectors = FaissVectors.normalize(rng.rand(count, dimension).astype('float32').tolist())
    store = FaissVectors(tmp_path)
    store.add(ids=[str(index) for index in range(count)], vectors=vectors,
              documents=[f'd{index}' for index in range(count)],
              metadatas=[{'source': 'fixture'} for _ in range(count)])
    assert store.count() == count
    # Querying with a known vector must return that vector's own id first.
    hits = store.search(vectors[7], limit=3)
    assert hits[0]['id'] == '7'
    assert hits[0]['distance'] == pytest.approx(0.0, abs=1e-5)
    store.flush()
    assert FaissVectors(tmp_path).count() == count
    assert FaissVectors(tmp_path).search(vectors[7], limit=1)[0]['id'] == '7'
