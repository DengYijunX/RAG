"""Tests for the selectable vector backend.

The default changed from Chroma to FAISS because Chroma's HNSW crashes this
interpreter above roughly 96 vectors at 1024 dimensions, which meant the
product's own index could not be built here. Both backends must satisfy the same
interface, and the choice must not change what is retrieved — that equality was
measured before the default moved, and these tests keep it from regressing
silently.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag.backends import FaissVectors, NormalizationError, build_backend, _normalise
from rag.kb_service import KnowledgeBaseService
from tests.test_store import get_test_config


def unit(rows):
    return _normalise(rows).tolist()


IDS = ['c1', 'c2', 'c3']
DOCUMENTS = ['甲', '乙', '丙']
METADATA = [{'source': 'a.md'}, {'source': 'a.md'}, {'source': 'b.md'}]
RAW = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]


def test_the_default_backend_is_faiss():
    assert get_test_config('.').vector_backend == 'faiss'
    assert isinstance(build_backend('faiss', '.'), FaissVectors)


def test_an_unknown_backend_is_refused_rather_than_defaulted():
    """Silently falling back would hide a typo in a deployment's configuration."""
    with pytest.raises(ValueError, match='未知的向量后端'):
        build_backend('pgvector', '.')


@pytest.mark.parametrize('name', ['faiss', 'FAISS', 'faiss-flat-ip', 'exact'])
def test_backend_names_are_case_insensitive_and_aliased(name):
    assert isinstance(build_backend(name, '.'), FaissVectors)


def test_add_search_count_round_trip(tmp_path):
    backend = FaissVectors(tmp_path)
    backend.add(IDS, unit(RAW), DOCUMENTS, METADATA)
    assert backend.count() == 3
    backend.flush()
    hits = backend.search(unit([[3.0, 1.0, 0.0]])[0], 3)
    assert [hit['id'] for hit in hits] == ['c3', 'c1', 'c2']
    assert hits[0]['content'] == '丙'
    assert hits[0]['source'] == 'b.md'


def test_searching_an_index_that_was_never_built_returns_nothing(tmp_path):
    """Parity with the Chroma backend, which returns [] for a missing collection.

    Raising here would make swapping the backend turn an empty knowledge base
    into a crash instead of the "no evidence" answer the product already has.
    """
    assert FaissVectors(tmp_path).search([1.0, 0.0], 5) == []


def test_a_reopened_index_with_records_but_no_vectors_returns_nothing(tmp_path):
    """A generation whose vectors were never written must answer with nothing.

    Covers the case the two "empty" paths differ on: records.jsonl exists (so the
    backend has chunk text) but index.faiss does not, so there is nothing to
    search. Chroma behaves the same way for a missing collection.
    """
    (tmp_path / 'records.jsonl').write_text(
        json.dumps({'id': 'c1', 'document': '甲', 'metadata': {'source': 'a.md'}},
                   ensure_ascii=False) + '\n', encoding='utf-8')
    backend = FaissVectors(tmp_path)
    assert backend.count() == 1
    assert backend.search([1.0, 0.0], 5) == []
    assert backend.search([1.0, 0.0], 0) == []


def test_a_reopened_backend_serves_the_same_order(tmp_path):
    first = FaissVectors(tmp_path)
    first.add(IDS, unit(RAW), DOCUMENTS, METADATA)
    first.flush()
    reopened = FaissVectors(tmp_path)
    hits = reopened.search(unit([[3.0, 1.0, 0.0]])[0], 3)
    assert [hit['id'] for hit in hits] == ['c3', 'c1', 'c2']


def test_the_index_records_that_it_is_exact(tmp_path):
    backend = FaissVectors(tmp_path)
    backend.add(IDS, unit(RAW), DOCUMENTS, METADATA)
    backend.flush()
    meta = json.loads((tmp_path / 'backend.json').read_text(encoding='utf-8'))
    assert meta['exact'] is True and meta['metric'] == 'cosine'


def test_the_query_is_normalised_so_length_does_not_matter(tmp_path):
    backend = FaissVectors(tmp_path)
    backend.add(IDS, unit(RAW), DOCUMENTS, METADATA)
    backend.flush()
    short = backend.search(unit([[3.0, 1.0, 0.0]])[0], 3)
    long = backend.search(unit([[30.0, 10.0, 0.0]])[0], 3)
    assert [hit['id'] for hit in short] == [hit['id'] for hit in long]


def test_stored_vectors_are_normalised_on_the_way_in(tmp_path):
    """A long stored vector must not outrank a short one by magnitude alone."""
    backend = FaissVectors(tmp_path)
    backend.add(IDS, [[100.0, 0.0, 0.0], [0.0, 0.1, 0.0], [1.0, 0.0, 0.0]],
                DOCUMENTS, METADATA)
    backend.flush()
    hits = backend.search([1.0, 0.0, 0.0], 3)
    assert hits[0]['distance'] == pytest.approx(0.0, abs=1e-6)


def test_distance_is_one_minus_cosine(tmp_path):
    backend = FaissVectors(tmp_path)
    backend.add(['only'], [[1.0, 0.0]], ['甲'], [{'source': 'a.md'}])
    backend.flush()
    hit = backend.search([1.0, 0.0], 1)[0]
    assert hit['distance'] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('vectors', [[[0.0, 0.0]], [[float('nan'), 1.0]],
                                     [[float('inf'), 1.0]]])
def test_degenerate_vectors_are_refused(vectors):
    """Scores from a zero or non-finite vector would be meaningless."""
    with pytest.raises(NormalizationError):
        _normalise(vectors)


def test_a_dimension_mismatch_is_refused(tmp_path):
    backend = FaissVectors(tmp_path)
    backend.add(['a'], [[1.0, 0.0]], ['甲'], [{'source': 'x'}])
    with pytest.raises(ValueError, match='维度不一致'):
        backend.add(['b'], [[1.0, 0.0, 0.0]], ['乙'], [{'source': 'x'}])


def test_flushing_without_an_index_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        FaissVectors(tmp_path).flush()


def test_the_service_uses_the_configured_backend(tmp_path):
    service = KnowledgeBaseService(replace(get_test_config('.'), vector_backend='faiss'),
                                   storage_root=tmp_path)
    assert isinstance(service.vector_backend(tmp_path / 'gen'), FaissVectors)


def test_an_explicit_factory_still_wins_over_configuration(tmp_path):
    """Tests and the acceptance check inject a backend; that must keep working."""
    marker = object()
    service = KnowledgeBaseService(get_test_config('.'), storage_root=tmp_path,
                                   vector_backend=lambda directory: marker)
    assert service.vector_backend(tmp_path / 'gen') is marker


def test_importing_the_product_package_does_not_import_chromadb():
    """A FAISS deployment must not pay for Chroma, nor break if it is absent."""
    import subprocess
    import sys

    code = ('import sys, rag.kb_service, rag.backends; '
            'print(any(n == "chromadb" or n.startswith("chromadb.") for n in sys.modules))')
    output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == 'False', output.stdout
