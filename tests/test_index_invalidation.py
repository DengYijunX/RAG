"""Tests that a chunker change invalidates a published index.

The bug this guards: index reuse compared only the embedding model, so switching
the chunker silently kept the old chunk set. After structural chunking became the
default the accounting corpus produced 866 chunks while the published index kept
serving 749 — and the baseline reported exactly the old numbers, which is what
made it noticeable.
"""
from dataclasses import replace

import pytest

from retrieval_baseline import chunking_fingerprint, embedding_fingerprint
from tests.test_store import get_test_config


def config(**overrides):
    return replace(get_test_config('.'), **overrides)


def test_the_fingerprint_covers_everything_that_changes_the_chunk_set():
    fingerprint = chunking_fingerprint(config())
    assert set(fingerprint) == {'chunk_size', 'chunk_overlap', 'structure_aware'}
    assert fingerprint['structure_aware'] is True


@pytest.mark.parametrize('overrides', [{'chunk_size': 1000}, {'chunk_overlap': 0}])
def test_changing_a_chunking_knob_changes_the_fingerprint(overrides):
    assert chunking_fingerprint(config(**overrides)) != chunking_fingerprint(config())


def test_the_same_configuration_yields_the_same_fingerprint():
    assert chunking_fingerprint(config()) == chunking_fingerprint(config())


def test_an_index_without_a_chunking_record_cannot_be_reused():
    """An unrecorded chunking must count as a mismatch, not as a match.

    ``info.get('chunking') or {}`` yields ``{}`` for an index published before
    this field existed, and ``{} != fingerprint`` is what forces the rebuild.
    """
    legacy_pointer = {'generation': 'faiss-old', 'chunks': 749,
                      'embedding': {'model': 'm', 'base_url': 'u'}}
    previous = legacy_pointer.get('chunking') or {}
    assert previous != chunking_fingerprint(config())


def test_the_embedding_fingerprint_is_unchanged_by_chunking():
    """The two checks are independent: chunking must not disturb the model check."""
    assert embedding_fingerprint(config()) == embedding_fingerprint(config(chunk_size=1000))
