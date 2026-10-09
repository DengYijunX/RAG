"""Neighbour expansion: assemble contiguous source text around a retrieved chunk.

The implementation now lives in the product package as ``rag.context`` — it is a
generic operation over index manifests, not an experiment detail, and keeping two
copies would let them drift. This module re-exports it and keeps the
evaluation-side helper that reads a published generation.

See [rag/context.py](../../../rag/context.py) for why expansion exists and what
it deliberately does not do.
"""
from pathlib import Path

from rag.context import Neighbourhood, NeighbourExpander, document_order  # noqa: F401


def load_manifest_chunks(path):
    """Read chunk records from a published evaluation generation."""
    import json

    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload['chunk_records'] if 'chunk_records' in payload else payload['chunks']

