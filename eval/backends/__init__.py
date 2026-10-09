"""Storage backends used only by evaluation.

The product's default backend is Chroma (see ``rag.kb_service.ChromaVectors``).
Backends here exist so the evaluation loop can run on machines where the
product's index cannot be built, without touching product code.
"""
from eval.backends.faiss_store import FaissVectors, NormalizationError  # noqa: F401

__all__ = ['FaissVectors', 'NormalizationError']
