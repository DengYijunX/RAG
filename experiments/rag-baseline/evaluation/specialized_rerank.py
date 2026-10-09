"""Reranking with a dedicated rerank endpoint.

The implementation now lives in the product package as ``rag.rerank``: the
product path needs it once deep-pool reranking is enabled, and keeping two
copies would let them drift. This module re-exports it so existing callers and
tests keep working.
"""
from rag.rerank import (RerankOutcome, HttpRerankClient, SpecializedReranker,  # noqa: F401
                        parse_rerank_response)
