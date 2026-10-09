"""Tests for the rerank wiring in the mode-comparison script.

The expensive part (the model call) is replaced by a stub, so these cover the
plumbing: candidate text must reach the reranker as text, the reranked head must
lead the order, and the untouched tail must keep recall from dropping.
"""
import pytest

from hybrid_retrieval import BM25Index, HybridRetriever
from retrieval_modes_experiment import chunk_text, ranking_for


class StubReranker:
    """Reverses the candidate order and records what it was handed."""

    def __init__(self):
        self.seen = []

    def rerank(self, question, candidates):
        self.seen.append((question, candidates))
        from llm_rerank import RerankOutcome
        ordered = list(reversed(candidates))
        return ordered, RerankOutcome(order=[c['id'] for c in ordered], applied=True, reason='ok')


def build_retriever():
    bm25 = BM25Index().add(['k1', 'k2'], ['关键词 甲', '关键词 乙'])

    def dense_search(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    return HybridRetriever(dense_search=dense_search, bm25=bm25), bm25


BY_ID = {f'd{index}': {'id': f'd{index}', 'content': f'正文 {index}'} for index in range(10)}
BY_ID.update({'k1': {'id': 'k1', 'content': '关键词 甲'}, 'k2': {'id': 'k2', 'content': '关键词 乙'}})


def test_chunk_text_accepts_strings_and_records():
    assert chunk_text('直接文本') == '直接文本'
    assert chunk_text({'id': 'x', 'content': '记录文本'}) == '记录文本'
    assert chunk_text(None) == ''
    assert chunk_text({'id': 'x'}) == ''
    # A record handed in as the text must not leak through as a dict.
    assert isinstance(chunk_text(BY_ID['d0']), str)


def test_rerank_mode_hands_text_not_records_to_the_reranker():
    retriever, bm25 = build_retriever()
    stub = StubReranker()

    def dense_hits(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    ranking_for('rerank', '问题', 10, dense_hits, bm25, retriever,
                by_id=BY_ID, reranker=stub, pool=4)
    question, payload = stub.seen[0]
    assert question == '问题'
    assert len(payload) == 4
    for candidate in payload:
        assert isinstance(candidate['content'], str)
        assert candidate['content']


def test_reranked_head_leads_and_the_tail_is_preserved():
    retriever, bm25 = build_retriever()
    stub = StubReranker()

    def dense_hits(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    # Depth 6, pool 3: the stub reverses ['d0','d1','d2'], so the head must be
    # ['d2','d1','d0'] and the untouched tail ['d3','d4','d5'] must follow.
    ids = ranking_for('rerank', '问题', 6, dense_hits, bm25, retriever,
                      by_id=BY_ID, reranker=stub, pool=3)
    assert ids[:3] == ['d2', 'd1', 'd0']
    assert ids[3:] == ['d3', 'd4', 'd5']
    assert len(ids) == len(set(ids))          # no duplicates across head and tail
    # Everything the hybrid arm returned is still present, so recall cannot drop.
    baseline = retriever.ranking('问题', 6, 6)
    assert set(ids) == set(baseline)


def test_pool_limits_how_many_candidates_the_model_sees():
    retriever, bm25 = build_retriever()
    stub = StubReranker()

    def dense_hits(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    ids = ranking_for('rerank', '问题', 10, dense_hits, bm25, retriever,
                      by_id=BY_ID, reranker=stub, pool=2)
    assert len(stub.seen[0][1]) == 2
    # The pool is what is reordered; the rest of the candidates still follow.
    assert ids[:2] == ['d1', 'd0']
    assert set(ids) == set(retriever.ranking('问题', 10, 10))


def test_without_a_reranker_rerank_mode_falls_back_to_hybrid():
    retriever, bm25 = build_retriever()

    def dense_hits(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    reranked = ranking_for('rerank', '问题', 5, dense_hits, bm25, retriever,
                           by_id=BY_ID, reranker=None, pool=5)
    hybrid = ranking_for('hybrid', '问题', 5, dense_hits, bm25, retriever)
    assert reranked == hybrid


def test_the_other_modes_ignore_the_reranker():
    retriever, bm25 = build_retriever()
    stub = StubReranker()

    def dense_hits(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    ranking_for('dense', '问题', 3, dense_hits, bm25, retriever,
                by_id=BY_ID, reranker=stub, pool=2)
    ranking_for('hybrid', '问题', 3, dense_hits, bm25, retriever,
                by_id=BY_ID, reranker=stub, pool=2)
    assert stub.seen == []
