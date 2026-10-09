"""Tests for BM25, tokenisation and reciprocal rank fusion.

The retrieval mode comparison in the report is only as trustworthy as these
primitives, so the expectations here are hand-computed rather than captured
from a previous run.
"""
import math

import pytest

from hybrid_retrieval import (BM25Index, HybridRetriever, RRF_K, reciprocal_rank_fusion,
                              tokenize)


def test_cjk_runs_become_bigrams_and_ascii_stays_whole():
    assert tokenize('第三十五条') == ['第三', '三十', '十五', '五条']
    assert tokenize('CAS-14-2017') == ['cas-14-2017']
    assert tokenize('支持 2.4GHz 频段') == ['2.4ghz', '支持', '频段']
    assert tokenize('') == [] and tokenize(None) == []


def test_single_cjk_character_is_kept():
    assert tokenize('甲') == ['甲']


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index().add(['a', 'b', 'c'], [
        '第35号 分部报告 第五条 业务分部',
        '第14号 收入 第十一条 履约义务',
        '完全无关的一段文字',
    ])
    assert index.search('分部报告', 3)[0] == 'a'
    assert index.search('履约义务', 3)[0] == 'b'
    # A query with no shared token returns nothing rather than an arbitrary doc.
    assert index.search('量子力学', 3) == []


def test_bm25_scores_are_positive_and_ordered():
    # Same term, shorter document wins: length normalisation is working.
    short = BM25Index().add(['a', 'b'], ['收入', '收入 确认 收入'])
    assert [identifier for identifier, _ in short.score('收入')] == ['a', 'b']
    assert short.score('收入')[0][1] > short.score('收入')[1][1] > 0
    # More occurrences of the term win when document length is equal.
    repeated = BM25Index().add(['a', 'b'], ['收入 确认', '收入'])
    assert [identifier for identifier, _ in repeated.score('收入')] == ['b', 'a']


def test_identical_documents_break_ties_deterministically():
    index = BM25Index().add(['b', 'a'], ['同样 文本', '同样 文本'])
    assert [identifier for identifier, _ in index.score('同样')] == ['a', 'b']


def test_empty_index_and_empty_query_are_safe():
    assert BM25Index().search('问题', 5) == []
    index = BM25Index().add(['a'], ['文本'])
    assert index.search('', 5) == []


def test_rrf_uses_rank_not_score():
    # 'x' is first in one list, second in the other; 'y' is first in one list only.
    fused = reciprocal_rank_fusion([['x', 'y'], ['z', 'x']])
    assert fused[0] == 'x'                       # appears in both
    expected_x = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    assert fused.index('x') == 0
    assert expected_x > 1 / (RRF_K + 1)          # beats a single first place
    assert set(fused) == {'x', 'y', 'z'}


def test_rrf_limit_and_weights():
    assert reciprocal_rank_fusion([['a', 'b'], ['b', 'a']], limit=1) == ['a']
    # Weighting the keyword arm to zero leaves the dense order intact.
    assert reciprocal_rank_fusion([['a', 'b'], ['b', 'a']], weights=[1.0, 0.0]) == ['a', 'b']
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([['a']], weights=[1.0, 1.0])


def test_rrf_document_found_by_only_one_arm_still_surfaces():
    """The point of fusion: a keyword-only hit must not be discarded."""
    fused = reciprocal_rank_fusion([['dense1', 'dense2'], ['keyword_only']], limit=3)
    assert 'keyword_only' in fused


def test_hybrid_retriever_keeps_both_arms_and_fuses():
    bm25 = BM25Index().add(['k1', 'shared'], ['关键词 命中', '共同 命中'])

    def dense_search(question, depth):
        return [{'id': 'd1'}, {'id': 'shared'}][:depth]

    retriever = HybridRetriever(dense_search=dense_search, bm25=bm25)
    dense, keyword = retriever.arms('共同 命中', 2)
    assert dense == ['d1', 'shared']
    assert keyword[0] == 'shared'
    fused = retriever.ranking('共同 命中', 2, 2)
    assert fused[0] == 'shared'          # in both arms, so it outranks 'd1'


def test_hybrid_depth_limits_the_candidate_pool():
    bm25 = BM25Index().add(['a', 'b', 'c'], ['词 一', '词 二', '词 三'])

    def dense_search(question, depth):
        return [{'id': f'd{index}'} for index in range(depth)]

    retriever = HybridRetriever(dense_search=dense_search, bm25=bm25)
    # depth 5 -> five dense ids and three keyword ids, all fused.
    assert len(retriever.ranking('词', 5, 10)) == 8
    assert len(retriever.ranking('词', 5, 2)) == 2      # limit applies after fusion
    # depth 1 -> one dense id plus the single best keyword hit.
    assert len(retriever.ranking('词', 1, 10)) == 2
    assert len(retriever.arms('词', 1)[1]) == 1
