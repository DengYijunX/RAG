"""Metric tests against hand-computed values.

Each expectation below is worked out by hand in the comment above it, so a
formula change has to disagree with arithmetic rather than with a previous
run of the same code.
"""
import math

import pytest

from eval import metrics


def test_recall_precision_hit_rate_on_a_known_ranking():
    # ranked = [a, b, c, d], relevant = {b, d}, k = 3
    ranked, relevant = ['a', 'b', 'c', 'd'], ['b', 'd']
    assert metrics.recall_at_k(ranked, relevant, 3) == 0.5      # 1 of 2 found
    assert metrics.precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert metrics.hit_rate_at_k(ranked, relevant, 3) == 1.0
    assert metrics.recall_at_k(ranked, relevant, 1) == 0.0      # b not in top 1
    assert metrics.hit_rate_at_k(ranked, relevant, 1) == 0.0


def test_reciprocal_rank_positions():
    assert metrics.reciprocal_rank(['a', 'b', 'c'], ['b']) == 0.5     # rank 2
    assert metrics.reciprocal_rank(['a', 'b'], ['a']) == 1.0          # rank 1
    assert metrics.reciprocal_rank(['a', 'b'], ['c']) == 0.0          # never hit
    assert metrics.reciprocal_rank([], ['c']) == 0.0


def test_average_precision_matches_hand_calculation():
    # ranked = [r1, x, r2], relevant = {r1, r2, r3}
    # hits at positions 1 and 3 -> (1/1 + 2/3) / 3 = 0.5555...
    value = metrics.average_precision(['r1', 'x', 'r2'], ['r1', 'r2', 'r3'])
    assert value == pytest.approx((1.0 + 2 / 3) / 3)
    assert metrics.mean_average_precision(['r1'], ['r1']) == 1.0


def test_ndcg_perfect_and_reversed_orderings():
    # Perfect: relevant doc first -> ideal == actual -> 1.0
    assert metrics.ndcg_at_k(['r1', 'x'], ['r1'], 2) == 1.0
    # Reversed: relevant doc second.
    # dcg = 1/log2(3); idcg = 1/log2(2) = 1  ->  1/log2(3)
    value = metrics.ndcg_at_k(['x', 'r1'], ['r1'], 2)
    assert value == pytest.approx(1 / math.log2(3))


def test_ndcg_counts_a_repeated_hit_only_once():
    # The same relevant passage returned twice must not earn gain twice.
    twice = metrics.ndcg_at_k(['r1', 'r1'], ['r1'], 2)
    once = metrics.ndcg_at_k(['r1', 'x'], ['r1'], 2)
    assert twice == pytest.approx(once)


def test_ndcg_respects_graded_gains():
    # Two relevant passages, graded: r1 worth 2, r2 worth 1.
    # Ranking [r2, r1] should score below the ideal [r1, r2].
    gains = {'r1': 2.0, 'r2': 1.0}
    ideal = metrics.ndcg_at_k(['r1', 'r2'], ['r1', 'r2'], 2, gains=gains)
    swapped = metrics.ndcg_at_k(['r2', 'r1'], ['r1', 'r2'], 2, gains=gains)
    assert ideal == 1.0
    assert swapped < 1.0


def test_unlabelled_questions_score_none_not_zero():
    for function in (metrics.recall_at_k, metrics.precision_at_k, metrics.hit_rate_at_k):
        assert function(['a'], [], 3) is None
    assert metrics.reciprocal_rank(['a'], []) is None
    assert metrics.average_precision(['a'], []) is None
    assert metrics.ndcg_at_k(['a'], [], 3) is None


def test_empty_results_are_scored_as_misses():
    assert metrics.recall_at_k([], ['r1'], 3) == 0.0
    assert metrics.precision_at_k([], ['r1'], 3) == 0.0
    assert metrics.hit_rate_at_k([], ['r1'], 3) == 0.0


def test_hit_dicts_and_objects_are_accepted():
    hits = [{'id': 'r1', 'content': 'x'}, {'chunk_id': 'r2'}]
    assert metrics.recall_at_k(hits, ['r1', 'r2'], 2) == 1.0

    class Hit:
        def __init__(self, identifier):
            self.id = identifier

    assert metrics.reciprocal_rank([Hit('a'), Hit('r1')], ['r1']) == 0.5


def test_unplaceable_slots_keep_their_rank():
    """Regression: a None slot must occupy its rank, not be skipped.

    The resolver marks a hit it cannot place with None. If the metric skips
    that slot, the next result is promoted from rank 2 to rank 1 and MRR is
    inflated — exactly the silent improvement this project refuses.
    """
    assert metrics.reciprocal_rank([None, 'r1'], ['r1']) == 0.5
    assert metrics.recall_at_k([None, 'r1'], ['r1'], 1) == 0.0
    assert metrics.recall_at_k([None, 'r1'], ['r1'], 2) == 1.0
    assert metrics.precision_at_k([None, 'r1'], ['r1'], 2) == 0.5
    assert metrics.ndcg_at_k([None, 'r1'], ['r1'], 2) == pytest.approx(1 / math.log2(3))
    # An empty id is treated the same way, since both mean "no passage here".
    assert metrics.reciprocal_rank([{'id': ''}, {'id': 'r1'}], ['r1']) == 0.5


def test_score_dataset_reports_how_many_questions_were_scored():
    samples = [('q1', ['a', 'r1'], ['r1']),          # hit at rank 2
               ('q2', ['x'], ['r2']),                 # miss
               ('q3', ['y'], [])]                     # unlabelled, excluded
    summary = metrics.score_dataset(samples, k=2)
    assert summary['questions'] == 2
    assert summary['mrr'] == pytest.approx((0.5 + 0.0) / 2)
    assert summary['recall@k'] == pytest.approx(0.5)


def test_score_dataset_with_no_labelled_questions_returns_none_metrics():
    summary = metrics.score_dataset([('q1', ['a'], [])], k=3)
    assert summary['questions'] == 0
    assert summary['recall@k'] is None
    assert summary['mrr'] is None


def test_registry_registers_retrieval_metrics_only():
    registry = metrics.default_registry(k=2)
    assert registry.names() == ['hit_rate@k', 'map', 'mrr', 'ndcg@k', 'precision@k', 'recall@k']
    with pytest.raises(KeyError):
        registry.register('recal@k', lambda sample: 0.0)
    with pytest.raises(NotImplementedError):
        registry.register('faithfulness', lambda sample: 0.0)


def test_registry_compute_averages_and_skips_none():
    registry = metrics.default_registry(k=2)
    samples = [(['r1'], ['r1']), (['x'], [])]     # second sample unlabelled
    output = registry.compute(['recall@k', 'mrr'], samples)
    assert output == {'recall@k': 1.0, 'mrr': 1.0}
    with pytest.raises(KeyError):
        registry.compute(['ndcg@9'], samples)
