"""Tests for slot-based retrieval metrics.

These exist because the de-duplicated passage list reports a passage found at
rank 9 as if it had been found early. A retriever that never puts the labelled
passage in the top results would otherwise look near-perfect.
"""
import pytest

from eval.metrics import (first_hit_slot, ndcg_in_top_k, precision_in_top_k, score_slot_metrics,
                          score_slots)


def test_passage_found_late_does_not_count_as_early():
    # The gold passage sits in the 9th retrieved chunk.
    slots = [[] for _ in range(8)] + [['P1']]
    assert score_slots(slots, ['P1'], k=4) == 0.0
    assert score_slots(slots, ['P1'], k=10) == 1.0
    assert first_hit_slot(slots, ['P1']) == 9


def test_passage_found_first_counts_immediately():
    slots = [['P1'], [], []]
    assert score_slots(slots, ['P1'], k=1) == 1.0
    assert first_hit_slot(slots, ['P1']) == 1


def test_multiple_gold_passages_partially_found():
    slots = [['P1'], [], ['P2'], []]
    assert score_slots(slots, ['P1', 'P2'], k=3) == 1.0
    assert score_slots(slots, ['P1', 'P2'], k=2) == 0.5
    assert first_hit_slot(slots, ['P1', 'P2']) == 1


def test_a_later_gold_passage_does_not_improve_the_first_rank():
    slots = [[], ['P2'], [], []]
    assert first_hit_slot(slots, ['P1', 'P2']) == 2
    assert score_slots(slots, ['P1', 'P2'], k=4) == 0.5


def test_unlabelled_questions_score_none():
    assert score_slots([[['P1']]], [], 4) is None
    assert first_hit_slot([[['P1']]], []) is None


def test_aggregate_reports_first_hit_and_mrr():
    per_question = [
        ([['P1']], ['P1']),        # first slot, rr 1.0
        ([[]], ['P1']),            # never found
        ([[], ['P1']], ['P1']),    # second slot, rr 0.5
        ([[]], []),                # unlabelled, excluded
    ]
    summary = score_slot_metrics(per_question, k=2)
    assert summary['scored'] == 3
    assert summary['recall_in_top_k'] == pytest.approx(2 / 3)
    assert summary['hit_rate_in_top_k'] == pytest.approx(2 / 3)
    assert summary['mean_first_hit_slot'] == pytest.approx((1 + 2) / 2)
    assert summary['mrr_first_hit'] == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_precision_counts_each_slot_once():
    # Three slots, two carry the gold passage, one carries an unrelated one.
    slots = [['P1'], ['P2'], []]
    assert precision_in_top_k(slots, ['P1'], 2) == 0.5
    assert precision_in_top_k(slots, ['P1'], 3) == pytest.approx(1 / 3)
    # A slot holding two gold passages still counts once for precision.
    assert precision_in_top_k([['P1', 'P2']], ['P1', 'P2'], 1) == 1.0


def test_ndcg_rewards_early_slots():
    early = [['P1'], [], [], []]
    late = [[], [], [], ['P1']]
    assert ndcg_in_top_k(early, ['P1'], 4) == 1.0
    assert ndcg_in_top_k(late, ['P1'], 4) < ndcg_in_top_k(early, ['P1'], 4)
    # A gold passage outside the cutoff earns nothing.
    assert ndcg_in_top_k(late, ['P1'], 3) == 0.0


def test_ndcg_ideal_uses_the_number_of_labels():
    # Two labels and two slots: filling both gives a perfect score.
    assert ndcg_in_top_k([['P1'], ['P2']], ['P1', 'P2'], 2) == 1.0
    # Four slots but only one label: hitting it first is already ideal.
    assert ndcg_in_top_k([['P1'], [], [], []], ['P1'], 4) == 1.0


def test_precision_and_ndcg_score_none_without_labels():
    assert precision_in_top_k([['P1']], [], 4) is None
    assert ndcg_in_top_k([['P1']], [], 4) is None
    summary = score_slot_metrics([([['P1']], [])], k=4)
    assert summary['precision_in_top_k'] is None and summary['ndcg_in_top_k'] is None


def test_aggregate_with_nothing_to_score():
    summary = score_slot_metrics([([[]], [])], k=4)
    assert summary['scored'] == 0
    assert summary['recall_in_top_k'] is None
    assert summary['mean_first_hit_slot'] is None


def test_the_gap_between_slot_and_deduplicated_scoring():
    """The exact failure this module was written to expose.

    A gold passage at rank 9: the de-duplicated list would place it first, so
    the naive metric says 1.0 while the slot metric says 0.0.
    """
    slots = [[] for _ in range(8)] + [['P1']]
    deduplicated = ['P1']            # what flattening + de-duplication yields
    assert score_slots(slots, ['P1'], k=4) == 0.0
    assert deduplicated[:4] == ['P1']  # the naive view claims rank 1
