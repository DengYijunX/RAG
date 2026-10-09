"""Tests for the dedicated rerank endpoint client and its validation.

The network call is injected, so these cover what decides whether a response may
be trusted. A rerank answer carries scores rather than a permutation, so the
dangerous cases are out-of-range indices (a score attributed to the wrong
candidate) and missing candidates (silently dropped context).
"""
from dataclasses import replace
from urllib.error import HTTPError

import pytest

from specialized_rerank import (HttpRerankClient, SpecializedReranker,
                                parse_rerank_response)
from tests.test_store import get_test_config


def candidates(count=4):
    return [{'id': f'c{index}', 'content': f'正文 {index}'} for index in range(1, count + 1)]


def config():
    return replace(get_test_config('.'), rerank_base_url='https://example.test/v1',
                   rerank_api_key='sk-test', rerank_model='qwen3.7-text-rerank')


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def rerank(self, query, documents, top_n=None):
        self.calls.append({'query': query, 'documents': list(documents), 'top_n': top_n})
        if self.error:
            raise self.error
        return self.payload


def reranker(payload=None, error=None):
    client = FakeClient(payload=payload, error=error)
    return SpecializedReranker(config(), client_factory=lambda: client), client


def test_scores_reorder_candidates_by_descending_relevance():
    instance, client = reranker({'results': [
        {'index': 2, 'relevance_score': 0.2},
        {'index': 0, 'relevance_score': 0.9},
        {'index': 3, 'relevance_score': 0.5},
        {'index': 1, 'relevance_score': 0.1},
    ]})
    order, outcome = instance.rerank('问题', candidates())
    assert [c['id'] for c in order] == ['c1', 'c4', 'c3', 'c2']
    assert outcome.applied is True and outcome.reason == 'ok'
    assert outcome.scores == {'c1': 0.9, 'c4': 0.5, 'c3': 0.2, 'c2': 0.1}
    assert client.calls[0]['query'] == '问题'


def test_a_partial_response_keeps_unscored_candidates_in_retrieval_order():
    """Dropping unscored candidates would shrink the context silently."""
    instance, _ = reranker({'results': [{'index': 2, 'relevance_score': 0.9}]})
    order, outcome = instance.rerank('问题', candidates())
    assert [c['id'] for c in order] == ['c3', 'c1', 'c2', 'c4']
    assert outcome.applied is True
    assert len(order) == 4


def test_ties_keep_the_retrieval_order():
    instance, _ = reranker({'results': [{'index': 1, 'relevance_score': 0.5},
                                        {'index': 0, 'relevance_score': 0.5}]})
    order, _ = instance.rerank('问题', candidates(2))
    assert [c['id'] for c in order] == ['c1', 'c2']


@pytest.mark.parametrize('payload,reason', [
    ({}, 'no_results'),
    ({'results': []}, 'no_results'),
    ({'results': 'nope'}, 'no_results'),
    ({'results': [{'index': 9, 'relevance_score': 0.5}]}, 'index_out_of_range:9'),
    ({'results': [{'index': -1, 'relevance_score': 0.5}]}, 'index_out_of_range:-1'),
    ({'results': [{'index': 0}]}, 'missing_index_or_score'),
    ({'results': [{'index': 0, 'relevance_score': 'high'}]}, 'score_not_a_number'),
    ({'results': [{'index': 0, 'relevance_score': 0.5},
                  {'index': 0, 'relevance_score': 0.9}]}, 'duplicate_index'),
])
def test_untrustworthy_responses_fall_back(payload, reason):
    instance, _ = reranker(payload)
    order, outcome = instance.rerank('问题', candidates())
    assert [c['id'] for c in order] == ['c1', 'c2', 'c3', 'c4']
    assert outcome.applied is False and reason in outcome.reason
    assert instance.stats()['failures'] == 1


def test_a_transport_error_falls_back_and_is_counted():
    instance, _ = reranker(error=HTTPError('https://example.test', 429, 'rate limited', {}, None))
    order, outcome = instance.rerank('问题', candidates())
    assert [c['id'] for c in order] == ['c1', 'c2', 'c3', 'c4']
    assert outcome.applied is False
    assert 'HTTPError' in outcome.reason
    assert instance.stats()['failures'] == 1


def test_fewer_than_two_candidates_skips_the_call():
    instance, client = reranker({'results': []})
    order, outcome = instance.rerank('问题', candidates(1))
    assert outcome.reason == 'too_few_candidates' and client.calls == []
    assert instance.stats()['calls'] == 0


def test_documents_are_truncated_before_sending():
    instance, client = reranker({'results': [{'index': 0, 'relevance_score': 1.0}]})
    instance.max_document_chars = 5
    instance.rerank('问题', [{'id': 'a', 'content': '甲' * 100}, {'id': 'b', 'content': '乙'}])
    assert all(len(document) <= 5 for document in client.calls[0]['documents'])


def test_stats_accumulate_over_calls():
    instance, _ = reranker({'results': [{'index': 0, 'relevance_score': 1.0}]})
    instance.rerank('q1', candidates(2))
    instance.rerank('q2', candidates(2))
    stats = instance.stats()
    assert stats['calls'] == 2 and stats['failures'] == 0 and stats['seconds'] >= 0


def test_parse_accepts_the_nested_output_shape():
    payload = {'output': {'results': [{'index': 1, 'score': 0.7}]}}
    assert parse_rerank_response(payload, 3) == [(1, 0.7)]


def test_parse_rejects_a_non_object_payload():
    with pytest.raises(ValueError):
        parse_rerank_response(['not', 'a', 'dict'], 3)


def test_client_targets_the_rerank_path_and_sends_the_model():
    client = HttpRerankClient('https://example.test/v1/', 'sk-x', 'qwen3.7-text-rerank')
    assert client.url == 'https://example.test/v1/rerank'
    assert client.model == 'qwen3.7-text-rerank'
    assert client.api_key == 'sk-x'


def test_config_exposes_whether_a_reranker_is_configured():
    assert config().rerank_configured is True
    blank = replace(config(), rerank_api_key='')
    assert blank.rerank_configured is False
