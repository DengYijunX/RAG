"""Tests for listwise LLM reranking.

The model call is faked, so these cover the logic that decides whether a reply
may be trusted: a reply that is not a complete permutation must fall back to the
retrieval order, because a partially valid reply would silently drop candidates
from the context.
"""
from types import SimpleNamespace

import pytest

from llm_rerank import LLMListwiseReranker, RerankOutcome, parse_permutation
from tests.test_store import get_test_config


def candidates(count=4):
    return [{'id': f'c{index}', 'content': f'片段内容 {index}'} for index in range(1, count + 1)]


class FakeClient:
    """Returns a canned reply; records the prompt it was given."""

    def __init__(self, reply=None, error=None, usage=(100, 20)):
        self.reply = reply
        self.error = error
        self.usage = usage
        self.prompts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages):
        self.prompts.append(messages[-1]['content'])
        if self.error:
            raise self.error
        prompt_tokens, completion_tokens = self.usage
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def reranker(reply=None, error=None, usage=(100, 20)):
    client = FakeClient(reply=reply, error=error, usage=usage)
    instance = LLMListwiseReranker(get_test_config('.'), client_factory=lambda: client)
    return instance, client


def test_a_valid_permutation_is_applied():
    instance, _ = reranker('3,1,4,2')
    order, outcome = instance.rerank('问题', candidates())
    assert [candidate['id'] for candidate in order] == ['c3', 'c1', 'c4', 'c2']
    assert outcome.applied is True and outcome.reason == 'ok'
    assert instance.stats()['calls'] == 1 and instance.stats()['failures'] == 0


@pytest.mark.parametrize('reply,reason', [
    ('', 'no_numbers'),
    ('很好，我建议 1,2', 'count_mismatch:2!=4'),
    ('1,1,2,3', 'duplicates'),
    ('1,2,3,9', 'not_the_same_set'),
    ('1,2,3,4,5', 'count_mismatch:5!=4'),
])
def test_untrustworthy_replies_fall_back(reply, reason):
    instance, _ = reranker(reply)
    order, outcome = instance.rerank('问题', candidates())
    assert [candidate['id'] for candidate in order] == ['c1', 'c2', 'c3', 'c4']
    assert outcome.applied is False and outcome.reason == reason
    assert instance.stats()['failures'] == 1


def test_a_provider_error_falls_back_and_is_counted():
    instance, _ = reranker(error=RuntimeError('provider down'))
    order, outcome = instance.rerank('问题', candidates())
    assert [candidate['id'] for candidate in order] == ['c1', 'c2', 'c3', 'c4']
    assert outcome.applied is False and outcome.reason == 'call_failed:RuntimeError'
    assert instance.stats()['failures'] == 1


def test_fewer_than_two_candidates_skips_the_call():
    instance, client = reranker('1')
    order, outcome = instance.rerank('问题', candidates(1))
    assert outcome.reason == 'too_few_candidates' and outcome.applied is False
    assert instance.stats()['calls'] == 0 and client.prompts == []
    assert instance.rerank('问题', [])[0] == []


def test_usage_and_latency_are_accumulated():
    instance, _ = reranker('2,1,3,4', usage=(150, 25))
    instance.rerank('问题', candidates())
    instance.rerank('问题2', candidates())
    stats = instance.stats()
    assert stats['calls'] == 2
    assert stats['prompt_tokens'] == 300 and stats['completion_tokens'] == 50
    assert stats['seconds'] >= 0


def test_parse_permutation_accepts_common_separators():
    for reply in ('1,2,3', '1 2 3', '1，2，3', '[1, 2, 3]', '1、2、3'):
        assert parse_permutation(reply, [1, 2, 3]) == ([1, 2, 3], '')


def test_parse_permutation_is_exact_about_the_set():
    # Duplicates and swaps that lose a candidate are both refused.
    assert parse_permutation('1,1,2', [1, 2, 3])[0] == []
    assert parse_permutation('1,2,4', [1, 2, 3])[0] == []


def test_prompt_numbers_candidates_and_truncates_long_ones():
    instance, client = reranker('1,2')
    long_text = '甲' * 5000
    instance.rerank('问题', [{'id': 'a', 'content': long_text}, {'id': 'b', 'content': '短'}])
    prompt = client.prompts[0]
    assert '[1]' in prompt and '[2]' in prompt
    assert '已截断' in prompt
    assert len(prompt) < 5000


def test_prompt_states_the_expected_count():
    instance, client = reranker('1,2,3')
    instance.rerank('问题', candidates(3))
    assert '3 个编号' in client.prompts[0]


def test_outcome_defaults():
    outcome = RerankOutcome()
    assert outcome.applied is False and outcome.order == [] and outcome.usage == {}
