"""Tests for the assistant first-pass answer review.

The rubric is explicit that assistant scores must not be presented as human
accuracy and that a model reply is untrusted input. These tests cover the
validation: a reply that cannot be trusted yields **no score**, never a zero,
because "the judge did not answer" and "the answer was wrong" differ.
"""
import json
from types import SimpleNamespace

import pytest

from answer_review import (ERROR_LABELS, AnswerJudge, ReviewOutcome, build_prompt,
                           extract_json, validate_reply)
from tests.test_store import get_test_config


def valid_payload(**overrides):
    payload = {'key_points': 2, 'citation_faithfulness': 1, 'behavior_match': 2,
               'error_labels': [], 'severity': 'none', 'justification': '覆盖了要点'}
    payload.update(overrides)
    return payload


class FakeClient:
    def __init__(self, reply=None, error=None, usage=(500, 60)):
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
            usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens))


def judge(reply=None, error=None):
    client = FakeClient(reply=reply, error=error)
    return AnswerJudge(get_test_config('.'), client_factory=lambda: client), client


def test_a_valid_reply_is_accepted_with_all_three_dimensions():
    instance, _ = judge(json.dumps(valid_payload(), ensure_ascii=False))
    outcome = instance.review('问题', '回答 [1]')
    assert outcome.valid is True and outcome.reason == 'ok'
    assert outcome.scores == {'key_points': 2, 'citation_faithfulness': 1, 'behavior_match': 2}
    assert outcome.severity == 'none'


def test_na_is_accepted_for_clarification_questions():
    instance, _ = judge(json.dumps(valid_payload(key_points='na', citation_faithfulness='NA',
                                                 behavior_match=2), ensure_ascii=False))
    outcome = instance.review('问题', '请补充型号')
    assert outcome.valid is True
    assert outcome.scores['key_points'] == 'na' and outcome.scores['citation_faithfulness'] == 'na'


def test_error_labels_must_come_from_the_rubric():
    instance, _ = judge(json.dumps(valid_payload(error_labels=['made_up_label'],
                                                 severity='high'), ensure_ascii=False))
    outcome = instance.review('问题', '回答')
    assert outcome.valid is False and 'invalid_error_labels' in outcome.reason


def test_labels_without_a_valid_severity_are_refused():
    instance, _ = judge(json.dumps(valid_payload(error_labels=['unsupported_claim'],
                                                 severity='catastrophic'), ensure_ascii=False))
    assert instance.review('问题', '回答').valid is False


def test_a_clean_reply_forces_severity_none():
    instance, _ = judge(json.dumps(valid_payload(error_labels=[], severity='high'),
                                   ensure_ascii=False))
    outcome = instance.review('问题', '回答')
    assert outcome.valid is True and outcome.severity == 'none'


@pytest.mark.parametrize('overrides,reason_part', [
    ({'key_points': 3}, 'invalid_score'),
    ({'key_points': 'two'}, 'invalid_score'),
    ({'key_points': None}, 'invalid_score'),
])
def test_out_of_range_scores_are_refused(overrides, reason_part):
    instance, _ = judge(json.dumps(valid_payload(**overrides), ensure_ascii=False))
    outcome = instance.review('问题', '回答')
    assert outcome.valid is False and reason_part in outcome.reason
    assert outcome.scores == {}          # no score at all, not a zero


def test_a_missing_dimension_is_refused():
    payload = valid_payload()
    del payload['behavior_match']
    instance, _ = judge(json.dumps(payload, ensure_ascii=False))
    assert instance.review('问题', '回答').valid is False


def test_prose_around_the_json_is_tolerated():
    reply = '好的，这是评分：\n' + json.dumps(valid_payload(), ensure_ascii=False) + '\n以上。'
    instance, _ = judge(reply)
    assert instance.review('问题', '回答').valid is True


def test_unparseable_replies_are_invalid_not_zero():
    for reply in ('', 'no json here', '{broken', 'null'):
        instance, _ = judge(reply)
        outcome = instance.review('问题', '回答')
        assert outcome.valid is False and outcome.scores == {}
        assert instance.stats()['invalid'] == 1


def test_a_provider_error_is_counted_as_invalid():
    instance, _ = judge(error=RuntimeError('provider down'))
    outcome = instance.review('问题', '回答')
    assert outcome.valid is False and 'call_failed' in outcome.reason
    assert instance.stats()['invalid_rate'] == 1.0


def test_stats_track_usage_and_invalid_rate():
    instance, _ = judge(json.dumps(valid_payload(), ensure_ascii=False))
    instance.review('问题', '回答')
    instance.review('问题', '回答')
    stats = instance.stats()
    assert stats['calls'] == 2 and stats['invalid'] == 0
    assert stats['prompt_tokens'] == 1000 and stats['completion_tokens'] == 120


def test_extract_json_handles_the_awkward_cases():
    assert extract_json('{"a": 1}') == {'a': 1}
    assert extract_json('前缀 {"a": 1} 后缀') == {'a': 1}
    assert extract_json('没有对象') is None
    assert extract_json('') is None
    assert extract_json('{不是合法 json}') is None


def test_validate_reply_rejects_non_objects():
    outcome = validate_reply(['not', 'a', 'dict'])
    assert outcome.valid is False and outcome.reason == 'reply_not_an_object'
    assert isinstance(outcome, ReviewOutcome)


def test_prompt_carries_evidence_and_reference_points():
    prompt = build_prompt('问题？', '回答 [1]',
                          [{'content': '片段正文', 'title': '文档甲', 'start': 10, 'end': 20}],
                          ['要点一', '要点二'], 'answer')
    assert '问题？' in prompt and '回答 [1]' in prompt
    assert '要点一' in prompt and '文档甲' in prompt
    assert '片段正文' in prompt and '10-20' in prompt


def test_prompt_says_so_when_nothing_was_retrieved():
    prompt = build_prompt('问题？', '本次检索片段不足以回答。', [], [], 'answer')
    assert '没有检索到任何片段' in prompt


def test_prompt_instructs_against_using_its_own_knowledge():
    """The rubric forbids keyword or model-knowledge substitutes for evidence."""
    prompt = build_prompt('问题？', '回答', [], [], 'answer')
    assert isinstance(prompt, str) and len(prompt) > 50


def test_error_labels_match_the_rubric():
    assert ERROR_LABELS == ('retrieval_missing', 'unsupported_claim', 'wrong_scope',
                            'missing_clarification', 'fabricated_action_or_data',
                            'annotation_issue')
