"""Listwise reranking with the configured chat model.

The project uses its existing chat model rather than a dedicated cross-encoder
reranker. The trade-off, recorded in ADR-009:

- a general LLM needs no second provider, key or dependency, and reorders the
  whole candidate list in one call;
- it is a *generative* reranker, so its output must be validated as a complete
  permutation of the candidates. Anything else falls back to the retrieval
  order, and the failure is reported rather than hidden;
- it cannot raise the recall ceiling: a passage absent from the candidate pool
  stays absent. That is why the experiment must distinguish "the labelled
  passage was never retrieved" from "it was retrieved and not promoted".

This is evaluation-side code: ``rag`` must not import it.
"""
from dataclasses import dataclass, field
import json
import re
import time

SYSTEM_PROMPT = (
    '你是检索结果重排器。用户给出一个问题与若干候选片段，你只输出候选编号的排列，'
    '按与问题的相关程度从高到低，最相关的排最前。'
    '只输出编号，用英文逗号分隔，例如：3,1,7,2。'
    '不要解释，不要输出任何其他文字，不要编造不存在的编号，必须包含全部候选编号且每个只出现一次。'
)

INTEGER_PATTERN = re.compile(r'\d+')


@dataclass
class RerankOutcome:
    """Result of one rerank call, including why it was or was not applied."""
    order: list = field(default_factory=list)
    applied: bool = False
    reason: str = ''
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    raw: str = ''


def parse_permutation(text, expected):
    """Parse a model reply into a complete permutation of ``expected``.

    Returns ``(order, reason)``; ``order`` is empty when the reply cannot be
    trusted. Accepted input is a comma/space separated list of the candidate
    indices. Duplicates, out-of-range numbers and missing candidates all fail:
    a partially valid reply would silently drop candidates from the context.
    """
    numbers = [int(value) for value in INTEGER_PATTERN.findall(text or '')]
    if not numbers:
        return [], 'no_numbers'
    if len(numbers) != len(expected):
        return [], f'count_mismatch:{len(numbers)}!={len(expected)}'
    if sorted(numbers) != sorted(expected):
        if len(set(numbers)) != len(numbers):
            return [], 'duplicates'
        return [], 'not_the_same_set'
    return numbers, ''


class LLMListwiseReranker:
    """Reranks a candidate list with the configured chat model.

    ``client_factory`` returns an OpenAI-compatible client; it is injected so
    tests can run without a network call. The default builds the same client
    the application uses.
    """

    def __init__(self, config, client_factory=None, max_candidate_chars=600):
        self.config = config
        self.max_candidate_chars = max_candidate_chars
        self._client_factory = client_factory or self._default_client
        self.calls = 0
        self.failures = 0
        self.total_seconds = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _default_client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.config.llm_base_url, api_key=self.config.llm_api_key,
                      timeout=60, max_retries=1)

    def build_prompt(self, question, candidates):
        """Number candidates 1..n and ask for their order.

        Candidate text is truncated so one long chunk cannot crowd out the rest
        of the list; the truncation is announced in the prompt so the model
        knows the excerpt is partial.
        """
        lines = [f'问题：{question}', '', '候选片段：']
        for index, candidate in enumerate(candidates, 1):
            text = candidate['content']
            truncated = len(text) > self.max_candidate_chars
            excerpt = text[:self.max_candidate_chars]
            marker = '（已截断）' if truncated else ''
            lines.append(f'[{index}] {excerpt}{marker}')
        lines.append('')
        lines.append(f'请输出 {len(candidates)} 个编号的排列，逗号分隔。')
        return '\n'.join(lines)

    def rerank(self, question, candidates):
        """Return the reranked candidate list, or the original order on failure."""
        if len(candidates) < 2:
            return list(candidates), RerankOutcome(order=[c['id'] for c in candidates],
                                                   reason='too_few_candidates')
        expected = list(range(1, len(candidates) + 1))
        prompt = self.build_prompt(question, candidates)
        started = time.time()
        self.calls += 1
        try:
            with self._client_factory() as client:
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                              {'role': 'user', 'content': prompt}])
            raw = response.choices[0].message.content or ''
            usage = getattr(response, 'usage', None)
        except Exception as exc:
            self.failures += 1
            self.total_seconds += time.time() - started
            return list(candidates), RerankOutcome(order=[c['id'] for c in candidates],
                                                   reason=f'call_failed:{type(exc).__name__}',
                                                   seconds=time.time() - started)
        elapsed = time.time() - started
        self.total_seconds += elapsed
        if usage is not None:
            self.total_prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self.total_completion_tokens += getattr(usage, 'completion_tokens', 0) or 0

        order, reason = parse_permutation(raw, expected)
        if not order:
            self.failures += 1
            return list(candidates), RerankOutcome(order=[c['id'] for c in candidates],
                                                   reason=reason, seconds=elapsed,
                                                   usage={'prompt': getattr(usage, 'prompt_tokens', None),
                                                          'completion': getattr(usage, 'completion_tokens', None)},
                                                   raw=raw[:200])
        reordered = [candidates[index - 1] for index in order]
        return reordered, RerankOutcome(order=[candidate['id'] for candidate in reordered],
                                        applied=True, reason='ok', seconds=elapsed,
                                        usage={'prompt': getattr(usage, 'prompt_tokens', None),
                                               'completion': getattr(usage, 'completion_tokens', None)},
                                        raw=raw[:200])

    def stats(self):
        return {'calls': self.calls, 'failures': self.failures,
                'seconds': round(self.total_seconds, 2),
                'prompt_tokens': self.total_prompt_tokens,
                'completion_tokens': self.total_completion_tokens}
