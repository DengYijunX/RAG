"""Reranking with a dedicated rerank endpoint (Qwen rerank on Aliyun MaaS).

This is a different contract from the listwise LLM reranker in ``llm_rerank.py``:

- the endpoint is ``POST {base}/rerank`` with ``{model, query, documents, top_n}``
  and answers ``{results: [{index, relevance_score}]}`` — not a chat completion;
- one call scores the whole candidate list, like the listwise approach, but the
  model is trained for relevance rather than for following instructions;
- because the response carries *scores*, not a permutation, a partial or
  malformed answer must not be trusted either: every returned index has to be
  inside the candidate range, and candidates the endpoint did not score keep
  their relative retrieval order behind the scored ones.

Contract with the caller is the same as the LLM reranker: the returned list is
always a complete permutation-friendly ordering (reranked head + untouched
tail), and any failure is reported rather than hidden.

Evaluation-side code: ``rag`` must not import it.
"""
from dataclasses import dataclass, field
import json
import time
import urllib.error
import urllib.request

@dataclass
class RerankOutcome:
    order: list = field(default_factory=list)
    applied: bool = False
    reason: str = ''
    seconds: float = 0.0
    scores: dict = field(default_factory=dict)


def parse_rerank_response(payload, expected_count):
    """Validate a rerank response into ``[(index, score)]``, or raise ValueError.

    ``index`` is 0-based in the OpenAI-compatible rerank shape. Out-of-range
    indices are refused rather than clamped: a clamped index would attribute a
    score to the wrong candidate.
    """
    if not isinstance(payload, dict):
        raise ValueError('response_not_an_object')
    results = payload.get('results')
    if results is None and isinstance(payload.get('output'), dict):
        results = payload['output'].get('results')
    if not isinstance(results, list) or not results:
        raise ValueError('no_results')
    parsed = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError('result_not_an_object')
        index = item.get('index')
        score = item.get('relevance_score', item.get('score'))
        if index is None or score is None:
            raise ValueError('missing_index_or_score')
        if not isinstance(index, int) or not 0 <= index < expected_count:
            raise ValueError(f'index_out_of_range:{index}')
        try:
            parsed.append((index, float(score)))
        except (TypeError, ValueError):
            raise ValueError(f'score_not_a_number:{score!r}') from None
    if len({index for index, _ in parsed}) != len(parsed):
        raise ValueError('duplicate_index')
    return parsed


class HttpRerankClient:
    """Minimal client for the rerank endpoint.

    Uses urllib rather than the OpenAI SDK: the SDK has no rerank resource on
    this provider, and this keeps the dependency surface unchanged. The URL is
    whatever ``config.rerank_base_url`` holds — ``rag.config.normalize_rerank_url``
    has already resolved it to the provider's native rerank path.
    """

    def __init__(self, base_url, api_key, model, timeout=60):
        self.url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query, documents, top_n=None):
        body = {'model': self.model, 'query': query, 'documents': list(documents)}
        if top_n:
            body['top_n'] = top_n
        request = urllib.request.Request(
            self.url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'},
            method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))


class SpecializedReranker:
    """Reranks candidates with a dedicated rerank model.

    ``client_factory`` is injected so the validation logic can be tested without
    a network call; the default builds an :class:`HttpRerankClient` from config.
    """

    def __init__(self, config, client_factory=None, max_document_chars=2000):
        self.config = config
        self.max_document_chars = max_document_chars
        if client_factory is None:
            client_factory = lambda: HttpRerankClient(  # noqa: E731
                config.rerank_base_url, config.rerank_api_key, config.rerank_model)
        self._client_factory = client_factory
        self.calls = 0
        self.failures = 0
        self.total_seconds = 0.0

    def rerank(self, question, candidates):
        """Return ``(ordered_candidates, outcome)``; order is always complete."""
        original = list(candidates)
        if len(original) < 2:
            return original, RerankOutcome(order=[c['id'] for c in original],
                                           reason='too_few_candidates')
        documents = [candidate['content'][:self.max_document_chars] for candidate in original]
        started = time.time()
        self.calls += 1
        try:
            payload = self._client_factory().rerank(question, documents)
            parsed = parse_rerank_response(payload, len(original))
        except Exception as exc:
            self.failures += 1
            self.total_seconds += time.time() - started
            return original, RerankOutcome(order=[c['id'] for c in original],
                                           reason=f'{type(exc).__name__}:{exc}'[:120],
                                           seconds=time.time() - started)
        elapsed = time.time() - started
        self.total_seconds += elapsed

        # Scored candidates first, best score first; ties keep retrieval order.
        scored = sorted(parsed, key=lambda item: (-item[1], item[0]))
        head_indices = [index for index, _ in scored]
        # Candidates the endpoint did not score keep their retrieval order.
        tail_indices = [index for index in range(len(original)) if index not in set(head_indices)]
        ordered = [original[index] for index in head_indices + tail_indices]
        return ordered, RerankOutcome(order=[candidate['id'] for candidate in ordered],
                                      applied=True, reason='ok', seconds=elapsed,
                                      scores={original[index]['id']: score
                                              for index, score in parsed})

    def stats(self):
        return {'calls': self.calls, 'failures': self.failures,
                'seconds': round(self.total_seconds, 2)}
