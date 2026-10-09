"""Retrieval and generation metrics.

Every metric is a pure function over recorded results, so it can be recomputed
from stored artefacts without repeating model calls (the same separation
WeKnora gets from registering metrics through ``metric_hook.go``).

Conventions, fixed here because they change the numbers:

- Metrics are computed per question and then averaged, so a question with no
  labels cannot inflate the score.
- ``k`` is the length of the ranked list actually available; a question that
  returned fewer than ``k`` results is scored against what it returned and is
  counted, not skipped.
- Ties in NDCG use the standard gain formulation with log2 discounts; a
  duplicate retrieval of the same relevant passage earns no extra gain,
  because ``dcg`` walks the ranked list without re-counting ids.
- A metric over zero scored questions returns ``None``, never 0.0: "no
  evidence" and "scored zero" are different findings.
"""
import math

# Metric names, in the same spirit as WeKnora's metric registry: a metric is
# registered by name, and the runner stays independent of which are enabled.
# Retrieval-side metrics (module 2/3 in docs/ROADMAP.md).
RETRIEVAL_METRICS = ('precision@k', 'recall@k', 'hit_rate@k', 'mrr', 'ndcg@k', 'map')
# Generation-side metrics (module 5 in docs/ROADMAP.md); not implemented yet.
GENERATION_METRICS = ('answer_accuracy', 'faithfulness', 'citation_precision',
                      'citation_recall', 'answer_relevancy')


class MetricRegistry:
    """Name -> callable registry; callables take recorded results, return floats."""

    def __init__(self):
        self._metrics = {}

    def register(self, name, function):
        if name not in RETRIEVAL_METRICS + GENERATION_METRICS:
            raise KeyError(f'未登记的指标：{name}')
        if name in GENERATION_METRICS:
            raise NotImplementedError(f'{name} 尚未实现；先完成检索侧评测')
        self._metrics[name] = function
        return function

    def names(self):
        return sorted(self._metrics)

    def compute(self, names, results):
        """Run the registered metrics over recorded results and average them.

        ``results`` is a sequence of samples; each metric receives one sample
        at a time, and a metric returning ``None`` is excluded from the mean
        rather than counted as zero.
        """
        unknown = [name for name in names if name not in self._metrics]
        if unknown:
            raise KeyError(f'指标尚未注册：{", ".join(unknown)}')
        output = {}
        for name in names:
            values = [self._metrics[name](sample) for sample in results]
            values = [value for value in values if value is not None]
            output[name] = sum(values) / len(values) if values else None
        return output


def default_registry(k=5):
    """Registry with the retrieval metrics bound to one cutoff ``k``."""
    registry = MetricRegistry()
    registry.register('recall@k', lambda sample: recall_at_k(*sample, k=k))
    registry.register('precision@k', lambda sample: precision_at_k(*sample, k=k))
    registry.register('hit_rate@k', lambda sample: hit_rate_at_k(*sample, k=k))
    registry.register('mrr', lambda sample: reciprocal_rank(*sample))
    registry.register('map', lambda sample: mean_average_precision(*sample))
    registry.register('ndcg@k', lambda sample: ndcg_at_k(*sample, k=k))
    return registry


def _ranked_ids(results):
    """Accept the runner's hit dicts, plain ids, or any object with an id."""
    ids = []
    for item in results or ():
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            value = item.get('id', item.get('passage_id', item.get('chunk_id')))
            if value is not None:
                ids.append(str(value))
        else:
            value = getattr(item, 'id', None)
            if value is not None:
                ids.append(str(value))
    return ids


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def recall_at_k(ranked_ids, relevant_ids, k):
    """Share of labelled relevant passages found in the top ``k``."""
    if not relevant_ids:
        return None
    top = set(_ranked_ids(ranked_ids)[:k])
    return len(top & set(relevant_ids)) / len(set(relevant_ids))


def precision_at_k(ranked_ids, relevant_ids, k):
    """Share of the returned top ``k`` that is labelled relevant."""
    if not relevant_ids:
        return None
    top = _ranked_ids(ranked_ids)[:k]
    if not top:
        return 0.0
    return sum(1 for identifier in top if identifier in set(relevant_ids)) / len(top)


def hit_rate_at_k(ranked_ids, relevant_ids, k):
    """1.0 when at least one labelled passage is in the top ``k``."""
    if not relevant_ids:
        return None
    return 1.0 if set(_ranked_ids(ranked_ids)[:k]) & set(relevant_ids) else 0.0


def reciprocal_rank(ranked_ids, relevant_ids):
    """1 / rank of the first labelled hit; 0.0 when none was returned."""
    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    for position, identifier in enumerate(_ranked_ids(ranked_ids), 1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked_ids, relevant_ids):
    """Mean of the precision values at each labelled hit."""
    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    hits = 0
    total = 0.0
    for position, identifier in enumerate(_ranked_ids(ranked_ids), 1):
        if identifier in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def dcg_at_k(gains, k):
    """Discounted cumulative gain; ``gains`` is the ranked list of gains."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains[:k], 1))


def ndcg_at_k(ranked_ids, relevant_ids, k, gains=None):
    """Normalised DCG against the ideal ordering of the labelled passages.

    ``gains`` optionally maps a passage id to its graded gain; without it the
    labels are treated as binary relevance.
    """
    if not relevant_ids:
        return None
    relevant = list(dict.fromkeys(relevant_ids))
    weight = (lambda identifier: float(gains.get(identifier, 1.0))) if gains else (lambda _: 1.0)
    ranked = _ranked_ids(ranked_ids)
    seen = set()
    actual = []
    for identifier in ranked[:k]:
        if identifier in relevant and identifier not in seen:
            seen.add(identifier)
            actual.append(weight(identifier))
        else:
            actual.append(0.0)
    ideal = sorted((weight(identifier) for identifier in relevant), reverse=True)
    best = dcg_at_k(ideal, k)
    if best <= 0:
        return 0.0
    return dcg_at_k(actual, k) / best


def mean_average_precision(ranked_ids, relevant_ids):
    """MAP for a single query is its average precision."""
    return average_precision(ranked_ids, relevant_ids)


def score_dataset(samples, k=5):
    """Aggregate the retrieval metrics over recorded samples.

    ``samples`` yields ``(question_id, ranked_ids, relevant_ids)``. The return
    value reports how many questions were scored so a metric over an empty or
    unlabelled set is visible as ``scored: 0`` rather than as a zero score.
    """
    metric_names = ('recall@k', 'precision@k', 'hit_rate@k', 'mrr', 'map', 'ndcg@k')
    registry = default_registry(k=k)
    prepared = []
    for _, ranked_ids, relevant_ids in samples:
        if not relevant_ids:
            continue
        prepared.append((ranked_ids, relevant_ids))
    return {'k': k, 'questions': len(prepared),
            **registry.compute(metric_names, prepared)}
