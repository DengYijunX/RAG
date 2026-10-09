"""Metric registry.

WeKnora registers its metrics through ``metric_hook.go`` and aggregates them
per evaluation run. The same idea applies here: a metric is a pure function
over recorded results, registered by name, so the runner stays independent of
which metrics are enabled.

Splitting them the way WeKnora does keeps the diagnosis useful: a retrieval
metric that drops and a generation metric that holds point at different
modules, which is the whole reason for scoring them separately.
"""
# Retrieval-side metrics (module 2/3 in docs/ROADMAP.md).
RETRIEVAL_METRICS = ('precision@k', 'recall@k', 'hit_rate@k', 'mrr', 'ndcg@k', 'map')
# Generation-side metrics (module 5 in docs/ROADMAP.md).
GENERATION_METRICS = ('answer_accuracy', 'faithfulness', 'citation_precision',
                      'citation_recall', 'answer_relevancy')


class MetricRegistry:
    """Name -> callable registry; callables take recorded results, return floats."""

    def __init__(self):
        self._metrics = {}

    def register(self, name, function):
        if name not in RETRIEVAL_METRICS + GENERATION_METRICS:
            raise KeyError(f'未登记的指标：{name}')
        self._metrics[name] = function
        return function

    def names(self):
        return sorted(self._metrics)

    def compute(self, names, results):
        """TODO: implement, with a test proving each metric on hand-computed input."""
        raise NotImplementedError('eval.metrics.MetricRegistry.compute 尚未实现')


def default_registry():
    """TODO: build the registry with this project's metric set.

    Decisions to record before implementing (docs/ROADMAP.md, 模块 6):

    - which metrics are reported per domain and never averaged together;
    - how ties and empty result lists are scored, since a missing value and a
      zero answer different questions;
    - that the 97 held-out questions stay unrun until the dev set is frozen.
    """
    raise NotImplementedError('eval.metrics.default_registry 尚未实现')
