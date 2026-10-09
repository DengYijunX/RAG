"""Evaluation runner: run the pipeline, record everything, aggregate.

Mirrors WeKnora's evaluation service: a task is submitted, QA samples run
concurrently under a bound, and per-stage artefacts are kept so metrics can be
recomputed without repeating the expensive calls.

Boundaries that matter for this project (docs/ROADMAP.md, 模块 6):

- One run fixes corpus, chunking, embedding model, generation model and
  parameters; only the intended variable changes.
- Retrieval artefacts (query embedding count, candidate lists, selected
  context, final request) and generation artefacts (raw answer, usage) are
  stored per sample, because a wrong answer has to be attributable to a stage.
- Results are written to disk with the configuration snapshot and source
  hashes, so a later reader can tell which code produced them.
"""
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Aggregated outcome of one run, plus the per-sample records behind it."""
    task_id: str = ''
    dataset: str = ''
    configuration: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    failures: list = field(default_factory=list)


def run_evaluation(service, dataset, *, metrics=(), output_dir=None, concurrency=1):
    """Run one evaluation task against a knowledge base.

    TODO: implement in this order, each step landing with its own test:
    1. record the configuration snapshot and refuse to run if it changed
       between samples;
    2. retrieve only (no generation) and write retrieval metrics;
    3. add generation behind an explicit flag, since it costs model calls;
    4. write per-sample stage artefacts next to the aggregate summary;
    5. persist to ``output_dir`` so a restart cannot lose history.
    """
    raise NotImplementedError('eval.runner.run_evaluation 尚未实现；本次只落地目录与接口占位')
