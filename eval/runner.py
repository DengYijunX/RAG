"""Evaluation runner: run retrieval, align hits, score, and store artefacts.

Mirrors WeKnora's evaluation service in shape: a task runs a set of samples,
per-stage artefacts are kept, and metrics are aggregated centrally. The
differences that matter for this project:

- Retrieval only by default. Generation costs model calls and is opt-in, so
  the default run is free and repeatable on every change.
- One run fixes knowledge base, dataset version, split and ``k``; the summary
  records them with source hashes, so a score can always be traced back to the
  code and data that produced it.
- Unresolved hits and unlabelled questions are reported, never silently
  counted as misses.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

from eval import metrics
from eval.alignment import PassageResolver


@dataclass
class SampleRecord:
    """Everything recorded for one question, enough to recompute its metrics."""
    question_id: str
    question: str
    split: str
    relevant_passage_ids: list = field(default_factory=list)
    ranked_passage_ids: list = field(default_factory=list)
    labels_per_hit: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    coverage: float = None
    error: str = ''


@dataclass
class EvaluationResult:
    """Aggregated outcome of one run, plus the per-sample records behind it."""
    task_id: str = ''
    knowledge_base: str = ''
    dataset: str = ''
    split: str = ''
    k: int = 0
    configuration: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    output_dir: str = ''

    def summary(self):
        return {'task_id': self.task_id, 'knowledge_base': self.knowledge_base,
                'dataset': self.dataset, 'split': self.split, 'k': self.k,
                'metrics': self.metrics, 'configuration': self.configuration,
                'questions': len(self.samples), 'failures': len(self.failures),
                'output_dir': self.output_dir}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run_evaluation(questions, retrieve, *, knowledge_base='', dataset='', split='',
                   k=5, resolver=None, output_dir=None, max_questions=None):
    """Run a retrieval-only evaluation and optionally persist its artefacts.

    ``questions`` is an iterable of ``eval.datasets.Question``; ``retrieve`` is
    ``question -> ranked hits`` where a hit carries ``id`` and ``content``.
    ``resolver`` maps hits to dataset passage ids; without one, hits must carry
    a ``passage_id`` themselves.
    """
    questions = list(questions)
    if max_questions is not None:
        questions = questions[:max_questions]
    if not questions:
        raise ValueError('评测题集为空，拒绝运行')

    task_id = f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{_digest(split or dataset)[:8]}'
    samples, failures = [], []
    for question in questions:
        try:
            hits = list(retrieve(question))
        except Exception as exc:
            # A failing question is recorded, not dropped: a crash is data.
            failures.append({'question_id': question.id, 'error': type(exc).__name__,
                             'message': str(exc)[:200]})
            samples.append(SampleRecord(question_id=question.id, question=question.question,
                                        split=question.split,
                                        relevant_passage_ids=list(question.relevant_passage_ids),
                                        error=type(exc).__name__))
            continue
        if resolver is not None:
            resolution = resolver.resolve(hits, question.relevant_passage_ids)
        else:
            resolution = None
        ranked = resolution.ranked_passage_ids if resolution else \
            [str(hit.get('passage_id', '')) for hit in hits if hit.get('passage_id')]
        samples.append(SampleRecord(
            question_id=question.id, question=question.question, split=question.split,
            relevant_passage_ids=list(question.relevant_passage_ids),
            ranked_passage_ids=ranked,
            labels_per_hit=[list(labels) for labels in resolution.labels_per_hit]
            if resolution else [],
            hits=[{'id': str(hit.get('id', '')), 'source': hit.get('source', ''),
                   'rank': hit.get('rank'), 'score': hit.get('score')} for hit in hits],
            unresolved=list(resolution.unresolved) if resolution else [],
            coverage=resolution.coverage if resolution else None))

    scored = [(sample.question_id, sample.ranked_passage_ids, sample.relevant_passage_ids)
              for sample in samples if not sample.error]
    summary = metrics.score_dataset(scored, k=k)
    # Slot-based metrics are the ones that answer the product question: is the
    # labelled passage inside the top k chunks we would hand to the model?
    # The de-duplicated passage list above reports a passage found at rank 9 as
    # if it were found early, so it is kept only as a secondary view.
    slot_summary = metrics.score_slot_metrics(
        [(sample.labels_per_hit, sample.relevant_passage_ids)
         for sample in samples if not sample.error and sample.labels_per_hit], k=k)
    unresolved_total = sum(len(sample.unresolved) for sample in samples)
    hit_total = sum(len(sample.hits) for sample in samples)
    result = EvaluationResult(
        task_id=task_id, knowledge_base=knowledge_base, dataset=dataset, split=split, k=k,
        configuration={'k': k, 'split': split, 'dataset': dataset, 'knowledge_base': knowledge_base,
                       'questions': len(questions), 'resolver': bool(resolver)},
        metrics={**summary, **slot_summary,
                 'hits': hit_total, 'unresolved_hits': unresolved_total,
                 'resolution_coverage': (1 - unresolved_total / hit_total) if hit_total else None,
                 'unlabelled_questions': sum(1 for sample in samples if not sample.relevant_passage_ids)},
        samples=samples, failures=failures)

    if output_dir:
        result.output_dir = str(write_artifacts(result, output_dir))
    return result


def write_artifacts(result, output_dir):
    """Persist the summary and the per-sample records.

    Runs are never overwritten. The task id is unique to a second, so an
    immediate repeat (the comparison script runs three modes back to back)
    would collide; a numeric suffix keeps both records instead of failing or
    silently replacing the earlier one.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / result.task_id
    suffix = 2
    while directory.exists():
        directory = root / f'{result.task_id}-{suffix}'
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'summary.json').write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with open(directory / 'samples.jsonl', 'w', encoding='utf-8') as handle:
        for sample in result.samples:
            handle.write(json.dumps({k: v for k, v in asdict(sample).items() if v not in ([], '', None)},
                                    ensure_ascii=False) + '\n')
    if result.failures:
        (directory / 'failures.json').write_text(
            json.dumps(result.failures, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return directory
