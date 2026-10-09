"""Evaluation framework for the generic RAG application.

This package is the skeleton of module 6 in the WeKnora-aligned plan
(see docs/ROADMAP.md). It separates three concerns that the current
``experiments/rag-baseline/evaluation`` scripts mix together:

- ``eval.datasets``  ground truth: questions, relevant passages, answers
- ``eval.metrics``   pure metric computation over already-recorded results
- ``eval.runner``    task scheduling: run the pipeline, collect, aggregate

Nothing here calls a model, opens a network connection or writes to the
vector store: the runner is the only module allowed to touch the application,
and it does so through the existing service objects.

Current status: placeholders only. Every public function raises
``NotImplementedError`` so that an unfinished path can never be mistaken for
a working one. The existing experiment scripts remain the runnable evidence
and are not replaced until this package has tests of its own.
"""
from eval.datasets import GroundTruth, load_dataset  # noqa: F401
from eval.runner import EvaluationResult, run_evaluation  # noqa: F401

__all__ = ['GroundTruth', 'load_dataset', 'EvaluationResult', 'run_evaluation']
