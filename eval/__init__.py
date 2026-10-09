"""Evaluation framework for the generic RAG application.

This package is module 6 in the WeKnora-aligned plan (docs/ROADMAP.md). It
separates three concerns that the experiment scripts under
``experiments/rag-baseline/evaluation`` mix together:

- ``eval.datasets``  ground truth: questions, passage labels, answers
- ``eval.metrics``   pure metric computation over already-recorded results
- ``eval.runner``    task scheduling: run retrieval, collect, aggregate, store

Layering rule: ``eval`` wraps the product, never the reverse. ``rag`` must not
import ``eval``; ``tests/test_eval_scaffold.py`` enforces that direction.

Status: the dataset reader and the retrieval metrics are implemented and
tested. The runner is still a placeholder and raises ``NotImplementedError``
so an unfinished path cannot be mistaken for a working one.
"""
from eval.datasets import (Dataset, Question, load_all, load_dataset,  # noqa: F401
                           load_knowledge_base_dataset)
from eval.runner import EvaluationResult, run_evaluation  # noqa: F401

__all__ = ['Dataset', 'Question', 'load_all', 'load_dataset', 'load_knowledge_base_dataset',
           'EvaluationResult', 'run_evaluation']
