"""Contract tests for the evaluation skeleton.

These do not test metrics (there are none yet). They test that the placeholder
cannot be mistaken for a working implementation, and that the layering rule in
docs/ROADMAP.md holds: ``eval`` may depend on ``rag``, never the reverse.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import eval as eval_package
from eval import datasets, metrics, runner

REPO = Path(__file__).resolve().parents[1]


def test_public_placeholders_raise():
    with pytest.raises(NotImplementedError):
        datasets.load_dataset('anything.jsonl')
    with pytest.raises(NotImplementedError):
        datasets.align_passages([], [])
    with pytest.raises(NotImplementedError):
        metrics.default_registry()
    with pytest.raises(NotImplementedError):
        metrics.MetricRegistry().compute([], [])
    with pytest.raises(NotImplementedError):
        runner.run_evaluation(object(), object())


def test_metric_names_are_registered_and_typos_rejected():
    assert 'recall@k' in metrics.RETRIEVAL_METRICS
    assert 'ndcg@k' in metrics.RETRIEVAL_METRICS
    assert 'faithfulness' in metrics.GENERATION_METRICS
    registry = metrics.MetricRegistry()
    with pytest.raises(KeyError):
        registry.register('recal@k', lambda results: 0.0)
    assert registry.names() == []


def test_dataclasses_describe_ground_truth():
    truth = eval_package.GroundTruth(query_id='q-1', question='问题', relevant_passage_ids=['p-1'])
    assert truth.split == 'dev' and truth.answer == ''
    passage = datasets.Passage(passage_id='p-1', text='正文')
    assert passage.document_id == '' and passage.metadata == {}
    result = eval_package.EvaluationResult()
    assert result.metrics == {} and result.samples == []


def test_eval_never_depends_on_rag_and_rag_never_imports_eval():
    """The layering rule: evaluation wraps the product, not the other way round."""
    for path in (REPO / 'eval').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        assert 'from rag' not in text and 'import rag' not in text, f'{path} 不应导入 rag'
    for path in (REPO / 'rag').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(('import eval', 'from eval')), \
                f'{path} 不应导入 eval'


def test_importing_eval_does_not_pull_in_the_application():
    code = ('import sys, eval; '
            'print(any(name == "rag" or name.startswith("rag.") for name in sys.modules))')
    output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == 'False'
