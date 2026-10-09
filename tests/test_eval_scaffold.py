"""Contract tests for the evaluation framework's layering and honesty.

The dataset reader, metrics, alignment and runner have their own tests. These
check the properties that keep the package trustworthy as it grows: the
layering rule holds, and importing the evaluation package does not drag in the
application or a model call.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import eval as eval_package
from eval import datasets, metrics, runner

REPO = Path(__file__).resolve().parents[1]


def test_dataset_errors_are_explicit(tmp_path):
    missing = tmp_path / 'nope.jsonl'
    with pytest.raises(datasets.DatasetError):
        datasets.load_dataset(missing)
    broken = tmp_path / 'broken.jsonl'
    broken.write_text('{not json}\n', encoding='utf-8')
    with pytest.raises(datasets.DatasetError):
        datasets.load_dataset(broken)
    duplicate = tmp_path / 'dup.jsonl'
    duplicate.write_text('{"id": "q1", "question": "a"}\n{"id": "q1", "question": "b"}\n',
                         encoding='utf-8')
    with pytest.raises(datasets.DatasetError):
        datasets.load_dataset(duplicate)


def test_empty_question_set_is_refused():
    with pytest.raises(ValueError):
        runner.run_evaluation([], lambda question: [])


def test_metric_names_are_registered_and_typos_rejected():
    assert 'recall@k' in metrics.RETRIEVAL_METRICS
    assert 'ndcg@k' in metrics.RETRIEVAL_METRICS
    assert 'faithfulness' in metrics.GENERATION_METRICS
    registry = metrics.MetricRegistry()
    with pytest.raises(KeyError):
        registry.register('recal@k', lambda results: 0.0)
    with pytest.raises(NotImplementedError):
        registry.register('faithfulness', lambda results: 0.0)
    assert registry.names() == []


def test_public_dataclasses_describe_the_records():
    question = eval_package.Question(id='q-1', question='问题', relevant_passage_ids=['p-1'])
    assert question.split == '' and question.answer == ''
    assert question.graded is True and question.relevance('p-1') == 1.0
    assert question.relevance('p-2') == 0.0
    dataset = eval_package.Dataset(name='d', questions=[question])
    assert len(dataset) == 1 and dataset.summary()['graded'] == 1
    assert eval_package.EvaluationResult().metrics == {}


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
    code = ('import sys, eval, eval.metrics, eval.datasets, eval.alignment, eval.runner; '
            'print(any(name == "rag" or name.startswith("rag.") for name in sys.modules))')
    output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == 'False'
