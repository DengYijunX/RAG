"""Tests for the retrieval evaluation runner.

The runner is exercised with a fake retrieval function, so the tests cover the
accounting of a run — alignment, failures, artefacts — without any model call.
"""
import json

import pytest

from eval.alignment import PassageResolver
from eval.datasets import Question
from eval.runner import run_evaluation, write_artifacts


def questions(count=3):
    return [Question(id=f'q{index}', question=f'问题{index}', split='dev',
                     relevant_passage_ids=[f'P{index}'])
            for index in range(1, count + 1)]


def fake_retrieve(mapping):
    """``mapping`` maps question id -> list of passage ids to return."""
    def retrieve(question):
        return [{'id': f'chunk-{question.id}-{rank}', 'content': f'文本 {passage}',
                 'source': 'doc.md', 'rank': rank}
                for rank, passage in enumerate(mapping.get(question.id, []), 1)]
    return retrieve


def resolver_for(count=3):
    # The fake retriever returns "文本 <passage id>", so digests must match that
    # exact string; build the resolver from what retrieval would return.
    return PassageResolver({f'P{index}': f'文本 P{index}' for index in range(1, count + 1)})


def test_perfect_run_scores_one():
    result = run_evaluation(questions(), fake_retrieve({'q1': ['P1'], 'q2': ['P2'], 'q3': ['P3']}),
                            knowledge_base='kb', split='dev', k=1, resolver=resolver_for())
    assert result.metrics['recall@k'] == 1.0
    assert result.metrics['mrr'] == 1.0
    assert result.metrics['questions'] == 3
    assert result.metrics['unresolved_hits'] == 0
    assert result.metrics['resolution_coverage'] == 1.0


def test_second_rank_hit_lowers_mrr_but_not_recall_at_two():
    """An unresolved hit still occupies its rank; it just earns no credit."""
    retrieve = fake_retrieve({'q1': ['X', 'P1'], 'q2': ['P2'], 'q3': []})
    resolver = PassageResolver({'P1': '文本 P1', 'P2': '文本 P2', 'P3': '文本 P3'})
    result = run_evaluation(questions(), retrieve, k=2, resolver=resolver)
    # q1: P1 resolved at rank 2 -> recall 1, rr .5
    # q2: P2 resolved at rank 1 -> recall 1, rr 1
    # q3: nothing returned      -> recall 0, rr 0
    # Dropping the unresolved first hit would have made q1's rr 1.0 instead.
    assert result.metrics['recall@k'] == pytest.approx(2 / 3)
    assert result.metrics['mrr'] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert result.metrics['unresolved_hits'] == 1
    assert result.metrics['questions'] == 3
    first = next(sample for sample in result.samples if sample.question_id == 'q1')
    assert first.ranked_passage_ids == [None, 'P1']
    assert first.unresolved == [{'id': 'chunk-q1-1', 'rank': 1,
                                 'reason': 'no_matching_passage'}]


def test_unlabelled_questions_are_excluded_and_counted():
    items = questions() + [Question(id='qX', question='没有标签', split='dev')]
    result = run_evaluation(items, fake_retrieve({'q1': ['P1'], 'q2': ['P2'], 'q3': ['P3']}),
                            k=1, resolver=resolver_for())
    assert result.metrics['questions'] == 3          # qX carries no labels
    assert result.metrics['unlabelled_questions'] == 1
    assert len(result.samples) == 4                  # but it is still recorded


def test_alignment_failures_are_recorded_not_hidden():
    retrieve = fake_retrieve({'q1': ['NOT-A-PASSAGE'], 'q2': ['P2'], 'q3': ['P3']})
    result = run_evaluation(questions(), retrieve, k=1, resolver=resolver_for())
    assert result.metrics['unresolved_hits'] == 1
    assert result.metrics['resolution_coverage'] == pytest.approx(2 / 3)
    assert result.metrics['recall@k'] == pytest.approx(2 / 3)
    failed = next(sample for sample in result.samples if sample.question_id == 'q1')
    assert failed.unresolved == [{'id': 'chunk-q1-1', 'rank': 1,
                                  'reason': 'no_matching_passage'}]
    assert failed.ranked_passage_ids == [None]


def test_a_throwing_question_is_recorded_as_a_failure_not_dropped():
    def retrieve(question):
        if question.id == 'q2':
            raise RuntimeError('provider down')
        return [{'id': 'c', 'content': f'文本 {question.relevant_passage_ids[0]}', 'rank': 1}]

    result = run_evaluation(questions(), retrieve, k=1, resolver=resolver_for())
    assert result.metrics['questions'] == 2
    assert result.failures == [{'question_id': 'q2', 'error': 'RuntimeError',
                                'message': 'provider down'}]
    assert len(result.samples) == 3


def test_hits_without_a_resolver_need_explicit_passage_ids():
    def retrieve(question):
        return [{'id': 'c1', 'content': '文本', 'passage_id': question.relevant_passage_ids[0]}]

    result = run_evaluation(questions(), retrieve, k=1)
    assert result.metrics['recall@k'] == 1.0


def test_max_questions_limits_the_run():
    result = run_evaluation(questions(5), fake_retrieve({}), k=1, resolver=resolver_for(5),
                            max_questions=2)
    assert result.metrics['questions'] == 2
    assert len(result.samples) == 2


def test_artifacts_are_written_once_and_not_overwritten(tmp_path):
    result = run_evaluation(questions(), fake_retrieve({'q1': ['P1'], 'q2': ['P2'], 'q3': ['P3']}),
                            knowledge_base='kb', dataset='d', split='dev', k=1,
                            resolver=resolver_for(), output_dir=tmp_path)
    directory = tmp_path / result.task_id
    assert directory.is_dir()
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    assert summary['knowledge_base'] == 'kb' and summary['metrics']['recall@k'] == 1.0
    lines = (directory / 'samples.jsonl').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])['question_id'] == 'q1'
    # A second run with the same task id must not silently overwrite the first.
    with pytest.raises(FileExistsError):
        write_artifacts(result, tmp_path)


def test_configuration_captures_what_the_score_depends_on():
    result = run_evaluation(questions(1), fake_retrieve({'q1': ['P1']}), knowledge_base='kb',
                            dataset='d', split='dev', k=3, resolver=resolver_for(1))
    assert result.configuration == {'k': 3, 'split': 'dev', 'dataset': 'd',
                                    'knowledge_base': 'kb', 'questions': 1, 'resolver': True}
    assert result.summary()['questions'] == 1
    assert result.task_id.count('-') == 1        # <UTC stamp>-<digest>，同一次运行唯一
