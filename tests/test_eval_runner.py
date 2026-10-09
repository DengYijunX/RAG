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


def test_mrr_only_moves_between_multiple_gold_passages():
    """Documents a real limit of MRR in this setup.

    The resolver only credits passages the question labels, so a chunk holding
    an unlabelled passage contributes no rank. With a single gold passage MRR
    can therefore only be 1.0 or 0.0 — it cannot see that a gold passage was
    found late. Use recall@k for coverage, and the per-sample
    ``labels_per_hit`` to reason about ordering.
    """
    resolver = PassageResolver({'P1': '文本 P1', 'P2': '文本 P2'})
    item = [Question(id='q1', question='q1', split='dev', relevant_passage_ids=['P1'])]

    def retrieve(question):
        return [{'id': 'c1', 'content': '文本 P2', 'rank': 1},
                {'id': 'c2', 'content': '文本 P1', 'rank': 2}]

    result = run_evaluation(item, retrieve, k=2, resolver=resolver)
    # P2 is not a label of this question, so it is not a rank position.
    assert result.samples[0].ranked_passage_ids == ['P1']
    assert result.samples[0].labels_per_hit == [[], ['P1']]
    assert result.metrics['mrr'] == 1.0


def test_two_gold_passages_give_mrr_between_zero_and_one():
    resolver = PassageResolver({'P1': '文本 P1', 'P2': '文本 P2'})
    item = [Question(id='q1', question='q1', split='dev', relevant_passage_ids=['P1', 'P2'])]

    def retrieve(question):
        return [{'id': 'c1', 'content': '文本 P2', 'rank': 1},
                {'id': 'c2', 'content': '文本 P1', 'rank': 2}]

    result = run_evaluation(item, retrieve, k=2, resolver=resolver)
    assert result.samples[0].ranked_passage_ids == ['P2', 'P1']
    assert result.metrics['recall@k'] == 1.0
    assert result.metrics['mrr'] == 1.0      # P2 is gold too, and it is first


def test_unlabelled_chunks_do_not_shift_labelled_ranks():
    """A chunk holding no labelled passage contributes no rank position."""
    resolver = PassageResolver({'P1': '文本 P1'})
    item = [Question(id='q1', question='q1', split='dev', relevant_passage_ids=['P1'])]

    def retrieve(question):
        return [{'id': 'c1', 'content': '完全无关的一段', 'rank': 1},
                {'id': 'c2', 'content': '文本 P1', 'rank': 2}]

    result = run_evaluation(item, retrieve, k=2, resolver=resolver)
    assert result.metrics['mrr'] == 1.0
    assert result.metrics['recall@k'] == 1.0
    # The unlabelled chunk is still reported, never hidden.
    sample = result.samples[0]
    assert sample.ranked_passage_ids == ['P1']
    assert sample.labels_per_hit == [[], ['P1']]
    assert sample.coverage == 0.5


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
    # A chunk containing no labelled passage contributes no rank position.
    assert failed.ranked_passage_ids == []
    assert failed.labels_per_hit == [[]]


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


def test_a_repeat_run_in_the_same_second_gets_its_own_directory(tmp_path):
    """Back-to-back modes share a task id; neither run may replace the other."""
    result = run_evaluation(questions(1), fake_retrieve({'q1': ['P1']}), k=1,
                            resolver=resolver_for(1))
    first = write_artifacts(result, tmp_path)
    second = write_artifacts(result, tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert second.name.startswith(result.task_id)
    assert (first / 'summary.json').exists() and (second / 'summary.json').exists()


def test_configuration_captures_what_the_score_depends_on():
    result = run_evaluation(questions(1), fake_retrieve({'q1': ['P1']}), knowledge_base='kb',
                            dataset='d', split='dev', k=3, resolver=resolver_for(1))
    assert result.configuration == {'k': 3, 'split': 'dev', 'dataset': 'd',
                                    'knowledge_base': 'kb', 'questions': 1, 'resolver': True}
    assert result.summary()['questions'] == 1
    assert result.task_id.count('-') == 1        # <UTC stamp>-<digest>，同一次运行唯一
