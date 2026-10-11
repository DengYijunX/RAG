"""Run the held-out questions through the product path and record them for review.

The evaluation harness retrieves through its own FAISS store; this drives the
product service instead, so the answers come from the configuration the product
actually ships with (FAISS backend, structural chunking, deep-pool rerank). The
judge is reused from the evaluation side — retrieval and generation are the
product's, only the first-pass scoring is borrowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'rag-baseline' / 'evaluation'))

from answer_quality_review import citation_checks
from answer_review import DIMENSIONS, AnswerJudge
from eval.alignment import load_passages, normalize_text
from eval.datasets import load_knowledge_base_dataset
from rag.config import load_config
from rag.kb_service import KnowledgeBaseService

BASELINE = ROOT / 'experiments' / 'rag-baseline'
RUNS = BASELINE / 'evaluation' / 'product_runs'
LABELS = {'accounting': BASELINE / 'accounting' / 'articles.jsonl',
          'smart-home': BASELINE / 'evidence.jsonl'}


def evidence_status(citations, reference_texts):
    if not reference_texts:
        return '该题无标注要点（澄清/拒答类）'
    cited = normalize_text(''.join(citation.get('content') or '' for citation in citations))
    present = [identifier for identifier, text in reference_texts.items()
               if normalize_text(text) and normalize_text(text) in cited]
    if len(present) == len(reference_texts):
        return f'上下文已含全部参考原文（{len(present)}/{len(reference_texts)}）'
    if present:
        missing = [key for key in reference_texts if key not in present]
        return (f'上下文仅含部分参考原文（{len(present)}/{len(reference_texts)}）：'
                f'缺 {missing}')
    return f'上下文未含参考原文（0/{len(reference_texts)}）：检索没给到'


def main(argv=None):
    parser = argparse.ArgumentParser(description='用产品路径跑保留集题目并记录')
    parser.add_argument('--kb', required=True, help='产品知识库名称')
    parser.add_argument('--corpus-kb', choices=['accounting', 'smart-home'], required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--version', default='v1')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    service = KnowledgeBaseService(config)
    bases = [base for base in service.registry.list() if base.name == args.kb]
    if not bases:
        raise SystemExit(f'找不到产品知识库 {args.kb}，请先注册并建索引')
    kb_id = bases[0].id
    active = service.active(kb_id)
    print(f'产品知识库 {kb_id} | generation={active["generation"]} chunks={active["chunks"]}')
    print(f'后端 {config.vector_backend} | 深池 {config.retrieval_depth} | '
          f'重排池 {config.rerank_pool} | 生效 {config.deep_rerank_enabled}', flush=True)

    dataset = load_knowledge_base_dataset(BASELINE, args.corpus_kb, version=args.version,
                                         split=args.split)
    questions = list(dataset.questions)
    if args.limit:
        questions = questions[:args.limit]
    passages = load_passages(LABELS[args.corpus_kb])
    judge = AnswerJudge(config)

    # citation_checks needs the source text, not just its name: it verifies that
    # each cited snippet really appears in the document it claims to come from.
    sources = {}
    for document in service.documents(kb_id):
        payload = service.source(kb_id, document['id'])
        sources[document['source']] = payload['content']
    print(f'载入 {len(sources)} 份原文用于引用核对', flush=True)

    records = []
    for question in questions:
        started = time.time()
        answer = service.query(kb_id, question.question)
        checks = citation_checks(answer.get('answer', ''), answer.get('citations', []),
                                 sources,
                                 question.metadata.get('expected_action') or '')
        outcome = judge.review(question.question, answer.get('answer', ''),
                               answer.get('citations', []),
                               question.metadata.get('expected_points') or [],
                               question.metadata.get('expected_action') or '')
        reference_texts = {identifier: passages.get(identifier, '')
                           for identifier in question.relevant_passage_ids}
        records.append({
            'question_id': question.id, 'question': question.question,
            'category': question.metadata.get('category'),
            'expected_action': question.metadata.get('expected_action'),
            'reference_points': question.metadata.get('expected_points') or [],
            'status': answer.get('status'), 'answer': answer.get('answer', ''),
            'citations': answer.get('citations', []),
            'evidence_status': evidence_status(answer.get('citations', []), reference_texts),
            'automatic_checks': checks, 'automatic_passed': all(checks.values()),
            'assistant_scores': outcome.scores,
            'assistant_error_labels': outcome.error_labels,
            'assistant_justification': outcome.justification,
            'assistant_valid': outcome.valid,
            'human_scores': {dimension: None for dimension in DIMENSIONS},
            'human_error_labels': [], 'human_reviewed': False,
            'seconds': round(time.time() - started, 2),
        })
        print(f"  {question.id:11} {answer.get('status'):22} "
              f"引用 {len(answer.get('citations', []))} 条 {records[-1]['seconds']}s", flush=True)

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    directory = RUNS / f'{args.corpus_kb}-{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    reviewed = [record for record in records if record['assistant_valid']]
    summary = {
        'product_knowledge_base': kb_id, 'corpus': args.corpus_kb,
        'dataset': f'{args.version}/{args.split}', 'questions': len(records),
        'vector_backend': config.vector_backend,
        'retrieval_depth': config.retrieval_depth, 'rerank_pool': config.rerank_pool,
        'deep_rerank_enabled': config.deep_rerank_enabled,
        'assistant_reviewed': len(reviewed),
        'automatic_checks_passed': sum(1 for r in records if r['automatic_passed']),
        'full_marks': {
            dimension: {
                'numerator': sum(1 for r in reviewed
                                 if (r['assistant_scores'].get(dimension) == 2)),
                'denominator': sum(1 for r in reviewed
                                   if r['assistant_scores'].get(dimension) not in (None, 'na')),
                'rate': None}
            for dimension in DIMENSIONS},
        'error_labels': {},
        'judge_stats': judge.stats(),
        'note': ('产品路径（非评测脚本）生成的答案；评分仍是助手初审，未经人工审阅。'
                 '样本为 v1 test 全量，按标准此后不得再用于调参。'),
    }
    for dimension, block in summary['full_marks'].items():
        block['rate'] = (block['numerator'] / block['denominator']
                         if block['denominator'] else None)
    for record in reviewed:
        for label in record['assistant_error_labels']:
            summary['error_labels'][label] = summary['error_labels'].get(label, 0) + 1

    (directory / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with open(directory / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print()
    print(json.dumps({**summary, 'judge_stats': judge.stats()}, ensure_ascii=False, indent=2))
    print(f'\n产物：{directory}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
