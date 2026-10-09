"""Attribute remaining failures to retrieval or reranking.

For each question, compares the hybrid candidate pool with the reranked order
and sorts the outcome into one of three buckets:

- ``never_retrieved``      the labelled passage is absent from the candidate
                           pool, so no reranker can recover it;
- ``not_promoted``         it is in the pool but the reranker left it outside
                           the top k;
- ``promoted``             it is inside the top k after reranking.

The distinction is the point: "reranking did not help" and "the passage was
never retrieved" call for completely different follow-up work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'experiments' / 'rag-baseline' / 'evaluation' / 'retrieval_runs'


def latest(knowledge_base, mode):
    found = None
    for run in sorted(RUNS.glob('2026*')):
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        if summary['knowledge_base'] == knowledge_base and summary['dataset'].endswith(f':{mode}'):
            found = run
    return found


def samples(run):
    return {row['question_id']: row for row in
            (json.loads(line) for line in
             (run / 'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}


def first_slot(labels_per_hit, gold_ids):
    gold = set(gold_ids)
    for position, labels in enumerate(labels_per_hit or [], 1):
        if gold & set(labels):
            return position
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='把剩余失败归因到检索还是重排')
    parser.add_argument('--kb', required=True)
    parser.add_argument('--k', type=int, default=4)
    args = parser.parse_args(argv)

    before = samples(latest(args.kb, 'hybrid'))
    after = samples(latest(args.kb, 'rerank'))
    buckets = {'never_retrieved': [], 'not_promoted': [], 'promoted': [], 'fixed': [], 'broken': []}
    rows = []

    for question_id, row in after.items():
        gold = row.get('relevant_passage_ids') or []
        if not gold:
            continue
        before_row = before.get(question_id, {})
        rank_before = first_slot(before_row.get('labels_per_hit'), gold)
        rank_after = first_slot(row.get('labels_per_hit'), gold)
        # Was the gold passage anywhere in the reranked pool (the full list we
        # handed the model)?
        in_pool = any(set(gold) & set(labels) for labels in (row.get('labels_per_hit') or []))
        if rank_after is not None and rank_after <= args.k:
            outcome = 'promoted'
            if rank_before is None or rank_before > args.k:
                outcome = 'fixed'
        elif in_pool:
            outcome = 'not_promoted'
        else:
            outcome = 'never_retrieved'
        if rank_before is not None and rank_before <= args.k and (rank_after is None or rank_after > args.k):
            outcome = 'broken'
        buckets[outcome].append(question_id)
        rows.append({'question_id': question_id, 'rank_before': rank_before, 'rank_after': rank_after,
                     'gold_in_candidates': in_pool, 'outcome': outcome})

    print(f'=== {args.kb}：hybrid vs rerank（k={args.k}）===')
    for name, ids in buckets.items():
        print(f'  {name:16} {len(ids):3}  {", ".join(ids[:8])}{" …" if len(ids) > 8 else ""}')
    print()
    print('  逐题明细（仅列出重排后仍未进前 k 的题）：')
    for row in rows:
        if row['outcome'] in ('never_retrieved', 'not_promoted', 'broken'):
            print(f"    {row['question_id']:10} 重排前={row['rank_before']} 重排后={row['rank_after']} "
                  f"候选池含金={row['gold_in_candidates']} -> {row['outcome']}")
    out = RUNS / f'attribution-{args.kb}.json'
    out.write_text(json.dumps({'knowledge_base': args.kb, 'k': args.k,
                               'counts': {name: len(ids) for name, ids in buckets.items()},
                               'rows': rows}, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    print(f'\n产物：{out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
