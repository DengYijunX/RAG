"""Diagnose the questions where dense retrieval found no labelled evidence.

Reads a stored retrieval run and prints, per zero-recall question, the labelled
passages and what was actually retrieved: source document, chunk text, and
whether the source document even contains the labelled passage. That last
column is what separates "retrieval failed" from "the label points at a
document this corpus never had".

    .venv/Scripts/python experiments/rag-baseline/evaluation/diagnose_zero_recall.py \
        --kb accounting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.alignment import load_passages, normalize_text

BASELINE = REPO / 'experiments' / 'rag-baseline'
RUNS = BASELINE / 'evaluation' / 'retrieval_runs'
CORPORA = {
    'smart-home': {'passages': BASELINE / 'evidence.jsonl',
                   'directory': BASELINE / 'smart-home' / 'corpus'},
    'accounting': {'passages': BASELINE / 'accounting' / 'articles.jsonl',
                   'directory': BASELINE / 'accounting' / 'markdown'},
}


def latest_run(knowledge_base):
    runs = sorted(RUNS.iterdir())
    for run in reversed(runs):
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        if summary['knowledge_base'] == knowledge_base:
            return run, summary
    raise SystemExit(f'没有 {knowledge_base} 的运行产物')


def load_chunks(knowledge_base):
    """Chunk id -> record, read from the generation the baseline published."""
    base = REPO / 'chroma_db' / 'eval-baseline'
    kb_dir = next((path for path in base.iterdir()
                   if path.is_dir() and path.name.startswith(f'eval-{knowledge_base}')), None)
    if kb_dir is None:
        raise SystemExit(f'未找到 {knowledge_base} 的评测知识库目录')
    pointer = json.loads((kb_dir / 'active.eval.json').read_text(encoding='utf-8'))
    records = kb_dir / pointer['generation'] / 'records.jsonl'
    return {record['id']: {'content': record['document'], **record['metadata']}
            for record in (json.loads(line) for line in
                           records.read_text(encoding='utf-8').splitlines() if line.strip())}


def zero_recall_samples(run, k):
    samples = [json.loads(line) for line in
               (run / 'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    output = []
    for sample in samples:
        gold = set(sample.get('relevant_passage_ids') or [])
        if not gold:
            continue
        found = set(sample.get('ranked_passage_ids', [])[:k]) & gold
        if not found:
            output.append(sample)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='排查零召回题目')
    parser.add_argument('--kb', choices=sorted(CORPORA), required=True)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--limit', type=int, default=10, help='每题展示的命中数')
    args = parser.parse_args(argv)

    corpus = CORPORA[args.kb]
    run, summary = latest_run(args.kb)
    passages = load_passages(corpus['passages'])
    documents = {path.name: path.read_text(encoding='utf-8')
                 for path in sorted(corpus['directory'].glob('**/*.md'))}
    chunks = load_chunks(args.kb)

    print(f'运行 {run.name} | {summary["dataset"]} | k={args.k} '
          f'| recall@k={summary["metrics"]["recall@k"]:.4f}')
    samples = zero_recall_samples(run, args.k)
    print(f'零召回题目：{len(samples)} / {summary["metrics"]["questions"]}')
    print()

    for sample in samples:
        gold = set(sample['relevant_passage_ids'])
        print('=' * 100)
        print(f'[{sample["question_id"]}] {sample["question"]}')
        for passage_id in sorted(gold):
            text = passages.get(passage_id, '')
            print(f'  标签 {passage_id}（{len(text)} 字）: {text[:90]!r}')
            # Does any source document contain this passage at all?
            hits = [name for name, content in documents.items()
                    if normalize_text(text) and normalize_text(text) in normalize_text(content)]
            print(f'    出处文档：{hits if hits else "语料中没有任何文档包含这段文字"}')
        print('  检索到的片段：')
        for hit in sample.get('hits', [])[:args.limit]:
            chunk = chunks.get(hit['id'], {})
            content = chunk.get('content', '')
            mark = '★' if hit['rank'] == 1 else ' '
            print(f'   {mark}[{hit["rank"]:2d}] {hit.get("source","")[:38]:38} '
                  f'{content[:70]!r}')
        print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
