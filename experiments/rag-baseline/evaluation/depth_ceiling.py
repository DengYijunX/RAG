"""At which retrieval depth does the gold passage first appear?

Reranking can only promote what is already in the candidate pool, so before
paying for a deep-pool rerank run this measures the ceiling: for each question,
the shallowest rank at which a gold-labelled chunk enters the ranking. If a gold
chunk is absent even at depth 200, no pool size will help and the fix has to be
chunking instead.

    python experiments/rag-baseline/evaluation/depth_ceiling.py --kb accounting
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.alignment import PassageResolver, load_passages
from eval.datasets import load_knowledge_base_dataset
from hybrid_retrieval import BM25Index, HybridRetriever
from rag.config import load_config
from rag.kb_service import KnowledgeBaseService
from retrieval_modes_experiment import chunk_text, load_published_index

BASELINE = REPO / 'experiments' / 'rag-baseline'
PROBE_DEPTH = 200


def gold_chunk_ids(resolver, chunks, passage_ids):
    """Chunk ids whose text resolves to one of the question's labels.

    Resolution is per chunk, so a chunk may carry several labels; the question is
    whether *any* of them matches.
    """
    wanted = set(passage_ids)
    found = set()
    for chunk in chunks:
        resolution = resolver.resolve([chunk], passage_ids=passage_ids)
        if resolution.labels_per_hit and set(resolution.labels_per_hit[0]) & wanted:
            found.add(chunk['id'])
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description='测量金标签首次进入排名的深度')
    parser.add_argument('--kb', choices=['smart-home', 'accounting'], required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--version', default='v1')
    parser.add_argument('--depth', type=int, default=PROBE_DEPTH)
    args = parser.parse_args(argv)

    config = load_config()
    store, chunks, cache = load_published_index(args.kb, config)
    by_id = {chunk['id']: chunk for chunk in chunks}
    service = KnowledgeBaseService(config, storage_root=REPO / 'chroma_db' / 'eval-baseline')

    def dense(question, depth):
        """Match the shape ``HybridRetriever`` expects: hits, not bare ids."""
        return [{'id': row['id'], 'score': row.get('distance')}
                for row in store.search(_vectors(question), depth)]

    # Question vectors are cached so a repeated question costs nothing.
    cache = dict(cache)
    from rag.config import load_config as _load

    def _vectors(question):
        if question not in cache:
            cache[question] = service.vectors([question])[0]
        return cache[question]

    hybrid = HybridRetriever(dense_search=dense, bm25=BM25Index(chunks))

    dataset = load_knowledge_base_dataset(BASELINE, args.kb, version=args.version,
                                          split=args.split)
    passages = load_passages(BASELINE / 'evidence.jsonl' if args.kb == 'smart-home'
                             else BASELINE / 'accounting' / 'articles.jsonl')
    resolver = PassageResolver(passages)

    print(f'知识库 {args.kb} | {args.version}/{args.split} | 深度上限 {args.depth}')
    buckets = {'≤4': 0, '5-10': 0, '11-20': 0, '21-50': 0, '51+': 0, '未出现': 0, '无标签': 0}
    for question in dataset.questions:
        if not question.relevant_passage_ids:
            buckets['无标签'] += 1
            continue
        gold = gold_chunk_ids(resolver, chunks, question.relevant_passage_ids)
        if not gold:
            buckets['未出现'] += 1
            print(f'  {question.id:12} 没有任何切片带该标签')
            continue
        ranking = hybrid.ranking(question.question, args.depth, args.depth)
        first = next((rank for rank, identifier in enumerate(ranking, 1)
                      if identifier in gold), None)
        if first is None:
            buckets['未出现'] += 1
            print(f'  {question.id:12} 前 {args.depth} 名内未出现（{len(gold)} 个切片带标签）')
            continue
        key = ('≤4' if first <= 4 else '5-10' if first <= 10 else '11-20' if first <= 20
               else '21-50' if first <= 50 else '51+')
        buckets[key] += 1
        if first > 4:
            print(f'  {question.id:12} 第 {first:3} 名（{len(gold)} 个切片带标签）')
    print()
    print('金标签首次出现的排名分布：', buckets)
    recoverable = sum(buckets[k] for k in ('5-10', '11-20', '21-50', '51+'))
    print(f'  可通过「加深池子 + 重排」挽救：{recoverable}')
    print(f'  加深池子也救不了（切片里就没有）：{buckets["未出现"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
