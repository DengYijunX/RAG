"""Retrieval-mode comparison under the slot metric: dense vs BM25 vs hybrid.

Fixes corpus, chunking, embedding model, candidate depth and k, then changes
only the retrieval mode, so differences are attributable to the mode.

    .venv/Scripts/python experiments/rag-baseline/evaluation/retrieval_modes_experiment.py \
        --kb accounting --split dev --k 4 --depth 50

Relationship to the earlier work: ``hybrid_experiment.py`` (ADR / HYBRID_REPORT)
already compared BM25 and RRF, but scored *evidence coverage of a 2000-character
context* and used its own inline tokenizer. This script reuses the tested
primitives in ``hybrid_retrieval.py`` and scores the same runs with the slot
metric added in ``eval.metrics`` — "is the labelled passage inside the top k
retrieved chunks" — which is the metric the baseline report is stated in. It
does not replace or re-score the earlier report; the two answer different
questions and both stay on disk.

No fusion weight is tuned here: both arms get weight 1.0, so a hybrid win (or
loss) is not the product of dev-set fitting.

This measures retrieval coverage, not answer quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.alignment import PassageResolver, load_passages
from eval.backends.faiss_store import FaissVectors
from eval.datasets import load_knowledge_base_dataset
from eval.runner import run_evaluation
from hybrid_retrieval import BM25Index, HybridRetriever
from llm_rerank import LLMListwiseReranker
from rag.config import load_config
from rag.kb_service import KnowledgeBaseService

BASELINE = REPO / 'experiments' / 'rag-baseline'
STORAGE = REPO / 'chroma_db' / 'eval-baseline'
ARTIFACTS = BASELINE / 'evaluation' / 'retrieval_runs'
MODES = ('dense', 'bm25', 'hybrid', 'rerank')

CORPORA = {
    'smart-home': BASELINE / 'evidence.jsonl',
    'accounting': BASELINE / 'accounting' / 'articles.jsonl',
}


def load_published_index(knowledge_base):
    """Reuse the generation published by ``retrieval_baseline.py``."""
    kb_dir = next((path for path in STORAGE.iterdir()
                   if path.is_dir() and path.name.startswith(f'eval-{knowledge_base}')), None)
    if kb_dir is None:
        raise SystemExit(f'未找到 {knowledge_base} 的评测知识库，请先运行 retrieval_baseline.py')
    pointer = json.loads((kb_dir / 'active.eval.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in
               (kb_dir / pointer['generation'] / 'records.jsonl')
               .read_text(encoding='utf-8').splitlines() if line.strip()]
    chunks = [{'id': record['id'], 'content': record['document'],
               'source': record['metadata']['source']} for record in records]
    cache = json.loads((kb_dir / 'embedding-cache.json').read_text(encoding='utf-8'))
    return FaissVectors(kb_dir / pointer['generation']), chunks, cache


def chunk_text(record, fallback=''):
    """Text of a chunk record.

    Accepts either a plain string or the ``{'id', 'content', ...}`` records the
    index publishes, and returns the text either way. Passing the record itself
    to the reranker was a real bug: the prompt builder then sliced a dict.
    """
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return record.get('content', fallback)
    return fallback


def ranking_for(mode, question, depth, dense_hits, bm25, hybrid, *, by_id=None,
                reranker=None, pool=None):
    """Ranked chunk ids for one mode.

    ``rerank`` first builds the hybrid ranking at the pool depth, asks the chat
    model for a permutation, and returns the reranked ids followed by the rest.
    Keeping the tail preserves recall while the head is reordered, so a change
    in the top-k metrics cannot come from having dropped candidates.
    """
    if mode == 'dense':
        return [hit['id'] for hit in dense_hits(question, depth)]
    if mode == 'bm25':
        return bm25.search(question, depth)
    if mode == 'hybrid':
        return hybrid.ranking(question, depth, depth)
    # Rerank: the model sees only the pool, but the returned ranking keeps the
    # whole candidate list — reranked pool first, then the untouched remainder.
    # Without the tail, choosing a small pool would cap recall, and a drop in
    # recall could not be told apart from a reranking mistake.
    candidates = hybrid.ranking(question, depth, depth)
    pool_ids = candidates[:pool or depth]
    if reranker is None or len(pool_ids) < 2:
        return candidates
    lookup = by_id or {}
    payload = [{'id': identifier, 'content': chunk_text(lookup.get(identifier))}
               for identifier in pool_ids]
    reordered, _ = reranker.rerank(question, payload)
    head = [candidate['id'] for candidate in reordered]
    seen = set(head)
    return head + [identifier for identifier in candidates if identifier not in seen]


def show(value, digits=4):
    """Format a metric that may be None (an unscoreable run, not a zero)."""
    return 'n/a' if value is None else f'{value:.{digits}f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description='dense / BM25 / hybrid 检索模式对照（槽位口径）')
    parser.add_argument('--kb', choices=sorted(CORPORA), required=True)
    parser.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    parser.add_argument('--version', default='v1', choices=['v1', 'v2'])
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--depth', type=int, default=50, help='每条臂的候选深度')
    parser.add_argument('--rerank-pool', type=int, default=20,
                        help='rerank 模式下交给模型排序的候选数（越多越贵）')
    parser.add_argument('--limit', type=int, default=None, help='只跑前 N 题（冒烟用）')
    parser.add_argument('--modes', default=','.join(MODES))
    args = parser.parse_args(argv)

    selected = [mode.strip() for mode in args.modes.split(',') if mode.strip()]
    unknown = [mode for mode in selected if mode not in MODES]
    if unknown:
        raise SystemExit(f'未知模式：{unknown}；可选 {MODES}')

    split = None if args.split == 'all' else args.split
    questions = load_knowledge_base_dataset(BASELINE, args.kb, split=split, version=args.version)
    passages = load_passages(CORPORA[args.kb])
    resolver = PassageResolver(passages)
    store, chunks, cache = load_published_index(args.kb)
    if not chunks:
        raise SystemExit('发布索引为空')
    by_id = {chunk['id']: chunk for chunk in chunks}

    print(f'知识库 {args.kb} | {args.version}/{args.split} | 题目 {len(questions)}'
          f' | 语料片段 {len(chunks)} | 候选深度 {args.depth} | k={args.k}', flush=True)
    bm25 = BM25Index().add([chunk['id'] for chunk in chunks],
                           [chunk['content'] for chunk in chunks])

    def dense_hits(question, depth):
        return store.search(cache[question], depth)

    hybrid = HybridRetriever(dense_search=dense_hits, bm25=bm25)
    reranker = LLMListwiseReranker(load_config()) if 'rerank' in selected else None

    results = {}
    for mode in selected:
        started = time.time()
        pools = {}

        def retrieve(question, mode=mode):
            ids = ranking_for(mode, question.question, args.depth, dense_hits, bm25, hybrid,
                              by_id=by_id, reranker=reranker, pool=args.rerank_pool)
            pools[question.id] = ids
            return [{'id': chunk_id, 'content': by_id[chunk_id]['content'],
                     'source': by_id[chunk_id]['source'], 'rank': rank}
                    for rank, chunk_id in enumerate(ids, 1) if chunk_id in by_id]

        run = run_evaluation(questions.questions, retrieve, knowledge_base=args.kb,
                             dataset=f'{args.version}/{args.split}:{mode}',
                             split=args.split or 'all', k=args.k, resolver=resolver,
                             output_dir=ARTIFACTS, max_questions=args.limit)
        elapsed = time.time() - started
        results[mode] = {'summary': run.summary(), 'pools': pools, 'seconds': elapsed,
                         'output_dir': run.output_dir}
        m = run.metrics
        print(f'  [{mode:6}] recall_in_top_{args.k}={show(m["recall_in_top_k"])} '
              f'precision_in_top_{args.k}={show(m["precision_in_top_k"])} '
              f'ndcg_in_top_{args.k}={show(m["ndcg_in_top_k"])} '
              f'mean_first_hit_slot={show(m["mean_first_hit_slot"], 3)} '
              f'mrr_first_hit={show(m["mrr_first_hit"])} ({elapsed:.1f}s)', flush=True)

    if 'dense' in results and 'bm25' in results:
        overlaps, keyword_only = [], 0
        for question in questions.questions:
            dense = set(results['dense']['pools'].get(question.id, []))
            keyword = set(results['bm25']['pools'].get(question.id, []))
            if dense or keyword:
                overlaps.append(len(dense & keyword) / max(1, len(dense | keyword)))
            if keyword - dense:
                keyword_only += 1
        print(f'  两臂候选池 Jaccard 均值 {sum(overlaps)/len(overlaps):.3f}；'
              f'{keyword_only}/{len(questions)} 题关键词臂带来向量臂没有的候选', flush=True)

    comparison = {mode: {**data['summary'], 'seconds': data['seconds']}
                  for mode, data in results.items()}
    if reranker is not None:
        stats = reranker.stats()
        print(f'  重排调用：{stats["calls"]} 次，回退 {stats["failures"]} 次，'
              f'prompt {stats["prompt_tokens"]} / completion {stats["completion_tokens"]} tokens，'
              f'累计 {stats["seconds"]}s', flush=True)
        comparison['rerank_stats'] = stats
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = ARTIFACTS / f'modes-{args.kb}-{stamp}.json'
    out.write_text(json.dumps({'knowledge_base': args.kb, 'dataset': f'{args.version}/{args.split}',
                               'k': args.k, 'depth': args.depth,
                               'rerank_pool': args.rerank_pool if reranker else None,
                               'modes': comparison},
                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n对照结果：{out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
