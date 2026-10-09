"""Does neighbour expansion recover the articles that chunking split apart?

The baseline's slot metric asks whether a *chunk* holds a labelled passage.
Several accounting questions fail that metric because the article is split
across adjacent chunks, so no chunk holds it. This script measures a second,
end-to-end question:

    does the text we would actually hand to the model contain the full article?

It scores both readings over the same retrieval, at radius 0 (no expansion)
and radius 1 (each retrieved chunk plus its contiguous neighbours), so the
effect of expansion is isolated from any retrieval change.

    .venv/Scripts/python experiments/rag-baseline/evaluation/context_expansion_experiment.py \
        --kb accounting --split dev --k 4 --modes hybrid --radius 0,1,2

Retrieval artifacts are reused; no model is called. This measures evidence
availability in the context, not answer quality.
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

from eval.alignment import load_passages, normalize_text
from eval.datasets import load_knowledge_base_dataset
from neighbour_context import NeighbourExpander
from retrieval_modes_experiment import CORPORA, load_published_index, ranking_for

from hybrid_retrieval import BM25Index, HybridRetriever

BASELINE = REPO / 'experiments' / 'rag-baseline'
ARTIFACTS = BASELINE / 'evaluation' / 'retrieval_runs'


def score_context(questions, expander, ranked_ids_for, radius, k):
    """Per-question outcome for one expansion radius.

    A question counts as complete when every labelled passage's full text is
    present in the normalised context that would be handed onward.
    """
    rows, complete, partial = [], 0, 0
    for question in questions.questions:
        gold = set(question.relevant_passage_ids)
        if not gold:
            continue
        ranked = ranked_ids_for(question)
        neighbourhoods = expander.expand_ranking(ranked, radius=radius, limit=k)
        text = normalize_text('\n'.join(item.content for item in neighbourhoods))
        found = sorted(passage_id for passage_id in gold
                       if normalize_text(PASSAGES[passage_id]) in text)
        rows.append({'question_id': question.id, 'gold': len(gold), 'found': len(found),
                     'passage_ids': found,
                     'chunks': [identifier for item in neighbourhoods for identifier in item.chunk_ids]})
        if len(found) == len(gold):
            complete += 1
        elif found:
            partial += 1
    total = len(rows)
    return {'scored': total, 'complete': complete, 'partial': partial,
            'missing': total - complete - partial,
            'complete_rate': complete / total if total else None, 'rows': rows}


PASSAGES = {}


def main(argv=None):
    global PASSAGES
    parser = argparse.ArgumentParser(description='邻段扩展是否恢复被切开的条文')
    parser.add_argument('--kb', choices=sorted(CORPORA), required=True)
    parser.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    parser.add_argument('--version', default='v1', choices=['v1', 'v2'])
    parser.add_argument('--k', type=int, default=4, help='送入上下文的片段数')
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--modes', default='hybrid', help='逗号分隔：dense,bm25,hybrid')
    parser.add_argument('--radius', default='0,1,2', help='逗号分隔的扩展半径')
    args = parser.parse_args(argv)

    split = None if args.split == 'all' else args.split
    questions = load_knowledge_base_dataset(BASELINE, args.kb, split=split, version=args.version)
    PASSAGES = load_passages(CORPORA[args.kb])
    store, chunks, cache = load_published_index(args.kb)
    by_id = {chunk['id']: chunk for chunk in chunks}
    expander = NeighbourExpander(chunks)
    bm25 = BM25Index().add([chunk['id'] for chunk in chunks],
                           [chunk['content'] for chunk in chunks])

    def dense_hits(question, depth):
        return store.search(cache[question], depth)

    hybrid = HybridRetriever(dense_search=dense_hits, bm25=bm25)
    radii = [int(value) for value in args.radius.split(',') if value.strip()]
    modes = [value.strip() for value in args.modes.split(',') if value.strip()]

    print(f'知识库 {args.kb} | {args.version}/{args.split} | 片段 {len(chunks)} | '
          f'送入 k={args.k} | 模式 {modes} | 半径 {radii}', flush=True)
    output = {'knowledge_base': args.kb, 'dataset': f'{args.version}/{args.split}',
              'k': args.k, 'depth': args.depth, 'modes': {}}

    for mode in modes:
        def ranked_ids_for(question, mode=mode):
            return ranking_for(mode, question.question, args.depth, dense_hits, bm25, hybrid)

        entry = {}
        for radius in radii:
            result = score_context(questions, expander, ranked_ids_for, radius, args.k)
            entry[f'radius_{radius}'] = {key: value for key, value in result.items() if key != 'rows'}
            entry[f'radius_{radius}']['rows'] = result['rows']
            print(f'  [{mode:6} radius={radius}] 完整命中 {result["complete"]}/{result["scored"]} = '
                  f'{result["complete_rate"]:.4f}  部分 {result["partial"]}  全缺 {result["missing"]}',
                  flush=True)
        output['modes'][mode] = entry

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = ARTIFACTS / f'context-{args.kb}-{stamp}.json'
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n产物：{out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
