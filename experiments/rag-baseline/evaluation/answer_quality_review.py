"""Generation-side review of the predeclared 20-question diagnostic sample.

Closes the biggest gap in the evaluation: every number so far has been a
retrieval proxy, and nothing checked whether the answer built from those
chunks is actually right.

What it does per question:

1. retrieves through the application service and generates an answer;
2. runs the automatic checks QUALITY_RUBRIC.md allows (citation ids exist,
   unique, sources belong to the knowledge base, snippets really appear in the
   source) — these check *form*, never meaning;
3. asks the configured chat model to apply the rubric as an **assistant first
   pass**, with validated output.

What it deliberately does not do: no keyword scoring, no character-coverage
scoring standing in for semantic grading, and no claiming human review. Every
record leaves ``human_*`` empty and the summary states that the sample is a
targeted diagnostic selection, not a population estimate.

    .venv/Scripts/python experiments/rag-baseline/evaluation/answer_quality_review.py \
        --kb smart-home
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_review import DIMENSIONS, AnswerJudge
from eval.alignment import normalize_text
from eval.datasets import load_knowledge_base_dataset
from rag.config import load_config
from rag.kb_service import KnowledgeBaseService
from retrieval_baseline import EmbeddingCache, embedding_fingerprint

BASELINE = REPO / 'experiments' / 'rag-baseline'
STORAGE = REPO / 'chroma_db' / 'eval-baseline'
OUTPUT = BASELINE / 'evaluation' / 'quality_runs'
SAMPLE = BASELINE / 'evaluation' / 'quality_sample.json'


def load_sample(path=SAMPLE):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('split') != 'dev':
        raise SystemExit('样本必须来自 dev 划分')
    return payload


def build_service(config):
    """Service over the published evaluation index, same embedding as scored."""
    kb_dirs = {path.name: path for path in STORAGE.iterdir() if path.is_dir()}
    return KnowledgeBaseService(config, storage_root=STORAGE), kb_dirs


def published_index(kb_dirs, knowledge_base, embedding_model, config):
    """The generation and evaluation indexes are separate pointers.

    ``retrieval_baseline.py`` publishes to ``active.eval.json`` with a FAISS
    generation; the application's ``active.json`` points at Chroma and is not
    written by the baseline. This script therefore retrieves through the
    published FAISS index and only borrows the service for citations and
    generation — otherwise every question fails with ``IndexUnavailable``.

    The returned cache embeds on demand: the sample contains questions that were
    not part of the baseline run, and a bare dict lookup would raise KeyError
    and be misread as a failed question.
    """
    for name, path in sorted(kb_dirs.items()):
        if not name.startswith(f'eval-{knowledge_base}'):
            continue
        pointer_path = path / 'active.eval.json'
        if not pointer_path.exists():
            continue
        pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
        built_with = (pointer.get('embedding') or {}).get('model')
        if built_with != embedding_model:
            continue
        cache = EmbeddingCache(path / 'embedding-cache.json',
                              embedding_fingerprint(config))
        if cache.discarded_reason:
            raise SystemExit(f'Embedding 缓存不可用：{cache.discarded_reason}')
        return name, pointer, path, cache
    raise SystemExit(f'未找到 {knowledge_base} 且由 {embedding_model} 建立的评测索引，'
                     f'请先运行 retrieval_baseline.py')


def query_with_published_index(service, kb_id, base_dir, generation, question, cache, top_k=4,
                               context_radius=0, depth=0, reranker=None, pool=20):
    """Top-k retrieve from the published FAISS generation, then generate.

    ``depth`` retrieves a deeper pool than ``top_k``; ``reranker`` then reorders
    that pool and the first ``top_k`` become the citations. Both are off by
    default so the baseline behaviour is unchanged. Retrieval metrics are not
    affected by either — this path only builds the prompt.
    """
    from eval.backends.faiss_store import FaissVectors
    from rag.context import expand_hits
    from rag.generation import load_chunk_records

    store = FaissVectors(base_dir / generation)
    vector = cache.vectors(service, [question])[0]
    search_depth = depth or top_k
    chunks = {chunk['id']: chunk for chunk in load_chunk_records(base_dir / generation)}
    hits = []
    for row in store.search(vector, search_depth):
        chunk = chunks.get(row['id'])
        if chunk is not None:
            hits.append({**chunk, 'score': row.get('distance')})

    if reranker is not None and len(hits) > 1:
        head = hits[:pool or len(hits)]
        payload = [{'id': hit['id'], 'content': hit.get('content') or ''} for hit in head]
        reordered, _ = reranker.rerank(question, payload)
        order = [item['id'] for item in reordered]
        remaining = [hit['id'] for hit in hits if hit['id'] not in set(order)]
        index = {hit['id']: hit for hit in hits}
        hits = [index[identifier] for identifier in order + remaining if identifier in index]

    hits = hits[:top_k]
    if context_radius:
        hits = expand_hits(hits, load_chunk_records(base_dir / generation), context_radius)
    citations = service.citations(kb_id, hits)
    if not citations:
        return {'status': 'insufficient_evidence', 'answer': '本次检索片段不足以回答。',
                'citations': []}
    return {'status': 'answer', 'answer': service.generate(question, citations),
            'citations': citations}


def cited_numbers(answer):
    """Citation markers in an answer, excluding document-issue brackets.

    Chinese standards write their document number as ``财会[2006]3号``, so a
    naive ``\\[(\\d+)\\]`` sweep reads the year as a citation. That produced a
    false "referenced id does not exist" failure on an answer that was in fact
    correct, so brackets holding a four-digit year are skipped — no citation
    list reaches a thousand entries, so a four-digit marker cannot be a real one.
    """
    numbers = []
    for value in re.findall(r'\[(\d+)\]', answer or ''):
        if len(value) >= 4:
            continue
        numbers.append(int(value))
    return numbers


def citation_checks(answer, citations, sources, expected_action=''):
    """Only the checks the rubric permits: existence, uniqueness, provenance.

    Marker presence is required only where the rubric requires citations. A
    clarification or abstention answer makes no substantive factual claim, so
    the rubric explicitly exempts it from citation requirements; demanding a
    marker there would fail a correct refusal.
    """
    exempt = expected_action in {'clarify', 'abstain'}
    ids = [citation['citation'] for citation in citations]
    referenced = cited_numbers(answer)
    checks = {
        'citation_ids_unique': len(ids) == len(set(ids)),
        'referenced_ids_exist': all(value in ids for value in referenced),
        'sources_in_selected_kb': all(citation['source'] in sources for citation in citations),
        'snippets_in_source': all(
            normalize_text(citation['content']) in normalize_text(sources.get(citation['source'], ''))
            for citation in citations),
    }
    if not exempt:
        checks['answer_has_citation_markers'] = bool(referenced) if citations else True
    return checks


def judge_samples(config, selected, service, kb_id, base_dir, generation, cache, sources,
                  passages, context_radius=0, depth=0, reranker=None, pool=20):
    judge = AnswerJudge(config)
    records = []
    for question in selected:
        started = time.time()
        try:
            response = query_with_published_index(service, kb_id, base_dir, generation,
                                                  question.question, cache,
                                                  context_radius=context_radius,
                                                  depth=depth, reranker=reranker, pool=pool)
            generated = {'status': response.get('status'), 'answer': response.get('answer', ''),
                         'citations': response.get('citations', [])}
        except Exception as exc:
            records.append({'question_id': question.id, 'error': type(exc).__name__,
                            'message': str(exc)[:200]})
            continue
        checks = citation_checks(generated['answer'], generated['citations'], sources,
                                 question.metadata.get('expected_action') or '')
        outcome = judge.review(question.question, generated['answer'],
                               generated['citations'],
                               question.metadata.get('expected_points') or [],
                               question.metadata.get('expected_action') or '')
        gold_present = sorted(
            passage_id for passage_id in question.relevant_passage_ids
            if normalize_text(passages.get(passage_id, '')) and any(
                normalize_text(passages[passage_id]) in normalize_text(citation['content'])
                or normalize_text(citation['content']) in normalize_text(passages[passage_id])
                for citation in generated['citations']))
        records.append({
            'question_id': question.id, 'question': question.question,
            'category': question.metadata.get('category'),
            'expected_action': question.metadata.get('expected_action'),
            'status': generated['status'], 'answer': generated['answer'],
            # Keep provenance fields, including whether the text was expanded by
            # neighbour context: dropping them made the expansion report read 0.
            'citations': [{**item, 'content': (item.get('content') or '')[:4000]}
                          for item in generated['citations']],
            'automatic_checks': checks, 'automatic_passed': all(checks.values()),
            'gold_passages_in_context': gold_present,
            'gold_passage_count': len(question.relevant_passage_ids),
            'assistant_scores': outcome.scores,
            'assistant_error_labels': outcome.error_labels,
            'assistant_severity': outcome.severity,
            'assistant_justification': outcome.justification,
            'assistant_valid': outcome.valid,
            'assistant_invalid_reason': '' if outcome.valid else outcome.reason,
            'human_scores': {dimension: None for dimension in DIMENSIONS},
            'human_error_labels': [],
            'human_reviewed': False,
            'seconds': round(time.time() - started, 2),
        })
    return records, judge.stats()


def summarize(records, knowledge_base, config, dataset_version):
    reviewed = [row for row in records if row.get('assistant_valid')]
    def rate(dimension, value):
        scored = [row for row in reviewed if row['assistant_scores'].get(dimension) not in (None, 'na')]
        hits = [row for row in scored if row['assistant_scores'][dimension] == value]
        return {'numerator': len(hits), 'denominator': len(scored),
                'rate': len(hits) / len(scored) if scored else None}
    labels = {}
    for row in reviewed:
        for label in row['assistant_error_labels']:
            labels[label] = labels.get(label, 0) + 1
    return {
        'knowledge_base': knowledge_base,
        'dataset_version': dataset_version,
        'embedding_model': config.embedding_model,
        'generation_model': config.llm_model,
        'questions': len(records),
        'assistant_reviewed': len(reviewed),
        'assistant_invalid': len(records) - len(reviewed),
        'automatic_checks_passed': sum(1 for row in records if row.get('automatic_passed')),
        'full_marks': {dimension: rate(dimension, 2) for dimension in DIMENSIONS},
        'error_labels': labels,
        'human_reviewed': 0,
        'note': ('助手初审，不是人工审阅；样本为定向诊断选样，不估计总体准确率、不计算置信区间。'
                 '自动检查只验证引用形式，不验证语义支持。'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='生成侧答案审阅（20 题定向诊断样本）')
    parser.add_argument('--kb', choices=['smart-home', 'accounting'], required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--sample', default=str(SAMPLE))
    parser.add_argument('--context-radius', type=int, default=None,
                        help='邻段扩展半径；缺省用配置 RAG_CONTEXT_RADIUS，0 表示关闭')
    parser.add_argument('--depth', type=int, default=0,
                        help='检索深度；大于 top_k 时先取深池再截断')
    parser.add_argument('--rerank', action='store_true',
                        help='对深池做专用 rerank 后取前 k（需配置 RERANK_*）')
    parser.add_argument('--pool', type=int, default=20, help='重排候选池大小')
    args = parser.parse_args(argv)

    config = load_config()
    context_radius = config.context_radius if args.context_radius is None else args.context_radius
    reranker = None
    if args.rerank:
        if not config.rerank_configured:
            raise SystemExit('未配置 RERANK_* ，无法重排')
        from specialized_rerank import SpecializedReranker

        reranker = SpecializedReranker(config)
    sample = load_sample(args.sample)
    requested = sample['knowledge_bases'][args.kb]
    dataset_version = sample.get('dataset_version', 'v2')
    dataset = load_knowledge_base_dataset(BASELINE, args.kb, version=dataset_version)
    lookup = {question.id: question for question in dataset.questions}
    missing = [key for key in requested if key not in lookup]
    if missing:
        raise SystemExit(f'样本题目不在 {dataset_version} 数据集中：{missing}')
    selected = [lookup[key] for key in requested]
    if any(question.split != 'dev' for question in selected):
        raise SystemExit('样本包含非 dev 题目')
    if args.limit:
        selected = selected[:args.limit]

    service, kb_dirs = build_service(config)
    kb_id, pointer, base_dir, cache = published_index(kb_dirs, args.kb, config.embedding_model,
                                                      config)
    sources = {}
    for document in service.documents(kb_id):
        text_path = service.base_dir(kb_id) / 'text' / f"{document['id']}.txt"
        if text_path.exists():
            sources[document['source']] = text_path.read_text(encoding='utf-8')

    from eval.alignment import load_passages
    passages = load_passages(BASELINE / 'evidence.jsonl' if args.kb == 'smart-home'
                             else BASELINE / 'accounting' / 'articles.jsonl')

    print(f'知识库 {args.kb} | 索引 {kb_id} generation={pointer["generation"]} '
          f'| 模型 {config.embedding_model} + {config.llm_model} | 题目 {len(selected)} '
          f'| 邻段扩展半径 {context_radius} | 深度 {args.depth or 4} '
          f'| 重排 {"on" if reranker else "off"}', flush=True)
    records, stats = judge_samples(config, selected, service, kb_id, base_dir,
                                   pointer['generation'], cache, sources, passages,
                                   context_radius=context_radius, depth=args.depth,
                                   reranker=reranker, pool=args.pool)
    summary = {**summarize(records, args.kb, config, dataset_version),
               'context_radius': context_radius, 'retrieval_depth': args.depth or 4,
               'reranked': bool(reranker), 'rerank_pool': args.pool if reranker else None}
    if reranker is not None:
        summary['rerank_stats'] = reranker.stats()
    summary['average_context_chars'] = round(
        sum(len(citation.get('content') or '')
            for record in records for citation in (record.get('citations') or []))
        / max(1, sum(len(record.get('citations') or []) for record in records)))
    summary['expanded_citations'] = sum(
        1 for record in records for citation in (record.get('citations') or [])
        if citation.get('expanded'))

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    directory = OUTPUT / f'{args.kb}-{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'summary.json').write_text(
        json.dumps({**summary, 'judge_stats': stats}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    with open(directory / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps({**summary, 'judge_stats': stats}, ensure_ascii=False, indent=2))
    print(f'\n产物：{directory}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
