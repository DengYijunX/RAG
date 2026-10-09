"""Dense-only retrieval baseline over the v1 corpus.

Answers one question with real numbers: given the v1 development questions,
how often does pure vector Top-K actually return the labelled passage, and how
high does it rank it?

    .venv/Scripts/python experiments/rag-baseline/evaluation/retrieval_baseline.py \
        --kb accounting --split dev --k 4 --top-k 10

What it does, in order:

1. registers a knowledge base and imports the corpus through the generic
   pipeline (no experiment metadata, no gold labels);
2. embeds every chunk once with the configured provider and writes a FAISS
   exact index plus an embedding cache, so re-runs cost nothing;
3. resolves retrieved chunks back to dataset passage ids
   (``evidence.jsonl`` for smart-home, ``articles.jsonl`` for accounting);
4. runs ``eval.runner`` over the split and stores summary + per-sample
   artefacts.

What it does *not* do: no generation, no answer-quality claim. The output is
retrieval coverage (recall@k, MRR, NDCG@k) and nothing else.

The re-run cache is the point: the expensive step is embedding, and re-running
the same corpus must not pay for it twice. ``--rebuild`` forces a rebuild.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.alignment import PassageResolver, load_passages
from eval.backends.faiss_store import FaissVectors
from eval.datasets import load_knowledge_base_dataset
from eval.runner import run_evaluation
from rag.config import load_config
from rag.kb_service import KnowledgeBaseService

BASELINE = REPO / 'experiments' / 'rag-baseline'
STORAGE = REPO / 'chroma_db' / 'eval-baseline'
ARTIFACTS = REPO / 'experiments' / 'rag-baseline' / 'evaluation' / 'retrieval_runs'

CORPORA = {
    # knowledge base -> (import directory, dataset passage index, question labels)
    'smart-home': {'directory': BASELINE / 'smart-home' / 'corpus',
                   'passages': BASELINE / 'evidence.jsonl'},
    'accounting': {'directory': BASELINE / 'accounting' / 'markdown',
                   'passages': BASELINE / 'accounting' / 'articles.jsonl'},
}


def existing_base_id(service, name):
    """Reuse the base previously registered under ``name``, if any.

    Registry ids are generated (``eval-smart-home-d1e98593``), so a re-run must
    look the id up by name rather than assume it.
    """
    for base in service.registry.list():
        if base.name == name:
            return base.id
    return None


class EmbeddingCache:
    """Text -> vector cache so a re-run does not pay the provider twice.

    Keyed by the exact text sent to the provider, so changing the chunking or
    the model still invalidates everything that matters. Persisted as JSON
    beside the index.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding='utf-8'))

    def __len__(self):
        return len(self.entries)

    def vectors(self, service, texts):
        missing = [text for text in texts if text not in self.entries]
        if missing:
            produced = service.vectors(missing)
            if len(produced) != len(missing):
                raise RuntimeError(f'Embedding 返回 {len(produced)} 条，期望 {len(missing)} 条')
            for text, vector in zip(missing, produced):
                self.entries[text] = vector
        return [self.entries[text] for text in texts]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding='utf-8')


def build_index(service, kb_id, cache, rebuild=False):
    """Publish a FAISS index for one knowledge base, reusing a previous build."""
    pointer = service.base_dir(kb_id) / 'active.eval.json'
    if pointer.exists() and not rebuild:
        info = json.loads(pointer.read_text(encoding='utf-8'))
        if (service.base_dir(kb_id) / info['generation']).is_dir():
            print(f'  复用已有索引 generation={info["generation"]} chunks={info["chunks"]}',
                  flush=True)
            return info

    chunks = service._chunks(kb_id)
    print(f'  建立新索引：{len(chunks)} 个片段，未命中缓存的部分会真实调用 Embedding', flush=True)
    vectors = cache.vectors(service, [chunk['content'] for chunk in chunks])
    cache.save()
    generation = f'faiss-{int(time.time()):x}'
    directory = service.base_dir(kb_id) / generation
    directory.mkdir(parents=True, exist_ok=True)
    store = FaissVectors(directory)
    store.add(ids=[chunk['id'] for chunk in chunks], vectors=vectors,
              documents=[chunk['content'] for chunk in chunks],
              metadatas=[{'source': chunk['source'], 'document_id': chunk['document_id']}
                         for chunk in chunks])
    store.flush()
    info = {'generation': generation, 'chunks': store.count(),
            'dimensions': len(vectors[0]), 'documents': len(service.documents(kb_id))}
    pointer.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'  索引完成 generation={generation} chunks={info["chunks"]} dim={info["dimensions"]}',
          flush=True)
    return info


def retrieve_via_faiss(service, kb_id, generation, question, top_k, cache):
    """Top-K retrieval straight from the FAISS generation, cache-first."""
    store = FaissVectors(service.base_dir(kb_id) / generation)
    vector = cache.vectors(service, [question])[0]
    lookup = {chunk['id']: chunk for chunk in service._chunks(kb_id)}
    output = []
    for rank, hit in enumerate(store.search(vector, top_k), 1):
        chunk = lookup.get(hit['id'], {})
        output.append({'id': hit['id'], 'content': hit['content'], 'source': hit['source'],
                       'rank': rank, 'score': hit['distance'],
                       'start': chunk.get('start'), 'end': chunk.get('end')})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Dense-only 检索基线（真实 Embedding + FAISS 精确索引）')
    parser.add_argument('--kb', choices=sorted(CORPORA), required=True)
    parser.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    parser.add_argument('--version', default='v1', choices=['v1', 'v2'])
    parser.add_argument('--k', type=int, default=4, help='指标截断（报告 Top-k 的 k）')
    parser.add_argument('--top-k', type=int, default=10, help='实际检索条数')
    parser.add_argument('--limit', type=int, default=None, help='只跑前 N 题（冒烟用）')
    parser.add_argument('--rebuild', action='store_true', help='强制重建索引')
    parser.add_argument('--dry-run', action='store_true', help='只检查数据与语料，不调用模型')
    args = parser.parse_args(argv)

    corpus = CORPORA[args.kb]
    split = None if args.split == 'all' else args.split
    questions = load_knowledge_base_dataset(BASELINE, args.kb, split=split, version=args.version)
    passages = load_passages(corpus['passages'])
    graded = [question for question in questions.questions if question.graded]
    print(f'知识库 {args.kb} | 数据集 {args.version}/{args.split} | 题目 {len(questions)}'
          f'（可判分 {len(graded)}）| 语料目录 {corpus["directory"].name}', flush=True)
    print(f'语料正文索引 {len(passages)} 条 passage', flush=True)
    if not corpus['directory'].is_dir():
        print('语料目录不存在，终止', flush=True)
        return 1

    config = load_config()
    service = KnowledgeBaseService(config, storage_root=STORAGE)
    resolved = {question.id: [label for label in question.relevant_passage_ids if label in passages]
                for question in questions.questions}
    missing = {qid: labels for qid, labels in resolved.items()
               if not labels and next(q for q in questions.questions if q.id == qid).graded}
    if missing:
        print(f'警告：{len(missing)} 道题的标签不在 passage 索引中，例如 '
              f'{list(missing)[:3]}', flush=True)

    if args.dry_run:
        print('dry-run：不调用模型。语料/标签检查通过。', flush=True)
        return 0

    name = f'eval-{args.kb}'
    kb_id = existing_base_id(service, name)
    if kb_id:
        print(f'复用已注册知识库 {kb_id}', flush=True)
    else:
        base = service.register(name, description=f'{args.kb} 检索基线（{args.version}）')
        kb_id = base.id
        print(f'注册知识库 {kb_id}', flush=True)

    if service._imports(kb_id) is None or args.rebuild:
        print('导入语料…', flush=True)
        report = service.import_sources(kb_id, corpus['directory'])
        print(f'  导入 {len(report.documents)} 份文档 / {len(report.chunks)} 片段'
              f'{"" if not report.skipped else f"，跳过 {len(report.skipped)}"}', flush=True)
    else:
        print(f'复用已导入清单：{len(service.documents(kb_id))} 份文档', flush=True)

    cache = EmbeddingCache(STORAGE / kb_id / 'embedding-cache.json')
    print(f'Embedding 缓存：{len(cache)} 条', flush=True)

    info = build_index(service, kb_id, cache, rebuild=args.rebuild)

    resolver = PassageResolver(passages)
    print('运行检索评测…', flush=True)

    def retrieve(question):
        return retrieve_via_faiss(service, kb_id, info['generation'], question.question,
                                  args.top_k, cache)

    result = run_evaluation(questions.questions, retrieve, knowledge_base=args.kb,
                            dataset=f'{args.version}/{args.split}', split=args.split or 'all',
                            k=args.k, resolver=resolver, output_dir=ARTIFACTS,
                            max_questions=args.limit)
    cache.save()

    summary = result.summary()
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n产物目录：{result.output_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
