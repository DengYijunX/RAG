"""End-to-end acceptance for the generic import flow (no source changes).

Run it against the real runtime to check that a new corpus can be registered,
imported, indexed, queried and traced back to its source without editing any
Python or front-end file:

    .venv/bin/python experiments/rag-baseline/evaluation/acceptance_third_corpus.py

It drives the FastAPI application through HTTP with ``TestClient``. No source
file is edited; instead the running application object is configured at run
time — the vector backend and generator are replaced so the check is offline
and free, which means this script proves the import and wiring contract, not
retrieval quality or provider behaviour.
Set ``RAG_ACCEPTANCE_REAL=1`` to keep the configured embedding provider and
Chroma instead (requires network access and a working native vector index).

Usage report and observed evidence live in docs/THIRD_CORPUS_ACCEPTANCE.md.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / 'test_docs' / 'third-corpus'
BASE_NAME = '阳台蔬菜种植手册'


def build_client(tmp):
    import rag.api as api
    from rag.kb_service import InMemoryVectors, KnowledgeBaseService

    service = KnowledgeBaseService(api.config, storage_root=tmp / 'knowledge')
    if not os.environ.get('RAG_ACCEPTANCE_REAL'):
        rows = {}
        service.vector_backend = lambda directory: InMemoryVectors(directory, rows)
        service.vectors = lambda texts: [[float(len(t) % 7), float(sum(ord(c) for c in t) % 97), 1.0]
                                         for t in texts]
        service.generate = lambda question, citations: '（验收替身回答）依据片段 [1]'
    api.bases = service
    if not os.environ.get('RAG_ACCEPTANCE_REAL'):
        # The built-in fixtures only need to be listed here; loading their
        # corpora and Chroma collections is not part of this contract check.
        api.knowledge = StubFixtures()
    from fastapi.testclient import TestClient
    return TestClient(api.app), service


class StubFixtures:
    """Reports the two evaluation fixtures without touching native Chroma."""

    def catalog(self):
        return [{'id': 'smart-home', 'name': '智能家居售后', 'documents': 5, 'ready': False, 'chunks': 0},
                {'id': 'accounting', 'name': '企业会计准则', 'documents': 59, 'ready': False, 'chunks': 0}]


def check(label, condition, detail=''):
    mark = 'PASS' if condition else 'FAIL'
    print(f'[{mark}] {label}{(" — " + detail) if detail else ""}')
    return bool(condition)


def main():
    import tempfile

    results = []
    with tempfile.TemporaryDirectory() as raw:
        client, service = build_client(Path(raw))
        print(f'语料目录：{CORPUS}')
        print(f'语料文件：{len(list(CORPUS.rglob("*.md")))} 个 Markdown')

        created = client.post('/knowledge-bases', json={'name': BASE_NAME,
                                                        'description': '独立主题，用于通用接入验收'})
        results.append(check('创建知识库（仅通过 HTTP，未改源码）', created.status_code == 201,
                             created.text[:120]))
        kb = created.json()['id']

        files = [('files', (path.name, path.read_bytes(), 'text/markdown'))
                 for path in sorted(CORPUS.rglob('*.md'))]
        imported = client.post(f'/knowledge-bases/{kb}/sources', files=files)
        payload = imported.json()
        results.append(check('导入第三语料文件', imported.status_code == 200 and payload['documents'] == len(files),
                            json.dumps(payload, ensure_ascii=False)[:160]))

        indexed = client.post(f'/knowledge-bases/{kb}/index')
        info = indexed.json()
        results.append(check('建立索引', indexed.status_code == 200 and info['chunks'] > 0,
                            json.dumps(info, ensure_ascii=False)[:160]))

        listing = client.get('/knowledge-bases').json()
        entry = next(row for row in listing if row['id'] == kb)
        results.append(check('知识库出现在列表中且可查询', entry['ready'] is True,
                            f"documents={entry['documents']} chunks={entry['chunk_count']}"))
        results.append(check('两个既有夹具仍可列出', {row['id'] for row in listing} >= {'smart-home', 'accounting'},
                            ','.join(sorted(row['id'] for row in listing))))

        answer = client.post(f'/knowledge-bases/{kb}/query', json={'question': '浇水怎么判断该不该浇'})
        body = answer.json()
        results.append(check('问答返回片段引用', answer.status_code == 200 and bool(body.get('citations')),
                            json.dumps(body, ensure_ascii=False)[:200]))

        if body.get('citations'):
            source = client.get(body['citations'][0]['url'])
            results.append(check('查看引用对应原文', source.status_code == 200 and bool(source.json()['content']),
                                 body['citations'][0]['url']))

        manifest = json.loads(service.manifest_path(kb).read_text(encoding='utf-8'))
        chunks = manifest['chunks']
        located = [c for c in chunks if c['start'] is not None]
        verified = all(
            (CORPUS / c['source']).read_text(encoding='utf-8')[c['start']:c['end']] == c['content']
            for c in located)
        results.append(check('每个片段都能在原文中复原', bool(chunks) and len(located) == len(chunks) and verified,
                             f'{len(located)}/{len(chunks)} 个片段带位置'))
        results.append(check('原文携带内容 hash',
                             all(len(d['content_sha256']) == 64 for d in manifest['documents'])))
        results.append(check('导入未静默丢弃文件', manifest['skipped'] == [] and len(manifest['documents']) == len(files),
                             json.dumps(manifest['skipped'], ensure_ascii=False)))

    print()
    print(f'结果：{sum(results)}/{len(results)} 项通过')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
