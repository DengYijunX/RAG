"""Inspect one dev question using the real application, without changing retrieval.

Records the actual model messages, retrieval ranks, provenance and source spans.
A separate Top-20 probe runs only after the answer, reusing its query vector.
Gold labels are used only in the offline comparison after the application call.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import uuid

import baseline as b
from rag.config import load_config
from rag.knowledge import KnowledgeService
from openai.resources.chat.completions import Completions

EVAL = Path(__file__).resolve().parent


class TracedService(KnowledgeService):
    def __init__(self, config):
        super().__init__(config)
        self.events = []
        self.vector = None
        self.hits = []
        self.col = None

    def vectors(self, texts):
        start = time.perf_counter()
        result = super().vectors(texts)
        self.vector = result[0]
        self.events.append({'stage': 'embedding', 'inputs': texts, 'dimensions': len(result[0]),
                            'model': self.config.embedding_model, 'seconds': time.perf_counter()-start})
        return result

    def collection(self, kb, info):
        self.col = super().collection(kb, info)
        service = self

        class Proxy:
            def count(self):
                return service.col.count()

            def query(self, **kwargs):
                start = time.perf_counter()
                result = service.col.query(**kwargs)
                service.events.append({'stage': 'online_vector_search', 'n_results': kwargs['n_results'],
                                       'ids': result['ids'][0], 'distances': result['distances'][0],
                                       'seconds': time.perf_counter()-start})
                return result

        return Proxy()

    def citations(self, kb, hits):
        self.hits = hits
        return super().citations(kb, hits)


def run(kb, question_id):
    questions_path = EVAL/'v2'/kb/'questions.jsonl'
    question = next(q for q in b.read(questions_path) if q['id'] == question_id)
    if question['split'] != 'dev':
        raise ValueError('This walkthrough accepts dev questions only')
    service = TracedService(load_config())
    corpus = service.corpus(kb)
    active = service.active(kb)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = EVAL/'traces'/kb/run_id
    out.mkdir(parents=True)
    manifest = {'status': 'running', 'knowledge_base': kb, 'question_id': question_id, 'split': 'dev',
                'purpose': 'single-case walkthrough, not an accuracy estimate', 'index_rebuilt': False,
                'index': active, 'llm_model': service.config.llm_model,
                'llm_url': service.config.llm_base_url, 'gold_used_for_retrieval_or_generation': False,
                'script_sha256': b.sha(Path(__file__)), 'questions_sha256': b.sha(questions_path),
                'application_sha256': {str(p): b.sha(b.REPO/p) for p in map(Path, ['rag/knowledge.py', 'rag/knowledge_routing.py', 'rag/config.py'])},
                'repository_commit': subprocess.check_output(['git', '-C', str(b.REPO), 'rev-parse', 'HEAD'], text=True).strip()}
    b.dump(out/'manifest.json', manifest)
    model_calls = []
    original_create = Completions.create

    def observe_create(client, *args, **kwargs):
        # Only model, messages and usage are saved, never client/auth headers.
        record = {'model': kwargs['model'], 'messages': kwargs['messages']}
        model_calls.append(record)
        start = time.perf_counter()
        result = original_create(client, *args, **kwargs)
        record.update(seconds=time.perf_counter()-start,
                      usage=result.usage.model_dump() if result.usage else None)
        return result

    try:
        start = time.perf_counter()
        with patch.object(Completions, 'create', observe_create):
            response = service.query(kb, question['question'])
        online_seconds = time.perf_counter()-start
        assert response['answer']
        b.dump(out/'online.json', {'question': question['question'], 'events': service.events,
                                  'hits': service.hits, 'model_calls': model_calls,
                                  'response': response, 'total_seconds': online_seconds})
        if model_calls:
            assert len(model_calls) == 1
            messages = model_calls[0]['messages']
            assert all(hit['content'] in messages[-1]['content'] for hit in service.hits)
            (out/'prompt.txt').write_text('\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in messages)+'\n')
        # Analysis starts after the real application response. Not fed back to LLM.
        spans = {key: {field: p[field] for field in ['source', 'start', 'end']} for key, p in corpus.parents.items()}
        probe = []
        if service.vector is not None:
            found = service.col.query(query_embeddings=[service.vector], n_results=min(20, service.col.count()), include=['distances'])
            lookup = {c['id']: c for c in corpus.chunks}
            probe = [{**lookup[key], 'rank': rank, 'distance': distance}
                     for rank, (key, distance) in enumerate(zip(found['ids'][0], found['distances'][0]), 1)]
        analysis = {'diagnostic_probe_not_used_for_answer': True, 'probe': probe,
                    'expected_action': question['expected_action'], 'expected_points': question['expected_points'],
                    'gold_review_status': question.get('review_status'),
                    'online_coverage': b.score(question['evidence_ids'], service.hits, spans),
                    'evidence': []}
        for key in question['evidence_ids']:
            parent = corpus.parents[key]
            containing = [c['id'] for c in corpus.chunks if b.score([key], [c], spans)['all_evidence_covered']]
            matches = [p['rank'] for p in probe if p['id'] in containing]
            analysis['evidence'].append({'id': key, 'source_evidence': parent,
                                         'original_chunks_covering_at_least_80_percent': containing,
                                         'probe_ranks_covering_at_least_80_percent': matches})
        b.dump(out/'diagnosis.json', analysis)
        b.dump(out/'corpus_chunks.json', {'knowledge_base': kb, 'chunks': corpus.chunks})
        manifest.update(status='completed', generation_calls=len(model_calls),
                        embedding_calls=sum(e['stage']=='embedding' for e in service.events),
                        output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name != 'manifest.json'})
        b.dump(out/'manifest.json', manifest)
        print(json.dumps({'output': str(out), 'answer': response['answer'], 'coverage': analysis['online_coverage'],
                          'evidence_ranks': {r['id']:r['probe_ranks_covering_at_least_80_percent'] for r in analysis['evidence']}}, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        b.dump(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', choices=list(b.CONFIG), required=True)
    parser.add_argument('--question-id', required=True)
    args = parser.parse_args()
    try:
        run(args.kb, args.question_id)
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__, 'http_status':getattr(exc,'status_code',None)}), file=sys.stderr)
        sys.exit(2)
