"""Dev-only dense Top-K ablation; domain routing is deliberately excluded.

Uses live application indexes and generate(), with labels only for later scoring.
No production defaults are changed. See topk_plan.json for the frozen scope.
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
from openai.resources.chat.completions import Completions
from rag.config import load_config
from rag.knowledge import KnowledgeService

EVAL = Path(__file__).resolve().parent


def inputs():
    plan = json.loads((EVAL/'topk_plan.json').read_text())
    previous = json.loads((EVAL/'quality_sample.json').read_text())
    assert plan['split'] == 'dev' and plan['top_k'] == [4, 8, 12]
    questions = {}
    for kb in b.CONFIG:
        rows = b.read(EVAL/'v2'/kb/'questions.jsonl')
        questions[kb] = [q for q in rows if q['split'] == 'dev']
        eligible = [q['id'] for q in questions[kb]
                    if q['id'] in previous['knowledge_bases'][kb]
                    and q['expected_action'] in ('answer', 'flag_conflict')]
        assert eligible == plan['generation_sample'][kb]
    return plan, questions


def aggregate(rows, ks):
    output = {}
    for kb in b.CONFIG:
        subset = [r for r in rows if r['knowledge_base'] == kb]
        scored = [r for r in subset if r['variants']['4']['metrics'] is not None]
        stats = {}
        for k in ks:
            values = [r['variants'][str(k)] for r in scored]
            gen = [r['variants'][str(k)]['generation'] for r in subset
                   if 'generation' in r['variants'][str(k)]]
            stats[str(k)] = {
                'retrieval_scored_questions': len(scored),
                'mean_evidence_unit_coverage': sum(v['metrics']['coverage_recall_at_k'] for v in values)/len(values),
                'all_evidence_covered_count': sum(v['metrics']['all_evidence_covered'] for v in values),
                'context_characters_mean_all_dev': sum(r['variants'][str(k)]['context_characters'] for r in subset)/len(subset),
                'by_category': {},
                'improved_vs_4': [r['question_id'] for r in scored if r['variants'][str(k)]['metrics']['coverage_recall_at_k'] > r['variants']['4']['metrics']['coverage_recall_at_k']],
                'regressed_vs_4': [r['question_id'] for r in scored if r['variants'][str(k)]['metrics']['coverage_recall_at_k'] < r['variants']['4']['metrics']['coverage_recall_at_k']],
                'nonmatching_top4_prefix': [r['question_id'] for r in subset if r['variants'][str(k)]['retrieved_ids'][:4] != r['variants']['4']['retrieved_ids']],
                'generation_count': len(gen),
                'generation_prompt_tokens_total': sum(g['usage']['prompt_tokens'] for g in gen) if gen else None,
                'generation_completion_tokens_total': sum(g['usage']['completion_tokens'] for g in gen) if gen else None,
                'generation_seconds_p50': b.percentile([g['seconds'] for g in gen], .5),
                'generation_seconds_p95': b.percentile([g['seconds'] for g in gen], .95),
            }
            for category in sorted({r['category'] for r in scored}):
                group = [r['variants'][str(k)]['metrics'] for r in scored if r['category'] == category]
                stats[str(k)]['by_category'][category] = {'count':len(group), 'mean_evidence_unit_coverage':sum(g['coverage_recall_at_k'] for g in group)/len(group), 'all_evidence_covered_count':sum(g['all_evidence_covered'] for g in group)}
        output[kb] = {'dev_questions':len(subset), 'unscored_for_retrieval':len(subset)-len(scored), 'variants':stats}
    return output


def run():
    plan, questions = inputs()
    service = KnowledgeService(load_config())
    indexes = {kb:service.active(kb) for kb in questions}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = EVAL/'topk_runs'/run_id
    out.mkdir(parents=True)
    b.dump(out/'plan.json', plan)
    manifest = {'status':'running', 'kind':'dense_topk_dev_ablation', 'split':'dev',
                'routing_enabled':False, 'index_rebuilt':False, 'indexes':indexes,
                'model':service.config.llm_model, 'llm_url':service.config.llm_base_url,
                'gold_used_for_retrieval_or_generation':False,
                'source_sha256':{str(p.relative_to(b.REPO)):b.sha(p) for p in
                    [Path(__file__), EVAL/'topk_plan.json', EVAL/'quality_sample.json', EVAL/'baseline.py',
                     b.REPO/'rag/knowledge.py', b.REPO/'rag/config.py',
                     *[EVAL/'v2'/kb/'questions.jsonl' for kb in questions]]},
                'repository_commit':subprocess.check_output(['git','-C',str(b.REPO),'rev-parse','HEAD'],text=True).strip(),
                'generation_calls':0, 'embedding_batches':[],
                'latency_conditions':'serial, no warmup, provider retries possible; model calls only, not production load test',
                'reviewer':'assistant review pending; human scores unset'}
    b.dump(out/'manifest.json', manifest)
    original_create = Completions.create
    rows = []
    try:
        for kb, qs in questions.items():
            corpus = service.corpus(kb)
            col = service.collection(kb, indexes[kb])
            assert col.count() == len(corpus.chunks)
            lookup = {c['id']:c for c in corpus.chunks}
            b.dump(out/(kb+'-chunks.json'), corpus.chunks)
            vectors = []
            for start in range(0, len(qs), 20):
                batch = qs[start:start+20]
                t = time.perf_counter()
                result = service.vectors([q['question'] for q in batch])
                assert len(result) == len(batch) and all(len(v)==indexes[kb]['dimensions'] for v in result)
                vectors.extend(result)
                manifest['embedding_batches'].append({'knowledge_base':kb, 'question_ids':[q['id'] for q in batch], 'seconds':time.perf_counter()-t})
            for qi, (q, vector) in enumerate(zip(qs, vectors)):
                row = {'knowledge_base':kb, 'question_id':q['id'], 'question':q['question'],
                       'category':q['category'], 'variants':{}}
                ks = plan['top_k']
                order = ks[qi%len(ks):]+ks[:qi%len(ks)]
                row['execution_order'] = order
                for k in order:
                    t = time.perf_counter()
                    found = col.query(query_embeddings=[vector], n_results=min(k,col.count()), include=['distances'])
                    hits = [lookup[key] for key in found['ids'][0]]
                    variant = {'retrieved_ids':found['ids'][0], 'distances':found['distances'][0],
                               'search_seconds':time.perf_counter()-t,
                               'context_characters':sum(len(h['content']) for h in hits)}
                    if q['id'] in plan['generation_sample'][kb]:
                        captures = []

                        def observe(client, *args, **kwargs):
                            record = {'model':kwargs['model'], 'messages':kwargs['messages']}
                            captures.append(record)
                            began = time.perf_counter()
                            response = original_create(client, *args, **kwargs)
                            record.update(seconds=time.perf_counter()-began,
                                          usage=response.usage.model_dump() if response.usage else None)
                            return response

                        citations = service.citations(kb, hits)
                        with patch.object(Completions, 'create', observe):
                            answer = service.generate(q['question'], citations)
                        assert len(captures)==1 and answer
                        assert all(h['content'] in captures[0]['messages'][-1]['content'] for h in hits)
                        variant['generation'] = {**captures[0], 'answer':answer, 'citations':citations}
                        manifest['generation_calls'] += 1
                    # Labels are consulted after retrieval and optional generation only.
                    variant['metrics'] = b.score(q['evidence_ids'], hits, corpus.parents) if q['expected_action'] in ('answer','flag_conflict') else None
                    row['variants'][str(k)] = variant
                rows.append(row)
                with (out/'results.jsonl').open('a') as f:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'kb':kb,'question':q['id'],'completed':len(rows),'generation_calls':manifest['generation_calls']}),flush=True)
            b.dump(out/'manifest.json', manifest)
        b.dump(out/'summary.json', aggregate(rows, plan['top_k']))
        manifest.update(status='completed', completed_questions=len(rows),
                        output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        b.dump(out/'manifest.json',manifest)
        print(json.dumps({'output':str(out),'completed_questions':len(rows)}),flush=True)
    except Exception as exc:
        manifest.update(status='failed', completed_questions=len(rows), error_type=type(exc).__name__, http_status=getattr(exc,'status_code',None))
        b.dump(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.validate_only:
            plan, questions = inputs()
            print(json.dumps({'dev_questions':{kb:len(qs) for kb,qs in questions.items()}, 'planned_generation_calls':sum(len(ids) for ids in plan['generation_sample'].values())*len(plan['top_k']), 'model_calls':0}))
        else:
            run()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}),file=sys.stderr)
        sys.exit(2)
