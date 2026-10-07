"""Replay dev Top-12 under one fixed budget; optionally expand source neighbors."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import uuid

import baseline as b
from context_budget import select_context
from openai.resources.chat.completions import Completions
from rag.config import load_config
from rag.knowledge import KnowledgeService

EVAL = Path(__file__).resolve().parent


def prepare():
    plan = json.loads((EVAL/'context_plan.json').read_text())
    source = EVAL/plan['source_run']
    previous = json.loads((source/'manifest.json').read_text())
    assert previous['status']=='completed' and previous['split']==plan['split']=='dev'
    for name, digest in previous['output_sha256'].items():
        assert b.sha(source/name)==digest
    questions = {kb:{q['id']:q for q in b.read(EVAL/'v2'/kb/'questions.jsonl') if q['split']=='dev'} for kb in b.CONFIG}
    original = b.read(source/'results.jsonl')
    service = KnowledgeService(load_config())
    old_plan = json.loads((source/'plan.json').read_text())
    assert plan['generation_sample']==old_plan['generation_sample']
    for kb in b.CONFIG:
        question_path = EVAL/'v2'/kb/'questions.jsonl'
        assert b.sha(question_path)==previous['source_sha256'][str(question_path.relative_to(b.REPO))]
        assert service.signature(kb)==previous['indexes'][kb]['signature']
        assert service.corpus(kb).chunks==json.loads((source/(kb+'-chunks.json')).read_text())
        assert {r['question_id'] for r in original if r['knowledge_base']==kb}==questions[kb].keys()
        assert set(plan['generation_sample'][kb])<=questions[kb].keys()
    assert len(original)==sum(len(q) for q in questions.values())
    return plan, source, original, questions, service


def summarize(rows, variants):
    summary = {}
    for kb in b.CONFIG:
        group = [r for r in rows if r['knowledge_base']==kb]
        scored = [r for r in group if r['variants'][variants[0]]['metrics'] is not None]
        summary[kb] = {'dev_questions':len(group), 'scored_questions':len(scored), 'variants':{}}
        for variant in variants:
            metrics = [r['variants'][variant]['metrics'] for r in scored]
            generations = [r['variants'][variant]['generation'] for r in group if 'generation' in r['variants'][variant]]
            stats = {'mean_evidence_unit_coverage':sum(m['coverage_recall_at_k'] for m in metrics)/len(metrics),
                     'all_evidence_covered_count':sum(m['all_evidence_covered'] for m in metrics),
                     'context_characters_mean':sum(r['variants'][variant]['context_characters'] for r in group)/len(group),
                     'context_characters_max':max(r['variants'][variant]['context_characters'] for r in group),
                     'generation_answers':len(generations),
                     'prompt_tokens_sum_per_arm':sum(g['usage']['prompt_tokens'] for g in generations),
                     'completion_tokens_sum_per_arm':sum(g['usage']['completion_tokens'] for g in generations),
                     'improved_vs_dense_budget':[], 'regressed_vs_dense_budget':[], 'by_category':{}}
            for r in scored:
                delta=r['variants'][variant]['metrics']['coverage_recall_at_k']-r['variants']['dense_budget']['metrics']['coverage_recall_at_k']
                if delta:
                    stats['improved_vs_dense_budget' if delta>0 else 'regressed_vs_dense_budget'].append(r['question_id'])
            for category in sorted({r['category'] for r in scored}):
                values=[r['variants'][variant]['metrics'] for r in scored if r['category']==category]
                stats['by_category'][category]={'count':len(values), 'mean_evidence_unit_coverage':sum(v['coverage_recall_at_k'] for v in values)/len(values), 'all_evidence_covered_count':sum(v['all_evidence_covered'] for v in values)}
            summary[kb]['variants'][variant]=stats
    return summary


def run():
    plan, source, original, questions, service = prepare()
    runid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = EVAL/'context_runs'/runid
    out.mkdir(parents=True)
    b.dump(out/'plan.json', plan)
    manifest = {'status':'running', 'split':'dev', 'kind':'fixed_budget_neighbor_ablation',
                'source_run':plan['source_run'], 'source_manifest_sha256':b.sha(source/'manifest.json'),
                'source_sha256':{str(p.relative_to(b.REPO)):b.sha(p) for p in
                    [Path(__file__), EVAL/'context_budget.py', EVAL/'context_plan.json', EVAL/'baseline.py', b.REPO/'rag/knowledge.py', b.REPO/'rag/config.py']},
                'repository_commit':subprocess.check_output(['git','-C',str(b.REPO),'rev-parse','HEAD'],text=True).strip(),
                'model':service.config.llm_model, 'llm_url':service.config.llm_base_url,
                'embedding_calls':0, 'generation_calls':0, 'generation_reuses':0,
                'routing_enabled':False, 'gold_used_for_selection_or_generation':False,
                'budget_scope':'chunk bodies in Python characters; not token budget, excludes title/system/query overhead'}
    b.dump(out/'manifest.json',manifest)
    create = Completions.create
    rows = []
    try:
        for i, old in enumerate(original):
            kb, qid = old['knowledge_base'], old['question_id']
            q = questions[kb][qid]
            assert old['question']==q['question']
            corpus = service.corpus(kb)
            candidates=old['variants'][str(plan['source_k'])]['retrieved_ids']
            row={'knowledge_base':kb,'question_id':qid,'question':q['question'],'category':q['category'],
                 'dense_candidate_ids':candidates,'variants':{}}
            variants=plan['variants'][::1 if i%2==0 else -1]
            row['execution_order']=variants
            cached={}
            for variant in variants:
                result=select_context(candidates,corpus.chunks,plan['budget_characters'],neighbors=variant=='neighbor_budget')
                hits=result.pop('hits')
                result['selected_ids']=[h['id'] for h in hits]
                if qid in plan['generation_sample'][kb]:
                    key=tuple(result['selected_ids'])
                    if key in cached:
                        result['generation']={**cached[key][1],'reused_from_variant':cached[key][0]}
                        manifest['generation_reuses']+=1
                    else:
                        calls=[]

                        def observe(client,*args,**kwargs):
                            record={'model':kwargs['model'],'messages':kwargs['messages']}
                            calls.append(record)
                            started=time.perf_counter()
                            response=create(client,*args,**kwargs)
                            record.update(seconds=time.perf_counter()-started,usage=response.usage.model_dump() if response.usage else None)
                            return response

                        citations=service.citations(kb,hits)
                        with patch.object(Completions,'create',observe):
                            answer=service.generate(q['question'],citations)
                        assert len(calls)==1 and answer
                        assert all(h['content'] in calls[0]['messages'][-1]['content'] for h in hits)
                        result['generation']={**calls[0],'answer':answer,'citations':citations,'reused_from_variant':None}
                        cached[key]=(variant,result['generation'])
                        manifest['generation_calls']+=1
                # Labels only enter scoring after selection/generation.
                result['metrics']=b.score(q['evidence_ids'],hits,corpus.parents) if q['expected_action'] in ('answer','flag_conflict') else None
                row['variants'][variant]=result
            rows.append(row)
            with (out/'results.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'completed':len(rows),'question':qid,'generation_calls':manifest['generation_calls']}),flush=True)
        b.dump(out/'summary.json',summarize(rows,plan['variants']))
        manifest.update(status='completed', completed_questions=len(rows), output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        b.dump(out/'manifest.json',manifest)
        print(json.dumps({'output':str(out)}),flush=True)
    except Exception as exc:
        manifest.update(status='failed',completed_questions=len(rows),error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
        b.dump(out/'manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:
        if args.validate_only:
            plan,source,rows,questions,service=prepare()
            print(json.dumps({'dev_questions':len(rows),'embedding_calls':0,'generation_calls':0,'max_planned_generation_calls':2*sum(len(ids) for ids in plan['generation_sample'].values())}))
        else:
            run()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}),file=sys.stderr)
        sys.exit(2)
