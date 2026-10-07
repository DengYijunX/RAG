"""Dev pilot: same expanded pool, ranking x boundary prompt factorial design."""
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
from context_budget import select_context
from context_experiment import prepare
from rerank_context import expanded_pool, rank_messages, validated_order, boundary_messages
from openai import OpenAI
from openai.resources.chat.completions import Completions

EVAL=Path(__file__).resolve().parent


def inputs():
    plan=json.loads((EVAL/'rerank_plan.json').read_text())
    previous,source,rows,questions,service=prepare()
    assert plan['split']=='dev' and plan['source_run']==previous['source_run']
    assert plan['generation_sample']==previous['generation_sample']
    return plan,source,rows,questions,service


def rank(service, question, pool, sources):
    messages=rank_messages(question,pool,sources)
    record={'messages':messages,'model':service.config.llm_model,'temperature':0,'max_tokens':1024,
            'response_format':{'type':'json_object'},'fallback':None}
    started=time.perf_counter()
    try:
        with OpenAI(base_url=service.config.llm_base_url,api_key=service.config.llm_api_key,timeout=60,max_retries=2) as client:
            response=client.chat.completions.create(model=record['model'],messages=messages,
                                                   temperature=0,max_tokens=1024,response_format=record['response_format'])
        raw=response.choices[0].message.content
        order,fallback=validated_order(raw,pool)
        record.update(raw=raw,order=order,fallback=fallback,usage=response.usage.model_dump() if response.usage else None)
    except Exception as exc:
        record.update(order=[c['id'] for c in pool],fallback='provider_error',
                      error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None),usage=None)
    record['seconds']=time.perf_counter()-started
    return record


def summarize(pool_rows,pilot,variants):
    result={}
    for kb in b.CONFIG:
        group=[r for r in pool_rows if r['knowledge_base']==kb]
        scored=[r for r in group if r['pool_metrics'] is not None]
        samples=[r for r in pilot if r['knowledge_base']==kb]
        item={'all_dev_questions':len(group),'retrieval_scored_questions':len(scored),
              'pool_chunks_mean':sum(len(r['pool_ids']) for r in group)/len(group),
              'pool_all_evidence_count':sum(r['pool_metrics']['all_evidence_covered'] for r in scored),
              'base_budget_all_evidence_count':sum(r['base_metrics']['all_evidence_covered'] for r in scored),
              'pilot_questions':len(samples),'ranking_calls':len(samples),
              'ranking_fallbacks':sum(bool(r['ranking']['fallback']) for r in samples),
              'ranking_prompt_tokens':sum((r['ranking']['usage'] or {}).get('prompt_tokens',0) for r in samples),
              'ranking_completion_tokens':sum((r['ranking']['usage'] or {}).get('completion_tokens',0) for r in samples),
              'ranking_seconds_p50':b.percentile([r['ranking']['seconds'] for r in samples],.5), 'variants':{}}
        for v in variants:
            values=[r['variants'][v] for r in samples]
            item['variants'][v]={'questions':len(values),
                'mean_evidence_unit_coverage':sum(r['metrics']['coverage_recall_at_k'] for r in values)/len(values),
                'all_evidence_covered_count':sum(r['metrics']['all_evidence_covered'] for r in values),
                'context_characters_mean':sum(r['context_characters'] for r in values)/len(values),
                'generation_prompt_tokens':sum(r['generation']['usage']['prompt_tokens'] for r in values),
                'generation_completion_tokens':sum(r['generation']['usage']['completion_tokens'] for r in values),
                'generation_seconds_p50':b.percentile([r['generation']['seconds'] for r in values],.5)}
        result[kb]=item
    return result


def run():
    plan,source,old,questions,service=inputs()
    out=EVAL/'rerank_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    b.dump(out/'plan.json',plan)
    dependencies=['rerank_experiment.py','rerank_context.py','rerank_plan.json','context_budget.py','context_experiment.py','context_plan.json','baseline.py']
    manifest={'status':'running','split':'dev','source_run':plan['source_run'],
              'source_manifest_sha256':b.sha(source/'manifest.json'),
              'source_sha256':{str(p.relative_to(b.REPO)):b.sha(p) for p in
                  [*[EVAL/name for name in dependencies], b.REPO/'rag/knowledge.py',b.REPO/'rag/config.py']},
              'repository_commit':subprocess.check_output(['git','-C',str(b.REPO),'rev-parse','HEAD'],text=True).strip(),
              'model':service.config.llm_model,'llm_url':service.config.llm_base_url,
              'ranking_calls':0,'generation_calls':0,'embedding_calls':0,
              'gold_used_for_pool_ranking_selection_generation':False,
              'scope':'all-dev offline pool coverage plus fixed 9-question LLM pilot; no full-dev reranker result'}
    b.dump(out/'manifest.json',manifest)
    pilot=[]; pool_rows=[]; create=Completions.create
    try:
        # This offline diagnostic is independent of the frozen pilot selection.
        for prior in old:
            kb,qid=prior['knowledge_base'],prior['question_id'];q=questions[kb][qid];corpus=service.corpus(kb)
            pool,origins=expanded_pool(prior['variants']['12']['retrieved_ids'],corpus.chunks)
            assert len(pool)<=36
            selected=select_context([c['id'] for c in pool],corpus.chunks,plan['budget_characters'])
            scored=q['expected_action'] in ('answer','flag_conflict')
            record={'knowledge_base':kb,'question_id':qid,'pool_ids':[c['id'] for c in pool],'origins':origins,
                    'pool_metrics':b.score(q['evidence_ids'],pool,corpus.parents) if scored else None,
                    'base_selected_ids':[c['id'] for c in selected['hits']],
                    'base_metrics':b.score(q['evidence_ids'],selected['hits'],corpus.parents) if scored else None}
            pool_rows.append(record)
        with (out/'pool_results.jsonl').open('w') as f:
            for r in pool_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        for prior in pool_rows:
            kb,qid=prior['knowledge_base'],prior['question_id']
            if qid not in plan['generation_sample'][kb]:continue
            q=questions[kb][qid];corpus=service.corpus(kb);lookup={c['id']:c for c in corpus.chunks}
            pool=[lookup[key] for key in prior['pool_ids']]
            row={'knowledge_base':kb,'question_id':qid,'question':q['question'],'pool_ids':prior['pool_ids'],
                 'ranking':rank(service,q['question'],pool,corpus.sources),'variants':{}}
            manifest['ranking_calls']+=1
            path=out/(kb+'-'+qid+'.json')
            b.dump(path,row)
            offset=len(pilot)%4
            order=plan['variants'][offset:]+plan['variants'][:offset]
            row['execution_order']=order
            for variant in order:
                ids=row['ranking']['order'] if variant.startswith('rank_') else row['pool_ids']
                selected=select_context(ids,corpus.chunks,plan['budget_characters'])
                hits=selected.pop('hits');citations=service.citations(kb,hits);calls=[]

                def observe(client,*args,**kwargs):
                    if variant.endswith('_boundary'):
                        kwargs['messages']=boundary_messages(kwargs['messages'],q['question'],citations,hits)
                    call={'model':kwargs['model'],'messages':kwargs['messages']}
                    calls.append(call);started=time.perf_counter()
                    response=create(client,*args,**kwargs)
                    call.update(seconds=time.perf_counter()-started,usage=response.usage.model_dump() if response.usage else None)
                    return response

                with patch.object(Completions,'create',observe):
                    answer=service.generate(q['question'],citations)
                assert len(calls)==1 and answer
                assert all(h['content'] in calls[0]['messages'][-1]['content'] for h in hits)
                selected.update(selected_ids=[c['id'] for c in hits],generation={**calls[0],'answer':answer,'citations':citations},
                                metrics=b.score(q['evidence_ids'],hits,corpus.parents))
                row['variants'][variant]=selected
                manifest['generation_calls']+=1
                b.dump(path,row);b.dump(out/'manifest.json',manifest)
            pilot.append(row)
            print(json.dumps({'question':qid,'pilot_completed':len(pilot),'generation_calls':manifest['generation_calls'],'rank_fallback':row['ranking']['fallback']}),flush=True)
        b.dump(out/'summary.json',summarize(pool_rows,pilot,plan['variants']))
        manifest.update(status='completed',pool_questions=len(pool_rows),pilot_questions=len(pilot),
                        output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        b.dump(out/'manifest.json',manifest)
        print(json.dumps({'output':str(out)}),flush=True)
    except Exception as exc:
        manifest.update(status='failed',pilot_completed=len(pilot),error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
        b.dump(out/'manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    try:
        if args.validate_only:
            plan,source,old,questions,service=inputs()
            print(json.dumps({'dev_pool_questions':len(old),'pilot_questions':sum(len(ids) for ids in plan['generation_sample'].values()),'planned_ranking_calls':9,'planned_generation_calls':36,'model_calls':0}))
        else:run()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}),file=sys.stderr)
        sys.exit(2)
