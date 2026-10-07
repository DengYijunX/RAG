"""Paired full/compact structure prompts with frozen selected passages."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
import uuid
from openai import OpenAI
import baseline as b
from section_experiment import inputs as section_inputs, EVAL
from rag.context_structure import compact_messages, compact_structure, enforce_input_byte_budget


def inputs():
    plan=json.loads((EVAL/'compact_plan.json').read_text())
    _,_,service,_,cases=section_inputs()
    source=EVAL/plan['source_run'];manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['split']==plan['split']=='dev'
    for name,digest in manifest['output_sha256'].items():assert b.sha(source/name)==digest
    rows=[]
    for kb,qid,question,hits,records,metrics in cases:
        prior=json.loads((source/(kb+'-'+qid+'.json')).read_text())
        assert prior['question']==question and prior['structure']==records and prior['metrics']==metrics
        assert prior['selected_ids']==[h['id'] for h in hits]
        citations=service.citations(kb,hits)
        assert citations==prior['variants']['structure']['generation']['citations']
        messages={'full_structure':prior['variants']['structure']['generation']['messages'],
                  'compact_structure':compact_messages(prior['variants']['boundary']['generation']['messages'],records)}
        sizes={v:enforce_input_byte_budget(msg,plan['compact_max_input_bytes'] if v=='compact_structure' else plan['full_max_input_bytes']) for v,msg in messages.items()}
        rows.append({'knowledge_base':kb,'question_id':qid,'question':question,'selected_ids':prior['selected_ids'],
                     'context_characters':prior['context_characters'],'metrics':metrics,'compact_structure':compact_structure(records),
                     'citations':citations,'messages':messages,'input_bytes':sizes,'variants':{}})
    assert len(rows)*2==plan['generation_calls']
    return plan,source,service,rows


def run():
    plan,source,service,rows=inputs()
    out=EVAL/'compact_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    deps=[EVAL/n for n in ['compact_experiment.py','compact_plan.json','section_experiment.py','structure_context.py','structure_plan.json','context_experiment.py','context_plan.json','baseline.py']]+[b.REPO/'rag'/n for n in ['context_structure.py','structure.py','knowledge.py','config.py']]
    manifest={'status':'running','split':'dev','source_run':plan['source_run'],'source_manifest_sha256':b.sha(source/'manifest.json'),
              'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.REPO,text=True).strip(),
              'source_sha256':{str(p.relative_to(b.REPO)):b.sha(p) for p in deps},'model':service.config.llm_model,
              'llm_url':service.config.llm_base_url,'generation_calls':0,'generation_attempts':0,'embedding_calls':0,'ranking_calls':0,
              'generation_reuses':0,'gold_used_for_prompt':False,'sdk_retry_limit':2}
    b.dump(out/'plan.json',plan);b.dump(out/'manifest.json',manifest)
    try:
        with OpenAI(base_url=service.config.llm_base_url,api_key=service.config.llm_api_key,timeout=60,max_retries=2) as client:
            for index,row in enumerate(rows):
                order=plan['variants'][::(-1 if index%2 else 1)];row['execution_order']=order
                path=out/(row['knowledge_base']+'-'+row['question_id']+'.json');b.dump(path,row)
                for v in order:
                    manifest['generation_attempts']+=1;b.dump(out/'manifest.json',manifest)
                    started=time.perf_counter()
                    response=client.chat.completions.create(model=service.config.llm_model,messages=row['messages'][v])
                    elapsed=time.perf_counter()-started;answer=response.choices[0].message.content
                    assert answer and response.usage is not None
                    row['variants'][v]={'answer':answer,'seconds':elapsed,'usage':response.usage.model_dump(),'finish_reason':response.choices[0].finish_reason}
                    manifest['generation_calls']+=1;b.dump(path,row);b.dump(out/'manifest.json',manifest)
                print(json.dumps({'question':row['question_id'],'completed':index+1,'generation_calls':manifest['generation_calls']}),flush=True)
        summary={v:{'questions':len(rows),'prompt_tokens':sum(r['variants'][v]['usage']['prompt_tokens'] for r in rows),
                    'completion_tokens':sum(r['variants'][v]['usage']['completion_tokens'] for r in rows),
                    'input_bytes_max':max(r['input_bytes'][v] for r in rows),
                    'generation_seconds_p50':b.percentile([r['variants'][v]['seconds'] for r in rows],.5)} for v in plan['variants']}
        b.dump(out/'summary.json',summary)
        manifest.update(status='completed',output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        b.dump(out/'manifest.json',manifest);print(json.dumps({'output':str(out)}),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None));b.dump(out/'manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    try:
        if args.validate_only:
            plan,_,_,rows=inputs();print(json.dumps({'cases':len(rows),'planned_generation_calls':plan['generation_calls'],'model_calls':0,
                                                  'max_bytes':{v:max(r['input_bytes'][v] for r in rows) for v in plan['variants']}}))
        else:run()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}),file=sys.stderr);sys.exit(2)
