"""Frozen-context structure ablation; source parser receives no gold labels."""
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
from context_experiment import prepare
from rerank_context import boundary_messages
from structure_context import build_index, metadata, structure_messages
from openai.resources.chat.completions import Completions

EVAL=Path(__file__).resolve().parent


def inputs():
    plan=json.loads((EVAL/'structure_plan.json').read_text())
    previous,_,_,questions,service=prepare()
    assert plan['split']=='dev' and plan['generation_sample']==previous['generation_sample']
    source=EVAL/plan['source_run'];manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['split']=='dev'
    for name,digest in manifest['output_sha256'].items():assert b.sha(source/name)==digest
    indexes={};snapshots={};cases=[]
    for kb,ids in plan['generation_sample'].items():
        corpus=service.corpus(kb);docs,locations=build_index(corpus.sources,corpus.chunks,kb)
        indexes[kb]=(docs,locations)
        for c in corpus.chunks:
            start,end=locations[c['id']];text=corpus.sources[c['source']]['content']
            assert text[start:end]==c['content']
            assert len(b.normalize(text[:start]))==c['start']
            assert len(b.normalize(text[:end]))==c['end']
        snapshots[kb]={'documents':{name:d.record() for name,d in docs.items()},'chunk_raw_spans':locations}
        lookup={c['id']:c for c in corpus.chunks}
        for qid in ids:
            q=questions[kb][qid]
            prior=json.loads((source/(kb+'-'+qid+'.json')).read_text())
            selected=prior['variants'][plan['selection_variant']]
            hits=[lookup[key] for key in selected['selected_ids']]
            assert prior['question']==q['question']
            assert sum(len(h['content']) for h in hits)<=plan['budget_characters']
            assert [c['content'] for c in selected['generation']['citations']]==[h['content'] for h in hits]
            cases.append((kb,qid,q['question'],hits,metadata(hits,docs,locations),selected['metrics']))
    return plan,source,service,snapshots,cases


def run():
    plan,source,service,snapshots,cases=inputs()
    out=EVAL/'structure_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    b.dump(out/'plan.json',plan);b.dump(out/'structure_snapshot.json',snapshots)
    deps=[EVAL/n for n in ['section_experiment.py','structure_context.py','structure_plan.json','context_experiment.py','context_plan.json','rerank_context.py','baseline.py']]+[b.REPO/'rag'/n for n in ['structure.py','knowledge.py','config.py']]
    manifest={'status':'running','split':'dev','source_run':plan['source_run'],'source_manifest_sha256':b.sha(source/'manifest.json'),
              'source_sha256':{str(p.relative_to(b.REPO)):b.sha(p) for p in deps},
              'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.REPO,text=True).strip(),
              'model':service.config.llm_model,'llm_url':service.config.llm_base_url,
              'embedding_calls':0,'ranking_calls':0,'generation_calls':0,'generation_reuses':0,
              'gold_used_for_structure_or_generation':False,'documents':sum(len(v['documents']) for v in snapshots.values()),
              'chunks':sum(len(v['chunk_raw_spans']) for v in snapshots.values())}
    b.dump(out/'manifest.json',manifest);rows=[];create=Completions.create
    try:
        for index,(kb,qid,question,hits,records,metrics) in enumerate(cases):
            citations=service.citations(kb,hits)
            row={'knowledge_base':kb,'question_id':qid,'question':question,'selected_ids':[h['id'] for h in hits],
                 'context_characters':sum(len(h['content']) for h in hits),'metrics':metrics,'structure':records,'variants':{}}
            order=plan['variants'][::(-1 if index%2 else 1)];row['execution_order']=order
            for variant in order:
                calls=[]
                def observe(client,*args,**kwargs):
                    msgs=boundary_messages(kwargs['messages'],question,citations,hits)
                    if variant=='structure':msgs=structure_messages(msgs,records)
                    kwargs['messages']=msgs
                    call={'model':kwargs['model'],'messages':msgs};calls.append(call);started=time.perf_counter()
                    response=create(client,*args,**kwargs)
                    call.update(seconds=time.perf_counter()-started,usage=response.usage.model_dump() if response.usage else None)
                    return response
                with patch.object(Completions,'create',observe):answer=service.generate(question,citations)
                assert len(calls)==1 and answer
                row['variants'][variant]={'generation':{**calls[0],'answer':answer,'citations':citations}}
                manifest['generation_calls']+=1
                b.dump(out/(kb+'-'+qid+'.json'),row);b.dump(out/'manifest.json',manifest)
            rows.append(row)
            print(json.dumps({'question':qid,'completed':len(rows),'generation_calls':manifest['generation_calls']}),flush=True)
        summary={v:{'questions':len(rows),'prompt_tokens':sum(r['variants'][v]['generation']['usage']['prompt_tokens'] for r in rows),
                    'completion_tokens':sum(r['variants'][v]['generation']['usage']['completion_tokens'] for r in rows),
                    'generation_seconds_p50':b.percentile([r['variants'][v]['generation']['seconds'] for r in rows],.5)} for v in plan['variants']}
        b.dump(out/'summary.json',summary)
        manifest.update(status='completed',output_sha256={p.name:b.sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        b.dump(out/'manifest.json',manifest);print(json.dumps({'output':str(out)}),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
        b.dump(out/'manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    try:
        if args.validate_only:
            plan,_,_,snapshots,cases=inputs()
            print(json.dumps({'documents':sum(len(v['documents']) for v in snapshots.values()),'chunks':sum(len(v['chunk_raw_spans']) for v in snapshots.values()),'cases':len(cases),'planned_generation_calls':2*len(cases),'model_calls':0}))
        else:run()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}),file=sys.stderr);sys.exit(2)
