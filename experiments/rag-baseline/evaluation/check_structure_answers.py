"""Diagnostic generation on a declared small dev sample, not an accuracy estimate."""
import argparse,json,time,sys
from pathlib import Path
import baseline as b
EVAL=Path(__file__).resolve().parent
SAMPLE={'smart-home':['Q040','Q043','Q051','Q008'],'accounting':['ACC-Q041','ACC-Q059','ACC-Q114','ACC-Q116','ACC-Q118']}

def run(kb):
    paths=[]
    for p in sorted((EVAL/'experiments'/kb).glob('*/manifest.json')):
        m=json.loads(p.read_text())
        if m['status']=='completed' and m.get('contextual_embedding'):paths.append((p,m))
    p,m=paths[-1]
    rows={r['question_id']:r for r in b.read(p.with_name('results.jsonl'))}
    settings=b.settings(True)
    assert settings==m['model_config']
    client=b.client(settings['llm_url'],'LLM_API_KEY')
    target=p.with_name('diagnostic_answers.jsonl')
    if target.exists():raise ValueError('Diagnostic file already exists; refusing duplicate paid calls')
    baseline_rows={r['question_id']:r for r in b.read((b.ROOT/m['baseline_manifest']).with_name('results.jsonl'))}
    for qid in SAMPLE[kb]:
        row=rows[qid]
        variants=['expanded']+(['structural'] if qid in ('ACC-Q041','ACC-Q059') else [])
        done={}
        for variant in variants:
            hits=row[variant]['hits']
            context='\n\n---\n\n'.join(f'[来源: {h["source"]}]\n{h["content"]}' for h in hits)
            if context in done:
                record={**done[context],'variant':variant,'reused_identical_context':True}
            else:
                start=time.perf_counter()
                response=client.chat.completions.create(model=settings['llm_model'],messages=[{'role':'system','content':'你是一个基于文档回答问题的助手。请根据提供的文档内容回答问题。如果文档中没有相关信息，请如实告知。'},{'role':'user','content':f'参考文档：\n\n{context}\n\n问题：{row["question"]}'}])
                answer=response.choices[0].message.content
                assert answer
                record={'question_id':qid,'question':row['question'],'variant':variant,'answer':answer,'baseline_answer':baseline_rows[qid]['answer'],'generation_seconds':time.perf_counter()-start,'usage':response.usage.model_dump() if response.usage else None,'context_chars':row[variant]['context_chars'],'reused_identical_context':False,'review_status':'diagnostic_assistant_review_pending_not_accuracy_score','selection':'8 baseline failure examples plus Q008 post-hoc structural regression diagnosis'}
                done[context]=record
            with target.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(kb,qid,variant,'saved',flush=True)
    b.dump(p.with_name('diagnostic_manifest.json'),{'status':'completed','knowledge_base':kb,'selected_question_ids':SAMPLE[kb],'selection_bias':True,'overall_accuracy_estimated':False,'model_config':settings,'script_sha256':b.sha(Path(__file__))})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kb',choices=list(SAMPLE),required=True);args=parser.parse_args()
    try:run(args.kb)
    except Exception as e:
        print(json.dumps({'status':'failed','error_type':type(e).__name__,'http_status':getattr(e,'status_code',None)}),file=sys.stderr);sys.exit(2)
