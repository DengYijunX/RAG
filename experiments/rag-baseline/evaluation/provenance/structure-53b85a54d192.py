"""Dev-only ablation: cached baseline -> structural children -> parent expansion."""
import argparse,json,time,uuid,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import baseline as b
from rag.loader import Document

EVAL=Path(__file__).resolve().parent

def units(kb):
    docs,_,index=b.prepare(kb)
    records=b.read(b.ROOT/kb/b.CONFIG[kb][1])
    parents={}; children=[]
    for r in records:
        span=index[r['id']]
        p={'id':r['id'],'source':span['source'],'start':span['start'],'end':span['end'],'content':r['text']}
        parents[p['id']]=p
        pieces=b.split([Document(content=p['content'],metadata={'source':p['source']})],500,50)
        cursor=0; normalized=b.normalize(p['content'])
        for i,piece in enumerate(pieces):
            needle=b.normalize(piece.content); start=normalized.find(needle,cursor)
            assert start>=0,(p['id'],i)
            cursor=start+1
            children.append({'id':f'struct:{p["id"]}:{i}','parent_id':p['id'],'source':p['source'],'start':p['start']+start,'end':p['start']+start+len(needle),'content':piece.content})
    assert b.score(list(index),children,index)['all_evidence_covered']
    return docs,parents,children,index

def expand(hits,parents):
    seen=set(); expanded=[]
    for hit in hits:
        pid=hit['parent_id']
        if pid not in seen:
            seen.add(pid); expanded.append(parents[pid])
    return expanded

def aggregate(rows,variant):
    subset=[r[variant] for r in rows if r[variant]['metrics'] is not None]
    allrows=[r[variant] for r in rows]
    def avg(key): return sum(r['metrics'][key] for r in subset)/len(subset) if subset else None
    bycat={}
    for cat in sorted({r['category'] for r in rows}):
        rs=[r[variant] for r in rows if r['category']==cat and r[variant]['metrics'] is not None]
        bycat[cat]={'scored_count':len(rs),'coverage_recall':sum(r['metrics']['coverage_recall_at_k'] for r in rs)/len(rs) if rs else None}
    return {'questions':len(rows),'scored_count':len(subset),'coverage_recall':avg('coverage_recall_at_k'),'all_evidence_coverage':avg('all_evidence_covered'),'context_chars_mean':sum(r['context_chars'] for r in allrows)/len(allrows),'context_chars_p95':b.percentile([r['context_chars'] for r in allrows],.95),'by_category':bycat}

def selected_baseline(kb):
    completed=[]
    for p in sorted((EVAL/'runs'/kb).glob('*/manifest.json')):
        m=json.loads(p.read_text())
        if m['status']=='completed' and m['split']=='dev' and m['full_split']:completed.append((p,m))
    if not completed:raise ValueError('No completed dev baseline')
    return completed[-1]

def check():
    report={}
    for kb in b.CONFIG:
        _,parents,children,index=units(kb)
        report[kb]={'parents':len(parents),'children':len(children),'max_parent_chars':max(len(p['content']) for p in parents.values()),'evidence_mapped':len(index),'all_evidence_covered':True}
    print(json.dumps(report,ensure_ascii=False,indent=2))

def run(kb):
    settings=b.settings()
    docs,parents,children,index=units(kb)
    questions=[q for q in b.validate_questions(kb,index) if q['split']=='dev']
    old_path,oldmanifest=selected_baseline(kb)
    assert oldmanifest['config']['embedding_model']==settings['embedding_model']
    assert oldmanifest['config']['embedding_url']==settings['embedding_url']
    assert oldmanifest['question_sha256']==b.sha(b.ROOT/'evaluation/v2'/kb/'questions.jsonl')
    for source,hashvalue in oldmanifest['source_files'].items(): assert b.sha(b.ROOT/source)==hashvalue
    old={r['question_id']:r for r in b.read(old_path.with_name('results.jsonl'))}
    assert set(old)=={q['id'] for q in questions}
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=EVAL/'experiments'/kb/runid;out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','kind':'structural_chunk_ablation','knowledge_base':kb,'split':'dev','question_count':len(questions),'model_config':settings,'child_max_chars':500,'child_overlap':50,'top_k_children':4,'parents':len(parents),'children':len(children),'gold_used_for_retrieval':False,'baseline_manifest':str(old_path.relative_to(b.ROOT)),'baseline_manifest_sha256':b.sha(old_path),'source_files':oldmanifest['source_files'],'questions_sha256':oldmanifest['question_sha256'],'script_sha256':b.sha(Path(__file__)),'baseline_helper_sha256':b.sha(Path(b.__file__)),'repository_commit':subprocess.check_output(['git','-C',str(b.REPO),'rev-parse','HEAD'],text=True).strip(),'changes':'Index independent FAQ/article bodies; split long bodies with original 500/50; no title prefix, no version filter, no reranker. Parent expansion changes only post-retrieval context. Preambles/chapter headings not indexed as separate units.','baseline_reused':True,'scoring_threshold':.8,'latency_scope':'structural query timings exclude batched query embedding; cached baseline timing not compared directly'}
    b.dump(out/'manifest.json',manifest)
    b.dump(out/'parents.json',parents)
    with (out/'children.jsonl').open('w') as f:
        for c in children:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    import chromadb
    from chromadb.config import Settings
    db=chromadb.PersistentClient(path=str(out/'chroma'),settings=Settings(anonymized_telemetry=False))
    col=db.create_collection('structural_children',embedding_function=None,metadata={'hnsw:space':'l2'})
    emb=b.client(settings['embedding_url'],'EMBEDDING_API_KEY')
    index_tokens=[];query_tokens=[];rows=[]
    started=time.perf_counter()
    try:
        for offset in range(0,len(children),20):
            batch=children[offset:offset+20]
            vectors,tokens=b.embed(emb,settings['embedding_model'],[c['content'] for c in batch])
            index_tokens.append(tokens)
            col.add(ids=[c['id'] for c in batch],embeddings=vectors,metadatas=[{'parent_id':c['parent_id']} for c in batch])
            if offset%200==0:print(f'{kb}: indexed {min(offset+20,len(children))}/{len(children)}',flush=True)
        manifest['index_seconds']=time.perf_counter()-started
        manifest['index_tokens']=sum(index_tokens) if all(t is not None for t in index_tokens) else None
        b.dump(out/'manifest.json',manifest)
        assert col.count()==len(children)
        qvectors=[];query_started=time.perf_counter()
        for offset in range(0,len(questions),20):
            vectors,tokens=b.embed(emb,settings['embedding_model'],[q['question'] for q in questions[offset:offset+20]])
            qvectors.extend(vectors);query_tokens.append(tokens)
        manifest['query_embedding_seconds']=time.perf_counter()-query_started
        manifest['query_embedding_tokens']=sum(query_tokens) if all(t is not None for t in query_tokens) else None
        # Cache vectors in ignored DB directory for follow-up ablations with identical queries.
        import numpy as np
        np.save(out/'chroma/query_vectors.npy',np.asarray(qvectors,dtype=np.float32))
        lookup={c['id']:c for c in children}
        for q,vector in zip(questions,qvectors):
            t=time.perf_counter()
            result=col.query(query_embeddings=[vector],n_results=4,include=['distances'])
            hits=[lookup[i] for i in result['ids'][0]];t1=time.perf_counter()
            full=expand(hits,parents);t2=time.perf_counter()
            row={'question_id':q['id'],'question':q['question'],'category':q['category'],'expected_action':q['expected_action'],'search_seconds':t1-t,'expansion_seconds':t2-t1}
            for name,context in [('baseline',old[q['id']]['hits']),('structural',hits),('expanded',full)]:
                metrics=b.score(q['evidence_ids'],context,index) if q['expected_action'] in ('answer','flag_conflict') else None
                row[name]={'hits':context,'metrics':metrics,'context_chars':sum(len(h['content']) for h in context)}
            assert row['baseline']['metrics']==old[q['id']]['metrics']
            if row['structural']['metrics'] is not None:
                assert row['expanded']['metrics']['coverage_recall_at_k']>=row['structural']['metrics']['coverage_recall_at_k']
            rows.append(row)
        with (out/'results.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        summary={name:aggregate(rows,name) for name in ('baseline','structural','expanded')}
        summary['changes_vs_baseline']={name:{'improved':[r['question_id'] for r in rows if r[name]['metrics'] and r[name]['metrics']['coverage_recall_at_k']>r['baseline']['metrics']['coverage_recall_at_k']],'regressed':[r['question_id'] for r in rows if r[name]['metrics'] and r[name]['metrics']['coverage_recall_at_k']<r['baseline']['metrics']['coverage_recall_at_k']]} for name in ('structural','expanded')}
        summary['generation_quality']='not measured by retrieval coverage; targeted answer checks separate'
        b.dump(out/'summary.json',summary)
        manifest.update(status='completed',completed_questions=len(rows));b.dump(out/'manifest.json',manifest)
        print(json.dumps({'output':str(out),'summary':summary},ensure_ascii=False,indent=2))
    except Exception as e:
        manifest.update(status='failed',error_type=type(e).__name__,http_status=getattr(e,'status_code',None));b.dump(out/'manifest.json',manifest)
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--kb',choices=list(b.CONFIG),default='smart-home');args=p.parse_args()
    try:check() if args.mode=='check' else run(args.kb)
    except Exception as e:
        print(json.dumps({'status':'failed','error_type':type(e).__name__,'http_status':getattr(e,'status_code',None)}),file=sys.stderr);return 2
    return 0
if __name__=='__main__':sys.exit(main())
