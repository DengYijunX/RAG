"""Unmodified repository splitter + Chroma L2 baseline, with source span scoring.
No gold answers enter retrieval or generation. Tests/synthetic runs are not model results.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time, uuid, subprocess
from importlib.metadata import version
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from rag.loader import load_from_dir
from rag.splitter import split, Chunk

URL='https://llm-k5ve8qaaz1tn5w7b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
CONFIG={'smart-home':('corpus','evidence.jsonl'),'accounting':('markdown','articles.jsonl')}
def read(path): return [json.loads(l) for l in path.read_text().splitlines()]
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(text): return re.sub(r'[\s#*]+','',text)
def union_size(intervals):
    end=-1; total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end)); end=max(end,b)
    return total

def prepare(kb):
    corpus,ename=CONFIG[kb]
    docs=load_from_dir(str(ROOT/kb/corpus),'*.md')
    chunks=split(docs,500,50)
    records=read(ROOT/kb/ename)
    by_source=defaultdict(list)
    for r in records:
        filename=Path(r['source']).name if kb=='smart-home' else r['document_id']+'.md'
        by_source[filename].append(r)
    index={}; normalized={}
    for d in docs:
        filename=Path(d.metadata['source']).name
        text=normalize(d.content); normalized[filename]=text
        cursor=0
        for r in by_source[filename]:
            needle=normalize(r['text']); pos=text.find(needle,cursor)
            if pos<0: raise ValueError(f'Evidence cannot be mapped: {r["id"]}')
            index[r['id']]={'start':pos,'end':pos+len(needle),'source':filename}
            cursor=pos+len(needle)
    cursors=defaultdict(int); output=[]
    for c in chunks:
        source=Path(c.metadata['source']).name
        needle=normalize(c.content); pos=normalized[source].find(needle,cursors[source])
        if pos<0: raise ValueError(f'Chunk cannot be mapped: {source} {c.metadata["chunk_index"]}')
        cursors[source]=pos+1
        output.append({'id':f'{kb}:{source}:{c.metadata["chunk_index"]}', 'source':source,'chunk_index':c.metadata['chunk_index'],'start':pos,'end':pos+len(needle),'content':c.content})
    assert len(index)==len(records)
    return docs,output,index

def score(gold_ids,hits,index,threshold=.8):
    if not gold_ids: return None
    overlaps={g:[] for g in gold_ids}; first=None
    for rank,h in enumerate(hits,1):
        for g in gold_ids:
            e=index[g]
            if e['source']==h['source']:
                a,b=max(e['start'],h['start']),min(e['end'],h['end'])
                if b>a: overlaps[g].append((a,b))
        if first is None and any(union_size(overlaps[g])/(index[g]['end']-index[g]['start']) >= threshold for g in gold_ids): first=rank
    ratios={g:union_size(overlaps[g])/(index[g]['end']-index[g]['start']) for g in gold_ids}
    covered=sum(r>=threshold for r in ratios.values())
    return {'evidence_coverage':ratios,'coverage_recall_at_k':covered/len(gold_ids),'all_evidence_covered':covered==len(gold_ids),'coverage_reciprocal_rank':1/first if first else 0}

def validate_questions(kb,index):
    rows=read(ROOT/'evaluation/v2'/kb/'questions.jsonl')
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['question'] for r in rows})==len(rows)
    sets={s:set() for s in ('dev','test')}; families={s:set() for s in sets}
    for q in rows:
        assert all(g in index for g in q['evidence_ids'])
        if q['expected_action'] in ('answer','flag_conflict'): assert q['evidence_ids']
        sets[q['split']].update(q['evidence_ids']); families[q['split']].update(q.get('standard_families',[]))
    assert not sets['dev'] & sets['test']
    assert not families['dev'] & families['test']
    return rows

def offline():
    report={'kind':'offline_mapping_validation','model_calls':0,'knowledge_bases':{}}
    for kb in CONFIG:
        docs,chunks,index=prepare(kb); qs=validate_questions(kb,index)
        full=score(list(index),chunks,index)
        assert full['all_evidence_covered']
        ceilings=[]
        for q in qs:
            if q['expected_action'] not in ('answer','flag_conflict'): continue
            # Necessary single-evidence ceiling only, not a joint top-K oracle.
            for g in q['evidence_ids']:
                e=index[g]
                best=sorted(chunks,key=lambda c: max(0,min(e['end'],c['end'])-max(e['start'],c['start'])) if c['source']==e['source'] else 0,reverse=True)[:4]
                if score([g],best,index)['coverage_recall_at_k']<1: ceilings.append({'question':q['id'],'evidence':g})
        report['knowledge_bases'][kb]={'documents':len(docs),'chunks':len(chunks),'evidence':len(index),'questions':len(qs),'all_evidence_mapped':True,'full_corpus_covers_all_evidence':True,'long_evidence_top4_warnings':ceilings}
        dump(ROOT/'evaluation/v2'/kb/'evidence_spans.json',index)
    dump(ROOT/'evaluation/v2/offline_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def settings(generate=False):
    # dotenv reads any existing user config; never print or persist credential values.
    from dotenv import load_dotenv
    load_dotenv(REPO/'.env',override=False)
    required=['EMBEDDING_API_KEY','EMBEDDING_MODEL']+(['LLM_API_KEY'] if generate else [])
    missing=[k for k in required if not os.environ.get(k)]
    if missing: raise ValueError('Missing configuration: '+', '.join(missing))
    endpoint=os.getenv('EMBEDDING_BASE_URL',URL)
    # User supplied this exact canonical URL earlier; only repair its known bare host.
    if endpoint == urlparse(URL).hostname:
        endpoint=URL
    parsed=urlparse(endpoint)
    if parsed.scheme!='https' or not parsed.hostname:
        raise ValueError('Invalid configuration: EMBEDDING_BASE_URL must be a complete HTTPS URL, not a bare hostname')
    return {'embedding_url':endpoint,'embedding_model':os.environ['EMBEDDING_MODEL'],'llm_url':os.getenv('LLM_BASE_URL','https://api.deepseek.com/v1'),'llm_model':os.getenv('LLM_MODEL','deepseek-chat')}

def client(url,key):
    from openai import OpenAI
    return OpenAI(base_url=url,api_key=os.environ[key],timeout=60,max_retries=2)

def embed(c,model,texts):
    result=c.embeddings.create(model=model,input=texts)
    data=sorted(result.data,key=lambda x:x.index)
    if [x.index for x in data]!=list(range(len(texts))): raise ValueError('Embedding response indexes inconsistent')
    vectors=[x.embedding for x in data]
    if not vectors or not vectors[0] or any(len(v)!=len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in vectors): raise ValueError('Embedding response shape/values invalid')
    return vectors,getattr(getattr(result,'usage',None),'total_tokens',None)

def preflight(generate=False):
    s=settings(generate)
    c=client(s['embedding_url'],'EMBEDDING_API_KEY')
    vectors,tokens=embed(c,s['embedding_model'],['Embedding connectivity check.'])
    report={'embedding':'passed','dimension':len(vectors[0]),'tokens':tokens,'configuration':s,'llm':'not_requested'}
    if generate:
        llm=client(s['llm_url'],'LLM_API_KEY')
        result=llm.chat.completions.create(model=s['llm_model'],messages=[{'role':'user','content':'Reply OK.'}],max_tokens=8)
        if not result.choices or not result.choices[0].message.content: raise ValueError('LLM returned empty response')
        report['llm']='passed'
    print(json.dumps(report,ensure_ascii=False,indent=2))

def percentile(values,p):
    if not values: return None
    return sorted(values)[max(0,math.ceil(len(values)*p)-1)]

def run(kb,split_name,generate,limit):
    s=settings(generate)
    docs,chunks,index=prepare(kb); allqs=validate_questions(kb,index)
    qs=[q for q in allqs if q['split']==split_name]
    if limit: qs=qs[:limit]
    assert qs
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=ROOT/'evaluation/runs'/kb/runid
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','kind':'real_api_baseline','knowledge_base':kb,'split':split_name,'question_count':len(qs),'full_split':len(qs)==sum(q['split']==split_name for q in allqs),'config':s,'chunk_size':500,'chunk_overlap':50,'top_k':4,'distance':'l2','coverage_threshold':.8,'question_sha256':sha(ROOT/'evaluation/v2'/kb/'questions.jsonl'),'source_files':{str(Path(d.metadata['source']).relative_to(ROOT)):sha(Path(d.metadata['source'])) for d in docs},'runner_sha256':sha(Path(__file__)),'generate':generate,'answers_manually_scored':False}
    manifest['repository_commit']=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    manifest['dependency_versions']={name:version(name) for name in ('chromadb','numpy','openai','langchain-text-splitters')}
    manifest['python_version']=sys.version.split()[0]
    dump(out/'manifest.json',manifest)
    import chromadb
    from chromadb.config import Settings
    db=chromadb.PersistentClient(path=str(out/'chroma'),settings=Settings(anonymized_telemetry=False))
    col=db.create_collection('rag_documents',embedding_function=None,metadata={'hnsw:space':'l2'})
    emb=client(s['embedding_url'],'EMBEDDING_API_KEY')
    llm=client(s['llm_url'],'LLM_API_KEY') if generate else None
    lookup={c['id']:c for c in chunks}
    rows=[]; index_tokens=[]; started=time.perf_counter()
    try:
        for offset in range(0,len(chunks),20):
            batch=chunks[offset:offset+20]
            vectors,tokens=embed(emb,s['embedding_model'],[c['content'] for c in batch])
            index_tokens.append(tokens)
            col.add(ids=[c['id'] for c in batch],embeddings=vectors,documents=[c['content'] for c in batch],metadatas=[{'source':c['source'],'chunk_index':c['chunk_index']} for c in batch])
        manifest['index_seconds']=time.perf_counter()-started
        manifest['indexed_chunks']=col.count()
        assert col.count()==len(chunks)
        for q in qs:
            t=time.perf_counter(); vector,tokens=embed(emb,s['embedding_model'],[q['question']]); t1=time.perf_counter()
            found=col.query(query_embeddings=vector,n_results=4,include=['distances'])
            hits=[lookup[i] for i in found['ids'][0]]; t2=time.perf_counter()
            scored=score(q['evidence_ids'],hits,index) if q['expected_action'] in ('answer','flag_conflict') else None
            row={'question_id':q['id'],'category':q['category'],'question':q['question'],'retrieved_ids':found['ids'][0],'distances':found['distances'][0],'hits':hits,'metrics':scored,'query_embedding_tokens':tokens,'embedding_seconds':t1-t,'vector_search_seconds':t2-t1}
            if llm:
                # Match current rag.generator prompt; no extra gold labels or special version hints.
                context='\n\n---\n\n'.join(f'[来源: {h["source"]}]\n{h["content"]}' for h in hits)
                answer=llm.chat.completions.create(model=s['llm_model'],messages=[{'role':'system','content':'你是一个基于文档回答问题的助手。请根据提供的文档内容回答问题。如果文档中没有相关信息，请如实告知。'},{'role':'user','content':f'参考文档：\n\n{context}\n\n问题：{q["question"]}'}])
                row['answer']=answer.choices[0].message.content
                if not row['answer']: raise ValueError('Empty LLM answer')
                row['generation_tokens']=answer.usage.model_dump() if answer.usage else None
                row['generation_seconds']=time.perf_counter()-t2
            row['total_seconds']=time.perf_counter()-t
            rows.append(row)
            with (out/'results.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        scored_rows=[r for r in rows if r['metrics'] is not None]
        grouped={}
        for category in sorted({r['category'] for r in rows}):
            subset=[r for r in scored_rows if r['category']==category]
            grouped[category]={'retrieval_scored_count':len(subset),'coverage_recall_at_4':sum(r['metrics']['coverage_recall_at_k'] for r in subset)/len(subset) if subset else None}
        summary={'completed_questions':len(rows),'retrieval_scored_questions':len(scored_rows),'coverage_recall_at_4':sum(r['metrics']['coverage_recall_at_k'] for r in scored_rows)/len(scored_rows) if scored_rows else None,'all_evidence_coverage_rate':sum(r['metrics']['all_evidence_covered'] for r in scored_rows)/len(scored_rows) if scored_rows else None,'coverage_mrr_at_4':sum(r['metrics']['coverage_reciprocal_rank'] for r in scored_rows)/len(scored_rows) if scored_rows else None,'by_category':grouped,'total_seconds_p50':percentile([r['total_seconds'] for r in rows],.5),'total_seconds_p95':percentile([r['total_seconds'] for r in rows],.95),'latency_conditions':'serial, no warmup, includes network/API retries; not production load testing','semantic_answer_accuracy':None,'refusal_accuracy':None,'review_required':True}
        dump(out/'summary.json',summary)
        # Human review sheet contains labels, separate from retrieval inputs.
        with (out/'answer_review.jsonl').open('w') as f:
            for q,r in zip(qs,rows): f.write(json.dumps({'question_id':q['id'],'question':q['question'],'expected_action':q['expected_action'],'expected_points':q['expected_points'],'answer':r.get('answer'),'correctness':None,'citation_support':None,'actual_action':None,'reviewer':None},ensure_ascii=False)+'\n')
        manifest['status']='completed'; dump(out/'manifest.json',manifest)
        print(json.dumps({'output':str(out),'summary':summary},ensure_ascii=False,indent=2))
    except Exception as e:
        manifest.update(status='failed',completed_questions=len(rows),error_type=type(e).__name__,http_status=getattr(e,'status_code',None))
        dump(out/'manifest.json',manifest)
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','preflight','run'])
    parser.add_argument('--kb',choices=list(CONFIG),default='smart-home')
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error('--limit must be positive')
    try:
        if args.mode=='prepare': offline()
        elif args.mode=='preflight': preflight(args.generate)
        else: run(args.kb,args.split,args.generate,args.limit)
    except Exception as e:
        # Do not dump API exception bodies or request headers (can expose credentials).
        safe={'status':'blocked_or_failed','error_type':type(e).__name__,'http_status':getattr(e,'status_code',None)}
        if isinstance(e,ValueError) and str(e).startswith(('Missing configuration:', 'Invalid configuration:')): safe['message']=str(e)
        print(json.dumps(safe,ensure_ascii=False),file=sys.stderr)
        return 2
    return 0
if __name__=='__main__': sys.exit(main())
