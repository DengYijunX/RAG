"""Offline integrity checks; these do NOT measure retrieval/answer quality."""
import hashlib,json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(path): return [json.loads(l) for l in path.read_text().splitlines()]
def validate(kb,evname):
    base=ROOT/kb
    rows=read(base/'questions.jsonl'); ev=read(base/evname); lookup={e['id']:e for e in ev}
    assert len(lookup)==len(ev)
    assert len(set(r['id'] for r in rows))==len(rows)
    assert len(set(r['question'] for r in rows))==len(rows)
    seen={s:set() for s in ('dev','test')}; families={s:set() for s in ('dev','test')}
    for r in rows:
        assert r['annotation_status']=='draft_needs_domain_review'
        assert r['expected_points'] and r['split'] in seen
        if r['expected_action']=='answer': assert r['evidence_ids']
        for e in r['evidence_ids']:
            assert e in lookup,(r['id'],e)
            seen[r['split']].add(e)
        families[r['split']].update(r.get('standard_families',[]))
        for ref in r.get('reference_excerpts',[]): assert ref['text']==lookup[ref['article_id']]['text']
        assert not set(r['evidence_ids']) & set(r.get('wrong_version_candidates',[]))
    assert not seen['dev'] & seen['test'], 'Evidence leakage'
    assert not families['dev'] & families['test'], 'Standard-family leakage'
    assert read(base/'questions.dev.jsonl')==[r for r in rows if r['split']=='dev']
    assert read(base/'questions.test.jsonl')==[r for r in rows if r['split']=='test']
    return {'questions':len(rows),'evidence':len(ev),'splits':dict(Counter(r['split'] for r in rows)),'checks':'unique IDs/questions, references, exact excerpts, evidence/family split isolation, split files'}

result={'smart_home':validate('smart-home','evidence.jsonl'),'accounting':validate('accounting','articles.jsonl')}
base=ROOT/'accounting'; docs=read(base/'documents.jsonl'); arts=read(base/'articles.jsonl')
assert len(docs)==59
assert not json.loads((base/'corpus_report.json').read_text())['issues']
for d in docs:
    assert hashlib.sha256((base/d['raw_path']).read_bytes()).hexdigest()==d['raw_sha256']
    text=(base/d['text_path']).read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==d['text_sha256']
    rendered=(base/d['markdown_path']).read_text()
    # Only markdown heading markers and layout whitespace may change.
    norm=lambda s:re.sub(r'\s+','',s)
    assert norm(re.sub(r'^#{1,3} ', '', rendered,flags=re.M))==norm(text),d['id']
    subset=[a for a in arts if a['document_id']==d['id']]
    assert [a['article_number'] for a in subset]==list(range(1,d['article_count']+1))
    for a in subset:
        assert a['text'] in text,(a['id'],'article text not contiguous in source')
        assert text.splitlines()[a['source_start_line']-1].strip().startswith(a['article_label'])
for d in read(ROOT/'smart-home'/'documents.jsonl'):
    assert hashlib.sha256((ROOT/'smart-home'/d['corpus_path']).read_bytes()).hexdigest()==d['sha256']
result['corpus_checks']='file hashes, markdown prose preservation, continuous article numbers, exact source text and source lines'
(ROOT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
