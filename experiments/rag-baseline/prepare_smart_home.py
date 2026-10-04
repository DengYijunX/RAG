import hashlib,json,shutil
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OUT=ROOT/'smart-home'; OUT.mkdir(exist_ok=True); (OUT/'corpus').mkdir(exist_ok=True)
rows=[json.loads(l) for l in (ROOT/'questions.jsonl').read_text().splitlines()]
evidence=[json.loads(l) for l in (ROOT/'evidence.jsonl').read_text().splitlines()]
docs=[]
for p in sorted((REPO/'文档').glob('*.md')):
    shutil.copyfile(p,OUT/'corpus'/p.name)
    docs.append({'source':str(p.relative_to(REPO)),'corpus_path':'corpus/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'knowledge_base':'smart_home','authority_verified':False})
# Include clarification evidence in split groups; no FAQ evidence crosses splits.
parent={e['id']:e['id'] for e in evidence}
def find(k):
    while parent[k]!=k: parent[k]=parent[parent[k]]; k=parent[k]
    return k
for r in rows:
    refs=r['evidence_ids']
    for ref in refs[1:]: parent[find(ref)]=find(refs[0])
groups=defaultdict(list)
for r in rows:
    refs=r['evidence_ids']; group=find(refs[0]) if refs else r['id']
    groups[group].append(r)
count=0
for group,rs in sorted(groups.items(),key=lambda item:hashlib.sha256(item[0].encode()).hexdigest()):
    split='test' if count+len(rs)<=20 else 'dev'
    if split=='test': count+=len(rs)
    for r in rs:
        r.update(knowledge_base='smart_home',split=split,split_group=group,generation_method='authored_from_source',annotation_status='draft_needs_domain_review')
for name,data in [('questions.jsonl',rows),('evidence.jsonl',evidence),('documents.jsonl',docs)]+[(f'questions.{s}.jsonl',[r for r in rows if r['split']==s]) for s in ('dev','test')]:
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
report={'documents':len(docs),'evidence_sections':len(evidence),'questions':len(rows),'categories':dict(Counter(r['category'] for r in rows)),'splits':{s:dict(Counter(r['category'] for r in rows if r['split']==s)) for s in ('dev','test')},'split_policy':'all shared FAQ evidence, including clarification cases, grouped; scenario coverage is uneven','source_defects':['X2 battery one-year quality-defect warranty conflicts with general three-month policy','X2 no Zigbee/WiFi interference statement conflicts with G2 2.4GHz interference guidance'],'annotation_status':'draft_needs_domain_review'}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
