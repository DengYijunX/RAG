"""Offline provenance and paired-input audit; no model calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import baseline as b
from section_experiment import inputs, EVAL
from structure_context import structure_messages

RUN=EVAL/'structure_runs/20261007T130324Z-65ea2859'


def audit():
    plan,source,service,snapshots,cases=inputs()
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['generation_calls']==18
    assert manifest['source_manifest_sha256']==b.sha(source/'manifest.json')
    for name,digest in manifest['output_sha256'].items():assert b.sha(RUN/name)==digest
    # Historical path structure_experiment.py was renamed after this run to avoid
    # replacing the older structural-retrieval experiment. Validate immutable Git.
    for name,digest in manifest['source_sha256'].items():
        data=subprocess.check_output(['git','show',manifest['repository_commit']+':'+name],cwd=b.REPO)
        assert hashlib.sha256(data).hexdigest()==digest
    assert json.loads((RUN/'structure_snapshot.json').read_text())==json.loads(json.dumps(snapshots))
    points=[]
    for kb,qid,question,hits,records,metrics in cases:
        r=json.loads((RUN/(kb+'-'+qid+'.json')).read_text())
        assert r['question']==question and r['selected_ids']==[h['id'] for h in hits]
        assert r['structure']==records and r['metrics']==metrics
        a=r['variants']['boundary']['generation'];c=r['variants']['structure']['generation']
        assert a['citations']==c['citations']==service.citations(kb,hits)
        assert c['messages']==structure_messages(a['messages'],records)
        # Source excerpts must exist within selected text of that exact source.
        for h,record in zip(hits,records,strict=True):
            selected=[v['content'] for v in hits if v['source']==h['source']]
            for block in record['blocks']:
                for title in block['heading_path']:
                    if title['title_excerpt'] is not None:assert any(title['title_excerpt'] in text for text in selected)
        questions={q['id']:q for q in b.read(EVAL/'v2'/kb/'questions.jsonl') if q['split']=='dev'}
        assert metrics==b.score(questions[qid]['evidence_ids'],hits,service.corpus(kb).parents)
        points.append({'question_id':qid,'body_characters':r['context_characters'],'all_evidence_covered':metrics['all_evidence_covered']})
    result={'run':str(RUN.relative_to(EVAL)),'status':'passed','source_code_verified_at_commit':manifest['repository_commit'],
            'output_hashes':len(manifest['output_sha256']),'documents':manifest['documents'],'literal_chunks':manifest['chunks'],
            'paired_questions':len(cases),'actual_generation_calls':18,'embedding_calls':0,'ranking_calls':0,
            'same_bodies_order_citations':True,'metadata_recomputed_from_source':True,'model_title_excerpts_in_selected_text':True,
            'test_questions_executed':0,'questions':points,'semantic_answer_quality':'requires separate review; not inferred from this audit'}
    b.dump(EVAL/'section_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':audit()
