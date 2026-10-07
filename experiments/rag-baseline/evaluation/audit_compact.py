"""Offline audit of paired inputs, retained structural fields and run provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import baseline as b
from compact_experiment import inputs, EVAL


def audit(run):
    run=run.resolve()
    plan,source,_,expected=inputs();manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['split']=='dev'
    assert manifest['source_manifest_sha256']==b.sha(source/'manifest.json')
    for name,digest in manifest['output_sha256'].items():assert b.sha(run/name)==digest
    for name,digest in manifest['source_sha256'].items():
        data=subprocess.check_output(['git','show',manifest['repository_commit']+':'+name],cwd=b.REPO)
        assert hashlib.sha256(data).hexdigest()==digest
    cases=[]
    for desired in expected:
        name=desired['knowledge_base']+'-'+desired['question_id']+'.json';r=json.loads((run/name).read_text())
        for field in ['question','selected_ids','citations','messages','input_bytes','compact_structure','metrics']:
            assert r[field]==desired[field]
        assert set(r['variants'])==set(plan['variants'])
        assert all(v['finish_reason']=='stop' for v in r['variants'].values())
        original=json.loads((source/name).read_text())['structure']
        view=r['compact_structure'];short={s[0]:s for s in view['sections']};mapping={};seen=set()
        for full,(citation,blocks) in zip(original,view['chunks'],strict=True):
            assert citation==full['citation']
            for block,packed in zip(full['blocks'],blocks,strict=True):
                a,end,sid,before,after,complete=packed
                assert [a,end]==block['chunk_relative_span']
                assert [before,after,complete]==[block['continues_from_previous'],block['continues_after'],block['section_complete_in_context']]
                current=sid
                for heading in reversed(block['heading_path']):
                    key=(full['document_id'],heading['id'])
                    assert current is not None and short[current][2]==heading['title_excerpt']
                    if key in mapping:assert mapping[key]==current
                    else:
                        assert current not in seen;mapping[key]=current;seen.add(current)
                    current=short[current][1]
                assert current is None
                if block['recognition']=='standalone_bold_heuristic':assert short[sid][3]=='bold'
        assert seen==set(short)
        cases.append({'question_id':r['question_id'],'input_bytes':r['input_bytes'],'prompt_tokens':{v:g['usage']['prompt_tokens'] for v,g in r['variants'].items()}})
    result={'run':str(run.relative_to(EVAL)),'status':'passed','paired_cases':len(cases),'generation_calls':manifest['generation_calls'],
            'code_commit':manifest['repository_commit'],'unchanged_bodies_order_citations':True,
            'retained_ancestry_title_visibility_and_boundary_states':True,'source_isolated_short_ids':True,
            'compact_input_byte_cap':plan['compact_max_input_bytes'],'heldout_calls':0,
            'token_counts':'actual provider usage, not byte estimate','semantic_quality':'see separate assistant review','cases':cases}
    b.dump(EVAL/'compact_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args();audit(args.run)
