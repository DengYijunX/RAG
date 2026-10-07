from copy import deepcopy
import pytest
from rag.context_structure import compact_structure, compact_messages, enforce_input_byte_budget, InputBudgetExceeded


def record(doc='a',citation=1,section='same',title='Visible',complete=False):
    return {'document_id':doc,'citation':citation,'source_sha256':'AUDIT_HASH',
            'blocks':[{'section_id':section,'heading_path':[] if section is None else [{'id':section,'title_excerpt':title}],
                       'recognition':'unknown' if section is None else 'markdown_atx','chunk_relative_span':[0,5],
                       'continues_from_previous':None if section is None else True,
                       'continues_after':None if section is None else False,'section_complete_in_context':complete}]}


def test_repeated_section_deduplicates_without_merging_documents_or_mutation():
    records=[record(),record(citation=2),record(doc='b',citation=3)]
    before=deepcopy(records);view=compact_structure(records)
    assert len(view['sections'])==2
    assert view['chunks'][0][1][0][2]==view['chunks'][1][1][0][2]
    assert view['chunks'][2][1][0][2]!=view['chunks'][0][1][0][2]
    assert records==before and 'AUDIT_HASH' not in str(view)


def test_unknown_and_hidden_titles_preserve_uncertainty():
    view=compact_structure([record(title=None),record(section=None,complete=None,citation=2)])
    assert view['sections'][0][2] is None
    assert view['chunks'][1][1][0]==[0,5,None,None,None,None]


def test_ancestry_and_heuristic_are_preserved():
    r=record();r['blocks'][0]['heading_path'].insert(0,{'id':'parent','title_excerpt':'Parent'})
    r['blocks'][0]['recognition']='standalone_bold_heuristic'
    view=compact_structure([r])
    assert view['sections']==[['S1',None,'Parent','markdown'],['S2','S1','Visible','bold']]
    assert view['chunks'][0][1][0]==[0,5,'S2',True,False,False]


def test_inconsistent_definition_fails():
    with pytest.raises(ValueError):compact_structure([record(),record(title='different')])
    r=record();r['blocks'][0]['section_id']='mismatch'
    with pytest.raises(ValueError):compact_structure([r])


def test_prompt_preserves_body_question_and_full_budget_is_enforced():
    boundary=[{'role':'system','content':'Rules'},{'role':'user','content':'原文🙂\n\n问题：多部分问题'}]
    messages=compact_messages(boundary,[record()])
    assert messages[1]['content'].startswith('原文🙂\n\n结构附注：')
    assert messages[1]['content'].endswith('\n\n问题：多部分问题')
    size=enforce_input_byte_budget(messages,10000)
    assert size>len(str(messages))
    assert enforce_input_byte_budget(messages,size)==size
    with pytest.raises(InputBudgetExceeded):enforce_input_byte_budget(messages,size-1)
    for value in [0,-1,True,1.5]:
        with pytest.raises(ValueError):enforce_input_byte_budget(messages,value)
