import baseline
from structure_context import build_index, metadata, structure_messages


def test_hidden_heading_text_never_enters_model_metadata():
    text='# Secret answer outside selection\nvisible body\n# Next\nend'
    start=text.index('visible');body='visible body'
    hit={'id':'c','source':'a.md','content':body,'start':start,'end':start+len(body)}
    docs,locations=build_index({'a.md':{'content':text}},[hit],'kb')
    records=metadata([hit],docs,locations)
    msg=structure_messages([{'content':'system'},{'content':body+'\n\n问题：q'}],records)
    assert 'Secret answer' not in str(msg)
    block=records[0]['blocks'][0]
    assert block['heading_path'][0]['title_excerpt'] is None
    assert block['continues_from_previous'] is True


def test_union_coverage_and_source_isolation():
    text='## Same\nfirst\nsecond';cut=text.index('second')
    chunks=[{'id':'a1','source':'a.md','content':text[:cut],'start':0,'end':cut},
            {'id':'a2','source':'a.md','content':text[cut:],'start':cut,'end':len(text)},
            {'id':'b1','source':'b.md','content':text[cut:],'start':cut,'end':len(text)}]
    docs,locations=build_index({s:{'content':text} for s in ['a.md','b.md']},chunks,'kb')
    records=metadata(chunks,docs,locations)
    assert all(r['blocks'][0]['section_complete_in_context'] for r in records[:2])
    assert records[2]['blocks'][0]['section_complete_in_context'] is False
    assert records[0]['blocks'][0]['section_id']!=records[2]['blocks'][0]['section_id']
    assert records[0]['blocks'][0]['heading_path'][0]['title_excerpt']=='Same'
