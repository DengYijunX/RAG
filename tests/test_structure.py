import pytest
from rag.structure import parse_structure, locate_chunks


def test_continuation_inherits_heading_until_next_same_level():
    text='# Guide\n\n## Rule\nfirst\n\ncontinued\n\n## Next\nother'
    doc=parse_structure(text,'doc')
    start=text.index('continued');end=text.index('## Next')
    result=doc.annotate(start,end)['blocks'][0]
    assert [doc.by_id[k].title for k in result['heading_path']]==['Guide','Rule']
    assert result['continues_from_previous'] is True
    assert result['continues_after'] is False
    assert result['section_complete_in_context'] is False
    whole=doc.annotate(start,end,[(text.index('## Rule'),start),(start,end)])['blocks'][0]
    assert whole['section_complete_in_context'] is True


def test_chunk_crosses_sections_and_truncated_conditions_remain_incomplete():
    text='## First\n1. one\n2. two\n3. three\n\n## Second\nnext'
    doc=parse_structure(text,'doc')
    r=doc.annotate(0,text.index('3. three'))['blocks'][0]
    assert r['continues_after'] and not r['section_complete_in_context']
    blocks=doc.annotate(text.index('2. two'),len(text))['blocks']
    assert len(blocks)==2 and blocks[0]['section_id']!=blocks[1]['section_id']


def test_parent_section_not_complete_when_only_its_title_is_present():
    text='# Parent\nintro\n## Child\nbody\n# Next\nend'
    d=parse_structure(text,'doc');end=text.index('## Child')
    b=d.annotate(0,end)['blocks'][0]
    assert b['continues_after'] and not b['section_complete_in_context']
    assert d.sections[0].end==text.index('# Next')


def test_fences_indented_code_and_blockquotes_are_not_headings():
    text='# Real\n```md\n## Fake\n```\n~~~~\n### Also fake\n~~~~\n    # Code\n> # Quote\n## Actual\nbody'
    d=parse_structure(text,'doc')
    assert [s.title for s in d.sections]==['Real','Actual']


def test_setext_unicode_crlf_and_closing_hashes_preserve_offsets():
    text='标题🙂\r\n=====\r\n正文\r\n## 次级 ##\r\n尾部'
    d=parse_structure(text,'doc')
    assert [(s.title,s.level) for s in d.sections]==[('标题🙂',1),('次级',2)]
    for s in d.sections:
        assert text[s.title_start:s.title_end]==s.title
    assert d.sections[1].line==4


def test_bold_heading_is_opt_in_and_marked_heuristic():
    text='**Question**\nanswer\n\n**Next**\nanswer'
    assert parse_structure(text,'doc').sections==[]
    d=parse_structure(text,'doc',bold_headings=True)
    assert len(d.sections)==2
    assert {s.kind for s in d.sections}=={'standalone_bold_heuristic'}


def test_unknown_structure_has_unknown_completeness():
    for d in [parse_structure('plain text','doc'),parse_structure('# literal','doc',format_name='text')]:
        b=d.annotate(0,len(d.text))['blocks'][0]
        assert b['section_id'] is None and b['section_complete_in_context'] is None
        assert b['continues_after'] is None


def test_source_identity_and_repeated_headings_are_distinct():
    text='# Same\na\n# Same\nb'
    a=parse_structure(text,'a');b=parse_structure(text,'b')
    assert len({s.id for s in a.sections})==2
    assert not {s.id for s in a.sections}&{s.id for s in b.sections}


def test_literal_chunk_positions_handle_overlap_and_repeated_text():
    assert locate_chunks('abcabc',['abc','bca','abc'])==[(0,3),(1,4),(3,6)]
    with pytest.raises(ValueError):locate_chunks('source',['missing'])
    with pytest.raises(ValueError):locate_chunks('source',[''])


def test_invalid_span_fails_and_whitespace_gaps_do_not_fake_truncation():
    d=parse_structure('## A\nfirst\n\nlast','doc')
    assert d.covered(0,len(d.text),[(0,10),(12,16)])
    assert not d.covered(0,len(d.text),[(0,5),(12,16)])
    with pytest.raises(ValueError):d.annotate(4,3)
    with pytest.raises(ValueError):d.annotate(0,2,[(0,999)])
