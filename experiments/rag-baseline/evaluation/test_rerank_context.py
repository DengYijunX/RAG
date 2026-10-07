import json
import pytest
from rerank_context import expanded_pool, parse_order, validated_order, boundary_messages


def chunk(key,source,pos):
    return {'id':key,'source':source,'start':pos,'end':pos+1,'content':key}


def test_pool_keeps_seeds_first_and_adds_only_one_hop_same_source():
    chunks=[chunk('a','doc',0),chunk('b','doc',1),chunk('c','doc',2),
            chunk('d','doc',3),chunk('other-version','other',1)]
    pool,origins=expanded_pool(['b','b'],chunks)
    assert [c['id'] for c in pool]==['b','c','a']
    assert origins['c']=={'kind':'next','seed':'b'}


@pytest.mark.parametrize('raw',[
    '{"order":["C01","C01"]}', '{"order":["C01"]}',
    '{"order":["C01","C99"]}', '{"order":[1,"C02"]}',
    'not json', '{"order":"C01"}', '[]',
])
def test_invalid_model_order_falls_back_without_dropping_or_inventing_candidates(raw):
    pool=[chunk('a','doc',0),chunk('b','doc',1)]
    assert validated_order(raw,pool)==(['a','b'],'invalid_order')


def test_valid_order_is_exact_permutation():
    pool=[chunk('a','doc',0),chunk('b','doc',1)]
    assert parse_order(json.dumps({'order':['C02','C01']}),pool)==['b','a']


def test_boundary_format_preserves_original_article_headings_and_final_question():
    hit={'id':'a','source':'doc','start':3,'end':12,'content':'续文\n### 第六条\n下一条正文'}
    c={'citation':1,'source':'doc','title':'Document','content':hit['content']}
    messages=boundary_messages([{'role':'system','content':'base'}], '用户问题', [c], [hit])
    assert hit['content'] in messages[-1]['content']
    assert messages[-1]['content'].endswith('问题：用户问题')
    assert '片段引用编号不是原文条款编号' in messages[0]['content']
    assert '第五条' not in messages[-1]['content']
