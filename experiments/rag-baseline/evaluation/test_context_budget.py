import pytest
from context_budget import select_context


def chunk(key, source, start, text):
    return dict(id=key, source=source, start=start, end=start+len(text), content=text)


def test_neighbors_follow_source_positions_not_input_order_or_other_versions():
    chunks = [chunk('tail', 'a-2020', 20, 'tail'), chunk('other', 'a-2021', 10, 'wrong'),
              chunk('head', 'a-2020', 0, 'head'), chunk('seed', 'a-2020', 10, 'seed')]
    got = select_context(['seed'], chunks, 20, neighbors=True)
    assert [c['id'] for c in got['hits']] == ['seed', 'tail', 'head']
    assert got['context_characters'] == 12


def test_whole_chunks_skip_oversized_seed_and_do_not_expand_it():
    chunks = [chunk('long', 'a', 0, '123456'), chunk('neighbor', 'a', 8, 'x'),
              chunk('small', 'b', 0, 'ok')]
    got = select_context(['long', 'small'], chunks, 5, neighbors=True)
    assert [c['id'] for c in got['hits']] == ['small']
    assert got['decisions'][0]['reason'] == 'budget'


def test_expansion_can_displace_relevant_later_seed_without_exceeding_budget():
    chunks = [chunk('a', 'doc', 0, 'aaaa'), chunk('b', 'doc', 4, 'bbbb'),
              chunk('c', 'other', 0, 'cccc')]
    dense = select_context(['a','c'], chunks, 8)
    expanded = select_context(['a','c'], chunks, 8, neighbors=True)
    assert [c['id'] for c in dense['hits']] == ['a','c']
    assert [c['id'] for c in expanded['hits']] == ['a','b']
    assert dense['context_characters'] == expanded['context_characters'] == 8


def test_repeated_seed_and_neighbor_are_counted_once():
    chunks = [chunk('a','doc',0,'aaaa'), chunk('b','doc',4,'bbbb')]
    got = select_context(['a','a','b'], chunks, 8, neighbors=True)
    assert [c['id'] for c in got['hits']] == ['a','b']
    assert got['context_characters'] == 8


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        select_context([], [], 0)
    c = chunk('a','doc',0,'x')
    with pytest.raises(ValueError):
        select_context(['a'], [c,c], 5)
    with pytest.raises(KeyError):
        select_context(['missing'], [c], 5)
