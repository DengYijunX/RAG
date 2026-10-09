"""Tests for neighbour expansion on the product path.

Expansion must not touch retrieval: same hits, same order, same identity. The
retrieved id stays the chunk that was actually ranked, so evaluation alignment
and citation identity survive; only the text and its provenance change.
"""
import pytest

from rag.context import NeighbourExpander, document_order, expand_hits


def chunks(*spec):
    """Build chunk records: (id, source, content)."""
    return [{'id': identifier, 'source': source, 'content': content,
             'start': index * 100, 'end': index * 100 + 99}
            for index, (identifier, source, content) in enumerate(spec)]


THREE = chunks(('c1', 'docA', '第一段'), ('c2', 'docA', '第二段'), ('c3', 'docA', '第三段'),
               ('d1', 'docB', '另一文档'))


def hits(*identifiers):
    return [{'id': identifier, 'source': 'docA', 'content': '原文', 'start': 0, 'end': 1}
            for identifier in identifiers]


def test_document_order_groups_by_source_preserving_order():
    grouped = document_order(THREE)
    assert grouped['docA'] == ['c1', 'c2', 'c3']
    assert grouped['docB'] == ['d1']


def test_radius_zero_returns_the_hit_unchanged():
    output = expand_hits(hits('c2'), THREE, radius=0)
    assert output[0]['content'] == '原文'
    assert 'expanded' not in output[0]


def test_radius_one_joins_the_immediate_neighbours():
    output = expand_hits(hits('c2'), THREE, radius=1)
    assert output[0]['content'] == '第一段\n第二段\n第三段'
    assert output[0]['expanded'] is True
    assert output[0]['expanded_span'] == ['c1', 'c2', 'c3']


def test_an_edge_chunk_only_expands_inward():
    assert expand_hits(hits('c1'), THREE, radius=1)[0]['expanded_span'] == ['c1', 'c2']
    assert expand_hits(hits('c3'), THREE, radius=1)[0]['expanded_span'] == ['c2', 'c3']


def test_expansion_never_crosses_into_another_document():
    """A fabricated seam between two documents would be worse than no expansion."""
    spans = [hit['expanded_span'] for hit in expand_hits(hits('c3', 'd1'), THREE, radius=2)]
    assert spans == [['c1', 'c2', 'c3'], ['d1']]
    assert all({'c1', 'c2', 'c3'}.isdisjoint(span) or 'd1' not in span for span in spans)


def test_retrieved_identity_and_order_are_preserved():
    output = expand_hits(hits('c3', 'c1'), THREE, radius=1)
    assert [hit['id'] for hit in output] == ['c3', 'c1']   # same order as retrieved
    assert output[0]['source'] == 'docA'


def test_an_unknown_id_passes_through_rather_than_crashing():
    output = expand_hits([{'id': 'gone', 'source': 'docA', 'content': '旧片段'}], THREE, radius=1)
    assert output[0]['content'] == '旧片段'
    assert 'expanded_span' not in output[0]


def test_the_single_chunk_case_is_not_marked_expanded():
    lonely = chunks(('only', 'docC', '唯一一段'))
    output = expand_hits(hits('only'), lonely, radius=1)
    assert output[0]['content'] == '唯一一段'
    assert output[0]['expanded'] is False


def test_a_single_chunk_document_is_not_expanded():
    output = expand_hits(hits('d1'), THREE, radius=1)
    assert output[0]['expanded'] is False
    assert output[0]['expanded_span'] == ['d1']


def test_expansion_does_not_mutate_the_input():
    original = hits('c2')
    expand_hits(original, THREE, radius=1)
    assert original[0]['content'] == '原文'


def test_expander_reports_an_empty_neighbourhood_for_unknown_ids():
    expander = NeighbourExpander(THREE)
    neighbourhood = expander.neighbourhood('missing', radius=1)
    assert neighbourhood.chunk_ids == [] and neighbourhood.content == ''
    assert neighbourhood.expanded is False


@pytest.mark.parametrize('identifier,radius,expected', [
    # c2 sits in the middle, so radius 1 reaches both sides.
    ('c2', 1, '第一段\n第二段\n第三段'),
    ('c2', 2, '第一段\n第二段\n第三段'),
    # c1 is at the edge, so radius 1 can only reach right.
    ('c1', 1, '第一段\n第二段'),
    ('c1', 2, '第一段\n第二段\n第三段'),
])
def test_radius_selects_how_far_it_reaches(identifier, radius, expected):
    assert expand_hits(hits(identifier), THREE, radius=radius)[0]['content'] == expected
