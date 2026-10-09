"""Tests for neighbour expansion.

The behaviour that matters: an article split across adjacent chunks becomes
retrievable once the retrieved chunk is expanded, and the expansion never
crosses into another source document.
"""
import pytest

from neighbour_context import NeighbourExpander, document_order, load_manifest_chunks


def chunks(*triples):
    return [{'id': identifier, 'source': source, 'content': content}
            for identifier, source, content in triples]


CORPUS = chunks(
    ('a1', 'doc-a.md', '第一条 甲'),
    ('a2', 'doc-a.md', '第二条 乙'),
    ('a3', 'doc-a.md', '第三条 丙'),
    ('b1', 'doc-b.md', '另一文档 丁'),
    ('b2', 'doc-b.md', '另一文档 戊'),
)


def test_document_order_groups_by_source():
    assert document_order(CORPUS) == {'doc-a.md': ['a1', 'a2', 'a3'],
                                      'doc-b.md': ['b1', 'b2']}


def test_radius_zero_is_the_chunk_itself():
    expander = NeighbourExpander(CORPUS)
    neighbourhood = expander.neighbourhood('a2', radius=0)
    assert neighbourhood.chunk_ids == ['a2']
    assert neighbourhood.content == '第二条 乙'
    assert neighbourhood.expanded is False


def test_radius_one_joins_adjacent_chunks_in_order():
    expander = NeighbourExpander(CORPUS)
    neighbourhood = expander.neighbourhood('a2', radius=1)
    assert neighbourhood.chunk_ids == ['a1', 'a2', 'a3']
    assert neighbourhood.content == '第一条 甲\n第二条 乙\n第三条 丙'
    assert neighbourhood.expanded is True


def test_expansion_does_not_cross_source_documents():
    expander = NeighbourExpander(CORPUS)
    assert expander.neighbourhood('a3', radius=1).chunk_ids == ['a2', 'a3']
    assert expander.neighbourhood('b1', radius=1).chunk_ids == ['b1', 'b2']
    assert 'doc-a' not in expander.neighbourhood('b1', radius=5).content


def test_radius_is_clamped_at_the_document_edges():
    expander = NeighbourExpander(CORPUS)
    assert expander.neighbourhood('a1', radius=5).chunk_ids == ['a1', 'a2', 'a3']
    assert expander.neighbourhood('b2', radius=9).chunk_ids == ['b1', 'b2']


def test_an_article_split_across_two_chunks_becomes_whole():
    article = '第五条 业务分部，是指企业内可区分的组成部分。'
    split = chunks(
        ('c1', 'doc.md', article[:12]),
        ('c2', 'doc.md', article[12:]),
    )
    expander = NeighbourExpander(split)
    assert article not in split[0]['content'] and article not in split[1]['content']
    # Either half, expanded, reconstructs the article.
    for identifier in ('c1', 'c2'):
        joined = expander.neighbourhood(identifier, radius=1).content.replace('\n', '')
        assert article in joined


def test_unknown_chunk_id_yields_an_empty_neighbourhood():
    expander = NeighbourExpander(CORPUS)
    neighbourhood = expander.neighbourhood('missing', radius=1)
    assert neighbourhood.chunk_ids == [] and neighbourhood.content == ''


def test_expand_ranking_deduplicates_neighbours():
    expander = NeighbourExpander(CORPUS)
    # Expanding a1 emits a1,a2; expanding a2 then contributes only a3.
    result = expander.expand_ranking(['a1', 'a2', 'b1'], radius=1)
    emitted = [identifier for item in result for identifier in item.chunk_ids]
    assert emitted == ['a1', 'a2', 'a3', 'b1', 'b2']
    assert len(emitted) == len(set(emitted))


def test_expand_ranking_keeps_rank_order_and_limit():
    expander = NeighbourExpander(CORPUS)
    result = expander.expand_ranking(['b1', 'a1'], radius=0, limit=1)
    assert len(result) == 1 and result[0].chunk_id == 'b1'
    assert expander.expanded_text(['b1', 'a1'], radius=0) == '另一文档 丁\n\n第一条 甲'


def test_missing_ids_are_skipped_not_fatal():
    expander = NeighbourExpander(CORPUS)
    assert expander.expand_ranking(['missing', 'a1'], radius=0)[0].chunk_id == 'a1'


def test_load_manifest_chunks_rejects_nothing_but_reads_both_shapes(tmp_path):
    import json

    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'chunks': [{'id': 'x'}]}), encoding='utf-8')
    assert load_manifest_chunks(path) == [{'id': 'x'}]
    path.write_text(json.dumps({'chunk_records': [{'id': 'y'}]}), encoding='utf-8')
    assert load_manifest_chunks(path) == [{'id': 'y'}]
