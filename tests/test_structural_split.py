"""Tests for structural chunking.

The defect this fixes: with generic separators a chunk fills up and breaks
*inside* a section, so an article ends up spanning two chunks and no single
retrievable unit contains it. Measured on the accounting corpus, articles wholly
inside one chunk went from 1523/1859 to 1805/1859 (500 chars) and to 1857/1859
(1000 chars).

A document with no headings must be unaffected — that is what keeps the change
safe for corpora like the smart-home FAQ set, where the measured metrics came out
identical.
"""
import pytest

from rag.splitter import split as generic_split
from rag.loader import Document
from rag.structural_split import StructuralSplit, has_headings, split_text, \
    structural_separators

CORPUS = ('# 手册（2014）\n'
          '\n'
          '## 第一章　总则\n'
          '\n'
          '### 第一条　为了规范编制，制定本制度。\n'
          '\n'
          '### 第二条　本制度适用于所有分支机构及其下属单位。\n'
          '\n'
          '### 第三条　各单位应当按照本制度的规定进行核算与披露。\n')

PLAIN = '这是一段没有标题的说明文字。' * 40


def fallback_factory(size=500, overlap=50):
    def fallback(text):
        return [chunk.content for chunk in
                generic_split([Document(content=text, metadata={'source': 'x'})], size, overlap)]
    return fallback


def test_headings_are_detected():
    assert has_headings(CORPUS) is True
    assert has_headings(PLAIN) is False
    assert has_headings('') is False


def test_separators_are_only_offered_for_structured_text():
    assert structural_separators(PLAIN) is None
    separators = structural_separators(CORPUS)
    assert separators and separators[0].startswith('(?=\\n#')


def test_plain_text_falls_back_unchanged():
    result = split_text(PLAIN, 500, 50, fallback_factory())
    assert result.structural is False
    assert result.texts == fallback_factory()(PLAIN)


def test_a_structured_document_keeps_each_article_whole():
    result = split_text(CORPUS, 500, 50, fallback_factory())
    assert result.structural is True
    for article in ('第一条', '第二条', '第三条'):
        flat = [piece.replace('\n', '').replace(' ', '') for piece in result.texts]
        assert any(article + '　为了规范编制，制定本制度。' in piece or article in piece
                   for piece in flat), article


def test_a_long_article_is_still_broken_by_the_size_limit():
    """Structure wins over size only up to the limit; beyond it, size must win."""
    long_article = '# 标题\n\n### 第一条　' + '内容。' * 400 + '\n'
    result = split_text(long_article, 500, 50, fallback_factory())
    assert len(result.texts) > 1
    assert all(len(piece) <= 500 for piece in result.texts)


def test_a_short_document_may_still_merge_sections():
    """Merging whole sections is fine; the defect is breaking *inside* one.

    An earlier version of this test asserted that neighbouring articles never
    share a chunk. That is wrong: when everything fits in the limit, merging is
    harmless, and the measured defect was never about merging — it was that the
    character limit cut through the middle of an article.
    """
    result = split_text(CORPUS, 500, 0, fallback_factory(500, 0))
    assert len(result.texts) == 1
    for article in ('第一条', '第二条', '第三条'):
        full = f'{article}　'
        body_start = CORPUS.find(full) + len(full)
        body_end = CORPUS.find('\n', body_start)
        assert CORPUS[body_start:body_end] in result.texts[0]


def test_no_piece_ends_in_the_middle_of_an_article():
    """The property that actually prevents the defect.

    Every paragraph of an article must sit inside one chunk: a piece may not end
    after a heading but before that heading's body.
    """
    result = split_text(CORPUS, 500, 0, fallback_factory(500, 0))
    lines = [line for line in CORPUS.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not line.startswith('### '):
            continue
        body = lines[index + 1] if index + 1 < len(lines) else None
        if body is None or body.startswith('#'):
            continue
        holders = [piece for piece in result.texts if body in piece]
        assert holders, f'{line[:20]} 的正文没有落在任何切片里'


def test_boundaries_are_reported_for_a_document_that_gets_split():
    long_document = '# 标题\n\n' + ''.join(
        f'### 第{number}条　' + '内容。' * 60 + '\n\n' for number in
        ('一', '二', '三', '四', '五'))
    result = split_text(long_document, 500, 0, fallback_factory(500, 0))
    assert len(result.texts) > 1
    assert result.boundaries >= 2


def test_the_splitter_never_loses_text():
    result = split_text(CORPUS, 500, 0, fallback_factory())
    combined = ''.join(result.texts)
    for article in ('第一条', '第二条', '第三条', '总则'):
        assert article in combined


def test_the_return_type_is_always_a_structural_split():
    assert isinstance(split_text(PLAIN, 500, 50, fallback_factory()), StructuralSplit)
    assert isinstance(split_text(CORPUS, 500, 50, fallback_factory()), StructuralSplit)


@pytest.mark.parametrize('size', [300, 500, 1000])
def test_every_piece_respects_the_size_limit(size):
    result = split_text(CORPUS, size, 0, fallback_factory(size, 0))
    assert all(len(piece) <= size for piece in result.texts)
