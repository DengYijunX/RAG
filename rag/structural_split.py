"""Structural chunking: never split a document where the document says it ends.

Why this exists: with a plain ``RecursiveCharacterTextSplitter`` the separators
are generic (blank line, newline, sentence ending), so a chunk can fill up and
break **inside** a section. Measured on the accounting corpus, article 第五条 of
CAS-35-2006 spans the seam between chunk ``:0000`` and ``:0001``: the article is
present in the document but no single chunk contains it. Nothing downstream can
recover that — no pool depth and no reranker can rank a passage that is not in
any retrievable unit, and the label lies across two units so it resolves to
neither.

The fix is not domain knowledge: the document already marks its own boundaries
with headings, and ``parse_structure`` has always parsed them. This module makes
the splitter honour them, by trying the document's own heading levels **before**
any generic separator. A Markdown heading, a PDF section or a plain-text
divider all qualify; no rule mentions Chinese articles or any specific standard.

Trade-off: honouring structure produces more, smaller chunks, because a short
section no longer gets merged with its neighbour. That is the point — it is what
keeps a section whole — and the retrieval effect is measured, not assumed.
"""
from dataclasses import dataclass, field
import re

# A Markdown ATX heading: up to three leading spaces, then 1-6 '#', then space.
ATX = re.compile(r'(?:^|\n)[ \t]{0,3}#{1,6}[ \t]+\S')


@dataclass
class StructuralSplit:
    texts: list = field(default_factory=list)
    #: True when the split fell on a document-declared boundary.
    structural: bool = False
    boundaries: int = 0


#: Version of the chunking algorithm. Bump it whenever the boundaries a chunk
#: may fall on change, so an import made by an older algorithm is not reused:
#: a chunk-set change invalidates both the import manifest and any index built
#: from it, and reusing either would silently serve the old chunking.
VERSION = 'structural-1'


def has_headings(text):
    return bool(ATX.search(text or ''))


def structural_separators(text):
    """Separators that try the document's own headings before any generic one.

    ``RecursiveCharacterTextSplitter`` applies separators in order: it splits on
    the first separator that appears and recurses into the pieces with the
    remaining ones. Putting a single "before any heading" pattern first therefore
    makes *every* heading a candidate boundary, at whatever level the document
    uses, before blank lines or sentence endings are considered.

    Breaking before every level matters: an earlier version matched only levels
    1-2 first, which let a whole chapter stay in one piece, so the running
    character limit still cut inside an article instead of at its heading.
    """
    if not has_headings(text):
        return None
    return [
        # Newline-anchored on purpose: a bare ``^`` made the splitter treat every
        # line start as a boundary, which is why the heading pattern looked like
        # it worked while producing the same chunks as the generic splitter.
        r'(?=\n#{1,6}[ \t])',
        '\n\n',
        '\n',
        '。',
        '.',
        '？',
        '?',
        ';',
        '；',
        ' ',
        '',
    ]


def split_text(text, chunk_size, chunk_overlap, fallback):
    """Split ``text`` at its own boundaries first, then by size.

    ``fallback`` is the generic splitter used for documents that declare no
    structure, so behaviour is unchanged for those.
    """
    separators = structural_separators(text)
    if separators is None:
        return StructuralSplit(texts=fallback(text), structural=False)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=separators, is_separator_regex=True)
    pieces = [piece for piece in splitter.split_text(text) if piece.strip()]
    boundaries = sum(1 for piece in pieces if ATX.match(piece))
    return StructuralSplit(texts=pieces, structural=True, boundaries=boundaries)
