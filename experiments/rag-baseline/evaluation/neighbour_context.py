"""Neighbour expansion: assemble contiguous source text around a retrieved chunk.

Why this exists: several accounting questions never retrieve their article
because the article is **split across adjacent chunks**. Measured on the four
failing questions (ACC-Q041/Q059/Q073/Q083), the labelled article is present in
the source document but **no single chunk contains it** — the longest run of
consecutive text found inside any chunk was about 80 characters of a 300–570
character article. When no chunk holds the passage, no amount of reranking or
fusion can recover it: the text is simply not in any retrievable unit.

What this module does: given a retrieved chunk, walk outward through the
neighbouring chunks of the *same source document* and return their contiguous
text. Neighbours are derived from document order in the manifest, not from
character offsets guessed here, so a PDF or a non-Markdown source cannot
produce a fabricated seam.

What it is not: it is not parent-child chunking. Retrieval still ranks single
chunks; expansion only affects the text handed onward. Splitting the index into
small retrieval units and larger context units is a separate design decision and
is not implemented here.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Neighbourhood:
    """A retrieved chunk plus its contiguous neighbours in the same source."""
    chunk_id: str
    source: str
    chunk_ids: list = field(default_factory=list)
    content: str = ''
    expanded: bool = False


def document_order(chunks):
    """Group chunk ids by source document, preserving the manifest's order.

    The importer writes one document's chunks consecutively and in document
    order, so grouping by source recovers the reading order. The order is
    validated by the caller's ability to reconstruct text, and never assumed to
    be a character offset.
    """
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk['source'], []).append(chunk['id'])
    return grouped


class NeighbourExpander:
    """Builds neighbourhoods from a manifest's chunk list."""

    def __init__(self, chunks):
        self.by_id = {chunk['id']: chunk for chunk in chunks}
        self.grouped = document_order(chunks)
        self.position = {}
        for source, ids in self.grouped.items():
            for index, identifier in enumerate(ids):
                self.position[identifier] = (source, index)

    def neighbourhood(self, chunk_id, radius=1):
        """Return the chunk plus up to ``radius`` neighbours on each side.

        A radius of 0 returns the chunk unchanged. Unknown ids are returned as
        an empty neighbourhood rather than raising: a stale retrieval artifact
        must not crash a scoring pass.
        """
        located = self.position.get(chunk_id)
        if located is None:
            return Neighbourhood(chunk_id=chunk_id, source='', chunk_ids=[], content='')
        source, index = located
        ids = self.grouped[source]
        start = max(0, index - radius)
        stop = min(len(ids), index + radius + 1)
        selected = ids[start:stop]
        content = '\n'.join(self.by_id[identifier]['content'] for identifier in selected)
        return Neighbourhood(chunk_id=chunk_id, source=source, chunk_ids=selected,
                             content=content, expanded=len(selected) > 1)

    def expand_ranking(self, ranked_ids, radius=1, limit=None, deduplicate=True):
        """Expand a ranked list, keeping rank order.

        With ``deduplicate`` (the default) a chunk that already appeared as a
        neighbour is not emitted again, but it still counts toward the returned
        neighbourhood of its own slot — the caller decides whether to score the
        neighbourhoods separately or the flattened list.
        """
        output, seen = [], set()
        for identifier in ranked_ids:
            neighbourhood = self.neighbourhood(identifier, radius)
            if not neighbourhood.chunk_ids:
                continue
            if deduplicate:
                fresh = [item for item in neighbourhood.chunk_ids if item not in seen]
                if not fresh:
                    continue
                seen.update(fresh)
                neighbourhood = Neighbourhood(
                    chunk_id=identifier, source=neighbourhood.source, chunk_ids=fresh,
                    content='\n'.join(self.by_id[item]['content'] for item in fresh),
                    expanded=len(fresh) > 1)
            output.append(neighbourhood)
            if limit and len(output) >= limit:
                break
        return output

    def expanded_text(self, ranked_ids, radius=1, limit=None):
        """Contiguous text of the top ``limit`` neighbourhoods, in rank order."""
        return '\n\n'.join(item.content for item in
                           self.expand_ranking(ranked_ids, radius, limit))


def load_manifest_chunks(path):
    """Read chunk records from a published evaluation generation."""
    import json

    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload['chunk_records'] if 'chunk_records' in payload else payload['chunks']
