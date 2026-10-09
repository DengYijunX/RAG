"""Assemble contiguous source text around a retrieved chunk.

Why this exists: several questions never retrieve their target article because
the article is **split across adjacent chunks**. The text is present in the
source document, but **no single chunk contains it**, so no ranking, fusion or
reranking step can recover it — it is not in any retrievable unit. The measured
case is the accounting corpus: of a 300–570 character article, the longest run
found inside any one chunk was about 80 characters.

What it does: given a retrieved chunk, walk outward through the neighbouring
chunks of the *same source document* and return their contiguous text.
Neighbours come from document order in the index manifest, not from character
offsets computed here, so a PDF or a non-Markdown source cannot produce a
fabricated seam.

What it is not: it is not parent-child chunking, and it does not change
retrieval. Ranking still happens over single chunks; expansion only affects the
text handed to generation. That separation is deliberate — retrieval metrics
must keep scoring individual chunks, otherwise expansion would silently inflate
them.
"""
from dataclasses import dataclass, field


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
    order, so grouping by source recovers the reading order.
    """
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk['source'], []).append(chunk['id'])
    return grouped


class NeighbourExpander:
    """Builds neighbourhoods from an index manifest's chunk list."""

    def __init__(self, chunks):
        self.by_id = {chunk['id']: chunk for chunk in chunks}
        self.grouped = document_order(chunks)
        self.position = {}
        for source, ids in self.grouped.items():
            for index, identifier in enumerate(ids):
                self.position[identifier] = (source, index)

    def neighbourhood(self, chunk_id, radius=1):
        """Return the chunk plus up to ``radius`` neighbours on each side.

        A radius of 0 returns the chunk unchanged. An unknown id yields an empty
        neighbourhood rather than raising: a stale retrieval artifact must not
        crash a request.
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
        neighbourhood of its own slot.
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


def expand_hits(hits, chunks, radius=1):
    """Rewrite retrieved hits so each carries its neighbourhood's text.

    Retrieval is untouched: the same hits, the same order, the same count. Only
    ``content`` and the provenance fields change, and ``expanded_span`` records
    which chunk ids the text now covers so a reader can tell an expanded hit
    from a plain one.

    Every returned item keeps its original ``id`` — that is the chunk that was
    actually retrieved — so evaluation alignment and citation identity stay
    intact.
    """
    if radius <= 0:
        return [dict(hit) for hit in hits]
    expander = NeighbourExpander(chunks)
    output = []
    for hit in hits:
        neighbourhood = expander.neighbourhood(hit['id'], radius)
        if not neighbourhood.chunk_ids:
            output.append(dict(hit))
            continue
        merged = dict(hit)
        merged['content'] = neighbourhood.content
        merged['expanded'] = neighbourhood.expanded
        merged['expanded_span'] = list(neighbourhood.chunk_ids)
        output.append(merged)
    return output
