"""Domain-independent experimental context selection; no relevance labels accepted."""
from collections import defaultdict


def select_context(candidate_ids, chunks, max_chars, neighbors=False):
    """Pack whole chunks; optionally visit each seed's next then previous chunk.

    Positions are source-local. A seed must fit before its neighbors are visited.
    Duplicate IDs are removed, but overlapping text is not merged. Character
    budget counts chunk bodies, not model tokens or citation/title overhead.
    """
    if max_chars < 1:
        raise ValueError('Context budget must be positive')
    lookup = {c['id']: c for c in chunks}
    if len(lookup) != len(chunks):
        raise ValueError('Chunk IDs must be unique')
    sources = defaultdict(list)
    for c in chunks:
        sources[c['source']].append(c)
    positions = {}
    for source, group in sources.items():
        group.sort(key=lambda c: (c['start'], c['end'], c['id']))
        for index, chunk in enumerate(group):
            positions[chunk['id']] = (source, index)
    hits, selected, decisions = [], set(), []
    used = 0

    def include(key, origin, seed):
        nonlocal used
        if key in selected:
            return True
        chunk = lookup[key]
        fits = used+len(chunk['content']) <= max_chars
        decisions.append({'id':key, 'seed':seed, 'origin':origin,
                          'accepted':fits, 'reason':'fits' if fits else 'budget'})
        if fits:
            hits.append(chunk)
            selected.add(key)
            used += len(chunk['content'])
        return fits

    for key in dict.fromkeys(candidate_ids):
        if not include(key, 'dense_seed', key):
            continue
        if neighbors:
            source, index = positions[key]
            for offset, origin in ((1, 'next'), (-1, 'previous')):
                adjacent = index+offset
                if 0 <= adjacent < len(sources[source]):
                    include(sources[source][adjacent]['id'], origin, key)
    return {'hits':hits, 'context_characters':used, 'decisions':decisions}
