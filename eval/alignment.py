"""Map retrieved chunks back to the dataset's passage ids.

The two id spaces are different by construction. A dataset labels relevance
with its own passage ids — ``D1-S01`` in ``evidence.jsonl``, ``CAS-10-2006-A002``
in ``articles.jsonl`` — while retrieval returns chunk ids derived from the
imported file. WeKnora solves this by matching retrieved text against the
ground-truth passages; that works, but silently scoring an unmatched chunk as
a miss hides the case where the label itself is stale.

This resolver therefore offers two levels and reports its misses:

- ``resolved``    hits it could place, with how it placed them
- ``unresolved``  hits it could not place, with the reason
- ``coverage``    share of hits that were placed

Matching order, strongest first:

1. an explicit ``passage_id`` on the hit (the importer can carry one later);
2. identical normalised text — the chunk *is* the passage;
3. the passage text appears inside the chunk — the common real case, because
   a retrieved chunk usually spans a heading plus several paragraphs while the
   dataset labels one of them.

Containment is matched longest-passage-first, and a chunk is credited only
when exactly one passage of the question's labels matches it; if two labels
are found in the same chunk, the hit is reported as ambiguous and scored
against neither. Matching is exact on normalised text (whitespace, ``#`` and
``*`` removed); there is no fuzzy threshold, because a near miss counted as a
hit is worse than a reported failure.
"""
from dataclasses import dataclass, field
import hashlib
import re


def normalize_text(text):
    """Collapse whitespace and markup so formatting differences do not matter."""
    return re.sub(r'[\s#*]+', '', text or '')


def text_digest(text):
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()


@dataclass
class Resolution:
    """Result of placing one question's hits in the dataset's id space.

    ``labels_per_hit`` keeps one entry per retrieved hit, in retrieval order,
    each holding every dataset passage that hit contains. ``ranked_passage_ids``
    flattens those entries in order and de-duplicates, which is the list the
    metrics consume: the first relevant passage found contributes its rank, and
    seeing it again later adds nothing.

    One hit often contains several labelled passages — an FAQ chunk covers a
    heading plus two answers — so forcing a single label per hit would hand the
    chunk to one label and hide the others, understating recall.
    """
    labels_per_hit: list = field(default_factory=list)
    resolved: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)

    @property
    def ranked_passage_ids(self):
        ranked, seen = [], set()
        for labels in self.labels_per_hit:
            for passage_id in labels:
                if passage_id not in seen:
                    seen.add(passage_id)
                    ranked.append(passage_id)
        return ranked

    @property
    def coverage(self):
        total = len(self.resolved) + len(self.unresolved)
        return len(self.resolved) / total if total else None

    def as_dict(self):
        return {'labels_per_hit': list(self.labels_per_hit),
                'resolved': list(self.resolved), 'unresolved': list(self.unresolved),
                'coverage': self.coverage}


class PassageResolver:
    """Places chunk hits into a dataset's passage id space.

    ``passages`` maps ``passage_id -> text``. Pass the dataset's passages once
    and resolve every question, or build one resolver per knowledge base.
    """

    def __init__(self, passages, max_containment_candidates=20000):
        self.passages = {str(key): value or '' for key, value in passages.items()}
        self.normalized = {key: normalize_text(text) for key, text in self.passages.items()}
        by_digest = {}
        for key, text in self.normalized.items():
            by_digest.setdefault(text_digest(text), []).append(key)
        # Text that identifies exactly one passage can be resolved by identity.
        self.unique = {digest: ids[0] for digest, ids in by_digest.items() if len(ids) == 1}
        self.duplicates = [(digest, tuple(ids)) for digest, ids in by_digest.items() if len(ids) > 1]
        self.ambiguous_digests = {digest for digest, _ in self.duplicates}
        self.max_containment_candidates = max_containment_candidates
        self._containment_skipped = 0

    def _contained(self, normalized_chunk, allowed):
        """Passages whose normalised text sits inside the chunk, longest first."""
        candidates = [key for key in allowed
                      if self.normalized[key] and len(self.normalized[key]) <= len(normalized_chunk)]
        if len(candidates) > self.max_containment_candidates:
            self._containment_skipped += 1
            return []
        candidates.sort(key=lambda key: -len(self.normalized[key]))
        return [key for key in candidates if self.normalized[key] in normalized_chunk]

    def resolve(self, hits, passage_ids=None):
        """Resolve ranked hits, keeping one entry per hit in retrieval order."""
        allowed = [str(item) for item in passage_ids] if passage_ids is not None \
            else list(self.passages)
        allowed = [key for key in allowed if key in self.passages]
        result = Resolution()
        for position, hit in enumerate(hits, 1):
            identifier = str(hit.get('id', ''))
            labels, how = self._place(hit, allowed)
            if labels:
                result.labels_per_hit.append(labels)
                result.resolved.append({'id': identifier, 'rank': position,
                                        'passage_ids': labels, 'match': how})
            else:
                result.labels_per_hit.append([])
                result.unresolved.append({'id': identifier, 'rank': position,
                                          'reason': self._reason(hit, allowed)})
        return result

    def _place(self, hit, allowed):
        direct = hit.get('passage_id')
        if direct and str(direct) in self.passages:
            return [str(direct)], 'explicit_id'
        digest = text_digest(hit.get('content', ''))
        # Text that several passages share identifies none of them. Refuse it
        # even when containment would match, rather than crediting a coin flip.
        if digest in self.ambiguous_digests:
            return [], ''
        exact = self.unique.get(digest)
        if exact and exact in allowed:
            return [exact], 'exact_text'
        contained = self._contained(normalize_text(hit.get('content', '')), allowed)
        if contained:
            return contained, 'contained_text'
        return [], ''

    def _reason(self, hit, allowed):
        digest = text_digest(hit.get('content', ''))
        if digest in self.ambiguous_digests:
            return 'ambiguous_passage_text'
        if self.unique.get(digest):
            return 'labelled_for_another_question'
        if self._contained(normalize_text(hit.get('content', '')), allowed):
            return 'contained_text'
        return 'no_matching_passage'


def load_passages(path, id_field='id', text_field='text'):
    """Read ``passage_id -> text`` from a JSONL corpus index.

    Used with ``evidence.jsonl`` (smart-home) and ``articles.jsonl``
    (accounting), which are the datasets' own passage files.
    """
    import json
    from pathlib import Path

    passages = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = record.get(id_field)
        if identifier is None:
            raise KeyError(f'{path} 的记录缺少 {id_field}')
        passages[str(identifier)] = record.get(text_field) or ''
    return passages
