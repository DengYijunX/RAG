"""Map retrieved chunks back to the dataset's passage ids.

The two id spaces are different by construction. A dataset labels relevance
with its own passage ids — ``D1-S01`` in ``evidence.jsonl``, ``CAS-10-2006-A002``
in ``articles.jsonl`` — while retrieval returns chunk ids derived from the
imported file. WeKnora solves this by matching retrieved text against the
ground-truth passages; that works, but silently scoring an unmatched chunk as
a miss hides the case where the label itself is stale.

This resolver therefore reports its misses:

- ``resolved``    chunk id -> passage id for every hit it could place
- ``unresolved``  hit ids it could not place, with the reason
- ``coverage``    share of hits that were placed

Matching is exact on normalized whitespace/newlines. No semantic matching, no
fuzzy thresholds: a near-miss that is scored as a hit is worse than a reported
failure. Passage texts are loaded from the dataset's own files, not inferred
from retrieval results.
"""
from dataclasses import dataclass, field
import hashlib
import re


def normalize_text(text):
    """Collapse whitespace so line-ending and spacing differences do not matter."""
    return re.sub(r'\s+', '', text or '')


def text_digest(text):
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()


@dataclass
class Resolution:
    """Result of placing one question's hits in the dataset's id space."""
    ranked_passage_ids: list = field(default_factory=list)
    resolved: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)

    def as_dict(self):
        return {'resolved': list(self.resolved), 'unresolved': list(self.unresolved),
                'coverage': self.coverage}

    @property
    def coverage(self):
        total = len(self.resolved) + len(self.unresolved)
        return len(self.resolved) / total if total else None


class PassageResolver:
    """Places chunk hits into a dataset's passage id space.

    ``passages`` maps ``passage_id -> text``. Both layouts are supported: pass
    the dataset's passages once and resolve every question, or build one
    resolver per knowledge base.
    """

    def __init__(self, passages):
        by_digest = {}
        for passage_id, text in passages.items():
            by_digest.setdefault(text_digest(text), []).append(str(passage_id))
        # Only digests that identify exactly one passage can be resolved by
        # text; anything else is ambiguous and must be reported, not guessed.
        self.unique = {digest: ids[0] for digest, ids in by_digest.items() if len(ids) == 1}
        self.duplicates = [(digest, tuple(ids)) for digest, ids in by_digest.items() if len(ids) > 1]
        self.ambiguous_digests = {digest for digest, _ in self.duplicates}

    def resolve(self, hits, passage_ids=None):
        """Resolve ranked hits, keeping one slot per hit.

        ``ranked_passage_ids`` has the same length as ``hits``; a hit that
        could not be placed becomes ``None`` so that the ranks after it are not
        shifted upwards. Dropping unresolved hits would silently improve every
        rank-based metric.
        """
        allowed = set(passage_ids) if passage_ids is not None else None
        result = Resolution()
        for hit in hits:
            identifier = str(hit.get('id', ''))
            direct = hit.get('passage_id')
            if direct and (allowed is None or str(direct) in allowed):
                result.resolved.append((identifier, str(direct)))
                result.ranked_passage_ids.append(str(direct))
                continue
            digest = text_digest(hit.get('content', ''))
            passage_id = self.unique.get(digest)
            if passage_id and (allowed is None or passage_id in allowed):
                result.resolved.append((identifier, passage_id))
                result.ranked_passage_ids.append(passage_id)
                continue
            reason = 'no_matching_passage'
            if digest in self.ambiguous_digests:
                reason = 'ambiguous_passage_text'
            elif self.unique.get(digest) and allowed is not None:
                reason = 'labelled_for_another_question'
            result.unresolved.append({'id': identifier, 'rank': len(result.ranked_passage_ids) + 1,
                                      'reason': reason})
            result.ranked_passage_ids.append(None)
        return result


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
