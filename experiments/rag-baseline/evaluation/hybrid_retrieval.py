"""BM25 keyword retrieval and reciprocal rank fusion.

Plain-Python prototype for the hybrid-retrieval comparison. Two design points
matter for this project's corpus:

**Tokenisation.** The corpora are Chinese with embedded Latin terms
(``CAS-14-2017``, ``2.4GHz``, ``X1``). There is no word segmenter in the
dependencies, so Chinese runs are indexed as overlapping character bigrams —
standard practice precisely because it needs no dictionary — while ASCII
sequences are kept as whole tokens. Queries are tokenised the same way, so a
question containing ``第三十五条`` matches article text ``第三十五条`` even when
nothing else aligns.

**Fusion.** RRF combines *ranks*, not scores, so BM25's unbounded scores are
never compared with cosine similarities. It is parameter-light (only ``k``),
which keeps the comparison against dense-only honest: no fusion weight was
tuned on the dev set.

This module is evaluation-side code: ``rag`` must not import it. It exists to
measure whether keyword retrieval adds anything before anything is promoted
into the product.
"""
from dataclasses import dataclass, field
import math
import re

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

ASCII_RUN = re.compile(r'[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*')
CJK_RUN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]+')


def tokenize(text):
    """Bigrams for CJK runs, whole tokens for ASCII runs.

    ``CAS-14-2017`` stays one token; ``第三十五条`` becomes ``第三``/``三十``/
    ``十五``/``五条``. Bigrams over-match a little by design: at this corpus
    size a missed exact term costs more than a loose one.
    """
    tokens = []
    lowered = (text or '').lower()
    for match in ASCII_RUN.finditer(lowered):
        tokens.append(match.group(0))
    for match in CJK_RUN.finditer(lowered):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed document set.

    Documents are the knowledge base's chunks; callers keep the chunk ids and
    the index only returns them ranked.
    """

    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.ids = []
        self.lengths = []
        self.frequencies = []
        self.document_frequency = {}
        self.average_length = 0.0

    def add(self, ids, texts):
        for identifier, text in zip(ids, texts):
            tokens = tokenize(text)
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.ids.append(identifier)
            self.lengths.append(len(tokens))
            self.frequencies.append(counts)
            for token in counts:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
        total = sum(self.lengths)
        self.average_length = total / len(self.lengths) if self.lengths else 0.0
        return self

    def __len__(self):
        return len(self.ids)

    def _idf(self, token):
        total = len(self.ids)
        frequency = self.document_frequency.get(token, 0)
        if frequency == 0:
            return 0.0
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def score(self, query):
        """Scores for every document, descending; zero-scoring docs are dropped."""
        tokens = tokenize(query)
        if not tokens or not self.ids:
            return []
        scores = [0.0] * len(self.ids)
        for token in dict.fromkeys(tokens):
            idf = self._idf(token)
            if idf == 0.0:
                continue
            for position, counts in enumerate(self.frequencies):
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                length = self.lengths[position] or 1
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))
                scores[position] += idf * frequency * (self.k1 + 1) / denominator
        ranked = [(identifier, score) for identifier, score in zip(self.ids, scores) if score > 0]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def search(self, query, limit):
        return [identifier for identifier, _ in self.score(query)[:limit]]


def reciprocal_rank_fusion(rankings, k=RRF_K, limit=None, weights=None):
    """Fuse ranked id lists by rank, not by score.

    ``rankings`` is a sequence of id lists, best first. A document's fused
    score is the sum of ``weight / (k + rank)`` over the lists it appears in.
    Ties break on the id so the output is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError('weights 与 rankings 数量不一致')
    fused = {}
    for ranking, weight in zip(rankings, weights):
        for rank, identifier in enumerate(ranking, 1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [identifier for identifier, _ in ordered[:limit]] if limit else [i for i, _ in ordered]


@dataclass
class HybridRetriever:
    """Dense + BM25 fused by RRF, with the two arms kept inspectable."""
    dense_search: object
    bm25: object
    k: int = RRF_K
    weights: list = field(default_factory=lambda: [1.0, 1.0])

    def arms(self, question, depth):
        """Return ``(dense_ids, keyword_ids)`` at the requested candidate depth."""
        dense = [hit['id'] for hit in self.dense_search(question, depth)]
        keyword = self.bm25.search(question, depth)
        return dense, keyword

    def ranking(self, question, depth, limit):
        dense, keyword = self.arms(question, depth)
        return reciprocal_rank_fusion([dense, keyword], k=self.k, limit=limit,
                                      weights=self.weights)
