"""Product vector backends.

Two implementations behind one interface (``add`` / ``search`` / ``count`` /
``flush``):

- :class:`FaissVectors` — exact inner-product index over unit vectors. This is
  the default because it is the only backend that builds reliably here: the HNSW
  implementation Chroma uses crashes the interpreter on this machine above
  roughly 96 vectors at 1024 dimensions, so the product's own index could not be
  built at all. Exact also means a query is reproducible.
- The Chroma backend stays available for deployments that want the approximate
  index, or that already hold Chroma collections.

Ranking parity was measured before the default changed: on 1024-dimension vectors
both backends returned the same order and the same scores to within 5e-7, and
Chroma's value is ``1 - cosine`` just like the FAISS backend reports. So changing
the backend does not silently change what is retrieved.

Layering note: this module is imported by the evaluation code, never the other way
round. ``eval`` may depend on ``rag``; ``rag`` must not depend on ``eval``.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

INDEX_FILE = 'index.faiss'
RECORDS_FILE = 'records.jsonl'
META_FILE = 'backend.json'


class NormalizationError(ValueError):
    """Vectors cannot be normalised, so cosine similarity would be wrong."""


def _normalise(vectors):
    """Return unit-length copies, refusing degenerate input.

    A zero vector has no direction; scoring it would produce a meaningless
    similarity, so it is refused rather than silently replaced with zeros.
    """
    import numpy as np

    matrix = np.asarray([list(vector) for vector in vectors], dtype='float64')
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise NormalizationError('向量形状无效')
    if not np.all(np.isfinite(matrix)):
        raise NormalizationError('向量含非有限值')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise NormalizationError('向量含零向量，无法计算余弦相似度')
    return matrix / norms


@dataclass
class FaissVectors:
    """Exact inner-product index over normalised vectors, persisted to disk.

    Exact rather than approximate on purpose: at this corpus size a flat index is
    fast enough (5000x1024 in 0.013 s) and returns the true nearest neighbours, so
    a score is not polluted by index approximation error and a query is
    reproducible.
    """

    directory: Path

    def __post_init__(self):
        self.directory = Path(self.directory)
        self.records = self._read_records()
        self._index = None

    # -- storage ------------------------------------------------------------
    def _read_records(self):
        path = self.directory / RECORDS_FILE
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def _write_records(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / (RECORDS_FILE + '.tmp')
        with open(temporary, 'w', encoding='utf-8') as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(self.directory / RECORDS_FILE)

    # -- index --------------------------------------------------------------
    def _load_index(self):
        import faiss
        import numpy as np

        if self._index is not None:
            return self._index
        path = self.directory / INDEX_FILE
        if not path.exists():
            return None
        # Read the bytes in Python rather than with faiss.read_index: FAISS opens
        # files with fopen, which fails on non-ASCII paths on Windows, and this
        # workspace lives under a Chinese user name.
        self._index = faiss.deserialize_index(
            np.frombuffer(path.read_bytes(), dtype='uint8'))
        if self._index.ntotal != len(self.records):
            raise ValueError(
                f'索引向量数 {self._index.ntotal} 与记录数 {len(self.records)} 不一致')
        return self._index

    def add(self, ids, vectors, documents, metadatas):
        import faiss

        matrix = _normalise(vectors).astype('float32')
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif self._index.d != matrix.shape[1]:
            raise ValueError(f'维度不一致：索引 {self._index.d}，新向量 {matrix.shape[1]}')
        self._index.add(matrix)
        for position, identifier in enumerate(ids):
            self.records.append({'id': identifier, 'document': documents[position],
                                 'metadata': dict(metadatas[position])})

    def search(self, vector, limit):
        index = self._load_index()
        # An index that was never built answers with nothing, like the Chroma
        # backend does for a missing collection. The two backends must behave the
        # same here, or swapping the backend would turn an empty knowledge base
        # into a crash.
        if index is None or index.ntotal == 0 or limit <= 0:
            return []
        query = _normalise([vector]).astype('float32')
        scores, positions = index.search(query, min(limit, index.ntotal))
        hits = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            record = self.records[position]
            # Inner product of unit vectors is cosine similarity; the
            # application's convention is a distance, so report 1 - cosine.
            hits.append({'id': record['id'], 'content': record['document'],
                         'source': record['metadata'].get('source', ''),
                         'distance': 1.0 - float(score)})
        return hits

    def count(self):
        return len(self.records)

    def flush(self):
        """Persist vectors and records; called after a build completes."""
        import faiss

        if self._index is None:
            raise ValueError('没有可写入的索引')
        self.directory.mkdir(parents=True, exist_ok=True)
        # serialize_index returns bytes we write ourselves; faiss.write_index
        # cannot open a non-ASCII path on Windows (see _load_index).
        (self.directory / INDEX_FILE).write_bytes(
            faiss.serialize_index(self._index).tobytes())
        self._write_records()
        (self.directory / META_FILE).write_text(
            json.dumps({'backend': 'faiss-flat-ip', 'dimensions': int(self._index.d),
                        'vectors': int(self._index.ntotal), 'metric': 'cosine',
                        'exact': True}, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        return self.directory

    @staticmethod
    def normalize(vectors):
        """Scale vectors to unit length; the caller keeps the original values."""
        return _normalise(vectors).tolist()


def build_backend(name, directory):
    """Return the backend named by configuration.

    Importing Chroma is deferred to this function so a FAISS deployment never
    pays for it, and so a broken Chroma install cannot prevent an import of the
    product package.
    """
    key = (name or 'faiss').strip().lower()
    if key in {'faiss', 'faiss-flat-ip', 'exact'}:
        return FaissVectors(directory)
    if key == 'chroma':
        from rag.chroma_backend import ChromaVectors

        return ChromaVectors(directory)
    raise ValueError(f'未知的向量后端：{name!r}（可选 faiss / chroma）')
