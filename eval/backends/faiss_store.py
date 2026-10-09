"""FAISS vector backend for offline evaluation.

Why this exists: Chroma's default index (HNSW) cannot be built on this
machine — hnswlib crashes with an access violation, and chromadb 1.x, which
drops the hnswlib dependency, crashes the same way. FAISS builds and queries
the same 1024-dimension data without trouble, so evaluation uses FAISS while
the product keeps Chroma as its default backend.

This is *not* a product backend. It lives under ``eval`` on purpose: the
evaluation package must not be imported by ``rag``, and tests/test_eval_scaffold.py
enforces that direction.

Callers must pass already-normalised vectors; the backend checks this and
raises rather than silently turning cosine similarity into a dot product of
unnormalised vectors.
"""
import json
import math
from pathlib import Path

INDEX_FILE = 'index.faiss'
RECORDS_FILE = 'records.jsonl'
META_FILE = 'backend.json'


class NormalizationError(ValueError):
    """Vectors must be L2-normalised before they reach a cosine index."""


def _require_normalised(vectors, tolerance=1e-3):
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or abs(norm - 1.0) > tolerance:
            raise NormalizationError(f'向量未归一化（L2 范数 {norm:.6f}），余弦相似度会失真')


class FaissVectors:
    """Exact inner-product index over normalised vectors, persisted to disk.

    Exact rather than approximate on purpose: at this corpus size a flat index
    is fast enough, and it returns the true nearest neighbours, so evaluation
    scores are not polluted by index approximation error.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = self._read_records()
        self._index = None

    # -- storage ------------------------------------------------------------
    def _read_records(self):
        path = self.directory / RECORDS_FILE
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

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
            raise FileNotFoundError(f'FAISS 索引不存在：{path}')
        # Read the bytes in Python rather than with faiss.read_index: FAISS
        # opens files with fopen, which fails on non-ASCII paths on Windows,
        # and this workspace lives under a Chinese user name.
        self._index = faiss.deserialize_index(
            np.frombuffer(path.read_bytes(), dtype='uint8'))
        if self._index.ntotal != len(self.records):
            raise ValueError(f'索引向量数 {self._index.ntotal} 与记录数 {len(self.records)} 不一致')
        return self._index

    def add(self, ids, vectors, documents, metadatas):
        import faiss
        import numpy as np

        vectors = [list(vector) for vector in vectors]
        _require_normalised(vectors)
        matrix = np.asarray(vectors, dtype='float32')
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif self._index.d != matrix.shape[1]:
            raise ValueError(f'维度不一致：索引 {self._index.d}，新向量 {matrix.shape[1]}')
        self._index.add(matrix)
        for position, identifier in enumerate(ids):
            self.records.append({'id': identifier, 'document': documents[position],
                                 'metadata': dict(metadatas[position])})

    def search(self, vector, limit):
        import numpy as np

        index = self._load_index()
        if index.ntotal == 0 or limit <= 0:
            return []
        vector = list(vector)
        _require_normalised([vector])
        scores, positions = index.search(np.asarray([vector], dtype='float32'),
                                         min(limit, index.ntotal))
        hits = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            record = self.records[position]
            # Inner product of normalised vectors is cosine similarity; the
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
        (self.directory / INDEX_FILE).write_bytes(faiss.serialize_index(self._index).tobytes())
        self._write_records()
        (self.directory / META_FILE).write_text(
            json.dumps({'backend': 'faiss-flat-ip', 'dimensions': int(self._index.d),
                        'vectors': int(self._index.ntotal), 'metric': 'cosine',
                        'exact': True}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return self.directory
    @staticmethod
    def normalize(vectors):
        """Scale vectors to unit length; the caller keeps the original values."""
        import numpy as np

        matrix = np.asarray([list(vector) for vector in vectors], dtype='float32')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise NormalizationError('存在零向量，无法归一化')
        return (matrix / norms).tolist()
