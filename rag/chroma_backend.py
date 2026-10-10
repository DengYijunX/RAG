"""Chroma-backed vector storage.

Kept as an option rather than the default: Chroma's HNSW index crashes the
interpreter on this machine above roughly 96 vectors at 1024 dimensions, so the
product's own index could not be built here at all. See ``rag.backends`` for the
default and for the measured ranking parity between the two.

This is the only module in ``rag`` that imports ``chromadb``, and it imports it
inside the method so that a FAISS deployment never loads it.
"""
from pathlib import Path


class ChromaVectors:
    """Chroma collection bound to one generation directory."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def add(self, ids, vectors, documents, metadatas):
        collection = self._collection(create=True)
        collection.add(ids=list(ids), embeddings=[list(vector) for vector in vectors],
                       documents=list(documents), metadatas=[dict(m) for m in metadatas])

    def search(self, vector, limit):
        collection = self._collection(create=False)
        if collection.count() == 0:
            return []
        result = collection.query(query_embeddings=[list(vector)],
                                  n_results=min(limit, collection.count()),
                                  include=['distances'])
        output = []
        for position, identifier in enumerate(result['ids'][0]):
            metadata = (result['metadatas'][0][position]
                        if result.get('metadatas') else {}) or {}
            document = result['documents'][0][position] if result.get('documents') else ''
            distance = result['distances'][0][position] if result.get('distances') else None
            output.append({'id': identifier, 'content': document,
                           'source': metadata.get('source', ''), 'distance': distance})
        return output

    def count(self):
        return self._collection(create=False).count()

    def flush(self):
        """No-op: Chroma persists on every add, so there is nothing to write.

        Present so both backends satisfy the same interface; the build code calls
        it unconditionally.
        """
        return self.directory

    def _collection(self, create):
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(self.directory),
                                           settings=Settings(anonymized_telemetry=False))
        if create:
            return client.get_or_create_collection('documents', embedding_function=None,
                                                   metadata={'hnsw:space': 'cosine'})
        return client.get_collection('documents', embedding_function=None)
