"""向量存储模块 —— ChromaDB PersistentClient 封装，负责 collection 管理和读写。"""

from typing import List
import chromadb
from chromadb.config import Settings

from rag.splitter import Chunk

# ChromaDB 全局单例，collection 名固定为 rag_documents
_client = None
_collection = None
COLLECTION_NAME = "rag_documents"


def _get_collection(config):
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=config.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def reset_collection(config):
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=config.persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.create_collection(name=COLLECTION_NAME)
    return _collection


def add(chunks: List[Chunk], vectors: List[List[float]], config) -> None:
    col = _get_collection(config)
    col.add(
        ids=[f"{c.metadata['source']}_{c.metadata['chunk_index']}" for c in chunks],
        documents=[c.content for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )


def search(query_vector: List[float], top_k: int, config) -> List[Chunk]:
    col = _get_collection(config)
    results = col.query(query_embeddings=[query_vector], n_results=top_k)

    chunks = []
    docs = results.get("documents")
    metas = results.get("metadatas")
    if docs and docs[0]:
        for i, doc in enumerate(docs[0]):
            meta = (metas[0][i] if metas and metas[0] else {})
            chunks.append(Chunk(content=doc, metadata=meta))
    return chunks
