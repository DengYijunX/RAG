from typing import List
from rag.splitter import Chunk
from rag.embedder import embed_single
from rag.store import search


def retrieve(query: str, config) -> List[Chunk]:
    query_vector = embed_single(query, config)
    return search(query_vector, config.top_k, config)
