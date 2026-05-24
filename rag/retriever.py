from typing import List, Optional
from rag.config import Config
from rag.splitter import Chunk
from rag.embedder import embed_single
from rag.store import search


def retrieve(query: str, config: Config, top_k: Optional[int] = None) -> List[Chunk]:
    query_vector = embed_single(query, config)
    k = top_k if top_k is not None else config.top_k
    return search(query_vector, k, config)
