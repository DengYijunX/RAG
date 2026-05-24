"""检索编排模块 —— 组合向量化 + 向量检索，一步返回相关 chunk 列表。"""

from typing import List, Optional
from rag.config import Config
from rag.splitter import Chunk
from rag.embedder import embed_single
from rag.store import search


def retrieve(query: str, config: Config, top_k: Optional[int] = None) -> List[Chunk]:
    """将查询文本向量化后从 ChromaDB 检索 top_k 个相似 chunk。"""
    query_vector = embed_single(query, config)
    k = top_k if top_k is not None else config.top_k
    return search(query_vector, k, config)
