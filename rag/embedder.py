"""向量化模块 —— 调用硅基流动 BGE-M3 API，OpenAI 兼容接口。"""

from typing import List
from openai import OpenAI
from rag.config import Config

# 全局单例 client，首次调用时懒加载
_client = None


def _get_client(config: Config):
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
        )
    return _client


def embed(texts: List[str], config: Config) -> List[List[float]]:
    client = _get_client(config)
    response = client.embeddings.create(
        model=config.embedding_model,
        input=texts,
    )
    return [d.embedding for d in response.data]


def embed_single(text: str, config: Config) -> List[float]:
    return embed([text], config)[0]
