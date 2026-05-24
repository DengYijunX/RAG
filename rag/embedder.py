from typing import List
from openai import OpenAI
from rag.config import Config

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
