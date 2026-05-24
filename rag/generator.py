from typing import List
from openai import OpenAI
from rag.splitter import Chunk


_client = None


def _get_client(config):
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
    return _client


def generate(query: str, chunks: List[Chunk], config) -> str:
    client = _get_client(config)

    context = "\n\n---\n\n".join(
        f"[来源: {c.metadata.get('source', 'unknown')}]\n{c.content}"
        for c in chunks
    )

    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你是一个基于文档回答问题的助手。请根据提供的文档内容回答问题。如果文档中没有相关信息，请如实告知。"
            },
            {
                "role": "user",
                "content": f"参考文档：\n\n{context}\n\n问题：{query}"
            },
        ],
    )
    return response.choices[0].message.content
