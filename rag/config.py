"""配置模块 —— 从 .env 文件和环境变量读取所有配置项。"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """RAG 系统全局配置，字段与 .env.example 一一对应。"""
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    persist_dir: str
    # Rerank is a different endpoint shape from chat/embeddings, so it carries
    # its own provider settings. Empty values mean "no reranker configured".
    rerank_base_url: str = ''
    rerank_api_key: str = ''
    rerank_model: str = ''
    # Neighbour expansion radius for the text handed to generation. 0 keeps each
    # hit as its own chunk; 1 adds the immediate neighbours in the same source.
    # Off by default: it was not shown to improve answers, and it makes the
    # answer-quality judge unreliable by lengthening the cited snippets.
    context_radius: int = 0
    # Vector backend. 'faiss' is the default because it is the only one that
    # builds reliably here — Chroma's HNSW crashes this interpreter above about
    # 96 vectors at 1024 dimensions, so the product's own index could not be
    # built at all. Both rank identically (measured), so this is a deployment
    # choice, not a quality one; 'chroma' remains available.
    vector_backend: str = 'faiss'
    # Retrieve this deep, then rerank the first ``rerank_pool`` and keep top_k.
    # 0 means "no deeper than top_k". Measured effect of 50/20 on the diagnostic
    # sample: answer completeness 7/10 -> 9/10 (accounting), 6/9 -> 9/9
    # (smart-home), context length almost unchanged, about 0.8 s per question.
    retrieval_depth: int = 0
    rerank_pool: int = 0

    @property
    def rerank_configured(self):
        return bool(self.rerank_base_url and self.rerank_api_key and self.rerank_model)

    @property
    def deep_rerank_enabled(self):
        """Deep-pool reranking is on only when a pool and a reranker are set."""
        return self.rerank_pool > 0 and self.rerank_configured


def _normalize_provider_url(value):
    """Accept either a full URL or a bare Aliyun MaaS host name.

    ``docs/CLOUD_STARTUP.md`` documents the host name alone; the
    OpenAI-compatible path has to be appended. A full URL passes through.
    """
    if value and value.startswith('llm-') and '://' not in value:
        return f'https://{value}/compatible-mode/v1'
    return value


def normalize_rerank_url(value):
    """Return the rerank endpoint URL for whatever the user configured.

    Aliyun MaaS serves reranking **only on the native path**, not under
    ``compatible-mode`` (verified: every compatible-mode path answers 404). So:

    - a URL already containing ``/services/rerank`` is used as given;
    - a ``compatible-mode`` URL has that suffix stripped and the native path
      appended, because that is the URL the user naturally copies from the
      embedding settings and it would otherwise 404;
    - a bare host gets the native path appended.
    """
    if not value:
        return ''
    stripped = value.rstrip('/')
    if '/services/rerank' in stripped:
        return stripped
    host = stripped.split('/compatible-mode')[0]
    return f'{host}/api/v1/services/rerank/text-rerank/text-rerank'


def load_config() -> Config:
    """加载 .env 并返回 Config 实例，缺失字段使用默认值。"""
    load_dotenv()
    return Config(
        embedding_base_url=_normalize_provider_url(
            os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "4")),
        persist_dir=os.getenv("PERSIST_DIR", "./chroma_db"),
        rerank_base_url=normalize_rerank_url(os.getenv("RERANK_BASE_URL", "")),
        rerank_api_key=os.getenv("RERANK_API_KEY", ""),
        rerank_model=os.getenv("RERANK_MODEL", ""),
        context_radius=int(os.getenv("RAG_CONTEXT_RADIUS", "0")),
        vector_backend=os.getenv("RAG_VECTOR_BACKEND", "faiss"),
        retrieval_depth=int(os.getenv("RAG_RETRIEVAL_DEPTH", "0")),
        rerank_pool=int(os.getenv("RAG_RERANK_POOL", "0")),
    )
