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
    context_radius: int = 0

    @property
    def rerank_configured(self):
        return bool(self.rerank_base_url and self.rerank_api_key and self.rerank_model)


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
    )
