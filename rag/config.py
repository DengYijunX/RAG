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


def load_config() -> Config:
    """加载 .env 并返回 Config 实例，缺失字段使用默认值。"""
    load_dotenv()
    return Config(
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "4")),
        persist_dir=os.getenv("PERSIST_DIR", "./chroma_db"),
    )
