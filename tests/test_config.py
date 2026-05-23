from rag.config import load_config, Config


def test_load_config_defaults():
    """测试默认值"""
    config = load_config()
    assert isinstance(config, Config)
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.llm_model == "deepseek-chat"
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.top_k == 4
    assert config.persist_dir == "./chroma_db"


def test_load_config_from_env(monkeypatch):
    """测试从环境变量读取"""
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    config = load_config()
    assert config.chunk_size == 800
    assert config.top_k == 6
    assert config.embedding_api_key == "test-key"
