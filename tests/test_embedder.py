from unittest.mock import patch, MagicMock
from rag.config import Config


def get_test_config():
    return Config(
        embedding_base_url="https://api.siliconflow.cn/v1",
        embedding_api_key="test-key",
        embedding_model="BAAI/bge-m3",
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="test-llm-key",
        llm_model="deepseek-chat",
        chunk_size=500,
        chunk_overlap=50,
        top_k=4,
        persist_dir="./chroma_db",
    )


@patch("rag.embedder.OpenAI")
def test_embed_returns_vectors(mock_openai):
    from rag.embedder import embed, embed_single
    import rag.embedder as mod

    # 清除内部缓存
    mod._client = None

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    config = get_test_config()
    vectors = embed(["hello", "world"], config)

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]

    v = embed_single("hello", config)
    assert v == [0.1, 0.2, 0.3]
