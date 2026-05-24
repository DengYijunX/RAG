from unittest.mock import patch, MagicMock


def get_test_config():
    from rag.config import Config
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


@patch("rag.generator.OpenAI")
def test_generate(mock_openai):
    from rag.generator import generate
    from rag.splitter import Chunk
    import rag.generator as mod

    mod._client = None

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "根据文档，答案是42。"
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )
    mock_openai.return_value = mock_client

    config = get_test_config()
    chunks = [
        Chunk(content="相关上下文", metadata={"source": "doc1.txt"}),
    ]
    answer = generate("问题是什么？", chunks, config)

    assert "42" in answer
    mock_client.chat.completions.create.assert_called_once()
