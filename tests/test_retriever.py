from unittest.mock import patch


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
        top_k=3,
        persist_dir="./chroma_db",
    )


@patch("rag.retriever.embed_single")
@patch("rag.retriever.search")
def test_retrieve(mock_search, mock_embed):
    from rag.retriever import retrieve
    from rag.splitter import Chunk

    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_chunks = [
        Chunk(content="相关文档内容", metadata={"source": "test.txt"}),
        Chunk(content="另一个文档", metadata={"source": "test2.txt"}),
    ]
    mock_search.return_value = mock_chunks

    config = get_test_config()
    results = retrieve("测试查询", config)

    mock_embed.assert_called_once_with("测试查询", config)
    mock_search.assert_called_once_with([0.1, 0.2, 0.3], 3, config)
    assert results == mock_chunks
    assert len(results) == 2
