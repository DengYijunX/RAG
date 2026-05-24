import tempfile
import shutil
from rag.config import Config
from rag.splitter import Chunk


def get_test_config(persist_dir):
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
        persist_dir=persist_dir,
    )


def test_add_and_search():
    import rag.store as mod
    mod._client = None
    mod._collection = None

    tmpdir = tempfile.mkdtemp()
    try:
        config = get_test_config(tmpdir)
        mod.reset_collection(config)

        chunks = [
            Chunk(content="今天天气很好", metadata={"source": "doc1.txt", "chunk_index": 0}),
            Chunk(content="人工智能发展迅速", metadata={"source": "doc2.txt", "chunk_index": 0}),
        ]
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        mod.add(chunks, vectors, config)

        results = mod.search([0.15, 0.25, 0.35], 2, config)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, Chunk)
            assert r.content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_reset_collection_clears_data():
    import rag.store as mod
    mod._client = None
    mod._collection = None

    tmpdir = tempfile.mkdtemp()
    try:
        config = get_test_config(tmpdir)
        mod.reset_collection(config)
        mod.add(
            [Chunk(content="test", metadata={"source": "x", "chunk_index": 0})],
            [[0.1, 0.2, 0.3]],
            config,
        )
        # 重置后应该为空
        mod.reset_collection(config)
        results = mod.search([0.1, 0.2, 0.3], 1, config)
        assert len(results) == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
