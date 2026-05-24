from unittest.mock import patch
import sys


def test_cli_requires_subcommand():
    sys.argv = ["rag"]
    try:
        from rag.cli import main
        main()
        assert False, "should raise"
    except SystemExit:
        pass


def test_cli_index(capsys):
    sys.argv = ["rag", "index", "/fake/path", "--glob", "*.md"]

    with patch("rag.loader.load_from_dir") as mock_load, \
         patch("rag.splitter.split") as mock_split, \
         patch("rag.embedder.embed") as mock_embed, \
         patch("rag.store.reset_collection") as mock_reset, \
         patch("rag.store.add") as mock_add:
        from rag.loader import Document
        from rag.splitter import Chunk

        mock_load.return_value = [
            Document(content="test doc", metadata={"source": "test.md"})
        ]
        mock_split.return_value = [
            Chunk(content="test chunk", metadata={"source": "test.md", "chunk_index": 0})
        ]
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        from rag.cli import main
        main()

    captured = capsys.readouterr()
    assert "索引完成" in captured.out


def test_cli_index_no_docs(capsys):
    sys.argv = ["rag", "index", "/empty/path"]

    with patch("rag.loader.load_from_dir") as mock_load:
        mock_load.return_value = []

        from rag.cli import main
        main()

    captured = capsys.readouterr()
    assert "未找到支持的文档文件" in captured.out


def test_cli_query(capsys):
    sys.argv = ["rag", "query", "什么是RAG？", "--top-k", "5"]

    with patch("rag.retriever.retrieve") as mock_retrieve, \
         patch("rag.generator.generate") as mock_generate:
        from rag.splitter import Chunk

        mock_chunks = [
            Chunk(content="RAG是检索增强生成", metadata={"source": "doc1.md"}),
            Chunk(content="RAG结合了检索和生成", metadata={"source": "doc2.md"}),
        ]
        mock_retrieve.return_value = mock_chunks
        mock_generate.return_value = "RAG是检索增强生成（Retrieval-Augmented Generation）的缩写。"

        from rag.cli import main
        main()

    captured = capsys.readouterr()
    assert "检索增强生成" in captured.out
    assert "引用来源" in captured.out


def test_cli_query_default_top_k(capsys):
    sys.argv = ["rag", "query", "什么是RAG？"]

    with patch("rag.retriever.retrieve") as mock_retrieve, \
         patch("rag.generator.generate") as mock_generate:
        from rag.splitter import Chunk

        mock_retrieve.return_value = []
        mock_generate.return_value = "无法回答"

        from rag.cli import main
        main()

    # When --top-k is not specified, retrieve is called with top_k=None
    # retrieve() handles None by falling back to config.top_k
    call_args = mock_retrieve.call_args
    assert call_args is not None


def test_cli_no_command():
    sys.argv = ["rag"]

    try:
        from rag.cli import main
        main()
        assert False, "should raise SystemExit"
    except SystemExit:
        pass
