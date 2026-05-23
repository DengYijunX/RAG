from rag.loader import Document
from rag.splitter import split, Chunk


def test_split_documents():
    docs = [
        Document(
            content="这是第一段内容。\n\n这是第二段内容。\n\n这是第三段，它比较长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长。",
            metadata={"source": "test.md"}
        )
    ]
    chunks = split(docs, chunk_size=30, chunk_overlap=5)

    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.content
        assert c.metadata["source"] == "test.md"
        assert "chunk_index" in c.metadata


def test_chunk_metadata_preserved():
    docs = [
        Document(
            content="A" * 100 + "\n\n" + "B" * 100,
            metadata={"source": "doc1.md", "page": 1}
        )
    ]
    chunks = split(docs, chunk_size=30, chunk_overlap=5)

    for c in chunks:
        assert c.metadata["source"] == "doc1.md"
        assert c.metadata["page"] == 1
