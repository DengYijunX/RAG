import os
from pathlib import Path
from rag.loader import load_file, load_from_dir, Document


def test_load_markdown_file():
    doc = load_file(Path("tests/fixtures/sample.md"))
    assert isinstance(doc, Document)
    assert "测试文档" in doc.content
    assert "Markdown" in doc.content


def test_load_text_file():
    doc = load_file(Path("tests/fixtures/sample.txt"))
    assert isinstance(doc, Document)
    assert "纯文本测试文件" in doc.content
    assert "第二行" in doc.content


def test_load_from_dir():
    docs = load_from_dir("tests/fixtures")
    assert len(docs) >= 2
    sources = [d.metadata["source"] for d in docs]
    assert any("sample.md" in s for s in sources)
    assert any("sample.txt" in s for s in sources)


def test_unsupported_extension_skipped(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("unsupported")
    docs = load_from_dir(str(tmp_path))
    assert len(docs) == 0
