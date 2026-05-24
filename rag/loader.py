"""文档加载模块 —— 支持 .md / .txt / .pdf，按文件后缀派发加载器。"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """加载后的原始文档，content 为全文，metadata 包含来源路径等信息。"""
    content: str
    metadata: dict


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_file(file_path: Path) -> Document:
    """根据文件后缀加载单个文件，PDF 走 PyPDF2，其余按 UTF-8 文本读取。"""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    else:
        return _load_text(file_path)


def load_from_dir(dir_path: str, glob_pattern: str = "*") -> List[Document]:
    """递归扫描目录，加载所有支持格式的文件。不支持的文件跳过并警告。"""
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    documents = []
    for file_path in sorted(path.rglob(glob_pattern)):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                doc = load_file(file_path)
                doc.metadata["source"] = str(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Warning: failed to load {file_path}: {e}")
    return documents


def _load_text(file_path: Path) -> Document:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(content=content, metadata={"source": str(file_path)})


def _load_pdf(file_path: Path) -> Document:
    from PyPDF2 import PdfReader
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    content = "\n".join(pages)
    return Document(content=content, metadata={"source": str(file_path)})
