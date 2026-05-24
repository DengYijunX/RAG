"""文本切片模块 —— 使用 RecursiveCharacterTextSplitter 将文档拆分为固定大小的 chunk。"""

from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import Document


@dataclass
class Chunk:
    """切片后的文本块，携带继承自原始文档的元数据 + chunk_index。"""
    content: str
    metadata: dict


def split(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    """将 Document 列表切分为 Chunk 列表，分隔符兼容中英文。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 按优先级尝试分割：段落 → 换行 → 中文句号 → 英文句号 → 空格 → 字符
        separators=["\n\n", "\n", "。", ".", "？", "?", " ", ""],
    )
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                content=text,
                metadata={**doc.metadata, "chunk_index": i}
            ))
    return chunks
