from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import Document


@dataclass
class Chunk:
    content: str
    metadata: dict


def split(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
