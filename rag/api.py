"""FastAPI HTTP 服务 —— 提供 RESTful 接口，托管前端静态页面。

端点：
    GET  /         → 前端页面
    POST /index    → 索引文档目录
    POST /query    → 检索问答
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from rag.config import load_config
from rag.knowledge import KnowledgeService, IndexUnavailable

app = FastAPI(title="RAG API", description="文档索引与检索问答")

config = load_config()
STATIC_DIR = Path(__file__).parent / "static"
knowledge = KnowledgeService(config)


class KnowledgeQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def meaningful_question(cls, value):
        if not value.strip():
            raise ValueError('请输入问题')
        return value.strip()


@app.get('/knowledge-bases')
def knowledge_bases():
    return knowledge.catalog()


@app.get('/knowledge-bases/{kb}/sources/{source_id}')
def knowledge_source(kb: str, source_id: str):
    try:
        return knowledge.source(kb, source_id)
    except KeyError:
        raise HTTPException(404, '未找到该知识库或来源') from None


@app.post('/knowledge-bases/{kb}/query')
def knowledge_query(kb: str, req: KnowledgeQuery):
    try:
        knowledge.corpus(kb)
    except KeyError:
        raise HTTPException(404, '未知知识库') from None
    try:
        return knowledge.query(kb, req.question)
    except IndexUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        # Provider exceptions may contain request credentials; never echo them.
        raise HTTPException(502, '问答服务暂时不可用，请稍后重试。') from None


class IndexRequest(BaseModel):
    path: str
    glob: str = "*"


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/index")
async def index_documents(req: IndexRequest):
    """索引指定目录下的文档到 ChromaDB，每次全量重建。"""
    from rag.loader import load_from_dir
    from rag.splitter import split
    from rag.embedder import embed
    from rag.store import reset_collection, add

    docs = load_from_dir(req.path, req.glob)
    if not docs:
        return JSONResponse(
            {"status": "empty", "documents": 0, "chunks": 0}
        )

    chunks = split(docs, config.chunk_size, config.chunk_overlap)

    batch_size = 20
    all_vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed([c.content for c in batch], config)
        all_vectors.extend(vectors)

    reset_collection(config)
    add(chunks, all_vectors, config)

    return {"status": "ok", "documents": len(docs), "chunks": len(chunks)}


@app.post("/query")
async def query_documents(req: QueryRequest):
    """检索相关文档片段并调用 LLM 生成回答，返回答案、来源和匹配片段。"""
    from rag.retriever import retrieve
    from rag.generator import generate

    top_k = req.top_k or config.top_k
    chunks = retrieve(req.question, config, top_k=top_k)

    answer = generate(req.question, chunks, config)

    sources = []
    seen = set()
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in seen:
            sources.append(src)
            seen.add(src)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": [
            {"content": c.content, "source": c.metadata.get("source", "")}
            for c in chunks
        ],
    }
