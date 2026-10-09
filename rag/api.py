"""FastAPI HTTP 服务 —— 通用知识库生命周期与问答。

通用流程（不依赖任何固定领域）：
    POST /knowledge-bases                     创建知识库
    GET  /knowledge-bases                     列出知识库及导入/索引状态
    POST /knowledge-bases/{kb}/sources        导入文件（上传或服务器目录）
    POST /knowledge-bases/{kb}/index          建立并按版本发布索引
    POST /knowledge-bases/{kb}/query          检索问答
    GET  /knowledge-bases/{kb}/sources/{id}   查看导入原文

保留原有的目录索引 / 问答端点作为兼容入口。
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from rag.config import load_config
from rag.kb_service import BuildFailed, IndexUnavailable, KnowledgeBaseService
from rag.knowledge import KnowledgeService
from rag.registry import KnowledgeBaseExists, KnowledgeBaseNotFound, RegistryError

app = FastAPI(title="RAG API", description="通用知识库：导入、索引与检索问答")

config = load_config()
STATIC_DIR = Path(__file__).parent / "static"
knowledge = KnowledgeService(config)
bases = KnowledgeBaseService(config)


class KnowledgeQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def meaningful_question(cls, value):
        if not value.strip():
            raise ValueError('请输入问题')
        return value.strip()


class CreateBase(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = ''
    language: str = 'zh'


class ImportRequest(BaseModel):
    path: str
    glob: str = '**/*'


def _unknown_base(exc):
    return HTTPException(404, '未找到该知识库或来源')


@app.get('/knowledge-bases')
def knowledge_bases():
    """Registered bases first, then the built-in evaluation fixtures."""
    return bases.bases() + knowledge.catalog()


@app.post('/knowledge-bases', status_code=201)
def create_base(req: CreateBase):
    try:
        base = bases.register(req.name, description=req.description, language=req.language)
    except KnowledgeBaseExists:
        raise HTTPException(409, '已存在同名知识库') from None
    except RegistryError as exc:
        raise HTTPException(422, str(exc)) from None
    return {'id': base.id, 'name': base.name, 'description': base.description,
            'language': base.language, 'documents': 0, 'chunks': 0,
            'imported': False, 'ready': False}


@app.get('/knowledge-bases/{kb}')
def knowledge_base(kb: str):
    for entry in bases.bases():
        if entry['id'] == kb:
            return entry
    raise HTTPException(404, '未知知识库')


@app.post('/knowledge-bases/{kb}/sources')
def import_sources(kb: str, files: list[UploadFile] | None = File(default=None),
                   path: str | None = Form(default=None), glob: str = Form(default='**/*')):
    """Import uploaded files, or every supported file under a server directory."""
    try:
        bases.base(kb)
    except KnowledgeBaseNotFound:
        raise _unknown_base(None) from None
    staging = None
    try:
        if files:
            staging = Path(tempfile.mkdtemp(prefix=f'import-{kb}-'))
            for upload in files:
                name = Path(upload.filename or '').name
                if not name or name.startswith('.'):
                    raise HTTPException(422, '文件名无效')
                target = staging / name
                with open(target, 'wb') as handle:
                    shutil.copyfileobj(upload.file, handle)
            directory = staging
        elif path:
            directory = Path(path)
            if not directory.is_dir():
                raise HTTPException(422, '目录不存在')
        else:
            raise HTTPException(422, '请上传文件或提供服务器目录')
        report = bases.import_sources(kb, directory, glob)
    except HTTPException:
        raise
    except NotADirectoryError:
        raise HTTPException(422, '目录不存在') from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return {'knowledge_base': kb, 'documents': len(report.documents), 'chunks': len(report.chunks),
            'skipped': report.skipped}


@app.post('/knowledge-bases/{kb}/index')
def build_index(kb: str):
    try:
        bases.base(kb)
    except KnowledgeBaseNotFound:
        raise _unknown_base(None) from None
    try:
        info = bases.build(kb)
    except IndexUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    except BuildFailed as exc:
        raise HTTPException(502, str(exc)) from None
    return {'knowledge_base': kb, 'generation': info['generation'], 'documents': info['documents'],
            'chunks': info['chunks'], 'dimensions': info.get('dimensions')}


@app.get('/knowledge-bases/{kb}/sources')
def knowledge_sources(kb: str):
    try:
        return bases.sources(kb)
    except KnowledgeBaseNotFound:
        raise _unknown_base(None) from None


@app.post('/knowledge-bases/{kb}/query')
def knowledge_query(kb: str, req: KnowledgeQuery):
    try:
        entry = next((row for row in bases.bases() if row['id'] == kb), None)
        if entry is not None:
            return bases.query(kb, req.question)
        knowledge.corpus(kb)
    except KnowledgeBaseNotFound:
        raise _unknown_base(None) from None
    except IndexUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except KeyError:
        raise _unknown_base(None) from None
    except HTTPException:
        raise
    except Exception:
        # Provider exceptions may contain request credentials; never echo them.
        raise HTTPException(502, '问答服务暂时不可用，请稍后重试。') from None
    try:
        return knowledge.query(kb, req.question)
    except IndexUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except KeyError:
        raise _unknown_base(None) from None
    except Exception:
        raise HTTPException(502, '问答服务暂时不可用，请稍后重试。') from None


@app.get('/knowledge-bases/{kb}/sources/{source_id}')
def knowledge_source(kb: str, source_id: str):
    try:
        bases.base(kb)
    except KnowledgeBaseNotFound:
        raise HTTPException(404, '未找到该知识库或来源') from None
    try:
        return bases.source(kb, source_id)
    except KeyError:
        try:
            return knowledge.source(kb, source_id)
        except KeyError:
            raise HTTPException(404, '未找到该知识库或来源') from None
    except IndexUnavailable as exc:
        raise HTTPException(409, str(exc)) from None


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
