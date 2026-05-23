# RAG CLI 模块设计规格

**日期**：2026-05-23  
**状态**：设计完成，待评审

---

## 1. 概述

一个 Python CLI 工具，支持**文档索引**和**RAG 查询**两个核心操作。用户将本地文档目录索引到 ChromaDB，然后通过自然语言检索并生成回答。

---

## 2. 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | Python 3.10+ | 生态最成熟 |
| 向量数据库 | ChromaDB | 本地轻量、零配置 |
| Embedding | 硅基流动 BGE-M3 API | OpenAI 兼容接口 |
| LLM | DeepSeek v4 Flash | OpenAI 兼容接口 |
| 文档加载 | 原生实现 + PyPDF2（可选） | 避免引入重型框架 |
| 文本切片 | langchain-text-splitters | 仅用切片器，不引入整个 LangChain |
| 配置 | python-dotenv + .env | 全量走 env |
| CLI 框架 | argparse（标准库） | 零额外依赖 |

---

## 3. 目录结构

```
rag/
  __init__.py
  cli.py            # CLI 入口：rag index <path> / rag query "<问题>"
  config.py         # .env 配置读取
  loader.py         # 按文件后缀派发加载器
  splitter.py       # 文本切片
  embedder.py       # 硅基流动 BGE-M3 Embedding
  store.py          # ChromaDB collection 管理
  retriever.py      # Embedding → 检索 → top-k
  generator.py      # DeepSeek 生成回答
requirements.txt
.env.example
```

每个文件不超过 120 行，单一职责，每个模块只暴露 1-2 个公开函数。

---

## 4. 数据流

### 索引流程 (`rag index <path>`)

```
CLI → loader.load_from_dir(path) → List[Document]
    → splitter.split(documents) → List[Chunk]
    → embedder.embed_batch(chunks) → List[Vector]
    → store.add(chunks, vectors) → 写入 ChromaDB
    → 输出统计信息
```

### 查询流程 (`rag query "<question>"`)

```
CLI → embedder.embed(query) → Vector
    → store.search(vector, top_k) → List[Chunk]
    → generator.generate(query, chunks) → str
    → 输出回答 + 引用来源
```

---

## 5. 模块接口

### config.py
```
load_config() → Config
# Config: openai_base_url, openai_api_key, embedding_model, 
#         llm_model, llm_base_url, llm_api_key, chunk_size,
#         chunk_overlap, top_k, persist_dir
```

### loader.py
```
load_from_dir(path: str) → List[Document]
# Document: content: str, metadata: dict (source, page, etc.)
# 支持: .md, .txt, .pdf（通过 PyPDF2）
```

### splitter.py
```
split(documents: List[Document], chunk_size: int, chunk_overlap: int) → List[Chunk]
# Chunk: content: str, metadata: dict
```

### embedder.py
```
embed(texts: List[str]) → List[List[float]]
embed_single(text: str) → List[float]
```

### store.py
```
get_collection() → ChromaDB Collection
add(chunks: List[Chunk], vectors: List[List[float]]) → None
search(vector: List[float], top_k: int) → List[Chunk]
```

### retriever.py
```
retrieve(query: str, top_k: int) → List[Chunk]
# 内部调用 embedder + store
```

### generator.py
```
generate(query: str, context_chunks: List[Chunk]) → str
# 拼装 prompt → 调 DeepSeek → 返回回答
```

### cli.py
```
main()
  subcommands:
    index <path> [--recursive] [--glob <pattern>]
    query "<question>" [--top-k <n>]
```

---

## 6. 配置项 (.env)

```
# Embedding（硅基流动）
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=BAAI/bge-m3

# LLM（DeepSeek）
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat

# 切片参数
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# 检索参数
TOP_K=4

# 本地存储
PERSIST_DIR=./chroma_db
```

---

## 7. 依赖清单

```
chromadb
langchain-text-splitters
python-dotenv
openai
PyPDF2
```

---

## 8. 非功能需求

- **幂等性**：同一文档目录多次索引，先清空旧数据再写入（避免重复）
- **错误处理**：不支持的文档格式跳过并警告，不中断流程
- **进度反馈**：索引过程中输出进度信息（已处理 N 个文件）
- **最小依赖**：除 langchain-text-splitters 和 PyPDF2 外，不引入大型框架

---

## 9. 不做的事情

- 不实现增量索引（每次全量重建）
- 不支持远程 ChromaDB 服务
- 不实现 Web UI
- 不实现对话历史管理（每次查询独立）
