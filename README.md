# RAG —— 文档检索增强生成

基于本地知识库的智能检索问答系统，支持 CLI 命令行和 HTTP 服务两种使用方式。

## 功能

- **多格式文档索引**：支持 `.md`、`.txt`、`.pdf`
- **向量化存储**：硅基流动 BGE-M3 Embedding + ChromaDB 本地持久化
- **LLM 问答生成**：DeepSeek 结合检索上下文生成回答
- **CLI 工具**：终端直接使用
- **Web 界面**：浏览器可视化操作

## 项目结构

```
RAG/
  rag/
    __init__.py        # 模块入口
    cli.py             # CLI 命令行（index / query）
    config.py          # .env 配置读取
    loader.py          # 多格式文档加载器
    splitter.py        # 文本切片
    embedder.py        # 向量化（BGE-M3 API）
    store.py           # ChromaDB 存储封装
    retriever.py       # 检索编排
    generator.py       # LLM 回答生成（DeepSeek）
    api.py             # FastAPI HTTP 服务
    static/
      index.html       # Web 前端界面
  tests/               # 单元测试
  requirements.txt
  .env.example         # 配置模板
```

## 快速开始

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
# Embedding（硅基流动）
EMBEDDING_API_KEY=sk-你的密钥

# LLM（DeepSeek）
LLM_API_KEY=sk-你的密钥
```

### 4. 索引文档

```bash
# CLI 方式
python -m rag.cli index 文档/

# 或通过 API
curl -X POST http://127.0.0.1:8080/index \
  -H "Content-Type: application/json" \
  -d '{"path": "文档/"}'
```

### 5. 检索问答

```bash
# CLI 方式
python -m rag.cli query "X1智能门锁怎么重置？"

# 或通过 API
curl -X POST http://127.0.0.1:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "X1智能门锁怎么重置？"}'
```

### 6. 启动 Web 界面

```powershell
$env:PYTHONUNBUFFERED=1
.venv\Scripts\python -m uvicorn rag.api:app --host 127.0.0.1 --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。

## API 文档

启动服务后访问 `http://127.0.0.1:8080/docs` 查看 Swagger 文档。

### POST /index

索引文档目录。

```json
{ "path": "文档/", "glob": "*" }
```

### POST /query

检索并生成回答。

```json
{ "question": "X1门锁怎么设置？", "top_k": 4 }
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| EMBEDDING_BASE_URL | 嵌入模型 API 地址 | `https://api.siliconflow.cn/v1` |
| EMBEDDING_MODEL | 嵌入模型名称 | `BAAI/bge-m3` |
| LLM_BASE_URL | LLM API 地址 | `https://api.deepseek.com/v1` |
| LLM_MODEL | LLM 模型名称 | `deepseek-chat` |
| CHUNK_SIZE | 文本切片大小 | 500 |
| CHUNK_OVERLAP | 切片重叠量 | 50 |
| TOP_K | 检索返回数量 | 4 |
| PERSIST_DIR | 向量库存储目录 | `./chroma_db` |

## 运行测试

```bash
python -m pytest tests/ -v
```

## 技术栈

- Python 3.10+
- ChromaDB —— 向量存储
- langchain-text-splitters —— 文本切片
- OpenAI SDK —— 统一调用 LLM 和 Embedding
- FastAPI + Uvicorn —— HTTP 服务
- PyPDF2 —— PDF 解析
