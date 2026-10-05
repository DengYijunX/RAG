# RAG —— 文档检索增强生成

基于本地知识库的智能检索问答系统，支持 CLI 命令行和 HTTP 服务两种使用方式。

项目目标是**通用 RAG**：新领域文档应通过导入流程接入，无需修改代码。以下售后与会计知识库是当前的评测和演示语料；当前原型仍有固定语料依赖，正按 [项目范围与实施计划](docs/PROJECT_SCOPE.md) 统一为通用知识库流程。

设计参考及与当前代码的差距见 [成熟项目对照](docs/ARCHITECTURE_REFERENCES.md)：LlamaIndex 的摄取流程、Haystack 的检索组件及 RAGFlow 的知识库 / 文档管理。运行时保持轻量依赖，逐项实现和验证所采用的能力。

从整体流程、输入输出与评测入手，阅读 [RAG 流程与评测路线](docs/RAG_FLOW_AND_EVALUATION.md)，其中包含用户指定的 Tencent/WeKnora 源码参考与逐步实践安排。

## 双知识库演示

页面现在可分别查询**智能家居售后**（5 份文档）与**企业会计准则**（59 份文档），展示编号引用、文件年份、涉及条号，并打开完整原文。两库使用独立索引；原来的自定义文档入口仍可使用。

会计问题明确给出准则、年份和单一条号时，优先完整条文；其他问题使用原 500/50、L2 Top-4 向量检索。未知型号、未收录年份及部分不明确的问题会返回澄清或资料不足。开发集候选检索实验见 [混合检索报告](experiments/rag-baseline/evaluation/HYBRID_REPORT.md)；96.77% 是会计开发集证据覆盖率，不是本页面的答案准确率。

从仓库根目录运行（需在环境或 `.env` 中设置 Embedding / LLM URL、模型名及密钥）：

```bash
# 安装现有依赖；本云环境已有 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# 首次或变更语料/Embedding 模型后建立索引，会调用 Embedding API
.venv/bin/python -m rag.knowledge index --kb all
# 检查索引状态；不调用外部模型
.venv/bin/python -m rag.knowledge status
# 启动 HTTP 服务
.venv/bin/python -m uvicorn rag.api:app --host 127.0.0.1 --port 8080
```

支持 `--kb smart-home` 或 `--kb accounting` 单库重建。默认索引保存在 `chroma_db/knowledge-bases/<知识库>/<版本>/`，不提交 Git；新索引全部成功后才切换，失败时旧索引仍可用。更换模型或语料时旧索引会被拒绝使用，需重建。完整启动与验证说明见 [云环境启动文档](docs/CLOUD_STARTUP.md)。

新增接口：`GET /knowledge-bases`、`POST /knowledge-bases/{kb}/query`（JSON 字段 `question`）和 `GET /knowledge-bases/{kb}/sources/{source_id}`。新问答接口返回 `status`、`answer`、`citations`，每个引用包含正文、标题、年份、涉及条号和原文 URL。`answer` 状态表示已生成回答，不表示所有结论已经独立核验。

本次验证：30 项应用测试通过，真实 Embedding 索引 26 / 749 个片段，两库 HTTP 问答与原文访问通过；浏览器切库、原文弹窗、移动端布局及文本安全展示通过。见 [验证记录](docs/KNOWLEDGE_VALIDATION.md)。保留测试集未运行；当前生成提示词增加了引用及证据不足约束，端到端质量需要后续单独评测。

## 功能

- **多格式文档索引**：支持 `.md`、`.txt`、`.pdf`
- **向量化存储**：可配置 OpenAI 兼容 Embedding + ChromaDB 本地持久化；当前云环境使用阿里云 qwen3.7-text-embedding-flash
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
