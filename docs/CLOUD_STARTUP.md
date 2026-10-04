# RAG 云环境启动

使用 `/workspace/RAG` 现有检出，不创建额外 worktree。遵守 `AGENTS.md`：按可验证的小步本地提交，默认不 push。环境维护保留用户改动；功能开发依照用户授权进行。

## 环境与模型

- Python 3.12，解释器 `/workspace/RAG/.venv/bin/python`；包管理使用 `uv`，缓存 `/workspace/.cache/uv`，不用只读用户主目录。
- 安装：从仓库根执行 `UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python .venv/bin/python -r requirements.txt pytest==9.1.1`。缺失虚拟环境时先执行 `uv venv --python python3.12 .venv`。
- 校验：`UV_CACHE_DIR=/workspace/.cache/uv uv pip check --python .venv/bin/python`；`.venv/bin/python -m pytest tests -q`。
- Embedding URL：`https://llm-k5ve8qaaz1tn5w7b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`，模型 `qwen3.7-text-embedding-flash`，1,024 维。配置加载器仅将此已授权裸主机名补全为完整 URL。
- LLM 默认 URL `https://api.deepseek.com/v1`，模型 `deepseek-chat`。`EMBEDDING_API_KEY`、`LLM_API_KEY` 使用现有环境绑定，不打印、复制或保存密钥值。

## 恢复索引和服务

所有命令从 `/workspace/RAG` 执行。

1. `.venv/bin/python -m rag.knowledge status` 检查两库。已有匹配索引时复用；不要每次启动都重建。
2. 首次或模型、语料变更后，执行 `.venv/bin/python -m rag.knowledge index --kb all`；也可指定 `smart-home` / `accounting`。此命令产生真实 Embedding 调用。成功后两库分别为 5 文档 / 26 片段、59 文档 / 749 片段。
3. 在持久命令会话启动 `.venv/bin/python -m uvicorn rag.api:app --host 127.0.0.1 --port 8080`。进程不会随快照保留。端口占用时检查已有服务，不停止未知进程。
4. 内部请求 `GET /knowledge-bases` 确认两库 `ready=true`，`GET /` 返回包含知识库选择器的页面。
5. 功能验证：`POST /knowledge-bases/smart-home/query`，正文 `{"question":"门锁滴滴响是不是没电了"}` 应返回 `status=clarify`，无引用且不调用模型。`POST /knowledge-bases/accounting/query`，正文 `{"question":"2025年收入准则第四条"}` 应返回 `insufficient_evidence`。
6. 需要验证模型链路时，发送 `{"question":"分别依据2006年与2017年收入准则第四条，收入确认的核心表述有何差别？"}` 到会计问答接口。应返回非空回答，前两个引用分别对应 2006 / 2017 第四条；打开每个引用的 `url` 检查原文。这一步调用 Embedding 和 LLM。

不提供 localhost 的云预览链接；本机请求用于内部验证。Chroma 无需独立数据库服务，既有遥测签名和 Pydantic 弃用警告不代表功能失败。

默认索引位于 `chroma_db/knowledge-bases`；可通过 `PERSIST_DIR` 变更父目录。两库及每次索引版本独立，成功后原子切换 `active.json`。不改写旧版本，也不自动删除。旧 CLI、自定义 `/index` 和 `/query` 仍走原接口，新知识库服务不使用其进程级 store 单例。

## 语料、实验与边界

固定语料来自仓库 `experiments/rag-baseline/smart-home/corpus`、`accounting/markdown` 及原文结构元数据；不索引题目、答案、报告或审阅文件。

优先阅读仓库内 `evaluation/BASELINE_REPORT.md`、`STRUCTURE_REPORT.md`、`HYBRID_REPORT.md`。旧外部目录 `/workspace/rag-project-prep` 保留历史缓存；应用不依赖它。离线检查：`.venv/bin/python -m unittest discover -s experiments/rag-baseline/evaluation -p 'test_*.py' -v`。已有 23 项评测工具检查通过。不要自动重跑昂贵实验或保留测试集。

开发集覆盖率不是答案准确率；服务增加了澄清规则和引用提示词，未做独立测试集的端到端质量验收。225 题候选集仍待人工审阅，97 题保留测试集未使用。会计文件年份和现行效力不可混为一谈。

## 可选浏览器验证

本环境已安装 Playwright 1.63.0，浏览器位于 `/workspace/.cache/ms-playwright`。需要重装时使用 `UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python .venv/bin/python playwright==1.63.0` 及 `PLAYWRIGHT_BROWSERS_PATH=/workspace/.cache/ms-playwright .venv/bin/python -m playwright install chromium`。浏览器验证时设置同一 `PLAYWRIGHT_BROWSERS_PATH`；基础应用运行不依赖 Playwright。
