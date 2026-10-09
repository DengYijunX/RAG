# 第三语料接入验收记录

日期：2026-10-09（Asia/Shanghai）。分支 `work`。本文只记录本次实际跑出的结果，未执行的部分单独标注。

## 验收目标（来自 `docs/PROJECT_SCOPE.md`）

使用者不修改 Python 或前端源码，即可创建第三个知识库、导入文档、建立索引、问答并查看原文；不同知识库与索引版本之间不误用数据；失败导入能清楚反馈。

## 语料

`test_docs/third-corpus/`，主题为**阳台蔬菜种植**，与既有两个领域（智能家居售后、企业会计准则）无重叠。6 篇 Markdown 为本项目自撰，不含第三方版权材料，一级标题依次为：阳台蔬菜种植手册、浇水与湿度、施肥、病虫害、采收、季节管理。

语料共 1019 字（每篇 151–188 字），因此每篇产生 1 个片段、合计 6 个。**这个规模只能证明接线正确，不能用于任何检索质量结论。**

## 执行方式

```powershell
$env:PYTHONPATH="C:\Users\AD钙\Desktop\Projects\RAG\tools\testenv"
.venv\Scripts\python.exe experiments\rag-baseline\evaluation\acceptance_third_corpus.py
```

脚本通过 FastAPI `TestClient` 走 HTTP 接口完成全流程，等价于页面上的操作序列：`POST /knowledge-bases` → `POST /knowledge-bases/{kb}/sources`（上传 6 个文件）→ `POST /knowledge-bases/{kb}/index` → `POST /knowledge-bases/{kb}/query` → `GET /knowledge-bases/{kb}/sources/{id}`。

**不修改任何业务源码文件**；它在运行时注入替身：替换 `vector_backend`（内存余弦存储）、`vectors`（确定性伪向量）与 `generate`（固定回答），并把 `api.knowledge` 换成只报告两个夹具的桩，以避免加载本机无法运行的原生向量索引。

## 实际结果（2026-10-09 运行，退出码 0）

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 创建知识库（仅走 HTTP） | 通过 | 返回 201，`id` 为随机短 ID，名称「阳台蔬菜种植手册」 |
| 导入第三语料 | 通过 | `documents=6, chunks=6, skipped=[]` |
| 建立索引 | 通过 | `chunks=6, dimensions=3`；`generation` 为 `uuid4().hex`，每次运行都不同 |
| 出现在知识库列表且可查询 | 通过 | `ready=true, chunk_count=6` |
| 两个既有夹具仍可列出 | 通过 | 列表含 `smart-home`、`accounting` 与新库（由桩提供，未加载真实夹具） |
| 问答返回片段引用 | 通过 | `status=answer`，引用带 `source_id` 与原文 URL |
| 查看引用对应原文 | 通过 | 原文内容非空，URL 形如 `/knowledge-bases/{kb}/sources/doc-…` |
| 每个片段都能在原文中复原 | 通过 | 6/6 个片段带位置，逐条满足 `原文[start:end] == content` |
| 原文携带内容 hash | 通过 | `documents[].content_sha256` 为 64 位十六进制（片段不带 hash 字段） |
| 导入未静默丢弃文件 | 通过 | `skipped == []`，且文档数等于上传文件数 |

合计 **10/10 项通过**。

## 自动测试

```powershell
$env:PYTHONPATH="C:\Users\AD钙\Desktop\Projects\RAG\tools\testenv"
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=".tmp\all" `
  --ignore=tests/test_knowledge.py --ignore=tests/test_store.py
```

结果：**83 项通过**（registry 11、importer 14、kb_service 14、api 12，其余为既有模块）。`tests/test_store.py` 单独运行为 2 项通过。

共收集 96 项；本机**无法运行** `tests/test_knowledge.py` 的 11 项（需要构建 749 片段的 Chroma 索引），另有 `test_store.py` 的 2 项需单独进程运行。

## 本次验收**没有**验证的内容

必须与上面的结论分开阅读，避免把流程检查当成质量结论：

- **没有使用真实 Embedding 与 Chroma 索引。** 本机 ChromaDB 依赖的原生扩展 hnswlib 在大批量写入时崩溃（`0xC0000005`），详见 `docs/ENVIRONMENT_WINDOWS.md`。验收脚本使用确定性的内存余弦存储与替身生成器，因此它证明的是**导入与接口契约**，不是真实检索效果。
- **没有验证答案质量。** 替身生成器直接返回固定文本，本次不产生任何准确率、覆盖率或 Token 数字。
- **没有用真实 PDF 走完整验收流程**；PDF 行为只有单元测试覆盖（页码来自解析器，字符位置一律不报告）。
- **没有验证两套既有语料走同一导入流程的回归**：`rag/knowledge.py` 仍按原有实验目录方式服务这两个夹具，通用列表在缺少该目录时只列出注册表内容（见 `docs/ARCHITECTURE_REFERENCES.md` 的 ADR-007）。

## 在能运行原生向量索引的环境复跑

```bash
# Docker 或 Linux 主机，仓库根目录
python -m venv .venv && .venv/bin/pip install -r requirements.txt
RAG_ACCEPTANCE_REAL=1 .venv/bin/python experiments/rag-baseline/evaluation/acceptance_third_corpus.py
.venv/bin/python -m pytest tests -q
```

设置 `RAG_ACCEPTANCE_REAL=1` 后脚本保留真实 Embedding 提供方与 Chroma，此时会真实调用模型并写入向量库，需要已配置的凭据与网络。
