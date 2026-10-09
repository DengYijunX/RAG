# 新会话交接（2026-10-09 晚，通用知识库注册与导入）

## 这一轮做了什么

在 `work` 分支上按 `docs/PROJECT_SCOPE.md` 的下一步顺序，完成「通用知识库注册 + 文档导入」的前四步，并做了第三语料接入验收。小步提交，未 push，未合并 main。

- `rag/registry.py`：持久化知识库注册表（ID/名称/描述/语言/可选扩展），原子写入，不存凭据。
- `rag/importer.py`：只读文件本身推导文档身份、内容 hash、原文字符区间与 Markdown 标题结构；PDF 不伪造位置；坏文件跳过不阻塞。
- `rag/kb_service.py`：注册→导入→建索引→问答→查看原文的通用生命周期，向量后端可注入；发布为成功后原子切换，失败保留旧索引；带 CLI（create/list/import/index/ask）。
- `rag/api.py`：`/knowledge-bases` 创建、`/{kb}/sources` 上传或目录导入、`/{kb}/index`、`/{kb}/query`、`/{kb}/sources/{id}` 查看原文。
- `rag/static/index.html`：下拉动态读取注册表，新增创建/导入/建索引操作，已无领域硬编码字符串。
- 测试：registry 11、importer 14、kb_service 14、api 12，本机共 83 项通过（`test_store` 单独 2 项通过；`test_knowledge` 的 11 项因原生向量索引不可用未运行）。
- 验收：第三语料 `test_docs/third-corpus/`（阳台蔬菜种植，自撰）10/10 通过，见 `docs/THIRD_CORPUS_ACCEPTANCE.md`。
- 设计出处：`docs/ARCHITECTURE_REFERENCES.md` 的 **ADR-007**（通用注册与导入）与 **ADR-008**（按 WeKnora 路线固定目标结构，评测框架先落占位）。
- 目标结构与后续路线：**`docs/ROADMAP.md`**（目录树、模块↔位置↔状态对应表、`eval/` 实施顺序、清理记录）。新会话先读这份。

## 本轮独立复核后修掉的问题

两位独立审查者（对抗式解耦审查 + 复现验证）查出并已修复：

1. `GET /knowledge-bases` 原先直接拼接 `knowledge.catalog()`，缺少 `experiments/rag-baseline` 时整个首屏请求 500；现在只依赖注册表，缺夹具仍正常返回（有测试）。
2. 导入会静默丢弃非 md/txt/pdf 文件；现在所有未导入文件都带原因进入 `skipped`，重复上传同名文件报 422，页面显示未导入清单。
3. PDF「报告页码」原先只是注释与测试名，没有字段；现在页码来自解析器实际页数，无页码时为 `null`。
4. `lines` 原先按换行数+1 计算，结尾换行会多算一行；改为物理行数。
5. 构建失败会残留半成品 generation 目录；现在失败即清理。
6. 检索原先分别读取 active/清单/向量库，存在窄竞态；现在每个 generation 自带片段快照，命中只在该 generation 内解析。
7. 未知知识库的 `GET .../sources` 原先返回 200 `[]`；现在 404。
8. 测试里若干近似恒真的断言（空循环、随机 ID 断言、未写入凭据却断言不落盘）已改为可失败的断言。

**本机两个硬限制**，动手前先读 `docs/ENVIRONMENT_WINDOWS.md`：`.venv` 基座已修好（CPython 3.11.15），跑测试必须注入 `tools/testenv/sitecustomize.py` 并加 `--basetemp`；ChromaDB 的原生 hnswlib 在本机崩溃（0xC0000005），本机**无法**跑真实索引与 `tests/test_knowledge.py`。

## 尚未完成（按优先级）

1. **在能跑原生向量索引的环境复跑真实验收**：Docker 或 Linux 主机上 `RAG_ACCEPTANCE_REAL=1` 跑 `acceptance_third_corpus.py`，并跑通 `tests/test_knowledge.py`（749 片段 Chroma 索引）。这是当前最大的未验证缺口。
2. **迁移既有两个夹具到同一导入流程**：`rag/knowledge.py` 仍读 `experiments/rag-baseline` 的 `evidence.jsonl` / `articles.jsonl`，会计条号路由是这条旧路径下的领域分支。迁移时把「条号/年份」变成显式启用的可选扩展，并保持 ADR-002~006 的实验可复现。
3. 领域扩展落地：基于导入元数据的精确定位（ADR-007 只预留了 `extensions` 字段）。
4. 检索增强接入通用路径：邻段扩展、重排、结构提示目前都只在实验脚本里，通用默认路径仍为纯向量 Top-K。
5. 文档禁用/删除、增量 upsert、Embedding 缓存均未实现。

## 不可遗忘的边界

- 通用流程不得依赖固定知识库 ID、产品型号、准则名称或评测答案；新语料必须走导入流程接入。
- 第三语料验收证明的是**导入与接口契约**，不是答案准确率；不要把 9/9 写成质量结论。
- 97 题保留测试仍未执行；不要为了更好结果重复调那九题。
- 默认本地提交，不 push；不要自动合并到 main。

---

# 上一轮交接（2026-10-09 早）

## 目标与协作约定

项目是可写入简历、可演示和可复现的**通用 RAG**。智能家居售后与企业会计准则只是两套评测夹具，不是固定产品边界。用户要求参考 Tencent/WeKnora 讲清流程、模块输入输出和评测；重要设计需记录具体参考出处与取舍。

阅读 `AGENTS.md`；小步实现、验证、commit。用户本轮明确要求 push，推送目标为 `origin/work`，远端为 `https://github.com/DengYijunX/RAG.git`。后续默认仍是本地提交，不自动 push。不要自动合并到 main。

## 当前状态

- 工作目录 `/workspace/RAG`，开发分支 `work`。本轮交接前最新功能/实验提交 `5b0fab7`；交接文件另作提交。
- FastAPI + Chroma + OpenAI 兼容 SDK 已有加载、切片、隔离索引、成功后原子切换、问答与引用原文界面。
- 现有两库：5份售后文档 / 26片段，59份会计文档 / 749片段。切片为500字符、50重叠。
- **尚未完成通用化**：`rag/knowledge.py` 仍依赖固定 `BASES`、实验目录、预生成元数据；页面与路由仍有领域硬编码。不要把已有实验工具当作应用默认能力。
- `rag/structure.py` 已实现原文字符位置、Markdown标题路径、续段及按已选上下文判断章节完整性。只支持保守Markdown子集；加粗行是显式启用的启发式，未知结构保持未知。
- `rag/context_structure.py` 已将完整审计记录与模型精简视图分离，使用去重目录、短节ID和分块数组；完整请求UTF-8字节预算超限时调用前失败。这不是精确Token预算，默认应用尚未接入。

## 最近完成的评测

顺序报告位于 `experiments/rag-baseline/evaluation/`：

1. `TOPK_REPORT.md`：128道dev检索对照，固定九题生成；覆盖提高不等于回答正确。
2. `CONTEXT_REPORT.md`：固定2000正文字符，邻段扩展会挤掉其他证据，两个语料表现不同。
3. `RERANK_REPORT.md`：候选重排原型与边界提示；九题必要证据覆盖改善，但仍有续段归属、截断和明显排序开销。
4. `SECTION_REPORT.md`：原文结构提示，九题18次生成；改善部分归属/截断说明，但输入Token为纯边界提示3.15倍。
5. **`COMPACT_REPORT.md`（最新）**：相同九题正文与顺序，完整/精简结构提示18次新生成。输入Token 43,807 → 20,907（减少52.3%），最大精简请求10,730字节，小于12,000字节预算。回答仍有边界误读、无依据的条件总数，不宣称质量无损或总体准确率提高。

最新运行：`evaluation/compact_runs/20261007T171547Z-9beed06a`（目录为UTC时间）；计划、请求、回答、用量、hash全部保存。`compact_case_review.jsonl` 是18条助手审阅，人工评分null；`compact_audit.json` 为技术审计。历史原始产物不覆盖。

需保留的失败：

- ACC-Q098：把“当前片段后面还有续段”误解为列表不全，虽然同节合并覆盖已完整。
- ACC-Q118：只见第五条前四项，两组均提醒截断，却仍猜条件总数为五。事实碰巧与全原文一致，不代表本次引用支持。
- ACC-Q059：完整结构组把“四类”排成五个并列项；精简组归属正确但概括粒度存在争议。
- Q043：前一轮漏了电量/强制重启，本轮两组均答全；单次生成波动不能归功于压缩。

225道候选题为dev128 / test97。**97题保留测试未执行**，标注仍待领域人工复核。证据覆盖不是答案准确率；不要自动重跑昂贵实验或为了更好结果重复调这九题。

## 新会话应继续的任务

转向**通用知识库注册与文档导入**，不要只继续九题提示词优化。

1. 检查 `rag/knowledge.py`、`rag/api.py`、`rag/static/index.html` 和现有测试，设计持久化知识库注册表、文档记录及导入接口。
2. 原文导入生成文档身份、内容hash、原文位置和片段结构，不读取 `evidence.jsonl`、`articles.jsonl` 或gold；先复用已有MD/TXT/PDF加载器，PDF无可靠结构时不伪造位置。
3. 按知识库隔离索引与来源；保持成功后发布、失败保留旧版本的行为。领域规则作为显式可选扩展。
4. 页面动态读取知识库。用第三种主题语料走创建→导入→索引→问答→查看原文，验收无需修改源码即可接入；两套现有语料走同一流程作回归。
5. 分步提交并记录设计来源；完整目标和验收条件见 `docs/PROJECT_SCOPE.md`、`docs/ARCHITECTURE_REFERENCES.md`。通用导入验证与答案质量验收分开。

## 恢复与检查

先读 `docs/CLOUD_STARTUP.md`，复用当前 `.venv` 和匹配索引。该文件中早期测试数量已过时：最近验证为45项应用测试、44项评测工具测试。

```bash
cd /workspace/RAG
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest experiments/rag-baseline/evaluation -q
.venv/bin/python -m rag.knowledge status
.venv/bin/python experiments/rag-baseline/evaluation/compact_experiment.py --validate-only
```

Embedding配置为阿里云已授权地址、`qwen3.7-text-embedding-flash`、1024维；生成模型为 `deepseek-chat`。凭据使用环境绑定，不打印、不提交、不复制到交接文本。新环境按启动文档恢复依赖，确认凭据是否已绑定；没有匹配索引时构建会产生真实Embedding调用。

先确认分支：新检出应使用远端 `work`，否则只读 `main` 会看不到本轮进展。同一工作区先检查 `git status`，不要覆盖未提交改动。
