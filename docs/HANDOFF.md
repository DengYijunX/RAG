# 新会话交接（2026-10-09）

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
