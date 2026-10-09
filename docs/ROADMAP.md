# 目标结构与后续路线（对齐 WeKnora 学习路线）

日期：2026-10-09。本文把「接下来按什么结构走、每个位置放什么」写成可执行的清单，并严格区分**已实现 / 仅实验 / 仅占位 / 未实现**。状态标记沿用 ADR-007 的口径：没验证过的不写成能力。

参考来源见 [ARCHITECTURE_REFERENCES.md](ARCHITECTURE_REFERENCES.md)。WeKnora 的七步路线（文档解析与 Chunking → Embedding 与索引 → 混合检索与融合 → Rerank 与上下文 → 生成与引用 → Evaluation → 消融与工程优化）与本项目现有模块的对应关系在第三节。

---

## 一、目标目录结构

`rag/` 是产品运行时代码；`eval/` 是评测框架骨架；`experiments/` 保留已跑过的实验证据。三者不互相依赖：`eval/` 可以调用 `rag/`，`rag/` 永不导入 `eval/`。

```
RAG/
├─ rag/                              产品运行时（分层见下）
│  ├─ host/                          【未实现】应用装配与依赖注入
│  ├─ config.py                      [已实现] .env 读取
│  ├─ loader.py                      [已实现] .md / .txt / .pdf → Document
│  ├─ splitter.py                    [已实现] 固定 500/50 切片
│  ├─ structure.py                   [已实现] Markdown 标题结构与片段定位
│  ├─ registry.py                    [已实现] 知识库注册表
│  ├─ importer.py                    [已实现] 文档身份 / hash / 位置 / skip 反馈
│  ├─ kb_service.py                  [已实现] 导入→建索引→检索→引用→生成
│  ├─ retrieve/                      【部分实现】检索层，混合与重排待落地
│  │  ├─ dense.py                    [已实现] 纯向量 Top-K（现位于 kb_service）
│  │  ├─ keyword.py                  【未实现】BM25 关键词检索
│  │  ├─ fusion.py                   【未实现】RRF 排名融合
│  │  └─ rerank.py                   【未实现】精排 → 阈值 → MMR 去重
│  ├─ generate/                      【部分实现】生成与引用
│  │  ├─ prompt.py                   [已实现] 现有提示词（在 kb_service 内）
│  │  └─ citation.py                 [已实现] 引用构造（在 kb_service 内）
│  ├─ api.py                         [已实现] HTTP 接口
│  ├─ static/index.html              [已实现] 页面
│  ├─ knowledge.py                   [待迁移] 双夹具专用路径，仍读 experiments/
│  ├─ knowledge_routing.py           [待迁移] 会计条号路由（领域专用，需显式启用）
│  ├─ context_structure.py           [仅实验] ADR-006 精简提示视图
│  └─ cli.py / store.py / retriever.py / embedder.py / generator.py
│                                    [兼容] 早期单库路径，服务 /index 与 /query
├─ eval/                             评测框架（本次新建，仅占位）
│  ├─ datasets.py                    【占位】GroundTruth / Passage / 数据集加载
│  ├─ metrics.py                     【占位】指标注册表与指标集合
│  └─ runner.py                       【占位】任务调度、逐题产物、聚合与落盘
├─ experiments/rag-baseline/         已有实验证据（不改写历史）
│  ├─ evaluation/*.py                 [已实现] baseline/topk/context/rerank/section/compact
│  ├─ evaluation/*_REPORT.md          [已实现] 各组实验结论与失败记录
│  └─ evaluation/acceptance_third_corpus.py  [已实现] 通用接入验收
├─ tests/                            共 102 项：89 项一次跑完（含 store 2 项），11 项需原生 Chroma 本机不可用
├─ docs/                             设计、交接、验收、环境
└─ test_docs/third-corpus/           第三语料夹具
```

**为什么把 `retrieve/` 与 `generate/` 拆成包**：它们现在是 `kb_service.py` 里的方法。混合检索、RRF、重排、MMR 四个能力都会落在这里，全塞进 `kb_service` 会让它继续膨胀（已 394 行）。拆分时保持 `kb_service` 只做编排，不搬业务逻辑进 `api.py`。

**为什么新增 `host/`**：目前 `api.py` 顶部直接构造 `KnowledgeService` 与 `KnowledgeBaseService` 两个全局对象，测试只能靠 monkeypatch 替换。抽出一个装配点后，接口与测试都指向同一个入口。

---

## 二、模块 ↔ 位置 ↔ 状态

| # | WeKnora 路线 | 本项目位置 | 状态 | 离"可信"还差什么 |
| --- | --- | --- | --- | --- |
| 1 | 文档解析与 Chunking | [rag/loader.py](../rag/loader.py)、[rag/importer.py](../rag/importer.py)、[rag/splitter.py](../rag/splitter.py)、[rag/structure.py](../rag/structure.py) | 已实现 | 解析成功率/表格还原无专项测试；父子分块未实现 |
| 2 | Embedding 与索引 | [rag/kb_service.py](../rag/kb_service.py)（`vectors` / `build` / `ChromaVectors`） | 已实现，**本机未验证** | 真实 Chroma 路径待 Docker 复跑；模型横向比较未做 |
| 3 | 混合检索与融合 | 实验：`experiments/.../hybrid_experiment.py`、`topk_experiment.py` | 仅实验 | BM25、RRF 未进产品路径；无生产配置 |
| 4 | Rerank 与上下文 | 实验：`rerank_experiment.py`、`context_budget.py`、`context_structure.py` | 仅实验 | 阈值、MMR 去重、父子扩展均未实现 |
| 5 | 生成与引用 | [rag/kb_service.py](../rag/kb_service.py)（`generate` / `citations`） | 已实现 | 只有固定提示词；Faithfulness / 引用准确率未测 |
| 6 | Evaluation | `eval/`（本次占位）；现有 `experiments/**` 脚本与报告 | 占位 + 脚本级 | 无统一数据集 schema、无指标注册表、无任务调度与持久化 |
| 7 | 消融与工程优化 | `experiments/**/*_REPORT.md`（5 份对照报告） | 已实验 | 未固化为可一键复跑的矩阵；延迟/成本未系统记录 |

---

## 三、Evaluation 包的实施顺序（本次只落地占位）

占位文件里已写明每步的验收方式，实现时不要跳步：

1. **数据集层** `eval/datasets.py`：定 JSONL schema（`query_id` / `question` / `relevant_passage_ids` / `answer` / `split`），加重复 ID 与未知字段的校验测试；把现有 dev128 / test97 读成 `GroundTruth`。
2. **ID 对齐**：评测答案用数据集 passage id，检索结果是 chunk，两个 id 空间不同。正确做法是从导入阶段就保留 `source_passage_id` 元数据并一路传到命中结果；文本匹配只作兜底并报告匹配失败，不静默算作未命中。
3. **指标层** `eval/metrics.py`：先实现检索侧 `recall@k`、`mrr`、`ndcg@k`，每个指标配手算样例测试；再实现生成侧，生成侧必须走人工标注或有评分规则约束的 LLM Judge，并抽样核验 Judge 本身。
4. **执行层** `eval/runner.py`：先只跑检索（不调用生成），再为生成加显式开关；每题保存召回列表、最终上下文、原始回答与用量，最后落盘配置快照与源码 hash。
5. **消融矩阵**：在固定语料、Embedding、模型、生成参数下，只改一个变量，至少覆盖 Dense-only → Hybrid → Hybrid+Rerank 三组，每组报告质量与延迟，报告写进 `experiments/`。

**纪律**（与 AGENTS.md 一致）：97 题保留测试在开发集冻结前不运行；两个领域分别报告指标，不合并成单一分数；证据覆盖率不等于答案准确率。

---

## 四、清理记录（2026-10-09）

本轮实际删除了本地产物 `.tmp/`（304 MB）、`.uv-cache/`（324 MB）、`.pytest_cache/`、`.dsh-acl-recovery/`，都在 `.gitignore` 内，不影响仓库内容。残留：沙箱创建的 `.tmp/pytest-of-AD钙` 等 6 个空目录与 `.uv-cache` 剩余 96.7 MB，当前进程无删除权限，需要时用管理员权限手工删。

**代码层面没有可删的死代码**：`context_structure.py` 被 `compact_experiment.py` 与测试引用，`enumerate_sources` 被 `test_importer.py` 引用，`cli.py`/`store.py`/`retriever.py`/`embedder.py`/`generator.py` 仍服务 `/index`、`/query` 与 `test_cli.py`。因此本轮不删文件，改为在本文标记各自状态。

**已知重复（待合并，不在本轮）**：`kb_service.vectors()/generate()` 与 `embedder.py`/`generator.py` 是同类功能的两份实现，前者服务知识库隔离，后者服务旧单库路径；合并时以 `host/` 装配点为准。
