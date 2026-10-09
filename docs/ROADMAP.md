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
├─ tests/                            共 102 项：91 项可运行，11 项需原生 Chroma（本机不可用）
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

## 三、Evaluation 包的实施进度

| 步骤 | 位置 | 状态 |
| --- | --- | --- |
| 1 数据集层 | [eval/datasets.py](../eval/datasets.py) | **已实现**：JSONL schema、重复 ID / 非法 JSON / 缺字段校验、`reference_excerpts` 提取 `article_id` 作为标签、v1/v2 显式选择 |
| 2 指标层（检索侧） | [eval/metrics.py](../eval/metrics.py) | **已实现**：`recall@k` / `precision@k` / `hit_rate@k` / `mrr` / `map` / `ndcg@k`，含手算样例 |
| 3 ID 对齐 | [eval/alignment.py](../eval/alignment.py) | **已实现**：按规范化文本精确匹配数据集 passage id；**未命中的槽位保留原排名**（`None`），不静默丢弃；歧义文本拒绝猜测并计入 `unresolved` |
| 4 执行层 | [eval/runner.py](../eval/runner.py) | **已实现（仅检索）**：逐题记录命中 / 未命中 / 失败，落盘 `summary.json` + `samples.jsonl`，同一 task_id 不覆盖 |
| 5 生成侧指标 | [eval/runner.py](../eval/runner.py) 之外的 `answer_review.py` + `answer_quality_review.py` | **已实现（助手初审）**：三维标准评分 + 自动引用检查；**人工审阅未做**，且实测评审自身不稳定 |
| 6 消融矩阵 | `experiments/**` | **仅实验**：5 份对照报告存在，但未固化为可一键复跑的矩阵 |

**已经跑出的第一份真实检索指标**：见 [RETRIEVAL_BASELINE_REPORT.md](../experiments/rag-baseline/evaluation/RETRIEVAL_BASELINE_REPORT.md)。v1 开发集纯向量 Top-10 检索，主指标是**金标签是否落在前 4 个片段内**：售后 `recall_in_top_k` = 0.810（35 道可判分），会计 = 0.856（67 道可判分）；`mean_first_hit_slot` 分别为 1.76 与 1.47。

**检索模式对照（dense / BM25 / hybrid-RRF）**：见 [RETRIEVAL_MODES_REPORT.md](../experiments/rag-baseline/evaluation/RETRIEVAL_MODES_REPORT.md)。结论**因语料而异**：会计 hybrid 最好（recall 0.856 → 0.893，NDCG 0.806 → 0.862），售后 **BM25 单路最好**（0.810 → 0.933），融合在售后反而略差。可迁移的结论是**融合收益取决于两条臂的分歧度**（会计候选池 Jaccard 0.247 → 有收益；售后 0.649 → 融合只是重排噪声）。RRF 权重未在开发集上调参。

**邻段扩展**：见 [CONTEXT_EXPANSION_REPORT.md](../experiments/rag-baseline/evaluation/CONTEXT_EXPANSION_REPORT.md)。根因是会计 4 道题的标注条文**完整存在于源文档但被 500 字切片切开，没有任何片段完整包含它**，所以排序手段无解。按同一源文档取相邻片段拼成连续文本后，**送入上下文的完整命中率**：会计 hybrid 0.866 → 0.910，售后 0.886 → 0.971。代价是上下文从约 1,800 字涨到约 3,900 字；radius 2 在会计无收益。**尚未接入产品默认路径**（需生成侧指标判断 3,900 字对答案的影响）。仍剩 3 道会计题失败，原因是目标片段根本没进候选，属检索问题。

**LLM 重排（listwise）**：见 [RERANK_LLM_REPORT.md](../experiments/rag-baseline/evaluation/RERANK_LLM_REPORT.md)，决策见 ADR-009。用**现有对话模型**（非专用 reranker）对 Top-20 候选输出排列，严格校验完整排列、非法即回退。会计 NDCG@4 0.862 → 0.920、售后 0.839 → 0.988，两库 `mean_first_hit_slot` 都变成 **1.000**。归因显示 `not_promoted` 与 `broken` 均为 0（提升确来自重排），会计剩余 5 题是 `never_retrieved`（候选池无金标签，重排无解）。**代价：每题 17–18 秒**。

**换阿里云 Embedding 后的三方对照**：见 [RERANK_PROVIDER_REPORT.md](../experiments/rag-baseline/evaluation/RERANK_PROVIDER_REPORT.md)。Embedding 换成 `qwen3.7-text-embedding-flash` 后两库都改善（售后 recall 0.810 → 0.976，会计 0.856 → 0.893）。**专用 rerank `qwen3.7-text-rerank` 与 LLM 重排质量接近但快一个数量级**：会计 NDCG 0.9193 vs 0.9152、耗时 87.2s vs 1534.3s（快约 17.6 倍）；售后专用略差（0.9664 vs 0.9880），差异只出现在 2 道「多实体对比」题上——推测专用 rerank 缺少「整份候选是否覆盖问题全部实体」的全局视角（**未验证**）。专用端点输出分数，同一输入下**可复现**；LLM 重排两次跑出的排列不同（NDCG 0.9197/0.9152）。**仍未接入产品默认路径**。

**生成侧答案审阅（助手初审）**：见 [ANSWER_QUALITY_REPORT.md](../experiments/rag-baseline/evaluation/ANSWER_QUALITY_REPORT.md)，标准见 [QUALITY_RUBRIC.md](../experiments/rag-baseline/evaluation/QUALITY_RUBRIC.md)。20 题预声明样本（每库 10 题，属 v2 数据集）：自动引用检查两库 **10/10 通过**、阻断缺陷 0；要点完整性满分率仅 **0.50**，错误标签以 `retrieval_missing`（2/3 题）为主，另有 4 处 `missing_clarification`。**两个限制必须一起读**：① 同一批答案两次评审，引用忠实性满分率从 1.0 变 0.78，**评审自身不稳定**，与标准「不得用 LLM 自评分代替语义评分」一致；② 样本是定向诊断选样，**不估计总体准确率**，且 `human_reviewed` 为 0。验收目标（≥90%/≥95%）针对**人工审阅**，与本报告数字不可直接比较。

注意区分：报告里同时保留的 `recall@k`（0.971 / 0.893）是**按标签去重、且在整条候选列表上**计算的视图，三种模式数值完全相同，**不得单独引用**。排序质量看 `ndcg_in_top_k`、`mean_first_hit_slot` 与 `precision_in_top_k`。

**三条容易出错的规则**（已写进代码与测试，改动前先读）：

1. **未命中的槽位不能丢**。检索返回 5 条、其中第 1 条无法对齐时，第 2 条的排名仍是 2，MRR 记 0.5 而不是 1.0。丢掉槽位会静默抬高所有基于排名的指标——这正是本项目拒绝的行为。
2. **歧义即拒绝**。若两段数据集正文规范化后完全相同，无法判断命中属于哪一段，计入 `unresolved` 并标 `ambiguous_passage_text`，不做猜测。
3. **无标注 ≠ 得零分**。题目没有标签时该题不计入指标（`questions` 计数可见），而不是计 0 分；空指标集返回 `None`。

**两套数据集必须先分清**（本次读源码时发现文档里的数字与文件不一致，已实测核对）：

| 版本 | 位置 | 题目数 | 用途 |
| --- | --- | --- | --- |
| v1 | `experiments/rag-baseline/`（售后在根、会计在自己目录） | **201** = 售后 60（dev 40 / test 20）+ 会计 141（dev 82 / test 59） | 已完成的实验都用它：dev 122 / test 79 |
| v2 | `experiments/rag-baseline/evaluation/v2/` | **225** = 售后 72（dev 43 / test 29）+ 会计 153（dev 85 / test 68） | 复核候选集，标注仍为 `draft_needs_domain_review`，**不是签署过的 gold** |

`docs/HANDOFF.md` 旧文写的"dev128 / test97 / 225 道候选题"与文件不符：128 与 97 来自更早的划分记录，实际 v1 是 122 / 79，v2 才是 225。读取时用 `load_knowledge_base_dataset(..., version='v1'|'v2')` 明确指定，不要靠目录猜测。

### 仍需注意

- `eval/runner.py` 与生成侧指标仍是占位，未新增任何质量数字。
- **纪律**（与 AGENTS.md 一致）：两个领域分别报告指标，不合并成单一分数；证据覆盖率不等于答案准确率；v2 未冻结前不把它当作 gold，也不为了更好成绩移动已有题目。
- 生成侧指标必须走人工标注或有评分规则约束的 LLM Judge，并抽样核验 Judge 本身。

**ID 对齐的做法**：评测答案用数据集 passage id（售后 `evidence.jsonl` 的 `D1-S01`、会计 `articles.jsonl` 的 `CAS-10-2006-A002`），检索返回的是导入片段 id。当前实现用规范化文本精确匹配来桥接，未命中的会进 `unresolved` 而不是算作未命中。更稳的做法仍是**从导入阶段保留 `source_passage_id` 并一路传到命中结果**：文本匹配对改写、截断和重复段落都脆弱，且匹配率本身会随切片策略变化。这条仍是待办。

---

## 四、清理记录（2026-10-09）

本轮实际删除了本地产物 `.tmp/`（304 MB）、`.uv-cache/`（324 MB）、`.pytest_cache/`、`.dsh-acl-recovery/`，都在 `.gitignore` 内，不影响仓库内容。残留：沙箱创建的 `.tmp/pytest-of-AD钙` 等 6 个空目录与 `.uv-cache` 剩余 96.7 MB，当前进程无删除权限，需要时用管理员权限手工删。

**代码层面没有可删的死代码**：`context_structure.py` 被 `compact_experiment.py` 与测试引用，`enumerate_sources` 被 `test_importer.py` 引用，`cli.py`/`store.py`/`retriever.py`/`embedder.py`/`generator.py` 仍服务 `/index`、`/query` 与 `test_cli.py`。因此本轮不删文件，改为在本文标记各自状态。

**已知重复（待合并，不在本轮）**：`kb_service.vectors()/generate()` 与 `embedder.py`/`generator.py` 是同类功能的两份实现，前者服务知识库隔离，后者服务旧单库路径；合并时以 `host/` 装配点为准。
