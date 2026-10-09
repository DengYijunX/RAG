# 成熟项目参考与设计决策

核对日期：2026-10-05（Asia/Shanghai）。本文把“实际依赖”“现有实现的对照”和“后续设计参考”分开，避免把常见做法相似说成已经借鉴过某个项目。

## 当前结论

此前明确使用的成熟组件是 `langchain-text-splitters` 的 `RecursiveCharacterTextSplitter`，见 [splitter.py](../rag/splitter.py)。现有设计说明强调模块拆分和最小依赖，但没有形成 LlamaIndex、Haystack、RAGFlow 的逐项参考记录。

本次核对后，以下来源成为后续通用 RAG 改造的明确参考。这里只阅读官方仓库的指定源码和文档，没有部署或跑测这些项目，不宣称其性能可以直接迁移。本项目既有原子索引切换和评测划分仍按自己的实现与实验解释，不追溯性地归功于这些来源。

## 参考来源

来源固定到 Git commit；文件 SHA256 见 [reference_sources.json](reference_sources.json)。HEAD 会变化，下面链接的版本不会随主分支移动。

用户指定的 Tencent/WeKnora 已另行分析，重点是插件化问答步骤、召回与重排分离、分阶段评测记录和指标局限，见 [RAG 流程与评测路线](RAG_FLOW_AND_EVALUATION.md)。这部分来源固定版本和哈希在 `weknora_sources.json`。

1. **LlamaIndex：摄取流水线与文档身份管理。**
   [IngestionPipeline 源码](https://github.com/run-llama/llama_index/blob/962940ddc079cc21701d28d1237c84c82a7c5164/llama-index-core/llama_index/core/ingestion/pipeline.py)。`run_transformations` 按变换与输入生成缓存键；`DocstoreStrategy` 区分去重、更新、更新并删除；文档 ID / hash 与文档存储参与更新判断。
2. **Haystack：检索结果融合的组件边界。**
   [DocumentJoiner 源码](https://github.com/deepset-ai/haystack/blob/87dba65f8c8b1f3906a36625615618ba29505971/haystack/components/joiners/document_joiner.py)。示例连接 BM25 与向量检索组件；Joiner 提供 concatenate、merge、RRF 等模式，以及权重和 top_k 配置。参考的是可组合、可独立测试的组件接口，不是断言融合必然提高效果。
3. **RAGFlow：知识库与文档的产品生命周期。**
   [知识库创建](https://github.com/infiniflow/ragflow/blob/2400ca8eb51432b1d304266d0de9ca232d9b53fa/docs/guides/dataset/dataset_list_and_creation.md)把名称、Embedding 模型、解析方式作为知识库配置；[文档管理](https://github.com/infiniflow/ragflow/blob/2400ca8eb51432b1d304266d0de9ca232d9b53fa/docs/guides/dataset/files_dataset_document_management.md)区分文件导入、解析、片段、状态和元数据，说明禁用 / 删除对检索的影响；[README](https://github.com/infiniflow/ragflow/blob/2400ca8eb51432b1d304266d0de9ca232d9b53fa/README.md)展示片段可视化和可追溯引用的产品方向。本文并未核验其全部后端实现。

## 如何体现在本项目

| 参考点 | 当前实际状态 | 后续设计决策 | 验证方式 |
| --- | --- | --- | --- |
| LlamaIndex：文档 ID、hash、摄取变换 | 已有语料 fingerprint、切片和全库版本重建；没有通用导入注册表或增量 upsert | 引入 DocumentRecord，保存稳定文档 ID、内容 hash、原文件及解析元数据；先做幂等导入和全量构建，增量更新另做版本 | 同内容重复导入不增加文档；同名不同内容按明确更新规则处理；hash 或模型变化阻止误用旧索引 |
| LlamaIndex：变换配置与缓存边界 | 固定 500/50；模型及语料变更检测已实现 | Parse → Split → Embed → Publish 分阶段；解析器、切片器及模型配置写入构建签名；缓存是后续可选优化 | 改切片配置后不能复用旧片段；构建中失败不发布半成品 |
| Haystack：Retriever / Joiner 分离 | BM25、RRF 已在实验脚本中对照；应用默认仍为向量与会计精确分支 | 通用 Retriever 接口返回带分数、来源和元数据的命中；融合及重排可插拔；默认策略由评测决定 | 相同问题、语料、上下文预算做消融；重排未实现时不标记已支持 |
| RAGFlow：知识库是配置化实体 | 新接口只列举固定 BASES；页面选项也是固定的 | 用注册表持久化知识库 ID、名称、解析与检索配置；页面动态读取；第三个库无需改代码 | 用独立主题资料走创建、导入、索引、问答全流程，验证两库历史数据不受影响 |
| RAGFlow：文件、解析结果、片段与状态分离 | 已有来源查看；依赖实验目录中的预生成元数据 | 导入文件生成来源和片段映射，不依赖评测 evidence 文件；提供可见的构建状态与失败原因 | 损坏文档、空文档及解析失败有明确反馈；文档与 chunk 数分别统计 |
| RAGFlow：片段和引用可检查 | 已实现编号引用、年份 / 条号、片段与原文弹窗 | 通用引用用文档身份及定位信息；年份、条号只在有元数据时展示 | 引用属于所选知识库、片段确实来自对应文档；语义支持关系另行审阅 |

## ADR-001：借鉴架构，保留轻量实现

**决策：** 继续使用 FastAPI、Chroma、OpenAI 兼容 SDK 和现有切片组件。暂不把完整 LlamaIndex / Haystack 框架或 RAGFlow 服务栈引入运行依赖。

**原因：** 当前目标是完成一个可讲清楚、可测试的通用 RAG 工程项目；现有依赖已跑通。直接迁移框架会同时改变解析、索引、检索和调用路径，使旧实验难以对照。借鉴组件边界、数据模型与生命周期，可以先消除领域硬编码，再按实际维护成本决定是否采用更多库。

**取舍：** 自己维护文档注册、构建状态和来源映射会增加代码与测试工作。应复用现有解析 / 存储组件，而不是逐项重写成熟库；需要复杂功能时先评估现成能力。引用外部代码时另行记录来源和遵守其许可，本次仅保存设计对照和来源元数据。

## 下一步的数据与接口约束（尚未实现）

- `KnowledgeBase`：ID、名称、解析配置、Embedding 标识、检索策略、可选领域扩展、活动索引版本。凭据由运行环境管理，不写入注册表。
- `DocumentRecord`：知识库 ID、文档 ID、原名、内容 hash、媒体类型、原文件位置和可选元数据。更名不应等于新建，内容更新应保留明确的版本关系。
- `ChunkRecord`：文档 ID、chunk ID、解析 / 切片版本、正文和定位信息。Markdown / TXT 可用字符或行位置；PDF 页码只有解析器实际提供时才显示。
- `IndexVersion`：知识库 ID、输入文档快照、构建签名、阶段状态和活动版本指针。先延用已经验证的成功后原子切换，不在本轮声称已支持分布式任务调度。
- 通用检索不读取测试问题或固定型号列表。会计精确条号定位作为显式启用的扩展，开关与结果分别评测。
- 新建 / 导入 / 构建 / 查询 / 来源访问统一走知识库 ID；售后和会计通过相同流程作为 fixture 接入，不为第三个领域新增条件分支。

这组约束落实后，再评估文档禁用 / 删除、增量 Embedding 缓存、异步任务、混合检索和重排的收益与成本；这些都不能因参考项目已有就宣称本项目也已具备。

## 持续维护

### ADR-002：开发集单题追踪

2026-10-07 落地于 [trace_question.py](../experiments/rag-baseline/evaluation/trace_question.py)，案例见 [实践一](RAG_TRACE_WALKTHROUGH.md)。具体参考是 [WeKnora evaluation.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/evaluation.go) 保存召回、重排与生成产物，以及 [metric_hook.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/metric_hook.go) 分阶段计算指标的边界。

采用点：记录真实应用各阶段输入输出，再以原文位置核对证据。取舍：仅在独立诊断进程中包装现有服务与 SDK，避免为了单例解释引入完整事件框架；这是开发评测工具，尚非线上全请求追踪。题目标注只在真实回答结束后评分，额外 Top-20 结果不回填给生成模型。

验证：Q043 实际运行记录了 1 次问题 Embedding、1 次生成与正式 Top-4；检查四条命中全文进入请求、输出文件哈希及源代码哈希一致，诊断发现第三组必要证据未进入正式上下文。工具只接受 dev 题；单例结果不用于宣称总体准确率。

后续重要设计改动记录：参考项目及具体出处、采用点、不采用的原因、落地文件、验证证据。已有实验收益仍按 [HYBRID_REPORT.md](../experiments/rag-baseline/evaluation/HYBRID_REPORT.md) 的领域与指标边界报告。该文档补足设计依据，不代表通用知识库改造已经完成。

2026-10-07 的 [Top-K 对照](../experiments/rag-baseline/evaluation/TOPK_REPORT.md) 延续 WeKnora `search.go` / `rerank.go` 分离召回和后处理、`metric_hook.go` 分层记录的参考点。当前仅实现纯向量 K 值实验，尚未引入重排服务或改变应用默认值；保存实际上下文与生成结果，验证覆盖提高不等于生成质量必然提高。下一步候选池与上下文预算分开仍属待验证设计，不能写成已实现能力。

### ADR-003：固定预算下验证邻段扩展，拒绝直接设为默认

2026-10-07 实验落地于 [context_budget.py](../experiments/rag-baseline/evaluation/context_budget.py)，结果见 [CONTEXT_REPORT.md](../experiments/rag-baseline/evaluation/CONTEXT_REPORT.md)。沿用已核验 WeKnora [search.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/chat_pipeline/search.go) / [rerank.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/chat_pipeline/rerank.go) 的阶段分离思路：将已召回候选与最终上下文分别保存、分别评分。前后相邻切片的具体贪心规则由本项目为控制实验编写，不声称来自 WeKnora 源码。

已实现的边界是实验工具：仅使用通用来源身份、字符位置、正文与预算，不接受题目标准答案或领域路由；整个应用尚未接入该选择器。固定 2000 正文字符，验证 128 道开发题、18 次生成、16 项测试。会计证据覆盖改善、售后明显退步，且出现条款归属矛盾，因此保留结果而不晋升默认。下一步相关性重排及结构边界组装仍待独立实现与评测；字符上限也尚未替代为生产 Token 预算。

### ADR-004：可回退的重排原型与独立生成对照

2026-10-07，参考 WeKnora [rerank.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/chat_pipeline/rerank.go) 的召回后独立重排和故障回退边界，在 [rerank_context.py](../experiments/rag-baseline/evaluation/rerank_context.py) / [rerank_experiment.py](../experiments/rag-baseline/evaluation/rerank_experiment.py) 实现实验原型。未使用其具体排序模型或移植算法：本项目用已有 LLM 输出候选标签的精确排列，非法结果或调用异常回退原顺序，记录额外 Token 与耗时。

同一候选池与 2000 字符预算下，把排序和生成的边界提示分为四组条件；提示格式仅保留原文标题并增加片段标记，不声称已解析完整章节结构。验证包括 128 题离线候选覆盖、固定九题的 9 次排序 / 36 次生成、27 项测试，以及实际请求与指标审计。结果见 [RERANK_REPORT.md](../experiments/rag-baseline/evaluation/RERANK_REPORT.md)：九题上下文覆盖改善，但存在实质生成错误、续段结构缺失和明显额外用量，暂不接入应用默认策略。全开发集重排、专用模型与通用结构元数据均待后续验证。

### ADR-005：从原文生成结构归属，独立验证生成收益

2026-10-07，在 [rag/structure.py](../rag/structure.py) 实现原文字符位置、Markdown 标题路径、跨节片段归属和选中上下文的章节完整性检查。沿用 ADR-002 中 WeKnora `evaluation.go` / `metric_hook.go` 的阶段产物与分层验证思路；具体 Markdown 子集解析器由本项目编写，不声称来自 WeKnora。模块不读取知识库领域、题目或标准答案。

取舍：仅支持 ATX、单行 Setext 和代码围栏排除；独立加粗行需显式启用，标记为启发式，不等同于正式标题。不支持完整 CommonMark 或语义条款识别；非 Markdown 归属未知。字符位置来自未经规范化的原文，无法字面定位时失败，不伪造来源。父章节完整性包含所有子章节，多个已选片段按来源合并覆盖后判断。

目前是可复用核心模块，尚未接入应用默认生成路径。10 项结构测试及全部 40 项应用测试通过。下一步固定已有九题检索正文，以新生成的对照组验证结构提示；模型可见标题只引用已选正文中的文字，防止元数据额外带入答案。收益及开销以实测报告为准。

实践五已完成，见 [SECTION_REPORT.md](../experiments/rag-baseline/evaluation/SECTION_REPORT.md)：64 份原文 / 775 个片段定位验证、九题 18 次生成，结构提示改善局部归属和截断说明，但产生漏答及片段外常识，输入 Token 为边界组 3.15 倍。实际生成适配器位于 `evaluation/structure_context.py`，运行入口为 `section_experiment.py`；旧 `structure_experiment.py` 保留原结构检索实验。40 项应用测试、44 项评测工具测试通过。尚未晋升应用默认；下一步优先精简模型可见字段，并在通用导入阶段保存完整审计结构。

### ADR-006：审计结构与模型提示分离

2026-10-08，延续 ADR-005 的原文结构模型及 WeKnora [metric_hook.go](https://github.com/Tencent/WeKnora/blob/bccb4b151bae403508da77fbb174efc79dc47c1a/internal/application/service/metric_hook.go) 的分阶段指标参考，在 [rag/context_structure.py](../rag/context_structure.py) 新增精简提示视图。完整 hash、位置和路径保留在实验审计记录；模型侧对章节去重、使用请求内短 ID，只保留可见标题、父节关系、片段归属及完整性。具体压缩格式由本项目设计，不宣称来自 WeKnora。

取舍：数组格式更短但需要字段解释，可能增加模型误读；必须以固定正文的真实生成对照验证，不能只看序列化大小。按文档身份隔离短 ID，保留未知状态与加粗启发式标记，不偷偷裁掉正文或截断标记。超出完整 messages 的 UTF-8 字节预算则调用前失败；没有官方模型 tokenizer 时不将该上限称为精确 Token 预算。服务端实际 Token 用量另行记录。默认应用暂不接入。

ADR-006 验证结果见 [COMPACT_REPORT.md](../experiments/rag-baseline/evaluation/COMPACT_REPORT.md)：九题18次重新生成，输入从43,807降至20,907 Token（−52.3%），请求最大10,730字节低于12,000字节精简上限；结构状态无损保留，但模型仍误读续段与完整性、猜测未展示条件总数。45项应用测试与44项评测工具测试通过，默认路径不变。下一步转向独立于评测文件的通用导入与第三语料接入，生成失败另存回归案例。

### ADR-007：知识库注册表与通用文档导入

2026-10-09，按 `docs/PROJECT_SCOPE.md` 的下一步实施顺序，新增 [rag/registry.py](../rag/registry.py)、[rag/importer.py](../rag/importer.py)、[rag/kb_service.py](../rag/kb_service.py)，并改造 [rag/api.py](../rag/api.py) 与 [rag/static/index.html](../rag/static/index.html)。

参考来源仍是本文开头固定的三处官方材料，**没有引入这些项目的框架或运行依赖**：
[RAGFlow 知识库创建](https://github.com/infiniflow/ragflow/blob/2400ca8eb51432b1d304266d0de9ca232d9b53fa/docs/guides/dataset/dataset_list_and_creation.md)把名称、Embedding 模型、解析方式作为知识库配置，本文据此把知识库做成注册表实体而非固定选项；
[RAGFlow 文档管理](https://github.com/infiniflow/ragflow/blob/2400ca8eb51432b1d304266d0de9ca232d9b53fa/docs/guides/dataset/files_dataset_document_management.md)区分文件、解析结果、片段与状态，本文据此把清单（manifest）与索引代（generation）分开；
[LlamaIndex IngestionPipeline](https://github.com/run-llama/llama_index/blob/962940ddc079cc21701d28d1237c84c82a7c5164/llama-index-core/llama_index/core/ingestion/pipeline.py)的文档 ID / hash 与变换缓存边界，本文据此把「文档身份 = 知识库 ID + 相对路径」「内容 hash」用于变更检测。以上是采用点，不是本项目已具备这些框架的能力。

**已实现**（有测试覆盖）：

- 注册表以 JSON 持久化知识库 ID、名称、描述、语言与可选 `extensions`，写入采用临时文件替换；不含任何凭据字段，凭据仍由环境变量提供。
- 导入只读文件本身：文档 ID 由知识库 ID 与相对路径派生，另有文件 SHA256 与解析后内容 SHA256；Markdown 记录 ATX/Setext 标题结构与章节归属，TXT 记录物理行数但不虚构标题；**PDF 的页码来自解析器实际给出的页数，字符位置一律不报告（`positions=false`），不伪造区间**。
- 每次导入的产出是「文档 + 片段 + 跳过项」三份清单：不支持的扩展名、解析失败、片段无法在原文定位，都会带原因进入 `skipped`，不会静默消失，也不会中断其余文件；行数按物理行计（结尾换行不算空行）。
- 片段位置复用 [rag/structure.py](../rag/structure.py) 的 `locate_chunks`：按顺序字面定位，定位失败即报错，不用规范化文本冒充原文位置。
- 索引构建按知识库隔离到独立 generation 目录，全部片段向量化并核对数量后才原子替换 `active.json`；任一环节失败抛 `BuildFailed` 并清理半成品目录，旧索引继续服务（测试覆盖）。每个 generation 保存自己的片段快照，检索只在该 generation 内解析命中，避免与并发导入交错。
- 签名包含文档 ID 与内容 hash 集合、Embedding 模型与地址、切片参数与距离度量；文档更新后旧索引被判定失效，需重新导入并重建。
- 接口按知识库 ID 工作：创建、文件上传或服务器目录导入、建索引、问答、查看原文；未知知识库 404、未建索引 503/409、提供方异常 502 且不回显凭据细节。`GET /knowledge-bases` 只依赖注册表，**缺少实验目录时仍正常返回已注册知识库**（有测试覆盖）；夹具项只标记 `fixture`，不把「型号 / 准则」等展示文案下发给通用列表。

**取舍与仍未完成的部分**（不得当成已实现）：

- `rag/knowledge.py` 与既有两个夹具**尚未迁移**：智能家居与会计仍按 `experiments/rag-baseline` 的预生成元数据加载，其会计条号路由仍是该路径下的**领域专用分支**；该目录缺失时这两个夹具不再出现在列表中，通用部分不受影响。通用路径（`rag/kb_service.py`）不含任何领域条件分支，两者目前在同一个 API 中共存。
- 领域扩展机制只预留了注册表字段，**尚未实现**基于导入元数据的精确定位；ADR-002 至 ADR-006 的检索增强（邻段扩展、重排、结构提示、精简提示）都还没有接入通用路径。检索默认仍是纯向量 Top-K。
- 本轮不实现文档删除/禁用、增量 upsert 与 Embedding 缓存；**每次导入整体替换该知识库的可检索集合**，每次构建为全量重建，这一点已写进接口文档与页面文案。
- 签名不含解析器与切片器的实现版本：只改切片逻辑而不改参数时，旧索引不会被判定失效。
- 向量后端在本机只能以注入的内存实现验证，真实 Chroma 路径未在本机跑通，原因见 [ENVIRONMENT_WINDOWS.md](ENVIRONMENT_WINDOWS.md)。

**验证方式**：11 项注册表测试、14 项导入测试、14 项生命周期测试、13 项接口测试（本机共 84 项通过，另有 11 项 `test_knowledge.py` 因原生向量索引不可用未运行）；第三语料（阳台蔬菜种植）端到端验收 10/10 通过，见 [THIRD_CORPUS_ACCEPTANCE.md](THIRD_CORPUS_ACCEPTANCE.md)。该验收证明的是导入与接口契约，**不是答案质量，也没有使用真实向量检索**。此外用 `uvicorn rag.api:app` 启动真实服务复核过列表与页面可访问。

### ADR-008：按 WeKnora 路线固定目标结构，评测框架先落占位

2026-10-09，用户要求参考 WeKnora 的七步学习路线（文档解析与 Chunking → Embedding 与索引 → 混合检索与融合 → Rerank 与上下文 → 生成与引用 → Evaluation → 消融与工程优化）说明后续结构并允许留占位。落地方式见 [ROADMAP.md](ROADMAP.md)：标出目标目录树、模块与位置的对应表、每个模块的现状与缺口，以及 `eval/` 的实现顺序。

**决策：** 新增 `eval/` 包（`datasets.py` / `metrics.py` / `runner.py`），只写数据类与接口说明，公开函数一律 `raise NotImplementedError`，并加 [tests/test_eval_scaffold.py](../tests/test_eval_scaffold.py) 断言占位不会被执行成功、指标名拼错会被拒、且 `eval` 不导入 `rag`、`rag` 不导入 `eval`（分层方向单向：评测包装产品，产品不依赖评测）。

**原因：** 现有评测能力散在 `experiments/rag-baseline/evaluation/` 的脚本里，没有统一的数据集 schema、指标注册和任务调度，无法直接回答"这次退化出在哪一层"。参考 WeKnora 的分层（数据集 / 指标 / 评测服务 + 指标 hook）先固定接口与边界，比继续堆脚本更容易审查；同时保留已跑过的实验产物，不改写历史结论。

**取舍：** 只落占位意味着这一轮**没有新增任何评测指标能力**，因此不写任何新数字；现有 5 份 `*_REPORT.md` 仍是唯一证据来源。占位用 `NotImplementedError` 而不是返回空值，是为了避免"看起来跑通了"。`rag/retrieve/`、`rag/generate/`、`rag/host/` 在 ROADMAP 中标为待拆分/未实现，本轮**不动代码结构**，避免在真实向量路径尚未在可用环境复跑前做大规模搬迁。

**验证方式：** 5 项脚手架测试通过；本机全套 89 项通过（84 + 5）。ROADMAP 的清理记录同时列出本轮删除的本地产物与"无可删死代码"的核查结果（`context_structure.py`、`enumerate_sources`、旧单库路径均仍被引用）。
