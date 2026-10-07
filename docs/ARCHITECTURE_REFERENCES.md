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
