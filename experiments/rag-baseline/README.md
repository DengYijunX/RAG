# RAG datasets and baseline checkpoint

This directory records the completed baseline before structural-chunking experiments.

- 64 source documents: 5 smart-home FAQs and 59 uploaded accounting standards.
- 225 v2 review-candidate questions; human domain review is still pending.
- 128 completed development-set retrieval and generation requests, with manifests, outputs and coverage metrics.
- Raw credentials, virtual environments and vector database caches are not part of this checkpoint.

## Run from the repository root

Use the repository virtual environment prepared during onboarding. The scripts below now resolve the checkout location themselves:

```bash
.venv/bin/python experiments/rag-baseline/validate_datasets.py
.venv/bin/python experiments/rag-baseline/check_loader.py
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py prepare
.venv/bin/python -m unittest discover -s experiments/rag-baseline/evaluation -p test_baseline.py -v
```

Real API calls require `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`; generation also requires `LLM_API_KEY` and optionally `LLM_BASE_URL`/`LLM_MODEL`. Supply credentials through environment settings, not tracked files. The script retains the previously authorized deployment URL as its fallback; override it for another deployment.

```bash
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py preflight --generate
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py run --kb smart-home --split dev --generate
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py run --kb accounting --split dev --generate
```

Existing reports and run manifests preserve the original cloud paths and hashes as historical evidence; these do not imply the new checkout must use those paths. The exact executed runner is preserved under `evaluation/provenance/`; the working runner differs only in how it locates the checkout. Future runs create new output directories. Stored Chroma caches were omitted, so reconstruct an index when rerunning.

Start with [the baseline report](evaluation/BASELINE_REPORT.md), [experiment instructions](evaluation/README.md), or [project plan](PROJECT_PLAN.md). The sections below preserve the original dataset handoff notes.

---

# 双知识库 RAG 数据集 v1

> 已完成两个开发集共128题的真实模型基线：[基线报告](evaluation/BASELINE_REPORT.md)。

> 更新：v2评测候选集已增至225题，并增加可执行基线脚本。见 [evaluation/README.md](evaluation/README.md)。下文保留v1语料与201题交付记录；原始语料未变。

目标：形成可复现、可评测、能解释技术取舍的 RAG 项目。根据用户选择，智能家居售后与企业会计准则使用独立知识库、独立评测和独立索引。当前交付的是语料与评测草稿，不是已完成的双知识库应用。

## 本次交付

| 知识库 | 原始资料 | 证据单元 | 评测题 | dev / test | 现有500字符、50重叠切片数 |
|---|---:|---:|---:|---:|---:|
| 智能家居售后 | 5份原有Markdown | 83个FAQ段落 | 60 | 40 / 20 | 26 |
| 企业会计准则 | old.zip中的59份DOCX | 1,859条条文 | 141 | 82 / 59 | 749 |

共64份原始文档、201道题。格式转换后的Markdown与原DOCX是同一资料，不重复计为新增文档。切片数量来自现有加载器实测，不代表检索准确率。

- [智能家居题目审阅稿](smart-home/REVIEW.md)
- [会计准则题目审阅稿](accounting/REVIEW.md)
- [数据完整性验证](validation.json)
- [现有加载器验证](loader_check.json)
- [项目推进与验收](PROJECT_PLAN.md)

## 文件布局与索引范围

```text
smart-home/
  corpus/                   # 只将这里的5份Markdown用于售后知识库索引
  documents.jsonl           # 文件清单、来源与SHA256
  evidence.jsonl            # FAQ证据、原文件及标题
  questions.jsonl           # 全部60题
  questions.dev.jsonl
  questions.test.jsonl
  REVIEW.md                 # 人工审阅入口
  report.json
accounting/
  raw/                      # 原DOCX保留
  text/                     # 保留Word软换行的提取正文
  markdown/                 # 只将这里的59份Markdown用于会计知识库索引
  documents.jsonl           # 准则编号、版本年份、哈希、来源状态
  articles.jsonl            # 条号、章/节、原文、提取文本行号
  questions*.jsonl           # 全部141题及dev/test划分
  REVIEW.md
  corpus_report.json
  questions_report.json
  scripts/                  # 可复现的转换和出题脚本
```

不要索引整个交付目录：评测问题、预期答案、审阅稿、脚本不能进入检索语料，否则会产生答案泄漏。根目录旧的FAQ生成中间文件不是最终评测入口；最终以 smart-home/ 与 accounting/ 下的文件为准。

## 题目设计

会计题包含59道显式条号定位题、32道语义/场景题、4道计算题、16道相邻版本对照题、10道多证据综合题、10道澄清题、10道无依据或工具边界题。定位题用于验证全59份文档的覆盖，不能用其通过率代替实际场景效果。版本对照题只比较指定条文，不假设两个年份相同条号具有相同主题。

售后题包含36道单文档题、8道多证据题、8道澄清题、6道拒答题、2道原文冲突题。特意覆盖X1/X2操作差异、C1无线/G2有线差异，以及电池保修范围冲突。

每题记录 question、category、expected_action、expected_points、evidence_ids、split、annotation_status。会计题还保留逐字 reference_excerpts 和 wrong_version_candidates。后者是可用于排查版本误召回的候选条文，并不意味着其语义一定矛盾，不能未经判断就当成负例标签。

全部题目是依据上传/仓库资料编写的初始草稿，状态为 draft_needs_domain_review；不是独立专家标注集，也不是来自真实用户日志。合成题设中的金额与业务事实仅用于测试推理，不写入权威知识文档。投入正式效果评测前，需要人工核对题意、证据充分性、预期要点和可接受的替代证据。

## 数据质量处理

- 59份DOCX均成功读取；未发现表格、图片或修订内容，因此本批次不需要OCR。脚本遇到这些对象会明确报错，不静默漏读。
- Word正文主要通过软换行组织。提取时保留 w:br，避免整篇内容拼接成一段。
- 处理了“一百零一”等条号，1,859条按文件分别连续；正文里的换行引用“第三章相关规定…”不会被误删为章节标题。
- 校验原文件哈希、Markdown正文保真、条文引用、引用摘录与正文一致、题目和ID唯一性。
- 会计语料涉及43个准则家族，其中15个有多年份版本。所有版本均保留，不按名称或大段重复内容去重。
- 每份文档保留 filename version_year；该字段不是经核实的施行时间。正文中的日期也不能代替完整的适用范围和过渡安排。
- 文件来自用户上传，尚未与财政部官网逐份核验。official_source_url 为空、authority_verified 为 false、current_effectiveness 为 unverified；没有为文件杜撰官方链接或现行效力。
- 售后资料保留原始冲突，不擅自修正文档，也不据此承诺免费维修。是否真实产品、是否现行政策也未独立核验。

## 分组与指标

会计集按准则家族分组，同一准则的所有年份、改写题、相关多准则综合题进入同一集合。售后集将共享FAQ证据的题目（包括澄清题）放在同一集合。校验确认dev/test之间没有共享证据ID，会计集也没有共享准则家族。

这是小语料上初始的分组划分，类别并不均衡：会计测试集仅1道拒答题，售后测试集也仅1道拒答题；不能用这种规模声称拒答能力稳定。实际上线前需补充独立人工问题、更多拒答及冲突案例。出题过程已经接触全语料，分组隔离不等于外部盲测。未来同义改写必须继承原组，不能随机分散。

两个知识库分别报告指标，且分题型报告；不把简单定位题和复杂场景题混成一个高分。

- 检索：按证据ID标注Recall@K = 命中必需证据数 / 必需证据总数；MRR按首次相关证据排名；多证据题补充全部证据命中率。拒答与澄清题不能当作有标准召回目标的普通题。
- 会计版本：统计错用版本比例、引用准则/年份/条号正确率。引用到同一规则的其他有效条文可由人工复核为替代证据。
- 生成：按预期要点审阅正确性和完整性，并核对每项结论的引文支撑。计算题同时检查金额、计算式与适用条件。
- 行为：区分answer、clarify、abstain和flag_conflict，统计按场景应答是否适当。
- 工程：记录检索、重排、生成各阶段耗时和Token量；固定并发、请求次数、冷热缓存条件后再比较延迟。

现有代码只返回文件路径和chunk_index；它尚不能直接用条号计算检索指标，也不会自动读取这里的版本元数据。后续需要增加条文/FAQ证据与检索片段映射、版本过滤或分库路由，不能将完整性校验通过写成检索评测通过。

## 复现

数据构建只使用Python标准库；现有RAG加载验证使用已安装的项目虚拟环境。

```bash
cd /workspace/rag-project-prep
# 重建会计语料（保留原上传zip的实际路径）
python accounting/scripts/prepare_corpus.py /path/to/old.zip
python accounting/scripts/build_questions.py
# 重建售后题目及最终分组；原始五份文档仍来自 /workspace/RAG/文档
python build_dataset.py
python prepare_smart_home.py
python validate_datasets.py
/workspace/RAG/.venv/bin/python check_loader.py
```

首次检查：原仓库19项测试通过；两个语料目录均被现有加载器成功读取。尚未进行Embedding、真实检索或LLM答案评测，当前环境缺少两个模型API密钥。

## 后续运行时隔离

数据已分目录。当前应用的ChromaDB client和collection是进程级单例，不支持在一个进程内切换不同PERSIST_DIR后保证正确隔离。接入时应先采用两个独立进程、两个独立PERSIST_DIR，或另行实现带知识库标识的存储适配；不能只改请求中的路径。

待密钥配置并授权运行真实模型评测后，索引路径分别指向 smart-home/corpus 和 accounting/markdown。评测用全新专用存储目录，防止全量重建覆盖已有用户索引。索引完成后再运行各自测试集；本交付未发起模型调用。
