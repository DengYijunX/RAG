# 新会话交接（2026-10-13）

分支 `work`，最新提交 `4fe225a`，**本地与远端一致**（`origin/work` = `https://github.com/DengYijunX/RAG.git`）。`work` 领先 `origin/main` 71 个提交，**未合并**。

先读 [ROADMAP.md](ROADMAP.md)（七步路线 + 模块状态 + 各轮结论）与 [ARCHITECTURE_REFERENCES.md](ARCHITECTURE_REFERENCES.md)（ADR-001~010 的决策与取舍）。本文件只讲**怎么跑、当前处于什么状态、下一步该做什么**。

---

## 一、项目是什么

可演示、可复现的**通用 RAG**。智能家居售后与企业会计准则只是两套**评测夹具**，不是产品边界——通用流程不得依赖固定知识库 ID、产品型号、准则名称或评测答案，新语料必须走导入流程接入、无需改源码（第三语料验收即为此）。

分层：`rag/` 产品运行时；`eval/` 评测框架（**可以**导入 `rag`，`rag` **不得**导入 `eval`，有测试强制）；`experiments/` 保留实验证据；`tests/` 464 项。

---

## 二、怎么把产品跑起来

三项配置（已写入 `.env`，`RAG_*` 为 2026-10-11 新增）：

```
RAG_VECTOR_BACKEND=faiss      # 默认；Chroma 的 HNSW 在本机约 96 个 1024 维向量即崩
RAG_RETRIEVAL_DEPTH=50        # 先取 50 个候选
RAG_RERANK_POOL=20            # 用专用 rerank 给前 20 个排序，再取 TOP_K=4
RAG_CONTEXT_RADIUS=0          # 邻段扩展；实测未被证明有益，保持关闭
```

```powershell
# 建知识库（会真实调用 Embedding）
.venv\Scripts\python.exe -m rag.kb_service create accounting
.venv\Scripts\python.exe -m rag.kb_service import accounting experiments\rag-baseline\accounting\markdown
.venv\Scripts\python.exe -m rag.kb_service index accounting
.venv\Scripts\python.exe -m rag.kb_service ask accounting "《企业会计准则第14号——收入（2017）》第四条的核心表述是什么？"
```

实测：会计语料 **59 份文档 / 866 片段 / 建索引 55 秒**；单题 **5–13 秒**（重排约 0.8 秒，其余为生成）。

**索引不随代码提交**（`chroma_db/` 被忽略），换机器须按上面重建。签名不匹配时 `service.active()` 会报"文档或模型配置已变化"。

### 跑测试（本机必须这样跑）

```powershell
$env:PYTHONPATH="<仓库>\tools\testenv"
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp="<仓库>\.tmp\x" --ignore=tests/test_knowledge.py
```

`PYTHONPATH` 指向 `tools/testenv/sitecustomize.py`（把临时目录重定向到工作区内，否则沙箱下不可写）；`--basetemp` 同理。**`tests/test_knowledge.py` 的 11 项因原生 Chroma 不可用长期不运行。**

---

## 三、当前状态：能力齐了，瓶颈在会计的检索召回

七步路线（[ROADMAP.md](ROADMAP.md) 有完整表）：① 解析与切片、② Embedding 与索引、③ 混合检索、④ Rerank、⑤ 生成与引用、⑥ Evaluation、⑦ 消融——**① ② ④ ⑤ ⑥ 已实现或已实验，③ 仅实验**。

### 关键数字（v1 数据集）

| 指标 | 开发集 | 保留集（test） |
| --- | --- | --- |
| 售后 检索 recall@4（深池+重排） | 0.9786 | **0.8684** |
| 会计 检索 recall@4（深池+重排） | 0.9104 | **0.8519** |
| 售后 答案要点完整性 | 0.947 | **0.947** |
| 会计 答案要点完整性 | 0.772（产品路径人工审阅） | **0.772** |

### 三个必须记住的结论

**1. 开发集到保留集有明显落差，且方法排序反转。** 开发集上"纯向量最好"（售后 0.9762 / 会计 0.9378）在保留集上却是**最差**（0.7237 / 0.7407）；深池+重排在两集都最好。原因是开发集被反复用于改动，纯向量基线也分享了"熟悉度"。**不要用开发集数字判断方法优劣。**

**2. 会计的缺口是检索侧，不是答题侧。** 产品路径保留集 59 题中，16 道未满分：**10 道"检索没给到"**（5 道原文完全没进上下文、3 道只进一半、**2 道版本错配**），**只有 3 道是答题侧**（证据齐全但没答全），另 5 道无要点题。

**3. 引用忠实性与行为匹配稳，完整性是短板。** 两库保留集上引用忠实性 0.982–1.000、行为匹配 1.000、**阻断缺陷 0**。

### 哪些手段有效、哪些无效（都实测过）

| 手段 | 结论 |
| --- | --- |
| 换 Embedding（bge-m3 → qwen3.7） | **有效**，两库都改善 |
| 结构感知切片 | **有效并已设默认**，会计 recall +4.5pp、售后零影响；令 4 道长期失败题中的 3 道可被检索 |
| 深池 50 + 专用重排 | **有效并已启用**，保留集比纯向量 +14.5pp / +11.1pp，代价 0.8 秒/题 |
| hybrid BM25+RRF | **因语料而异**（会计有效、售后反而略差），仅实验 |
| LLM listwise 重排 | 质量与专用重排接近但**每题 17–18 秒**，已弃用 |
| 邻段扩展 | **未被证明有益**，默认关闭（曾误判为"诱发编造"，已撤回，见 ADR/报告） |
| 加大切片到 1000 | 会计更好、售后略降；**未改默认**，属待权衡项 |

---

## 四、下一步（按杠杆排序）

**1. 提高会计的检索召回（最大杠杆）。** 10 道题的标注原文没进前 4，其中 **2 道是"版本错配"**——检索拿到的条文来自错误年份（`CAS-37-2014` 第八条与 `CAS-37-2017` 第八条**逐字相同**）。这提示**纯向量难以区分版本**，方向可能是关键词/元数据过滤或条款级标识，而不只是加深池子。

**2. 复查 v2 标注**（225 题，`draft_needs_domain_review`），作为**新的保留样本**——v1 test 已被观察两轮，按标准不得再用于调参。

**3. 迁移 `rag/knowledge.py`**：两个夹具仍读 `experiments/rag-baseline` 的预生成元数据，会计条号路由是这条旧路径下的领域分支。

**4. 在能跑原生 Chroma 的环境复跑**（Docker/Linux）：产品默认已改 FAISS，Chroma 作为可选后端保留，其真实路径仍未在可用环境验证。

---

## 五、不可遗忘的边界（踩过的坑）

**测量纪律**

- **检索指标改善 ≠ 答案改善**。邻段扩展在检索侧明显改善（完整命中 0.866→0.910），生成侧无收益。每个改动都要生成侧验证。
- **开发集为"关"的开关，默认值测试必须 stub 掉 `load_dotenv`**，否则断言对象从"代码默认值"漂移成"本机 `.env` 的值"。
- **判定器自身不稳定**：同一份回答两次评分可差 1 级。单次评审数字是线索，不是结论。
- **长上下文会让判定器失准**：引用片段变长后它会定位不到法条，报"无依据"（曾因此误判 8 处）。
- **允许撤回结论，但要留痕**。已撤销的结论（"更多上下文诱发编造"）保留在报告里并写明错在哪一步——错误的方法论比错误的结论更值得记录。

**工程纪律**

- 改动配置必须让下游失效：`ImportReport` 记切片身份、索引签名记片数与算法版本、Embedding 缓存在换模型时拒绝复用。**这三个洞都真的漏过**，症状都是"改了配置而指标一模一样"。
- 发布前必须 `backend.flush()`：Chroma 每次 `add()` 立即落盘掩盖了这个缺失，换 FAISS 后发布的 generation 只有 manifest，每次查询都答"不足以回答"。
- **重复定义会静默覆盖**：曾有一个 `query()` 被类里更靠后的同名方法覆盖，改动静默失效。`tests/test_no_shadowed_definitions.py` 用 AST 扫描防止再发生。
- **判定"有没有证据"要比对来源文件**：同一准则多个年份版本的部分条文逐字相同，只比对文本会把 2014 版当 2017 版。

**协作与边界**

- 小步提交；**默认本地提交，用户明确要求时才 push**；**不自动合并到 main**。
- 通用流程不得依赖固定知识库 ID、型号、准则名或评测答案。
- 第三语料验收证明的是**导入与接口契约**，不是答案准确率。
- **v1 test 79 题此后不得再用于调参**；后续开发需新划分保留样本。
- 审阅者是**项目所有者而非独立领域审阅者**（`human_review_meta.reviewer_is_independent = false`），这不足以构成独立签署。

---

## 六、模型与服务商配置（实测自 `.env`）

| 用途 | 提供方 | 模型 |
| --- | --- | --- |
| Embedding | 阿里云 MaaS（compatible-mode） | `qwen3.7-text-embedding-flash`（1024 维） |
| Rerank | 阿里云 MaaS（**原生端点**，不是 compatible-mode） | `qwen3.7-text-rerank` |
| 生成 / LLM 重排 | DeepSeek | `deepseek-v4-flash` |

**两个实测坑**：

1. Embedding 在 `compatible-mode/v1` 下可用，**rerank 在该路径下全部 404**，必须走 `api/v1/services/rerank/text-rerank/text-rerank`。`rag.config.normalize_rerank_url` 会自动转换；`rerank_path_probe.py` 可定位路径。
2. 阿里云 Embedding 有**批大小上限**（20 被拒），默认 `EMBEDDING_BATCH_SIZE=10`。

先跑探针确认端点：

```powershell
.venv\Scripts\python.exe experiments\rag-baseline\evaluation\provider_probe.py
```

**换 Embedding 即索引失效，必须重建**；Embedding 缓存按模型指纹失效。

---

## 七、实验产物地图

`experiments/rag-baseline/evaluation/` 下：

| 目录 | 内容 |
| --- | --- |
| `retrieval_runs/` | 检索指标运行（`summary.json` + `samples.jsonl`，记录切片与 embedding 指纹） |
| `quality_runs/` | 评测脚本的答案与三套评分（`assistant_*` / `human_*` / `lead_*`） |
| `product_runs/` | **产品路径**的答案与人工评分（2026-10-11 新增） |
| `context_runs/`、`hybrid_runs/`、`rerank_runs/`、`topk_runs/`、`structure_runs/`、`compact_runs/`、`runs/`、`experiments/`、`provenance/` | 早期实验产物（历史证据，不改写） |

**报告按时间倒序读**：`PRODUCT_REVIEW_REPORT` → `HUMAN_REVIEW_REPORT` → `HELDOUT_TEST_REPORT` → `STRUCTURAL_CHUNKING_REPORT` → `CONTEXT_WIRING_REPORT` → `RERANK_PROVIDER_REPORT` → `PROMPT_CLARIFICATION_REPORT` → `ANSWER_QUALITY_REPORT` → `RETRIEVAL_MODES_REPORT` → 更早的 TOPK/CONTEXT/RERANK/SECTION/COMPACT/HYBRID/STRUCTURE/BASELINE。

**评分标准是 [QUALITY_RUBRIC.md](../experiments/rag-baseline/evaluation/QUALITY_RUBRIC.md)**：三维 0/1/2 + 6 类错误标签，动手评分前先读它。

---

## 八、本机环境限制（动手前必读 [ENVIRONMENT_WINDOWS.md](ENVIRONMENT_WINDOWS.md)）

- **Chroma 的原生 hnswlib 在本机崩溃**（约 96 个 1024 维向量即 `0xC0000005`），这是产品后端改为 FAISS 的直接原因；`tests/test_knowledge.py` 因此长期不运行。
- `.venv` 基座为 CPython 3.11.15（原 Anaconda 基座已删，`pyvenv.cfg` 已重指）。
- 跑测试必须注入 `tools/testenv/sitecustomize.py` 并加 `--basetemp`（见第二节）。
- 工作目录含中文用户名，`faiss.write_index` 会失败——已改为 `serialize_index` + Python 写字节。
- git 不在 PATH：`C:\software\Git\cmd\git.exe`。

---

## 九、数据集计数（2026-10-09 实测更正）

| 版本 | 语料 | dev | test |
| --- | --- | --- | --- |
| v1（已交付，所有实验都用它） | 售后 | 40 | 20 |
| v1 | 会计 | 82 | 59 |
| v2（候选，`draft_needs_domain_review`） | 售后 | 43 | 29 |
| v2 | 会计 | 85 | 68 |

读取用 `load_knowledge_base_dataset(..., version='v1'|'v2')` **显式指定**。文档里旧数字"225 道候选题为 dev128 / test97"与文件不符，已按实测更正。

---

# 归档：更早轮次的交接（保留以便追溯）

以下内容来自 2026-10-09 及更早的交接，**其中部分描述已被后续改动取代**（例如"会计 749 片段"现为 866、"尚未完成通用化"已由 ADR-007 完成、"97 题保留测试未执行"已于 2026-10-10 执行）。保留原文是因为其中的**失败记录**仍是证据。

- 独立复核修掉的问题清单（`GET /knowledge-bases` 500、静默丢弃文件、PDF 页码、`lines` 多算一行、失败残留 generation、检索竞态、未知库返回 200、近似恒真断言）——均已修复并有测试。
- 早期九题提示词实验：`COMPACT_REPORT.md`（输入 Token 43,807 → 20,907）等，其中的**失败案例仍有效**：ACC-Q098 把续段误读为列表不全；ACC-Q118 只见第五条前四项却猜条件总数为五（事实碰巧正确，不代表引用支持）；ACC-Q059 把"四类"排成五个并列项；Q043 单次生成波动不能归功于压缩。
- 早期环境恢复步骤（`/workspace/RAG` 路径、`docs/CLOUD_STARTUP.md`）已过时，以本文件第二节为准。
