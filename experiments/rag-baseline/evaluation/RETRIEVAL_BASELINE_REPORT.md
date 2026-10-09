# 纯向量检索基线（v1 开发集）

日期：2026-10-09（Asia/Shanghai）。分支 `work`。这是项目**第一份真实检索指标**，替代此前只有"证据覆盖率"的记录。

## 这份报告回答什么

给定 v1 开发集的题目，**纯向量 Top-K 检索**能不能把数据集标注的证据片段找回来、排在哪。

它**不回答**答案对不对，也不涉及生成模型。指标是检索覆盖，不是回答准确率。

## 方法

```powershell
$env:PYTHONPATH="C:\Users\AD钙\Desktop\Projects\RAG\tools\testenv"
.venv\Scripts\python.exe experiments\rag-baseline\evaluation\retrieval_baseline.py `
    --kb smart-home --split dev --k 4 --top-k 10
.venv\Scripts\python.exe experiments\rag-baseline\evaluation\retrieval_baseline.py `
    --kb accounting --split dev --k 4 --top-k 10
```

- **语料**：售后 5 份文档 / 26 片段；会计 59 份文档 / 749 片段（500/50 切片）
- **索引**：FAISS 精确内积（归一化向量 = 余弦），非近似索引，因此打分不含索引近似误差
- **Embedding**：配置中的真实提供方（1024 维）；向量按文本缓存，重跑不重复计费
- **问答标签**：售后用 `evidence.jsonl` 的 `D1-Sxx`；会计用 `articles.jsonl` 的 `CAS-…-Axxx`，与检索索引用同一套 id
- **对齐**：命中片段与标签按规范化文本匹配——先完全相等，再判断标签文本是否**包含**在片段内（真实情况多为此类，因为一个片段常跨标题与多段正文）；一个片段可同时命中多个标签，全部计入
- **产物**：`experiments/rag-baseline/evaluation/retrieval_runs/<UTC时间>-<hash>/`，含 `summary.json` 与逐题 `samples.jsonl`

## 结果（v1 dev，k=4，实际检索 Top-10）

| 指标 | 售后（35 道可判分） | 会计（67 道可判分） |
| --- | --- | --- |
| recall@4 | **0.9714** | **0.8930** |
| hit_rate@4 | 0.9714 | 0.9254 |
| precision@4 | 0.9714 | 0.9254 |
| ndcg@4 | 0.9714 | 0.9001 |
| MRR | 0.9714 | 0.9254 |
| 逐题 recall@4 满分 | 34 / 35 | 58 / 67 |
| 逐题 recall@4 零分 | 1 / 35 | 5 / 67 |
| 首命中槽位均值 | 1.76（74% 在第 1 位） | 1.47（81% 在第 1 位） |
| 首命中槽位 > 4 的题 | 4 | 3 |
| 未能判分的题 | 5 / 40 | 15 / 82 |

## 怎么读这些数字（重要）

1. **这是"找回标注证据"的比例，不是答案正确率。** 检索到正确片段不代表生成的回答正确；反之也成立。生成侧指标仍未实现。
2. **MRR 与 NDCG 在这套设置下信息量有限。** 对齐层只给"该题标注过的片段"计分，因此单标签题的 MRR 只能是 1.0 或 0.0；上表 MRR 与 hit_rate 数值相同正是这个原因。真正表达排序质量的是**首命中槽位**（1.47 / 1.76，即多数题第 1 名就命中）。
3. **precision@4 与 hit_rate@4 相同**，因为每题多数只有一个金标签，前 4 名里只要有一个命中，precision 也是 1。
4. **`resolution_coverage` 只有 0.09–0.11，这不是缺陷。** 语料中大量片段不承载任何标注证据（售后 26 个片段中 23 个含证据段落，其余 3 个不含；检索每题的 Top-10 里多数片段与该题标签无关），它们在指标里正确地不计分，但都会被记录在 `unresolved` 中，没有被隐藏。
5. **标注仍是草案。** 售后与会计题目的 `annotation_status` 均为 `draft_needs_human_review` / `draft_needs_domain_review`。本报告的数字依赖这些草案标注，人工复核后需要重跑。

## 失败案例（下一步的输入）

- 会计 **5 道零分题**、售后 **1 道零分题**：Top-10 里完全没有标注证据。需要逐题查看是标注过窄（证据其实在别处）还是检索确实失败。
- 两库各有 **3–4 道题首命中排在第 4 名之后**：召回存在但排序偏后，是重排（Rerank）的候选收益点。
- 会计的分区差异尚未细分（本标准 vs 跨版本比较题），明细在 `samples.jsonl` 中可按 `relevant_passage_ids` 与命中顺序逐题核对。

## 复现与边界

- 脚本与产物均已提交；重跑读取 Embedding 缓存，不产生新的模型调用。`--rebuild` 才会重新向量化。
- 语料或切片参数变化会改变片段边界，需要重建索引并重跑；本机无法使用 Chroma 的 HNSW 索引（见 [ENVIRONMENT_WINDOWS.md](../../../docs/ENVIRONMENT_WINDOWS.md)），故使用 FAISS 精确索引；产品默认后端仍是 Chroma，两者未做对比，不得据此宣称哪个更快或更准。
- **未执行**：v2 候选集（225 题）、保留测试集（v1 test 79 题）、生成侧指标、跨知识库对比。开发集调参后如需报告泛化，必须在保留测试集上另跑一次。
