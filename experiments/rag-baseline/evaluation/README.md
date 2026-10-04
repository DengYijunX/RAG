# 原始方案基线评测

> 最新状态：128 题开发集已完成基线、结构切片及混合检索对照，23 项评测工具测试通过。会计“精确定位 + 向量回退”候选的证据覆盖率为 96.77%，售后保留原方案；候选尚未接入应用，测试集未运行。见 [基线报告](BASELINE_REPORT.md)、[结构切片报告](STRUCTURE_REPORT.md) 和 [混合检索报告](HYBRID_REPORT.md)。

## 当前状态

v2候选评测集共225题：售后72题（dev 43/test 29），会计153题（dev 85/test 68）。每库新增12道资料缺失/工具边界题，各测试集拒答题从1道增加到10道。原有题目的集合归属未变，避免为了得到更好成绩移动已有题目。

这是资料对照初审后的 review-candidate，不是专家签署的最终gold。每题仍为 draft_needs_domain_review。人工审阅可填写 `v2/<知识库>/human_review.csv`；没有人工签名时不宣称人工复核完成。正式冻结时应核对拒答是否确无依据、题意及答案是否唯一、证据是否充分及可替代证据。

## 验证结果

- 原有RAG仓库19项测试此前通过。
- 基线工具 11 项、结构切片工具 3 项及混合检索工具 9 项测试通过：覆盖评分、向量校验、配置检查、真实 Chroma 跨库隔离、父条目去重、源文跨度、问题元数据路由和上下文预算等。
- 83个FAQ和1,859条会计证据全部映射到原始500/50切片，分别26和749片。完整语料覆盖全部证据。
- 确认v2的dev/test无共享证据，会计无共享准则家族；所有原有题目的划分保持不变。
- ChromaDB 0.6.3存在PostHog接口不兼容的遥测日志及Pydantic弃用警告；上述实际存储检索测试通过。未改应用源码或依赖声明。
- 真实模型调用通过：Embedding 为 qwen3.7-text-embedding-flash（1,024 维），LLM 为 deepseek-chat。先前域名绑定和认证问题已解决。

## 配置

已知 Embedding URL：
`https://llm-k5ve8qaaz1tn5w7b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`

runner优先读取环境的EMBEDDING_BASE_URL，未设置时使用以上用户指定值。模型名称必须显式配置EMBEDDING_MODEL，不能继承仓库的BAAI/bge-m3默认值。密钥只从环境或用户既有.env读取，不写入数据包、输出或脚本。

当前使用用户已配置的阿里云 Embedding 与 DeepSeek LLM 认证绑定。历史上曾发生旧域名绑定和裸主机名 URL 问题，已通过实际调用验证修复；不需要重复配置密钥。runner 仅将上述已授权阿里云裸主机名补全为完整 URL，其他无协议的 URL 提前报错。

只测检索需要Embedding密钥和模型；加 `--generate` 后还需要LLM_API_KEY。LLM默认采用仓库的DeepSeek URL及deepseek-chat，可用LLM_BASE_URL、LLM_MODEL覆盖。

## 命令

从仓库根目录执行；解释器使用已有项目虚拟环境。`run` 和 `preflight` 会调用外部模型。

```bash
# 不调用外部模型
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py prepare
.venv/bin/python -m unittest discover -s experiments/rag-baseline/evaluation -p 'test_*.py' -v

# 配置就绪后：小请求验证真实模型与向量维度
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py preflight
# 同时验证生成模型
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py preflight --generate

# 先在dev运行；每次创建新存储目录，不重置用户索引
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py run --kb smart-home --split dev --generate
.venv/bin/python experiments/rag-baseline/evaluation/baseline.py run --kb accounting --split dev --generate
# 省略--generate只评检索，不产生答案准确率
```

运行参数和评测版本确定后，才将 `--split dev` 改为 `--split test`。`--limit N`仅用于小样本流程诊断；结果manifest会标记是否覆盖完整集合，不能当作完整基线。

## 方法与限制

- 沿用原仓库加载器和切片函数，500字符/50重叠、Chroma默认L2距离、Top-4。Embedding使用用户指定部署，实际模型名进入manifest；不同模型结果不可冒充同模型的检索算法提升。
- 基线不做结构切片、混合检索、重排或版本过滤。结构切片实验使用独立脚本与输出，见对照报告。
- 两个知识库独立索引。每次run使用知识库名+时间戳+随机ID生成全新目录，绕开原应用进程级单例的串库风险。
- 只索引指定corpus/markdown中的文档，绝不索引题目、预期答案或审阅文件。生成提示词使用原仓库同样的系统指令与检索上下文格式；来源路径缩短为文件名，gold标签不会发送给LLM。
- `coverage_recall_at_4`定义为Top-4片段合并后覆盖至少80%规范化证据字符的gold证据比例。`coverage_mrr_at_4`是首次达到某条gold证据80%覆盖的前缀排名倒数。它们是可审计的字符覆盖代理指标，**不是人工相关性标注的通用Recall/MRR，也不是答案正确率**。重叠字符只计一次，错误文件或年份不会因文本相似被计入对应证据。
- 多证据题另报全部证据覆盖率；复杂场景与显式条号定位题分开统计。澄清和拒答题不计入检索覆盖均值，答案行为留人工评分。
- 覆盖短条文不一定证明答案正确；长条文中只找到关键句也可能已足够回答，却达不到80%。正式检索评测应进一步标注关键证据句和相关chunk，再与此代理指标对照。
- 不自动用关键词命中判答案正确，不让同一个生成模型自评后直接输出“准确率”。`answer_review.jsonl`提供问题、答案、预期行为和要点，评分字段保持null等待审阅。
- P50/P95来自串行请求、无预热，包含网络及API重试；样本量有限，不代表线上并发性能。
- `prepare`还检查单条证据在最大重叠4片内是否可达阈值，这只是单证据必要条件，不是多证据题联合Top-4的理想上限。

## 输出

`evaluation/runs/<知识库>/<运行ID>/` 包含manifest、独立Chroma目录、逐题results.jsonl、summary.json和answer_review.jsonl。异常运行标记failed并保留已完成题目；不把部分完成视为成功。HTTP错误只保存类型和状态，不保存请求头或可能含凭据的原始异常正文。

真实基线保存在 `runs/`，结构切片结果保存在 `experiments/`，典型回答审阅保存在 `structure_case_review.jsonl`。答案准确率和拒答准确率仍未评分，Token 记录及其范围见结构切片报告。Chroma 数据和查询向量缓存不进入 Git，已保存结果可直接审阅，重建索引需真实模型调用。
