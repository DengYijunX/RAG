"""Summarize completed real runs, preserving metric limitations and audit scope."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
runs={}
for kb in ('smart-home','accounting'):
    candidates=sorted((ROOT/'runs'/kb).glob('*/manifest.json'))
    if not candidates: raise RuntimeError('Missing real run: '+kb)
    p=candidates[-1]; m=json.loads(p.read_text())
    if m['status']!='completed': raise RuntimeError('Run is not complete: '+kb)
    rows=[json.loads(l) for l in p.with_name('results.jsonl').read_text().splitlines()]
    if len(rows)!=m['question_count'] or not m['full_split']: raise RuntimeError('Incomplete split: '+kb)
    runs[kb]={'path':p.parent,'manifest':m,'rows':rows,'summary':json.loads(p.with_name('summary.json').read_text())}

audit_specs=[
 ('accounting','ACC-Q114','missing_version_evidence','对比收入准则两个版本第四条时，只召回2006年目标证据，未能完成2017年部分的对照。','按准则编号、版本和条号增强召回，比较关键词检索与版本覆盖约束。'),
 ('accounting','ACC-Q116','missing_multi_clause_evidence','租赁综合题只覆盖第三十二条，缺少一般确认规则和现金流分类条款。','多条文综合查询需扩展候选并保证多个子问题的证据覆盖。'),
 ('accounting','ACC-Q118','off_target_retrieval','收入确认与计量问题两条目标证据均未召回，回答转向知识产权许可等局部规则。','增加关键词/语义融合并对照重排，避免局部特例替代一般规则。'),
 ('smart-home','Q040','partial_retrieval','跨四种设备的5GHz问题只覆盖X2/G2目标证据；回答未能给出X1/C1结论。','多对象问题需要覆盖约束或查询分解，比较更大候选集和重排。'),
 ('smart-home','Q043','unsupported_assumption_and_missing_evidence','遗漏售后申请步骤的证据；答案把未说明原因的设置键故障直接认定为非人为故障。','召回申请流程，并要求生成保留保修条件，不擅自认定故障原因。'),
 ('smart-home','Q051','missing_model_scope','问题未给门锁型号，回答直接采用X1蜂鸣规则，未注明适用型号或先澄清。','对缺少关键型号的提问增加澄清或限定适用范围。'),
 ('accounting','ACC-Q041','partial_article_retrieval','分部报告第五条只覆盖约44.6%的原文，回答仅列首项因素。','按章/条切片，较长条文使用父子片段或检索后扩展。'),
 ('accounting','ACC-Q059','partial_article_retrieval','2014年职工薪酬第二条仅覆盖约51.1%，未覆盖完整定义展开。','保留条文关系并在回答完整条文请求时扩展上下文。')]
audits=[]
for kb,qid,kind,observation,next_step in audit_specs:
    r=next(r for r in runs[kb]['rows'] if r['question_id']==qid)
    audits.append({'knowledge_base':kb,'question_id':qid,'reviewer':'assistant qualitative spot review; not expert score','issue':kind,'observation':observation,'proposed_experiment':next_step,'question':r['question'],'answer':r['answer'],'retrieved_ids':r['retrieved_ids'],'metrics':r['metrics']})
(ROOT/'baseline_case_review.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audits))

lines=['# 双知识库开发集基线报告','',f'生成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")}。', '',
'Embedding认证及生成模型认证已通过。以下来自完整真实API运行，两个知识库和存储目录独立。使用v2 review-candidate，尚无人工领域签署。测试集未运行，保留后续固定方案的评估用途。','',
'## 配置','',
'- Embedding：qwen3.7-text-embedding-flash，1024维。','- LLM：deepseek-chat。','- 原仓库切片：500字符，重叠50；纯向量Chroma L2 Top-4。未加入BM25、rerank、结构切片或版本过滤。','- 每个知识库内部串行，两库同时运行。P50/P95包含网络及重试，无预热；不是生产并发压力测试。','',
'## 完成情况和覆盖指标','',
'| 知识库 | 完成题数 | 参与证据评分 | 平均证据覆盖率@4 | 全部证据覆盖率 | 覆盖MRR@4 | P50秒 | P95秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for kb,label in [('smart-home','智能家居售后'),('accounting','企业会计准则')]:
    s=runs[kb]['summary']
    lines.append(f'| {label} | {s["completed_questions"]} | {s["retrieval_scored_questions"]} | {s["coverage_recall_at_4"]:.1%} | {s["all_evidence_coverage_rate"]:.1%} | {s["coverage_mrr_at_4"]:.3f} | {s["total_seconds_p50"]:.2f} | {s["total_seconds_p95"]:.2f} |')
lines.extend(['','“证据覆盖”指Top-4片段合并后覆盖至少80%的规范化目标证据字符。覆盖率按每题目标证据命中比例取平均；全部覆盖率要求每题全部证据都达到阈值；覆盖MRR是首次达到某条证据阈值的检索前缀排名倒数。重叠字符去重，其他文件/版本不冒充指定证据。','',
'这些是字符覆盖代理指标，不是人工相关性Recall/MRR，更不是答案准确率。长条文即使命中关键句仍可能未达到阈值；因此后续应补充关键证据句标注。澄清/拒答题不参与此项均值，回答行为仍待人工评分。','',
'## 分题型结果','', '| 知识库 | 类别 | 证据评分题数 | 平均覆盖率@4 |','|---|---|---:|---:|'])
for kb,data in runs.items():
    for cat,s in data['summary']['by_category'].items():
        value='不适用，待行为评审' if s['coverage_recall_at_4'] is None else f'{s["coverage_recall_at_4"]:.1%}'
        lines.append(f'| {kb} | {cat} | {s["retrieval_scored_count"]} | {value} |')
lines.extend(['','## 定性抽查案例','', '以下8例为助手对已生成答案与证据的定性复核，不构成全体答案的人工评分。完整答案和检索片段见对应run。',''])
for a in audits:
    lines.extend([f'- **{a["knowledge_base"]} / {a["question_id"]}**：{a["observation"]} 建议：{a["proposed_experiment"]}'])
lines.extend(['','## 下一步实验','',
'1. 优先实验按FAQ/条文边界切片，对长条文增加父子关系和上下文扩展；保持Embedding模型及问题集合不变。',
'2. 针对多型号、多证据问题评估候选数与覆盖约束；再单独对比BM25+向量融合及rerank，不同时改所有变量。',
'3. 增加生成端型号澄清与条件约束；售后故障原因未知时不能自动承诺免费维修。',
'4. 按人工审阅表核对答案、引用和应答行为，形成可用的答案质量标签。会计版本约束须单独评估，不将不同年份差异直接当作矛盾。',
'5. 选定改进配置后再评估保留测试集。当前只有开发集结果，不能用于声称对未知真实用户问题具有同等效果。','',
'## 运行产物',''])
for kb,data in runs.items():
    rel=data['path'].relative_to(ROOT)
    lines.extend([f'- {kb}：[配置与文件哈希]({rel}/manifest.json)、[逐题结果]({rel}/results.jsonl)、[汇总指标]({rel}/summary.json)、[人工答案审阅表]({rel}/answer_review.jsonl)。'])
lines.extend(['','原仓库源码和依赖声明未修改。评测工具及产物位于仓库外的 /workspace/rag-project-prep/evaluation。已记录模型、Python/依赖版本、代码提交、输入哈希和独立索引目录。','',
'Embedding索引阶段的Token用量未落盘，本次不报告全流程Token总量或货币成本。生成返回的逐题Token用量保留在results.jsonl。模型输出没有自动转换为“准确率”，相关字段仍为null。',''])
(ROOT/'BASELINE_REPORT.md').write_text('\n'.join(lines))
status=json.loads((ROOT/'STATUS.json').read_text())
status.update(actual_model_baseline='completed_dev_both_knowledge_bases',completed_questions=sum(len(v['rows']) for v in runs.values()),blocking_action=None,missing=[],human_domain_review='pending; qualitative spot review saved separately',performance_metrics={kb:data['summary'] for kb,data in runs.items()})
(ROOT/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(ROOT/'BASELINE_REPORT.md')
