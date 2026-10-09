"""Merge the lead's judgement into the review runs and compare it with the assistant.

The rubric requires assistant scores and reviewer scores to be stored separately
and never conflated. This script:

1. validates every judgement against the rubric's closed sets (a bad value is an
   error, not a data point);
2. writes the judgement into ``records.jsonl`` as ``lead_scores`` /
   ``lead_error_labels``, deliberately **not** into the ``human_*`` fields —
   an AI judgement is not a human sign-off;
3. reports how far the assistant first pass agreed, per question and per
   dimension, so the disagreement can be inspected rather than averaged away.

    python experiments/rag-baseline/evaluation/merge_judgement.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'rag-baseline' / 'evaluation'))

from answer_review import DIMENSIONS, ERROR_LABELS

BASELINE = REPO / 'experiments' / 'rag-baseline'
RUNS = BASELINE / 'evaluation' / 'quality_runs'
JUDGEMENT = RUNS / 'lead_judgement.jsonl'


def load_judgement(path):
    if not path.is_file():
        raise SystemExit(f'找不到判断文件：{path}')
    entries = {}
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f'第 {lineno} 行不是合法 JSON：{exc}') from None
        identifier = entry.get('question_id')
        if not identifier:
            raise SystemExit(f'第 {lineno} 行缺少 question_id')
        if identifier in entries:
            raise SystemExit(f'{identifier} 重复出现')
        for dimension in DIMENSIONS:
            value = entry.get(dimension)
            if isinstance(value, str) and value.strip().lower() == 'na':
                entry[dimension] = 'na'
            elif value not in (0, 1, 2):
                raise SystemExit(f'{identifier}.{dimension} 取值非法：{value!r}（只接受 0/1/2/na）')
        labels = entry.get('error_labels') or []
        unknown = [label for label in labels if label not in ERROR_LABELS]
        if unknown:
            raise SystemExit(f'{identifier} 出现标准外标签：{unknown}')
        entry['error_labels'] = sorted(set(labels))
        entries[identifier] = entry
    return entries


def agreement(records, knowledge_base):
    """Compare the lead judgement with the assistant first pass."""
    paired = []
    for record in records:
        lead = record.get('lead_scores') or {}
        assistant = record.get('assistant_scores') or {}
        if not lead or not assistant:
            continue
        per_dimension = {}
        for dimension in DIMENSIONS:
            left, right = lead.get(dimension), assistant.get(dimension)
            per_dimension[dimension] = {
                'lead': left, 'assistant': right,
                'match': left is not None and left == right,
                'both_scored': left not in (None, 'na') and right not in (None, 'na'),
            }
        lead_labels = set(record.get('lead_error_labels') or [])
        assistant_labels = set(record.get('assistant_error_labels') or [])
        paired.append({
            'question_id': record['question_id'],
            'dimensions': per_dimension,
            'lead_labels': sorted(lead_labels),
            'assistant_labels': sorted(assistant_labels),
            'labels_only_assistant': sorted(assistant_labels - lead_labels),
            'labels_only_lead': sorted(lead_labels - assistant_labels),
        })
    comparable = [p for p in paired
                  if all(p['dimensions'][d]['both_scored'] for d in DIMENSIONS)]
    matches = sum(1 for p in comparable
                  if all(p['dimensions'][d]['match'] for d in DIMENSIONS))
    per_dimension_rate = {}
    for dimension in DIMENSIONS:
        scored = [p for p in paired if p['dimensions'][dimension]['both_scored']]
        hits = [p for p in scored if p['dimensions'][dimension]['match']]
        per_dimension_rate[dimension] = {
            'numerator': len(hits), 'denominator': len(scored),
            'rate': len(hits) / len(scored) if scored else None}
    return {
        'knowledge_base': knowledge_base,
        'questions': len(paired),
        'fully_comparable': len(comparable),
        'all_three_match': matches,
        'per_dimension_match': per_dimension_rate,
        'label_disagreements': [p for p in paired
                                if p['labels_only_assistant'] or p['labels_only_lead']],
        'note': ('领队判断与助手初审的一致性核对。助手评分不是人工签署，'
                 '此处的一致性也不等于正确率。'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='合并领队判断并核对与助手初审的一致性')
    parser.add_argument('--judgement', default=str(JUDGEMENT))
    args = parser.parse_args(argv)

    entries = load_judgement(Path(args.judgement))
    overall = {}
    for run in sorted(path for path in RUNS.iterdir() if path.is_dir()):
        records = [json.loads(line) for line in
                   (run / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        matched = 0
        for record in records:
            entry = entries.get(record['question_id'])
            if entry is None:
                continue
            record['lead_scores'] = {dimension: entry[dimension] for dimension in DIMENSIONS}
            record['lead_error_labels'] = entry['error_labels']
            record['lead_note'] = entry.get('note', '')
            record['lead_source'] = 'ai_lead_judgement_not_human_review'
            matched += 1
        if not matched:
            continue
        with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

        report = agreement(records, run.name.split('-')[0])
        (run / 'agreement_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        def rate(dimension, value):
            scored = [r for r in records
                      if (r.get('lead_scores') or {}).get(dimension) not in (None, 'na')]
            hits = [r for r in scored if r['lead_scores'][dimension] == value]
            return {'numerator': len(hits), 'denominator': len(scored),
                    'rate': len(hits) / len(scored) if scored else None}

        labels_count = {}
        for record in records:
            for label in record.get('lead_error_labels') or []:
                labels_count[label] = labels_count.get(label, 0) + 1
        summary = {
            'knowledge_base': report['knowledge_base'], 'run': run.name,
            'judged_by': 'AI lead judgement (not human review)',
            'judged': matched, 'questions': len(records),
            'lead_full_marks': {dimension: rate(dimension, 2) for dimension in DIMENSIONS},
            'lead_error_labels': labels_count,
            'agreement_with_assistant': {
                'fully_comparable': report['fully_comparable'],
                'all_three_match': report['all_three_match'],
                'per_dimension_match': report['per_dimension_match']},
            'note': ('该评分由 AI 领队作出，写入 lead_* 字段，**不占用 human_* 字段**；'
                     '人工签署仍为空。一致性核对见 agreement_report.json。'),
        }
        (run / 'lead_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        overall[run.name] = summary

    if not overall:
        raise SystemExit('没有任何运行匹配到判断文件')
    for name, summary in overall.items():
        print(f'== {name}')
        for dimension, value in summary['lead_full_marks'].items():
            print(f'   {dimension:24} 满分 {value["numerator"]}/{value["denominator"]}'
                  f' = {value["rate"]}')
        agreement_part = summary['agreement_with_assistant']
        print(f'   与助手初审（三维可比的 {agreement_part["fully_comparable"]} 题中，'
              f'三项全同 {agreement_part["all_three_match"]} 题）')
        for dimension, value in agreement_part['per_dimension_match'].items():
            print(f'     {dimension:24} 一致 {value["numerator"]}/{value["denominator"]}'
                  f' = {value["rate"]}')
        print(f'   错误标签：{summary["lead_error_labels"]}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
