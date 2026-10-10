"""Record a human review of a quality run and compare it with the assistant pass.

This is the step the rubric requires and the project had never completed: a
reviewer's scores, stored separately from the assistant's, with the provenance of
the review itself recorded — who reviewed, on what basis, and what it does *not*
establish. An owner confirming an AI's assessment is not independent domain
review, and the artefact has to say so rather than let a later reader assume
otherwise.

It also reports agreement per dimension, including the direction of every
disagreement, because the count alone hides whether the assistant runs strict or
lenient.

    python experiments/rag-baseline/evaluation/human_review.py \
        --run accounting-20261010T094718Z --scores human_scores.json
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

RUNS = REPO / 'experiments' / 'rag-baseline' / 'evaluation' / 'quality_runs'

#: How this review came about. Recorded verbatim so nobody later reads it as
#: independent domain sign-off.
DEFAULT_PROVENANCE = {
    'reviewer': 'project owner',
    'reviewer_is_independent': False,
    'basis': ('owner reviewed the answers presented in the session and confirmed them; '
              'scores were assigned by the AI lead applying QUALITY_RUBRIC.md and '
              'accepted by the owner, except the four items the owner was asked to '
              'adjudicate explicitly'),
    'sample': '20-question predeclared diagnostic sample (not a random sample)',
    'estimates_population': False,
}


def load_scores(path):
    """Validate a score file into ``{question_id: entry}``.

    Every value is checked against the rubric's closed sets: a typo must fail
    loudly rather than become a published number.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'找不到评分文件：{path}')
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
            if isinstance(value, str) and value.strip().lower() in {'na', 'n/a'}:
                entry[dimension] = 'na'
            elif value not in (0, 1, 2):
                raise SystemExit(f'{identifier}.{dimension} 取值非法：{value!r}（只接受 0/1/2/na）')
        labels = entry.get('error_labels') or []
        unknown = [label for label in labels if label not in ERROR_LABELS]
        if unknown:
            raise SystemExit(f'{identifier} 出现标准外标签：{unknown}；允许 {list(ERROR_LABELS)}')
        entry['error_labels'] = sorted(set(labels))
        entries[identifier] = entry
    if not entries:
        raise SystemExit('评分文件为空')
    return entries


def apply_review(run, entries, provenance):
    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    known = {record['question_id'] for record in records}
    unknown = sorted(set(entries) - known)
    if unknown:
        raise SystemExit(f'评分里的题目不在本次运行中：{unknown}')
    applied = 0
    for record in records:
        entry = entries.get(record['question_id'])
        if entry is None:
            continue
        record['human_scores'] = {dimension: entry[dimension] for dimension in DIMENSIONS}
        record['human_error_labels'] = entry['error_labels']
        record['human_note'] = entry.get('note', '')
        record['human_reviewed'] = True
        record['human_review_meta'] = dict(provenance)
        applied += 1
    with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return records, applied


def summarise(records, provenance):
    def block(field, dimension, value):
        scored = [r for r in records
                  if (r.get(field) or {}).get(dimension) not in (None, 'na')
                  and (r.get(field) or {}).get(dimension) is not None]
        hits = [r for r in scored if r[field][dimension] == value]
        return {'numerator': len(hits), 'denominator': len(scored),
                'rate': len(hits) / len(scored) if scored else None}

    review_field = 'human_scores'
    labels = {}
    for record in records:
        for label in record.get('human_error_labels') or []:
            labels[label] = labels.get(label, 0) + 1

    agreement, disagreements = {}, []
    for dimension in DIMENSIONS:
        pairs = [(r.get('human_scores', {}).get(dimension),
                  r.get('assistant_scores', {}).get(dimension), r['question_id'])
                 for r in records]
        comparable = [p for p in pairs if p[0] not in (None, 'na') and p[1] not in (None, 'na')]
        matches = [p for p in comparable if p[0] == p[1]]
        agreement[dimension] = {'numerator': len(matches), 'denominator': len(comparable),
                                'rate': len(matches) / len(comparable) if comparable else None}
        for human, assistant, identifier in comparable:
            if human != assistant:
                disagreements.append({
                    'question_id': identifier, 'dimension': dimension,
                    'human': human, 'assistant': assistant,
                    'direction': 'assistant_stricter' if assistant < human else 'assistant_lenient'})

    full = [r for r in records if r.get('human_reviewed')]
    return {
        'human_reviewed': len(full), 'questions': len(records),
        'human_full_marks': {dimension: block(review_field, dimension, 2) for dimension in DIMENSIONS},
        'human_error_labels': labels,
        'agreement_with_assistant': agreement,
        'disagreements': disagreements,
        'provenance': dict(provenance),
        'note': ('人工评分（项目所有者确认）与助手初审分开存储；一致性只说明两者是否一致，'
                 '不构成准确率。样本为定向诊断选样，不估计总体准确率。'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='录入人工评审并与助手初审核对')
    parser.add_argument('--run', required=True, help='运行目录名')
    parser.add_argument('--scores', required=True, help='JSONL 评分文件')
    parser.add_argument('--out', default=None, help='汇总输出路径（默认写入运行目录）')
    args = parser.parse_args(argv)

    run = RUNS / args.run
    if not run.is_dir():
        raise SystemExit(f'找不到运行目录：{run}')
    entries = load_scores(args.scores)
    records, applied = apply_review(run, entries, DEFAULT_PROVENANCE)
    summary = summarise(records, DEFAULT_PROVENANCE)
    target = Path(args.out) if args.out else run / 'human_summary.json'
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'已录入 {applied}/{len(records)} 题')
    for dimension, value in summary['human_full_marks'].items():
        rate = value['rate']
        print(f"  人工 {dimension:24} {value['numerator']}/{value['denominator']}"
              + (f' = {rate:.3f}' if rate is not None else ''))
    print('  与助手初审一致性：')
    for dimension, value in summary['agreement_with_assistant'].items():
        rate = value['rate']
        print(f"    {dimension:24} {value['numerator']}/{value['denominator']}"
              + (f' = {rate:.3f}' if rate is not None else ''))
    if summary['disagreements']:
        print('  分歧：')
        for item in summary['disagreements']:
            print(f"    {item['question_id']:11} {item['dimension']:22} "
                  f"人工 {item['human']} vs 助手 {item['assistant']}（{item['direction']}）")
    else:
        print('  无分歧')
    print(f'\n产物：{target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
