"""Export human review sheets for a generation-side review run, and import them back.

The rubric requires domain reviewers to sign off per question, with assistant
scores kept separate and never reported as human accuracy. This script produces
two views of one run:

- ``review.md``  — question, generated answer, the cited snippets, the dataset's
  reference points and the assistant's first pass, for reading;
- ``review.csv`` — the same content in one row per question, with empty
  ``human_*`` columns to fill in.

``--import`` reads a filled sheet back, validates every value against the
rubric, and writes ``human_summary.json``. An out-of-range score or an unknown
error label is refused loudly: a typo must not silently become a data point.

    .venv/Scripts/python experiments/rag-baseline/evaluation/review_sheet.py \
        --kb accounting --export
    .venv/Scripts/python experiments/rag-baseline/evaluation/review_sheet.py \
        --run quality_runs/accounting-20261010T000000Z --import
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_review import DIMENSIONS, ERROR_LABELS, SEVERITIES
from eval.datasets import load_knowledge_base_dataset

BASELINE = REPO / 'experiments' / 'rag-baseline'
QUALITY_RUNS = BASELINE / 'evaluation' / 'quality_runs'
SAMPLE = BASELINE / 'evaluation' / 'quality_sample.json'

CSV_COLUMNS = [
    'question_id', 'split', 'category', 'expected_action',
    'question', 'reference_points', 'assistant_answer', 'cited_snippets',
    'assistant_key_points', 'assistant_citation_faithfulness', 'assistant_behavior_match',
    'assistant_error_labels', 'assistant_justification',
] + [f'human_{dimension}' for dimension in DIMENSIONS] + ['human_error_labels', 'human_note']


def latest_run(knowledge_base):
    candidates = sorted(path for path in QUALITY_RUNS.iterdir()
                        if path.is_dir() and path.name.startswith(f'{knowledge_base}-'))
    if not candidates:
        raise SystemExit(f'没有 {knowledge_base} 的审阅产物，请先运行 answer_quality_review.py')
    return candidates[-1]


def resolve_run(value):
    path = Path(value)
    if not path.is_absolute():
        for candidate in (Path.cwd() / path, BASELINE / 'evaluation' / path, QUALITY_RUNS / path.name):
            if candidate.is_dir():
                return candidate
    if path.is_dir():
        return path
    raise SystemExit(f'找不到运行目录：{value}')


def load_context(knowledge_base, version):
    dataset = load_knowledge_base_dataset(BASELINE, knowledge_base, version=version)
    return {question.id: question for question in dataset.questions}


def snippet_block(citations):
    lines = []
    for citation in citations or []:
        location = f"{citation.get('start')}-{citation.get('end')}"
        lines.append(f"[{citation['citation']}] {citation.get('source', '')} @{location}")
        lines.append(citation.get('content', ''))
        lines.append('')
    return '\n'.join(lines).strip()


def rows_for(run, knowledge_base, version):
    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    questions = load_context(knowledge_base, version)
    rows = []
    for record in records:
        question = questions.get(record['question_id'])
        metadata = question.metadata if question else {}
        scores = record.get('assistant_scores') or {}
        rows.append({
            'question_id': record['question_id'],
            'split': question.split if question else '',
            'category': metadata.get('category', ''),
            'expected_action': record.get('expected_action') or metadata.get('expected_action', ''),
            'question': record.get('question', ''),
            'reference_points': '；'.join(metadata.get('expected_points') or []),
            'assistant_answer': record.get('answer', ''),
            'cited_snippets': snippet_block(record.get('citations')),
            'assistant_key_points': scores.get('key_points', 'invalid'),
            'assistant_citation_faithfulness': scores.get('citation_faithfulness', 'invalid'),
            'assistant_behavior_match': scores.get('behavior_match', 'invalid'),
            'assistant_error_labels': '；'.join(record.get('assistant_error_labels') or []),
            'assistant_justification': record.get('assistant_justification', ''),
            **{f'human_{dimension}': '' for dimension in DIMENSIONS},
            'human_error_labels': '',
            'human_note': '',
        })
    return rows


def export(run, knowledge_base, version, target=None):
    rows = rows_for(run, knowledge_base, version)
    target = Path(target) if target else run
    target.mkdir(parents=True, exist_ok=True)

    csv_path = target / 'review.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        f'# 人工审阅表：{knowledge_base}（{version}）', '',
        f'运行目录：`{run.name}`', '',
        '填写规则：每个维度给 0 / 1 / 2，或 `na`（澄清、拒答等标准规定不适用的情形）。'
        '错误标签只能从标准给出的 6 类里选，多个用「、」分隔，没有就留空。',
        '',
        '待填写的表格：[review.csv](review.csv)（用 Excel 或文本编辑器打开均可）。', '',
    ]
    for row in rows:
        markdown += [
            f"## {row['question_id']}（{row['category']} · 预期 {row['expected_action']}）", '',
            f"**问题**：{row['question']}", '',
            f"**参考要点**：{row['reference_points'] or '（无）'}", '',
            f"**模型回答**：\n\n{row['assistant_answer'] or '（空）'}", '',
            '**引用片段**：', '', '```', row['cited_snippets'] or '（无引用）', '```', '',
            f"**助手初审**：完整性 {row['assistant_key_points']} / "
            f"引用忠实性 {row['assistant_citation_faithfulness']} / "
            f"行为匹配 {row['assistant_behavior_match']}"
            + (f"；标签 {row['assistant_error_labels']}" if row['assistant_error_labels'] else ''),
            '', f"**助手理由**：{row['assistant_justification'] or '（无）'}", '',
            '**人工评分**：完整性 ____ / 引用忠实性 ____ / 行为匹配 ____；'
            '错误标签 ____；备注 ____', '', '---', '',
        ]
    (target / 'review.md').write_text('\n'.join(markdown), encoding='utf-8')
    print(f'已导出 {len(rows)} 题：')
    print(f'  表格 {csv_path}')
    print(f'  阅读版 {target / "review.md"}')
    return 0


def parse_score(raw, field):
    value = (raw or '').strip().lower()
    if value in {'', 'na', 'n/a', '不适用'}:
        return 'na' if value else None
    if value in {'0', '1', '2'}:
        return int(value)
    raise SystemExit(f'{field} 的值无法识别：{raw!r}（只接受 0 / 1 / 2 / na）')


def import_sheet(run, knowledge_base, version, csv_path=None):
    csv_path = Path(csv_path) if csv_path else run / 'review.csv'
    if not csv_path.is_file():
        raise SystemExit(f'找不到审阅表：{csv_path}')
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    missing_columns = [column for column in CSV_COLUMNS if column not in (rows[0] if rows else {})]
    if missing_columns:
        raise SystemExit(f'审阅表缺少列：{missing_columns}')

    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {record['question_id']: record for record in records}
    reviewed, incomplete = 0, []
    for row in rows:
        question_id = row['question_id']
        if question_id not in by_id:
            raise SystemExit(f'审阅表里的题目不在本次运行中：{question_id}')
        scores = {}
        for dimension in DIMENSIONS:
            scores[dimension] = parse_score(row.get(f'human_{dimension}'), f'{question_id}.{dimension}')
        labels_raw = (row.get('human_error_labels') or '').strip()
        labels = [part.strip() for part in labels_raw.replace('、', ',').replace('；', ',')
                  .replace(';', ',').split(',') if part.strip()]
        unknown = [label for label in labels if label not in ERROR_LABELS]
        if unknown:
            raise SystemExit(f'{question_id} 出现标准外的错误标签：{unknown}；'
                             f'允许：{list(ERROR_LABELS)}')
        filled = [value for value in scores.values() if value is not None]
        if not filled:
            incomplete.append(question_id)
            continue
        by_id[question_id]['human_scores'] = scores
        by_id[question_id]['human_error_labels'] = labels
        by_id[question_id]['human_note'] = (row.get('human_note') or '').strip()
        by_id[question_id]['human_reviewed'] = all(value is not None for value in scores.values())
        reviewed += 1

    with open(run / 'records.jsonl', 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def rate(dimension, value):
        scored = [r for r in records if r.get('human_scores', {}).get(dimension) not in (None, 'na')]
        hits = [r for r in scored if r['human_scores'][dimension] == value]
        return {'numerator': len(hits), 'denominator': len(scored),
                'rate': len(hits) / len(scored) if scored else None}

    labels_count = {}
    for record in records:
        for label in record.get('human_error_labels') or []:
            labels_count[label] = labels_count.get(label, 0) + 1
    summary = {
        'knowledge_base': knowledge_base, 'dataset_version': version, 'run': run.name,
        'human_reviewed': reviewed, 'incomplete': incomplete,
        'human_full_marks': {dimension: rate(dimension, 2) for dimension in DIMENSIONS},
        'human_error_labels': labels_count,
        'note': ('人工评分，与助手初审分开记录；样本为定向诊断选样，不估计总体准确率。'
                 '未填写的题目不计入任何比率。'),
    }
    (run / 'human_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n已写入 {run / "human_summary.json"}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='导出/回收人工审阅表')
    parser.add_argument('--kb', choices=['smart-home', 'accounting'])
    parser.add_argument('--run', help='运行目录名或路径（默认取最新）')
    parser.add_argument('--version', default='v2')
    parser.add_argument('--import', dest='do_import', action='store_true',
                        help='回收已填写的人工审阅结果')
    parser.add_argument('--csv', help='指定审阅表路径')
    args = parser.parse_args(argv)

    if args.run:
        run = resolve_run(args.run)
        knowledge_base = args.kb or run.name.split('-')[0]
    else:
        if not args.kb:
            raise SystemExit('请给出 --kb 或 --run')
        run = latest_run(args.kb)
        knowledge_base = args.kb

    if args.do_import:
        return import_sheet(run, knowledge_base, args.version, args.csv)
    return export(run, knowledge_base, args.version, run)


if __name__ == '__main__':
    raise SystemExit(main())
