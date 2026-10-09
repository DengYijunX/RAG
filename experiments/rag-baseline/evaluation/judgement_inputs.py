"""Load a quality run and print the judgement inputs, joinable by script.

Kept in the experiment directory (next to the other eval tools) so it can be run
the normal way: ``python experiments/rag-baseline/evaluation/judgement_inputs.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eval.datasets import load_knowledge_base_dataset

BASELINE = REPO / 'experiments' / 'rag-baseline'
RUNS = BASELINE / 'evaluation' / 'quality_runs'


def load_dataset_index():
    index = {}
    for knowledge_base in ('smart-home', 'accounting'):
        for question in load_knowledge_base_dataset(BASELINE, knowledge_base,
                                                    version='v2').questions:
            index[question.id] = {
                'knowledge_base': knowledge_base,
                'expected_action': question.metadata.get('expected_action', ''),
                'category': question.metadata.get('category', ''),
                'reference_points': question.metadata.get('expected_points') or [],
            }
    return index


def latest_run(knowledge_base):
    candidates = sorted(path for path in RUNS.iterdir()
                        if path.is_dir() and path.name.startswith(f'{knowledge_base}-'))
    if not candidates:
        raise SystemExit(f'没有 {knowledge_base} 的运行')
    return candidates[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description='打印逐题判断所需的全部输入')
    parser.add_argument('--kb', choices=['smart-home', 'accounting'], required=True)
    parser.add_argument('--snippet-chars', type=int, default=1200,
                        help='每条引用展示的字符数')
    parser.add_argument('--answer-chars', type=int, default=0, help='0 表示不截断')
    args = parser.parse_args(argv)

    run = latest_run(args.kb)
    records = [json.loads(line) for line in
               (run / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    index = load_dataset_index()
    print(f'# 运行 {run.name}，共 {len(records)} 题')
    for record in records:
        meta = index.get(record['question_id'], {})
        answer = record.get('answer') or ''
        if args.answer_chars:
            answer = answer[:args.answer_chars]
        print()
        print('=' * 96)
        print(f"[{record['question_id']}] 预期行为={meta.get('expected_action')} "
              f"类别={meta.get('category')}")
        print(f"问题：{record['question']}")
        for point in meta.get('reference_points') or []:
            print(f"  · 要点：{point}")
        print('-' * 96)
        print(answer)
        print('-' * 96)
        for citation in record.get('citations') or []:
            print(f"引用[{citation['citation']}] {citation.get('source')} "
                  f"@{citation.get('start')}-{citation.get('end')}")
            print((citation.get('content') or '')[:args.snippet_chars])
        print(f"助手初审={record.get('assistant_scores')} 标签="
              f"{record.get('assistant_error_labels')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
