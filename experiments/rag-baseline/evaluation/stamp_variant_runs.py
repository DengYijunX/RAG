"""Label the chunking-variant runs, which were produced without recording which they were.

A run that does not say which configuration produced it is not evidence. These
four were produced by the chunking comparison before the label was recorded, so
the label is restored from the session's own record of what was run when.

Usage: ``python experiments/rag-baseline/evaluation/stamp_variant_runs.py``
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'experiments' / 'rag-baseline' / 'evaluation' / 'retrieval_runs'

# Session record of the chunking comparison, in the order it was run.
VARIANTS = {
    '20261010T085954Z-8a17302d': ('structure500', 500, True),
    '20261010T090331Z-8a17302d': ('structure1000', 1000, True),
    '20261010T090600Z-8a17302d': ('structure500', 500, True),
    '20261010T090801Z-8a17302d': ('structure1000', 1000, True),
}


def main():
    stamped = 0
    for name, (label, size, structure_aware) in VARIANTS.items():
        directory = RUNS / name
        path = directory / 'summary.json'
        if not path.is_file():
            print(f'跳过 {name}：没有 summary.json')
            continue
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary['chunking'] = {'variant': label, 'chunk_size': size,
                               'structure_aware': structure_aware}
        summary['chunking_source'] = 'stamped_from_session_record'
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
        stamped += 1
    print(f'已标注 {stamped} 个变体运行')


if __name__ == '__main__':
    main()
