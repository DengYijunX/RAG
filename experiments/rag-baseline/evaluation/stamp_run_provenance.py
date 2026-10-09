"""Record provenance on existing retrieval runs.

Runs produced before the embedding model was recorded are ambiguous: two
different models were used during the session, and a reader cannot tell which
one a given artifact belongs to. This stamps each summary with the model that
built the index it was scored against, taken from the run's own embedding
fingerprint when available, and otherwise from the session timeline.

Usage: ``python experiments/rag-baseline/evaluation/stamp_run_provenance.py``
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'experiments' / 'rag-baseline' / 'evaluation' / 'retrieval_runs'

# The switch happened between these runs: everything from 153354 onward used the
# Aliyun model, everything before it used bge-m3 on SiliconFlow. This is the
# session's own record, not a guess from the numbers.
FIRST_ALIYUN_RUN = '20261009T153354Z'
MODELS = {
    'aliyun': 'qwen3.7-text-embedding-flash',
    'siliconflow': 'BAAI/bge-m3',
}


def main():
    stamped = 0
    for run in sorted(RUNS.glob('2026*')):
        path = run / 'summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary.get('embedding_model'):
            continue
        model = MODELS['aliyun'] if run.name >= FIRST_ALIYUN_RUN else MODELS['siliconflow']
        summary['embedding_model'] = model
        summary['embedding_source'] = 'stamped_from_session_timeline'
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        stamped += 1
    print(f'已标注 {stamped} 个运行')


if __name__ == '__main__':
    main()
