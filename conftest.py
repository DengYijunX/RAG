"""Test-session setup.

The evaluation prototypes under ``experiments/rag-baseline/evaluation`` are
scripts, not an installed package, so their directory is put on ``sys.path``
for the tests that exercise them directly (BM25, RRF, experiment entry points).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'rag-baseline' / 'evaluation'))
