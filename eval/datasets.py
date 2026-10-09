"""Ground-truth definitions shared by the dataset, metric and runner layers.

WeKnora defines ground truth in five Parquet files whose join key is
``query_id``; a sample carries ``question``, ``relevant_passage_ids`` and a
reference ``answer``, which lets retrieval errors and generation errors be
scored separately. This project keeps the same split but stores it in JSONL,
matching the format the existing dev/test splits already use.
"""
from dataclasses import dataclass, field


@dataclass
class Passage:
    """One retrievable unit the dataset refers to.

    ``passage_id`` is the dataset's own identifier. It is deliberately not a
    chunk id: chunking may change while the ground truth stays fixed, so the
    mapping between the two is resolved at run time (see ``eval.runner``).
    """
    passage_id: str
    text: str = ''
    document_id: str = ''
    metadata: dict = field(default_factory=dict)


@dataclass
class GroundTruth:
    """One graded question plus the evidence that should answer it."""
    query_id: str
    question: str
    relevant_passage_ids: list = field(default_factory=list)
    answer: str = ''
    split: str = 'dev'
    metadata: dict = field(default_factory=dict)


def load_dataset(path, split=None):
    """Read a JSONL ground-truth file.

    Expected fields per line: ``query_id``, ``question``,
    ``relevant_passage_ids``, ``answer``; extras go to ``metadata``.
    ``split`` filters on the ``split`` field when given.

    TODO: implement, then cover with a fixture file and a schema test that
    rejects duplicate ``query_id`` and unknown fields.
    """
    raise NotImplementedError('eval.datasets.load_dataset 尚未实现；本次只落地目录与接口占位')


def align_passages(ground_truth, retrieved, *, by='text'):
    """Map retrieved chunks back to dataset passage ids.

    WeKnora matches retrieved text against ground-truth passages because
    chunk ids and passage ids live in different id spaces. Matching by text
    is fragile for rewritten or truncated content, so the planned order is:
    explicit ``source_passage_id`` metadata first, normalized text second,
    and a reported failure when neither resolves.

    TODO: implement after the importer carries a passage id through to the
    chunk record.
    """
    raise NotImplementedError('eval.datasets.align_passages 尚未实现')
