"""Read chunk records from a published index generation, either format.

Two publishers write different files:

- the application's ``build()`` writes ``manifest.json`` with ``chunk_records``;
- the evaluation baseline writes ``records.jsonl``, one JSON object per line,
  where the text field is named ``document``.

Expansion needs document order, which differs between the two, so a caller that
assumed one format would work on the application and fail on the evaluation
index (and vice versa). Supporting both is the point of this function.
"""
import json
from pathlib import Path

MANIFEST = 'manifest.json'
RECORDS = 'records.jsonl'


def load_chunk_records(generation_dir):
    """Return chunk records with ``id`` / ``source`` / ``content`` keys.

    Raises FileNotFoundError when neither file exists: a generation without
    chunk records cannot support expansion, and silently returning an empty list
    would make every hit look unexpandable.
    """
    directory = Path(generation_dir)
    manifest = directory / MANIFEST
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        records = payload.get('chunk_records')
        if records is None:
            records = payload.get('chunks')
        if records is None:
            raise KeyError(f'{manifest} 既没有 chunk_records 也没有 chunks')
        return [_normalize(record) for record in records]
    records_path = directory / RECORDS
    if records_path.is_file():
        records = [json.loads(line) for line in
                   records_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        return [_normalize(record) for record in records]
    raise FileNotFoundError(f'{directory} 下既没有 {MANIFEST} 也没有 {RECORDS}')


def _normalize(record):
    """Map a record to the ``id`` / ``source`` / ``content`` shape expansion uses."""
    source = record.get('source')
    if source is None:
        source = (record.get('metadata') or {}).get('source', '')
    content = record.get('content')
    if content is None:
        content = record.get('document', '')
    return {**record, 'source': source, 'content': content}
