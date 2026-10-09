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

# Fields a consumer needs that the baseline stores inside ``metadata``.
PROVENANCE_KEYS = ('source', 'document_id')


def load_chunk_records(generation_dir):
    """Return chunk records with ``id`` / ``source`` / ``content`` / ``document_id``.

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
    """Map a record to the shape consumers use.

    ``content`` is called ``document`` in the baseline's records, and
    ``source`` / ``document_id`` live under ``metadata`` there. Promotion is
    always from metadata if present, so an authoritative top-level value is
    never overwritten by a stale copy.
    """
    metadata = record.get('metadata') or {}
    normalized = {**record, 'source': record.get('source') or metadata.get('source', ''),
                  'content': record.get('content') or record.get('document', '')}
    for key in PROVENANCE_KEYS:
        if metadata.get(key) is not None:
            normalized[key] = metadata[key]
    return normalized
