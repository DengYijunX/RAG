"""Generic document import for a knowledge base.

The importer only uses the file itself: it derives a stable document identity,
a content hash, the raw character span of every chunk and, for Markdown, the
heading ancestry. It never reads evaluation fixtures (``evidence.jsonl``,
``articles.jsonl``), gold labels or pre-generated metadata, so a brand new
corpus is imported by pointing at a directory.

Positions are reported only when the parser can prove them. PDF text is
extracted per page without reliable character offsets, so those records carry
page numbers and ``positions: false`` instead of invented spans.
"""
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path

from rag.loader import SUPPORTED_SUFFIXES, load_file
from rag.splitter import split
from rag.structural_split import VERSION as CHUNKING_VERSION
from rag.structural_split import split_text
from rag.structure import locate_chunks, parse_structure

MANIFEST_VERSION = 1


class ImportError_(Exception):
    """Raised when a source cannot be imported without inventing data."""


class UnsupportedSource(ImportError_):
    pass


@dataclass
class DocumentInput:
    """A file selected for import, before any parsing."""
    path: Path
    relative: str
    media_type: str
    size: int
    sha256: str


@dataclass
class ImportReport:
    """Import outcome for one knowledge base."""
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    #: Identity of the chunking that produced ``chunks``.
    chunking: dict = field(default_factory=dict)

    def as_dict(self):
        return {'manifest_version': MANIFEST_VERSION,
                'chunking': self.chunking,
                'documents': [asdict(d) if is_dataclass(d) else d for d in self.documents],
                'chunks': self.chunks,
                'skipped': self.skipped,
                'document_count': len(self.documents),
                'chunk_count': len(self.chunks)}


def media_type_for(path):
    suffix = Path(path).suffix.lower()
    return {'.md': 'text/markdown', '.txt': 'text/plain', '.pdf': 'application/pdf'}.get(suffix)


def enumerate_sources(directory, glob_pattern='**/*'):
    """Deterministic list of importable files; unsupported files are skipped."""
    available, _ = scan_sources(directory, glob_pattern)
    return available


def scan_sources(directory, glob_pattern='**/*'):
    """Split a directory into importable files and everything left out.

    The second return value lists files that were deliberately not imported
    (unsupported extension), so callers can report them instead of silently
    dropping them from an import that claims to have read the directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f'{directory} is not a directory')
    sources, unsupported = [], []
    for path in sorted(root.glob(glob_pattern)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        media_type = media_type_for(path)
        if not media_type or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            unsupported.append({'source': relative, 'reason': 'unsupported_media_type'})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(DocumentInput(path=path, relative=relative, media_type=media_type,
                                     size=path.stat().st_size, sha256=digest))
    return sources, unsupported


def document_id_for(kb_id, relative):
    """Stable document identity: same file in the same base, same id, forever.

    Public because published indexes and citations both depend on it, and a
    second implementation would drift. Deliberately not built-in ``hash``: this
    must survive a process restart.
    """
    return 'doc-' + hashlib.sha256(f'{kb_id}\0{relative}'.encode('utf-8')).hexdigest()[:16]


def _chunk_id(document_id, index):
    return f'{document_id}:{index:04d}'


def _line_count(raw):
    """Physical line count; a trailing newline does not add an empty line."""
    return len(raw.splitlines()) if raw else 0


def _apply_document(record, raw, document_id, chunk_size, chunk_overlap, media_type):
    """Fill source span and structure for one document; never fabricate a span."""
    record['parser'] = 'markdown' if media_type == 'text/markdown' else 'text'
    record['positions'] = False
    record['sections'] = []
    record['lines'] = _line_count(raw)
    record['chunk_size'] = chunk_size
    record['chunk_overlap'] = chunk_overlap
    if not raw.strip():
        record['empty'] = True
        return
    format_name = 'markdown' if media_type == 'text/markdown' else 'text'
    structure = parse_structure(raw, document_id, format_name=format_name)
    record['sections'] = [{'id': s.id, 'level': s.level, 'title': s.title, 'path': s.path,
                           'kind': s.kind, 'start': s.start, 'end': s.end} for s in structure.sections]
    record['structure_sha256'] = structure.sha256
    record['positions'] = True


def _chunks_for(document, record, kb_id, document_id, chunk_size, chunk_overlap,
                structure_aware=True):
    from rag.loader import Document as Loaded

    loaded = Loaded(content=document['content'], metadata={'source': document['source']})
    produced = split([loaded], chunk_size, chunk_overlap)
    texts = [c.content for c in produced]
    if structure_aware:
        # Honour the document's own headings before any generic separator, so a
        # section is not cut in half by the character limit. Measured on the
        # accounting corpus: articles wholly inside one chunk 1523/1859 -> 1805/1859.
        def fallback(text):
            return [chunk.content for chunk in
                    split([Loaded(content=text, metadata={'source': document['source']})],
                          chunk_size, chunk_overlap)]

        result = split_text(document['content'], chunk_size, chunk_overlap, fallback)
        if result.structural:
            texts = result.texts
    if record['positions']:
        spans = locate_chunks(document['content'], texts)
    else:
        spans = [None] * len(texts)
    chunks = []
    for index, (text, span) in enumerate(zip(texts, spans)):
        chunks.append({'id': _chunk_id(document_id, index), 'document_id': document_id,
                       'kb_id': kb_id, 'source': document['source'], 'chunk_index': index,
                       'content': text,
                       'start': span[0] if span else None, 'end': span[1] if span else None})
    return chunks


def _pdf_pages(path):
    """Page count from the PDF itself; only reported when the parser gives it."""
    from PyPDF2 import PdfReader

    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def chunking_config(chunk_size, chunk_overlap, structure_aware=True):
    """Identity of the chunking that produced a chunk set.

    Recorded in the import manifest so a consumer can tell whether an existing
    import was made by the current chunker. Without it, changing the chunker
    leaves the old manifest in place and every downstream step keeps using the
    old chunks while believing it is current — which is what happened when
    structural chunking became the default.
    """
    return {'version': CHUNKING_VERSION, 'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap, 'structure_aware': bool(structure_aware)}


def import_documents(kb_id, directory, chunk_size=500, chunk_overlap=50, glob_pattern='**/*',
                     structure_aware=True):
    """Parse a directory into document records and located chunks.

    Every file the directory yields ends up either in ``documents`` or in
    ``skipped`` with a reason: unsupported extensions, unparsable files and
    files whose chunks cannot be located in the source are all reported, and
    none of them abort the import of the remaining files.
    """
    report = ImportReport()
    report.chunking = chunking_config(chunk_size, chunk_overlap, structure_aware)
    directory = Path(directory).resolve()
    sources, unsupported = scan_sources(directory, glob_pattern)
    report.skipped.extend(unsupported)
    for source in sources:
        try:
            loaded = load_file(source.path)
            document_id = document_id_for(kb_id, source.relative)
            record = {'id': document_id, 'kb_id': kb_id, 'source': source.relative,
                      'media_type': source.media_type, 'size': source.size, 'sha256': source.sha256,
                      'content_sha256': hashlib.sha256(loaded.content.encode()).hexdigest(),
                      'characters': len(loaded.content)}
            document = {'source': source.relative, 'content': loaded.content}
            if source.media_type == 'application/pdf':
                record.update({'parser': 'pdf', 'positions': False, 'sections': [], 'lines': None,
                               'pages': _pdf_pages(source.path), 'chunk_size': chunk_size,
                               'chunk_overlap': chunk_overlap})
            else:
                _apply_document(record, loaded.content, document_id, chunk_size, chunk_overlap,
                                source.media_type)
            chunks = _chunks_for(document, record, kb_id, document_id, chunk_size, chunk_overlap,
                                 structure_aware)
        except Exception as exc:
            report.skipped.append({'source': source.relative, 'reason': type(exc).__name__})
            continue
        record['chunks'] = len(chunks)
        report.documents.append(record)
        report.chunks.extend(chunks)
    return report


def write_manifest(report, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)
    return target


def read_manifest(target):
    return json.loads(Path(target).read_text(encoding='utf-8'))
