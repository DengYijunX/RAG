"""Persistent knowledge-base registry.

A knowledge base is a user-created, named container. Nothing in this module
knows about a domain, a product model or an evaluation fixture: entries only
carry identity and import-time configuration, so any new corpus is added by
writing a registry record rather than by editing code.

Design reference: RAGFlow treats a knowledge base as a configured entity
(name, embedding model, parse method) instead of a fixed application option;
LlamaIndex separates document identity from the transformation pipeline.
See docs/ARCHITECTURE_REFERENCES.md (ADR-007).
"""
from dataclasses import asdict, dataclass, field
import json
import re
import secrets
import time
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_FILE_NAME = 'knowledge-bases.json'


class RegistryError(Exception):
    """Registry input is invalid or refers to a missing entry."""


class KnowledgeBaseNotFound(RegistryError):
    pass


class KnowledgeBaseExists(RegistryError):
    pass


@dataclass
class KnowledgeBase:
    """Identity plus import configuration; credentials stay in the environment."""
    id: str
    name: str
    description: str = ''
    language: str = 'zh'
    created_at: str = ''
    updated_at: str = ''
    extensions: dict = field(default_factory=dict)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def normalize_name(name):
    return re.sub(r'\s+', ' ', str(name or '')).strip()


def new_id(name=None):
    """Readable, collision-resistant identifier usable directly in a URL."""
    slug = re.sub(r'[^a-z0-9]+', '-', normalize_name(name).lower()).strip('-')[:24]
    return f'{slug}-{secrets.token_hex(4)}' if slug else secrets.token_hex(8)


class Registry:
    """JSON-backed registry; every write rewrites the file atomically."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / DEFAULT_FILE_NAME

    def _read(self):
        if not self.path.exists():
            return {'schema': SCHEMA_VERSION, 'knowledge_bases': []}
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema') != SCHEMA_VERSION:
            raise RegistryError(f"Unsupported registry schema: {data.get('schema')!r}")
        return data

    def _write(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.path)

    def list(self):
        return [KnowledgeBase(**entry) for entry in self._read()['knowledge_bases']]

    def get(self, kb_id):
        for base in self.list():
            if base.id == kb_id:
                return base
        raise KnowledgeBaseNotFound(kb_id)

    def exists(self, kb_id):
        return any(base.id == kb_id for base in self.list())

    def create(self, name, description='', language='zh', extensions=None):
        clean = normalize_name(name)
        if not clean:
            raise RegistryError('Knowledge base name is required')
        if len(clean) > 80:
            raise RegistryError('Knowledge base name is too long')
        data = self._read()
        if any(normalize_name(entry['name']) == clean for entry in data['knowledge_bases']):
            raise KnowledgeBaseExists(clean)
        base = KnowledgeBase(id=new_id(clean), name=clean, description=str(description or '').strip(),
                             language=str(language or 'zh').strip() or 'zh',
                             created_at=_now(), updated_at=_now(),
                             extensions=dict(extensions or {}))
        data['knowledge_bases'].append(asdict(base))
        self._write(data)
        return base

    def update(self, kb_id, *, name=None, description=None, language=None, extensions=None):
        data = self._read()
        for entry in data['knowledge_bases']:
            if entry['id'] != kb_id:
                continue
            if name is not None:
                clean = normalize_name(name)
                if not clean:
                    raise RegistryError('Knowledge base name is required')
                if any(other['name'] == clean and other['id'] != kb_id for other in data['knowledge_bases']):
                    raise KnowledgeBaseExists(clean)
                entry['name'] = clean
            if description is not None:
                entry['description'] = str(description).strip()
            if language is not None:
                entry['language'] = str(language).strip() or 'zh'
            if extensions is not None:
                entry['extensions'] = dict(extensions)
            entry['updated_at'] = _now()
            self._write(data)
            return KnowledgeBase(**entry)
        raise KnowledgeBaseNotFound(kb_id)
