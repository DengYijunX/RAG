"""Generic knowledge-base lifecycle tests.

They exercise register -> import -> build -> query -> read source with an
injected in-memory vector store, so the flow is verified without the native
ChromaDB/HNSW extension. The corpus is a third, unrelated domain: no product
model, accounting standard or evaluation fixture appears anywhere.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag.kb_service import (BuildFailed, IndexUnavailable, InMemoryVectors, KnowledgeBaseService,
                            normalize)
from rag.registry import KnowledgeBaseNotFound
from tests.test_store import get_test_config

CORPUS = {
    'lines/line-1.md': "# 1号线\n\n## 首末班车\n\n1号线首班车 6:00，末班车 23:00。\n\n## 换乘\n\n可在中心站换乘2号线。\n",
    'lines/line-2.md': "# 2号线\n\n## 首末班车\n\n2号线首班车 6:10，末班车 22:45。\n\n## 票价\n\n起步价 2 元。\n",
    'policy.md': "# 乘客守则\n\n## 禁带物品\n\n禁止携带易燃易爆物品进站。\n",
}


def write_corpus(root):
    directory = Path(root) / 'third-corpus'
    for relative, text in CORPUS.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return directory


def fake_vectors(texts):
    """Deterministic pseudo-embedding: character-bucket counts, never a model."""
    return [[float(len(t) % 7), float(sum(ord(c) for c in t) % 97), 1.0] for t in texts]


@pytest.fixture
def service(tmp_path):
    vectors = {}
    instance = KnowledgeBaseService(
        get_test_config(str(tmp_path / 'store')), storage_root=tmp_path / 'knowledge',
        vector_backend=lambda directory: InMemoryVectors(directory, vectors))
    instance.vectors = fake_vectors
    instance.generate = lambda question, citations: '依据片段回答 [1]'
    return instance


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path)


def register_and_import(service, corpus, name='城市轨道交通运营'):
    base = service.register(name, description='第三语料')
    service.import_sources(base.id, corpus)
    return base


def test_register_appears_in_bases_without_import(service):
    base = service.register('城市轨道交通运营')
    entry = next(row for row in service.bases() if row['id'] == base.id)
    assert entry['name'] == '城市轨道交通运营'
    assert entry['documents'] == 0 and entry['imported'] is False and entry['ready'] is False


def test_import_then_build_then_query(service, corpus):
    base = register_and_import(service, corpus)
    imported = next(row for row in service.bases() if row['id'] == base.id)
    assert imported['documents'] == 3
    assert imported['chunks'] > 0 and imported['ready'] is False

    info = service.build(base.id)
    assert info['chunks'] == imported['chunks']
    assert info['documents'] == 3
    assert info['dimensions'] == 3
    ready = next(row for row in service.bases() if row['id'] == base.id)
    assert ready['ready'] is True and ready['chunk_count'] == info['chunks']

    response = service.query(base.id, '2号线首班车几点')
    assert response['status'] == 'answer'
    assert response['citations']
    citation = response['citations'][0]
    assert citation['source'].endswith('.md')
    assert citation['source_id'].startswith('doc-')
    assert citation['url'].startswith(f'/knowledge-bases/{base.id}/sources/')


def test_chunk_locations_match_source_text(service, corpus):
    base = register_and_import(service, corpus)
    manifest = json.loads(service.manifest_path(base.id).read_text(encoding='utf-8'))
    assert manifest['chunks'], 'import must produce chunks to locate'
    for chunk in manifest['chunks']:
        assert chunk['start'] is not None and chunk['end'] > chunk['start']
        text = (corpus / chunk['source']).read_text(encoding='utf-8')
        assert text[chunk['start']:chunk['end']] == chunk['content']


def test_source_view_returns_imported_text_and_spans(service, corpus):
    base = register_and_import(service, corpus)
    service.build(base.id)
    sources = service.sources(base.id)
    assert len(sources) == len(CORPUS)
    payload = service.source(base.id, sources[0]['source_id'])
    assert payload['content']
    assert normalize(payload['content']) in normalize((corpus / payload['source']).read_text(encoding='utf-8'))
    with pytest.raises(KeyError):
        service.source(base.id, 'not-a-document')


def test_reimport_same_files_is_stable(service, corpus):
    base = register_and_import(service, corpus)
    first = json.loads(service.manifest_path(base.id).read_text(encoding='utf-8'))
    service.import_sources(base.id, corpus)
    second = json.loads(service.manifest_path(base.id).read_text(encoding='utf-8'))
    assert [d['id'] for d in first['documents']] == [d['id'] for d in second['documents']]
    assert first['chunk_count'] == second['chunk_count']


def test_document_change_invalidates_index(service, corpus):
    """The manifest is the imported snapshot: editing a file needs a re-import."""
    base = register_and_import(service, corpus)
    service.build(base.id)
    (corpus / 'policy.md').write_text(CORPUS['policy.md'] + '\n## 新增\n\n携带宠物需装笼。\n',
                                      encoding='utf-8')
    service.import_sources(base.id, corpus)
    with pytest.raises(IndexUnavailable):
        service.active(base.id)
    with pytest.raises(IndexUnavailable):
        service.query(base.id, '禁带物品')
    rebuilt = service.build(base.id)
    assert rebuilt['chunks'] > 0
    assert service.query(base.id, '禁带物品')['status'] == 'answer'


def test_model_change_invalidates_index(service, corpus):
    base = register_and_import(service, corpus)
    service.build(base.id)
    changed = KnowledgeBaseService(replace(service.config, embedding_model='other-model'),
                                   storage_root=service.storage_root,
                                   vector_backend=service.vector_backend)
    with pytest.raises(IndexUnavailable):
        changed.active(base.id)


def test_failed_build_keeps_previous_generation(service, corpus):
    base = register_and_import(service, corpus)
    first = service.build(base.id)
    healthy = service.vectors
    service.vectors = lambda texts: (_ for _ in ()).throw(RuntimeError('provider down'))
    with pytest.raises(BuildFailed):
        service.build(base.id)
    assert service.active(base.id)['generation'] == first['generation']
    service.vectors = healthy
    assert service.query(base.id, '1号线首班车')['status'] == 'answer'


def test_build_without_import_and_query_without_index(service):
    base = service.register('空知识库')
    with pytest.raises(IndexUnavailable):
        service.build(base.id)
    with pytest.raises(IndexUnavailable):
        service.active(base.id)


def test_unknown_base_is_rejected(service):
    with pytest.raises(KnowledgeBaseNotFound):
        service.import_sources('missing', '.')
    with pytest.raises(KnowledgeBaseNotFound):
        service.base('missing')
    with pytest.raises(KnowledgeBaseNotFound):
        service.sources('missing')
    with pytest.raises(KnowledgeBaseNotFound):
        service.build('missing')


def test_import_replaces_the_searchable_set(service, corpus):
    """A later import defines the whole set; files no longer present drop out."""
    base = register_and_import(service, corpus)
    assert len(service.documents(base.id)) == 3
    subset = Path(corpus).parent / 'subset'
    subset.mkdir()
    (subset / 'policy.md').write_text(CORPUS['policy.md'], encoding='utf-8')
    service.import_sources(base.id, subset)
    assert [d['source'] for d in service.documents(base.id)] == ['policy.md']


def test_failed_build_leaves_no_stale_generation(service, corpus):
    base = register_and_import(service, corpus)
    service.vectors = lambda texts: (_ for _ in ()).throw(RuntimeError('provider down'))
    with pytest.raises(BuildFailed):
        service.build(base.id)
    leftovers = [path for path in service.base_dir(base.id).iterdir()
                 if path.is_dir() and len(path.name) == 32]
    assert leftovers == []


def test_bases_are_isolated(service, corpus):
    first = register_and_import(service, corpus, '库一')
    second = service.register('库二')
    other_corpus = Path(corpus).parent / 'second-corpus'
    other_corpus.mkdir()
    (other_corpus / 'other.md').write_text('# 其他\n\n完全不同主题的内容。\n', encoding='utf-8')
    service.import_sources(second.id, other_corpus)
    assert len(service.documents(first.id)) == 3
    assert len(service.documents(second.id)) == 1
    assert service.documents(first.id)[0]['id'] != service.documents(second.id)[0]['id']
    with pytest.raises(KeyError):
        service.source(second.id, service.documents(first.id)[0]['id'])


def test_no_domain_specific_branch_is_required(service, corpus):
    """A brand new domain must work through the same calls, with no code edit."""
    base = service.register('花卉养护手册')
    assert base.extensions == {}
    stored = json.loads((service.storage_root / 'knowledge-bases.json').read_text(encoding='utf-8'))
    entry = next(row for row in stored['knowledge_bases'] if row['id'] == base.id)
    # Nothing domain-specific is written for a new base, and no fixture is
    # reachable through it: its documents come only from its own import.
    assert entry == {'id': base.id, 'name': '花卉养护手册', 'description': '',
                     'language': 'zh', 'created_at': entry['created_at'],
                     'updated_at': entry['updated_at'], 'extensions': {}}
    service.import_sources(base.id, corpus)
    service.build(base.id)
    assert {d['source'] for d in service.documents(base.id)} == set(CORPUS)
    assert service.query(base.id, '什么时候浇水')['status'] == 'answer'
