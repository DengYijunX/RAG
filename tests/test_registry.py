import json

import pytest

from rag.registry import (KnowledgeBase, KnowledgeBaseExists, KnowledgeBaseNotFound,
                          Registry, RegistryError, new_id, normalize_name)


def test_create_persists_and_reloads(tmp_path):
    registry = Registry(tmp_path)
    created = registry.create('城市轨道交通运营', description='第三语料')
    assert created.id and created.name == '城市轨道交通运营'
    assert created.created_at.endswith('Z')
    reloaded = Registry(tmp_path).get(created.id)
    assert reloaded == created
    assert len(Registry(tmp_path).list()) == 1


def test_ids_are_url_safe_and_unique(tmp_path):
    registry = Registry(tmp_path)
    first = registry.create('Metro Ops')
    second = registry.create('Metro Ops 2')
    assert first.id != second.id
    assert all(character.isalnum() or character == '-' for character in first.id)
    assert new_id() != new_id()


def test_duplicate_name_rejected(tmp_path):
    registry = Registry(tmp_path)
    registry.create('运营手册')
    with pytest.raises(KnowledgeBaseExists):
        registry.create(' 运营手册 ')
    with pytest.raises(KnowledgeBaseExists):
        registry.create('运营手册')


def test_blank_and_overlong_names_rejected(tmp_path):
    registry = Registry(tmp_path)
    for bad in ['', '   ', None]:
        with pytest.raises(RegistryError):
            registry.create(bad)
    with pytest.raises(RegistryError):
        registry.create('x' * 81)


def test_missing_base_raises(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(KnowledgeBaseNotFound):
        registry.get('does-not-exist')
    assert registry.exists('does-not-exist') is False


def test_update_changes_fields_and_keeps_id(tmp_path):
    registry = Registry(tmp_path)
    created = registry.create('旧名称')
    updated = registry.update(created.id, name='新名称', description='说明', language='en')
    assert updated.id == created.id
    assert updated.name == '新名称'
    assert updated.language == 'en'
    assert registry.get(created.id).description == '说明'


def test_update_rejects_duplicate_name(tmp_path):
    registry = Registry(tmp_path)
    first = registry.create('甲')
    registry.create('乙')
    with pytest.raises(KnowledgeBaseExists):
        registry.update(first.id, name='乙')


def test_registry_file_is_versioned_and_atomic(tmp_path):
    registry = Registry(tmp_path)
    registry.create('甲')
    payload = json.loads((tmp_path / 'knowledge-bases.json').read_text(encoding='utf-8'))
    assert payload['schema'] == 1
    assert list(payload['knowledge_bases'][0]) >= ['id', 'name', 'created_at']
    assert not list(tmp_path.glob('*.tmp'))


def test_unsupported_schema_is_refused(tmp_path):
    (tmp_path / 'knowledge-bases.json').write_text(json.dumps({'schema': 99, 'knowledge_bases': []}))
    with pytest.raises(RegistryError):
        Registry(tmp_path).list()


def test_registry_keeps_extensions_without_adding_credentials(tmp_path):
    """The registry stores what it is given; credentials stay in the environment."""
    registry = Registry(tmp_path)
    registry.create('甲', extensions={'retrieval': 'vector', 'api_key': 'sk-should-not-be-added'})
    stored = json.loads((tmp_path / 'knowledge-bases.json').read_text(encoding='utf-8'))
    entry = stored['knowledge_bases'][0]
    assert entry['extensions'] == {'retrieval': 'vector', 'api_key': 'sk-should-not-be-added'}
    # No credential field is invented by the registry itself.
    assert set(entry) == {'id', 'name', 'description', 'language', 'created_at', 'updated_at',
                          'extensions'}
    assert 'embedding_api_key' not in entry and 'llm_api_key' not in entry


def test_name_normalization():
    assert normalize_name('  a   b  ') == 'a b'
    assert normalize_name(None) == ''
    assert KnowledgeBase(id='x', name='y').language == 'zh'
