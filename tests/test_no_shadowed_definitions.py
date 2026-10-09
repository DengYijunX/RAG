"""Guard against a duplicate method definition silently shadowing a change.

While wiring deep-pool reranking, a second ``query()`` definition further down
the class shadowed the edited one: the file looked correct, the tests of the new
helper passed, and the feature simply never ran. Python does not warn. This test
turns that class of mistake into a failure, and also checks that the AST's last
binding is the only one.
"""
import ast
from pathlib import Path

import pytest

RAG = Path(__file__).resolve().parents[1] / 'rag'


def classes_in(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def duplicate_methods(path):
    """Method names defined more than once in a class — the last silently wins."""
    duplicates = []
    for node in classes_in(path):
        seen = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault(item.name, []).append(item.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                duplicates.append((node.name, name, lines))
    return duplicates


def module_functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    seen = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seen.setdefault(node.name, []).append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


MODULES = sorted(RAG.glob('*.py'))


def test_there_is_something_to_check():
    assert MODULES, f'{RAG} 下没有找到模块'


@pytest.mark.parametrize('path', MODULES, ids=lambda path: path.name)
def test_no_class_defines_a_method_twice(path):
    duplicates = duplicate_methods(path)
    assert not duplicates, f'{path.name} 重复定义方法（后者会静默覆盖前者）：{duplicates}'


@pytest.mark.parametrize('path', MODULES, ids=lambda path: path.name)
def test_no_module_defines_a_function_twice(path):
    duplicates = module_functions(path)
    assert not duplicates, f'{path.name} 重复定义函数：{duplicates}'


def test_the_checker_actually_detects_a_duplicate(tmp_path):
    """A guard that cannot fail is worthless, so prove it fires."""
    sample = tmp_path / 'sample.py'
    sample.write_text('class A:\n'
                      '    def f(self):\n        return 1\n'
                      '    def f(self):\n        return 2\n', encoding='utf-8')
    assert duplicate_methods(sample) == [('A', 'f', [2, 4])]
