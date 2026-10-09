"""Tests for the automatic citation checks.

Two of these exist because the first version of the checker produced false
failures on correct answers: a Chinese document number (``财会[2006]3号``) was
read as a citation marker, and an abstention answer was required to cite
evidence the rubric does not ask it to cite.
"""
import pytest

from answer_quality_review import citation_checks, cited_numbers

SOURCES = {'doc.md': '第一条 正文内容。第二条 更多内容。'}


def citations(*items):
    return [{'citation': index, 'source': 'doc.md', 'content': content}
            for index, content in enumerate(items, 1)]


def test_plain_markers_are_read():
    assert cited_numbers('结论成立[1]，另见[2]。') == [1, 2]
    assert cited_numbers('没有引用') == []


def test_document_numbers_are_not_citation_markers():
    """``财会[2006]3号`` is an issuance number, not a citation."""
    answer = '本片段是《企业会计准则第14号——收入（2006）》，文号为财会[2006]3号[1]。'
    assert cited_numbers(answer) == [1]


def test_a_four_digit_year_bracket_does_not_break_the_check():
    answer = '财会〔2017〕22号[1][2] 与 财会[2006]3号[3]'
    checks = citation_checks(answer, citations('第一条 正文内容。', '第二条 更多内容。', '第一条 正文内容。'),
                             SOURCES, 'answer')
    assert checks['referenced_ids_exist'] is True
    assert all(checks.values())


def test_citations_must_exist_in_the_citation_list():
    checks = citation_checks('结论[3]', citations('第一条 正文内容。'), SOURCES, 'answer')
    assert checks['referenced_ids_exist'] is False


def test_duplicate_citation_ids_are_caught():
    duplicated = [{'citation': 1, 'source': 'doc.md', 'content': '第一条 正文内容。'},
                  {'citation': 1, 'source': 'doc.md', 'content': '第二条 更多内容。'}]
    assert citation_checks('x[1]', duplicated, SOURCES, 'answer')['citation_ids_unique'] is False


def test_sources_outside_the_knowledge_base_are_caught():
    outside = [{'citation': 1, 'source': 'other.md', 'content': '正文'}]
    checks = citation_checks('x[1]', outside, SOURCES, 'answer')
    assert checks['sources_in_selected_kb'] is False


def test_snippets_must_appear_in_the_source():
    checks = citation_checks('x[1]', citations('这段不在源文里。'), SOURCES, 'answer')
    assert checks['snippets_in_source'] is False


def test_a_substantive_answer_without_markers_fails():
    checks = citation_checks('直接给结论，没有任何编号。', citations('第一条 正文内容。'),
                             SOURCES, 'answer')
    assert checks['answer_has_citation_markers'] is False


@pytest.mark.parametrize('action', ['clarify', 'abstain'])
def test_exempt_answers_are_not_required_to_cite(action):
    """The rubric exempts clarification and abstention from citation demands."""
    answer = '本次检索片段不足以回答这个问题，建议联系人工客服核实。'
    checks = citation_checks(answer, citations('第一条 正文内容。'), SOURCES, action)
    assert 'answer_has_citation_markers' not in checks
    assert all(checks.values())


def test_a_substantive_claim_inside_an_abstention_is_still_checked():
    """Exemption drops the marker requirement, not provenance checking."""
    checks = citation_checks('结论[9]', citations('第一条 正文内容。'), SOURCES, 'abstain')
    assert checks['referenced_ids_exist'] is False


def test_without_citations_the_marker_check_is_vacuous():
    checks = citation_checks('', [], SOURCES, 'answer')
    assert checks['answer_has_citation_markers'] is True
