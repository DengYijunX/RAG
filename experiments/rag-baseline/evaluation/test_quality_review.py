import unittest
from types import SimpleNamespace
import quality_review as q


class QualityChecks(unittest.TestCase):
    def test_held_out_sample_is_rejected_before_any_model_call(self):
        sample = {'split': 'dev', 'knowledge_bases': {'kb': ['id']}}
        with self.assertRaises(ValueError):
            q.select_questions('kb', sample, [{'id': 'id', 'split': 'test'}])

    def test_missing_reference_and_wrong_source_fail(self):
        corpus = SimpleNamespace(sources={'a': {'normalized': '实际原文'}})
        response = {'answer': '主张[2]', 'citations': [{'citation': 1, 'source': 'b', 'content': '实际原文'}]}
        checks = q.citation_checks(response, corpus)
        self.assertFalse(checks['passed'])
        self.assertFalse(checks['checks']['referenced_ids_exist'])
        self.assertFalse(checks['checks']['all_sources_in_selected_kb'])

    def test_existing_reference_does_not_claim_semantic_correctness(self):
        corpus = SimpleNamespace(sources={'a': {'normalized': '实际原文'}})
        response = {'answer': '任意主张[1]', 'citations': [{'citation': 1, 'source': 'a', 'content': '实际原文'}]}
        result = q.citation_checks(response, corpus)
        self.assertTrue(result['passed'])
        self.assertFalse(result['semantic_faithfulness_measured'])


if __name__ == '__main__':
    unittest.main()
