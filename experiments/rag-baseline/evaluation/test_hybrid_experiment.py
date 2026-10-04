import unittest
import hybrid_experiment as h


DOCS = [
    {'id': 'CAS-14-2006', 'standard_id': '14', 'version_year': 2006, 'title': '企业会计准则第14号——收入（2006）'},
    {'id': 'CAS-14-2017', 'standard_id': '14', 'version_year': 2017, 'title': '企业会计准则第14号——收入（2017）'},
    {'id': 'CAS-09-2014', 'standard_id': '09', 'version_year': 2014, 'title': '企业会计准则第9号——职工薪酬（2014）'},
]


class RoutingTests(unittest.TestCase):
    def test_named_standard_beats_incidental_statement_title(self):
        docs = DOCS + [
            {'id':'CAS-21-2018', 'standard_id':'21', 'version_year':2018, 'title':'企业会计准则第21号——租赁（2018）'},
            {'id':'CAS-31-2006', 'standard_id':'31', 'version_year':2006, 'title':'企业会计准则第31号——现金流量表（2006）'},
        ]
        r = h.route_accounting('按2018年租赁准则，本金利息在现金流量表归入哪里？', docs)
        self.assertEqual(r['sources'], ['CAS-21-2018.md'])

    def test_version_and_article_are_query_derived(self):
        r = h.route_accounting('按2006年与2017年收入准则第四条比较', DOCS)
        self.assertEqual(r['sources'], ['CAS-14-2006.md', 'CAS-14-2017.md'])
        self.assertEqual(r['articles'], [4])
        self.assertTrue(r['exact_enabled'])

    def test_unknown_year_does_not_fall_back_to_existing_version(self):
        r = h.route_accounting('2025年收入准则第4条', DOCS)
        self.assertEqual(r['sources'], [])
        self.assertTrue(r['unmatched_explicit_scope'])

    def test_unspecified_version_never_defaults_to_latest(self):
        r = h.route_accounting('收入准则第二条', DOCS)
        self.assertEqual(len(r['sources']), 2)
        self.assertFalse(r['exact_enabled'])
        self.assertTrue(r['version_unspecified'])

    def test_multiple_families_or_articles_disable_ambiguous_exact_join(self):
        for query in ['2014年第9号第二条与2017年第14号第四条', '2017年收入准则第四条与第十四条']:
            self.assertFalse(h.route_accounting(query, DOCS)['exact_enabled'])

    def test_chinese_numbers_and_article_not_confused_with_standard(self):
        r = h.route_accounting('2014年第九号准则第一百零二条', DOCS)
        self.assertEqual(r['families'], ['9'])
        self.assertEqual(r['articles'], [102])
        self.assertEqual(r['sources'], ['CAS-09-2014.md'])


class RankingTests(unittest.TestCase):
    def test_exact_lookup_missing_clause_does_not_substitute_another_version(self):
        source = 'CAS-14-2006.md'
        chunk = {'id': 'chunk', 'source': source, 'start': 0, 'end': 4, 'content': '收入定义'}
        retriever = h.Retriever('accounting', [chunk], {}, [], DOCS, None)
        retriever.dense = lambda vector, sources=None, depth=20: ['chunk'] if sources is None or source in sources else []
        for query in ['2025年收入准则第四条', '2006年收入准则第999条']:
            result, _ = retriever.retrieve(query, [0.0])
            self.assertEqual(result['exact_then_dense'], [])
        result, _ = retriever.retrieve('什么是收入', [0.0])
        self.assertEqual(result['exact_then_dense'], result['dense_replay'])

    def test_bm25_and_rrf_handle_empty_or_disjoint_terms(self):
        index = h.BM25({'a': '贷款资本化', 'b': '设备售后维修'})
        self.assertEqual(index.rank('资本化', ['a', 'b']), ['a'])
        self.assertEqual(index.rank('WiFi', ['a', 'b']), [])
        self.assertEqual(h.fuse(['a', 'b'], ['b', 'a'])[0], 'a')

    def test_packing_respects_source_spans_and_budget(self):
        hits = [{'id': str(i), 'source': 'doc', 'start': i*800, 'end': (i+1)*800, 'content': '字'*800} for i in range(3)]
        hits.insert(1, dict(hits[0]))
        result = h.pack(hits)
        self.assertEqual([r['id'] for r in result], ['0', '1'])
        self.assertLessEqual(sum(len(r['content']) for r in result), 2000)
        self.assertEqual(h.pack([{'source':'doc', 'start':0, 'end':2001, 'content':'字'*2001}]), [])


if __name__ == '__main__':
    unittest.main()
