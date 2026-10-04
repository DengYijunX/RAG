import unittest
import structure_experiment as s

class ExpansionTests(unittest.TestCase):
    def test_deduplicates_parents_preserving_retrieval_order(self):
        parents={'p1':{'id':'p1'},'p2':{'id':'p2'}}
        hits=[{'parent_id':'p2'},{'parent_id':'p2'},{'parent_id':'p1'}]
        self.assertEqual(s.expand(hits,parents),[parents['p2'],parents['p1']])
    def test_missing_parent_fails_instead_of_inventing_context(self):
        with self.assertRaises(KeyError):s.expand([{'parent_id':'missing'}],{})
    def test_relocated_baseline_scores_are_unchanged(self):
        for kb in s.b.CONFIG:
            _,parents,children,index=s.units(kb)
            p,_=s.selected_baseline(kb)
            questions={q['id']:q for q in s.b.validate_questions(kb,index)}
            for row in s.b.read(p.with_name('results.jsonl')):
                q=questions[row['question_id']]
                expected=s.b.score(q['evidence_ids'],row['hits'],index) if q['expected_action'] in ('answer','flag_conflict') else None
                self.assertEqual(expected,row['metrics'])
            for child in children:
                parent=parents[child['parent_id']]
                self.assertEqual(child['source'],parent['source'])
                self.assertGreaterEqual(child['start'],parent['start'])
                self.assertLessEqual(child['end'],parent['end'])
                self.assertLessEqual(len(child['content']),500)
                self.assertEqual(s.b.normalize(parent['content'])[child['start']-parent['start']:child['end']-parent['start']],s.b.normalize(child['content']))

if __name__=='__main__':unittest.main()
