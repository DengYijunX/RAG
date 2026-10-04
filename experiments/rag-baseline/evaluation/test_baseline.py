import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import baseline

class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.index={'e1':{'source':'one.md','start':0,'end':100},'e2':{'source':'two.md','start':0,'end':100}}
    def test_duplicates_do_not_inflate_coverage(self):
        hits=[{'source':'one.md','start':0,'end':50}]*4
        result=baseline.score(['e1'],hits,self.index)
        self.assertEqual(result['evidence_coverage']['e1'],.5)
        self.assertEqual(result['coverage_recall_at_k'],0)
    def test_complementary_chunks_and_rank(self):
        hits=[{'source':'one.md','start':0,'end':45},{'source':'one.md','start':40,'end':85}]
        result=baseline.score(['e1','e2'],hits,self.index)
        self.assertEqual(result['coverage_recall_at_k'],.5)
        self.assertEqual(result['coverage_reciprocal_rank'],.5)
        self.assertFalse(result['all_evidence_covered'])
    def test_wrong_version_or_source_gets_no_credit(self):
        result=baseline.score(['e1'],[{'source':'wrong-version.md','start':0,'end':100}],self.index)
        self.assertEqual(result['coverage_recall_at_k'],0)
    def test_boundary_and_empty_gold(self):
        self.assertEqual(baseline.score(['e1'],[{'source':'one.md','start':0,'end':80}],self.index)['coverage_recall_at_k'],1)
        self.assertIsNone(baseline.score([],[],self.index))

class ApiAdapterTests(unittest.TestCase):
    def test_response_indexes_reordered(self):
        fake=SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw:SimpleNamespace(data=[SimpleNamespace(index=1,embedding=[2.,3.]),SimpleNamespace(index=0,embedding=[0.,1.])],usage=None)))
        vectors,tokens=baseline.embed(fake,'unit-test',['first','second'])
        self.assertEqual(vectors,[[0.,1.],[2.,3.]])
        self.assertIsNone(tokens)
    def test_missing_index_is_rejected(self):
        fake=SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw:SimpleNamespace(data=[SimpleNamespace(index=1,embedding=[2.])],usage=None)))
        with self.assertRaises(ValueError): baseline.embed(fake,'unit-test',['first'])
    def test_nonfinite_is_rejected(self):
        fake=SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw:SimpleNamespace(data=[SimpleNamespace(index=0,embedding=[float('nan')])],usage=None)))
        with self.assertRaises(ValueError): baseline.embed(fake,'unit-test',['first'])
    def test_bare_hostname_reports_url_field(self):
        with patch.dict(baseline.os.environ,{'EMBEDDING_API_KEY':'unit-test-not-a-credential','EMBEDDING_MODEL':'unit-test','EMBEDDING_BASE_URL':'example.com'},clear=True),patch('dotenv.load_dotenv'):
            with self.assertRaisesRegex(ValueError,'Invalid configuration: EMBEDDING_BASE_URL'): baseline.settings()
    def test_known_bare_hostname_uses_user_authorized_full_url(self):
        hostname=baseline.urlparse(baseline.URL).hostname
        with patch.dict(baseline.os.environ,{'EMBEDDING_API_KEY':'unit-test-not-a-credential','EMBEDDING_MODEL':'unit-test','EMBEDDING_BASE_URL':hostname},clear=True),patch('dotenv.load_dotenv'):
            self.assertEqual(baseline.settings()['embedding_url'],baseline.URL)
    def test_empty_model_fails_before_call(self):
        with patch.dict(baseline.os.environ,{},clear=True),patch('dotenv.load_dotenv'):
            with self.assertRaisesRegex(ValueError,'EMBEDDING_MODEL'): baseline.settings()

class StorageTests(unittest.TestCase):
    def test_independent_chroma_directories(self):
        import chromadb
        from chromadb.config import Settings
        with TemporaryDirectory() as path:
            left=chromadb.PersistentClient(path=str(Path(path)/'home'),settings=Settings(anonymized_telemetry=False))
            right=chromadb.PersistentClient(path=str(Path(path)/'accounting'),settings=Settings(anonymized_telemetry=False))
            a=left.create_collection('rag_documents',embedding_function=None)
            b=right.create_collection('rag_documents',embedding_function=None)
            a.add(ids=['home-only'],embeddings=[[1.,0.]],documents=['home'])
            b.add(ids=['accounting-only'],embeddings=[[0.,1.]],documents=['accounting'])
            self.assertEqual(a.query(query_embeddings=[[1.,0.]],n_results=1)['ids'],[['home-only']])
            self.assertEqual(b.query(query_embeddings=[[0.,1.]],n_results=1)['ids'],[['accounting-only']])
            left.delete_collection('rag_documents')
            self.assertEqual(b.count(),1)

if __name__=='__main__': unittest.main()
