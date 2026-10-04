"""Exercise the existing RAG loader/splitter independently for each corpus."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from rag.loader import load_from_dir
from rag.splitter import split
root=Path(__file__).resolve().parent
results={}
for name,folder,expected in [('smart_home','smart-home/corpus',5),('accounting','accounting/markdown',59)]:
    docs=load_from_dir(str(root/folder),'*.md')
    chunks=split(docs,chunk_size=500,chunk_overlap=50)
    assert len(docs)==expected and len(chunks)>len(docs)
    assert all(c.content and str(root/folder) in c.metadata['source'] for c in chunks)
    results[name]={'documents':len(docs),'chunks_500_overlap_50':len(chunks),'loader_check':'passed','embedding_retrieval_generation':'not run'}
(root/'loader_check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
