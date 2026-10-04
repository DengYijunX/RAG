"""Extract user-supplied DOCX text without executing document content."""
import argparse, hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from xml.etree import ElementTree as ET

OUT = Path(__file__).resolve().parents[1]
NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def tag(name): return '{'+NS+'}'+name

def chinese_number(s):
    digits = dict(zip('零一二三四五六七八九',range(10)))
    total = value = 0
    for c in s:
        if c in digits: value=digits[c]
        else:
            total += (value or 1)*{'十':10,'百':100,'千':1000}[c]
            value=0
    return total+value

def dump_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def main(archive):
    for name in ('raw','text','markdown'):
        (OUT/name).mkdir(exist_ok=True)
    docs, articles, issues = [], [], []
    with ZipFile(archive) as z:
        assert sum(i.file_size for i in z.infolist()) < 100_000_000
        names = [n for n in z.namelist() if n.endswith('.docx')]
        assert len(names)==len(set(PurePosixPath(n).name for n in names)), 'Duplicate basenames'
        for name in sorted(names):
            p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts
            raw=z.read(name)
            with ZipFile(BytesIO(raw)) as dz:
                assert sum(i.file_size for i in dz.infolist()) < 20_000_000
                xml=dz.read('word/document.xml')
                assert b'<!DOCTYPE' not in xml and b'<!ENTITY' not in xml
                root=ET.fromstring(xml)
                for unsupported in ('tbl','drawing','del','ins','altChunk'):
                    assert not list(root.iter(tag(unsupported))), f'Unsupported {unsupported}: {name}'
                parts=[]
                for para in root.findall('./'+tag('body')+'/'+tag('p')):
                    text=''.join((el.text or '') if el.tag==tag('t') else '\n' if el.tag==tag('br') else '\t' if el.tag==tag('tab') else '' for el in para.iter())
                    parts.append(text)
                text='\n'.join(parts)+'\n'
            filename=p.name
            year=int(re.search(r'（(\d{4})）',filename)[1])
            number=re.search(r'第(\d+)号',filename)
            standard='basic' if number is None else number[1].zfill(2)
            doc_id=f'CAS-{standard}-{year}'
            lines=text.splitlines()
            title=lines[0]
            doc_articles=[]
            chapter=None
            section=None
            current=None
            for lineno,line in enumerate(lines,1):
                s=line.strip()
                if re.match(r'^第[零一二三四五六七八九十百]+章',s) and not re.search('[，。；]',s):
                    chapter=s
                    section=None
                    continue
                if re.match(r'^第[零一二三四五六七八九十百]+节',s) and not re.search('[，。；]',s):
                    section=s
                    continue
                m=re.match(r'^第([零一二三四五六七八九十百]+)条',s)
                if m:
                    n=chinese_number(m[1])
                    current={'id':f'{doc_id}-A{n:03}','document_id':doc_id,'standard_id':standard,'version_year':year,'article_number':n,'article_label':m[0],'chapter':chapter,'section':section,'source_file':filename,'source_start_line':lineno,'lines':[]}
                    doc_articles.append(current)
                if current is not None and s:
                    current['lines'].append(line)
                    current['source_end_line']=lineno
            nums=[a['article_number'] for a in doc_articles]
            if nums!=list(range(1,len(nums)+1)):
                issues.append({'document_id':doc_id,'issue':'nonsequential_articles','numbers':nums})
            assert doc_articles
            for a in doc_articles:
                a['text']='\n'.join(a.pop('lines'))
                a['text_sha256']=hashlib.sha256(a['text'].encode()).hexdigest()
            (OUT/'raw'/filename).write_bytes(raw)
            (OUT/'text'/f'{doc_id}.txt').write_text(text,encoding='utf-8')
            # Change only headings; preserve extracted prose and document order.
            rendered=[]
            for i,line in enumerate(lines):
                s=line.strip()
                prefix='# ' if i==0 else '## ' if re.match(r'^第[零一二三四五六七八九十百]+章',s) else '### ' if re.match(r'^第[零一二三四五六七八九十百]+条',s) else ''
                rendered.append(prefix+line)
            (OUT/'markdown'/f'{doc_id}.md').write_text('\n\n'.join(rendered)+'\n',encoding='utf-8')
            docs.append({'id':doc_id,'standard_id':standard,'version_year':year,'title':title,'source_type':'user_upload','source_archive_member':name,'raw_path':'raw/'+filename,'raw_sha256':hashlib.sha256(raw).hexdigest(),'text_path':f'text/{doc_id}.txt','text_sha256':hashlib.sha256(text.encode()).hexdigest(),'markdown_path':f'markdown/{doc_id}.md','article_count':len(doc_articles),'text_characters':len(text),'official_source_url':None,'authority_verified':False,'current_effectiveness':'unverified','year_semantics':'filename version year, not verified effective date'})
            articles.extend(doc_articles)
    assert len(docs)==59
    dump_jsonl(OUT/'documents.jsonl',docs)
    dump_jsonl(OUT/'articles.jsonl',articles)
    groups=defaultdict(list)
    for d in docs: groups[d['standard_id']].append(d['version_year'])
    stats={'documents':len(docs),'standard_families':len(groups),'articles':len(articles),'text_characters':sum(d['text_characters'] for d in docs),'years':dict(Counter(d['version_year'] for d in docs)),'multi_version_families':{k:sorted(v) for k,v in groups.items() if len(v)>1},'exact_duplicate_files':len(docs)-len(set(d['raw_sha256'] for d in docs)),'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'issues':issues}
    (OUT/'corpus_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    main(parser.parse_args().archive)
