"""Source-derived Markdown structure and raw-character chunk provenance.

Conservative subset: ATX headings, single-line Setext headings (3+ markers),
fenced-code exclusion, and opt-in standalone bold-line heading heuristics.
This is not a complete CommonMark parser or a semantic clause detector.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import re


@dataclass
class Section:
    id: str
    level: int
    title: str
    title_start: int
    title_end: int
    start: int
    heading_end: int
    block_end: int
    end: int
    line: int
    path: list[str]
    kind: str


class DocumentStructure:
    def __init__(self, text, document_id, sections, format_name):
        self.text = text
        self.document_id = document_id
        self.sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.sections = sections
        self.by_id = {s.id:s for s in sections}
        self.format_name = format_name
        self.line_starts = [0]+[m.end() for m in re.finditer('\n', text)]

    def record(self):
        return {'document_id':self.document_id,'source_sha256':self.sha256,
                'offset_unit':'Python Unicode character, raw source, half-open',
                'format':self.format_name,'length':len(self.text),
                'sections':[asdict(s) for s in self.sections]}

    def covered(self, start, end, spans):
        """Only whitespace may be absent from a completely covered range."""
        cursor = start
        for a,b in sorted(spans):
            a,b = max(a,start),min(b,end)
            if b<=a:
                continue
            if a>cursor and self.text[cursor:a].strip():
                return False
            cursor=max(cursor,b)
        return not self.text[cursor:end].strip()

    def annotate(self, start, end, context_spans=None):
        if not 0<=start<end<=len(self.text):
            raise ValueError('Invalid raw source span')
        spans=[(start,end)] if context_spans is None else context_spans
        if any(not 0<=a<b<=len(self.text) for a,b in spans):
            raise ValueError('Invalid context source span')
        blocks=[]
        prefix_end=self.sections[0].start if self.sections else len(self.text)
        intervals=[(0,prefix_end,None)]+[(s.start,s.block_end,s) for s in self.sections]
        for a,b,s in intervals:
            a,b=max(a,start),min(b,end)
            if b<=a or not self.text[a:b].strip():
                continue
            blocks.append({'raw_span':[a,b],'lines':[bisect_right(self.line_starts,a),bisect_right(self.line_starts,b-1)],
                           'section_id':s.id if s else None,'heading_path':s.path if s else [],
                           'recognition':s.kind if s else 'unknown',
                           'continues_from_previous':bool(self.text[s.start:a].strip()) if s else None,
                           'continues_after':bool(self.text[b:s.end].strip()) if s else None,
                           'section_complete_in_context':self.covered(s.start,s.end,spans) if s else None})
        return {'document_id':self.document_id,'source_sha256':self.sha256,
                'raw_span':[start,end],'blocks':blocks}


def parse_structure(text, document_id, *, format_name='markdown', bold_headings=False):
    """Build heading ancestry from source syntax; no domain or evaluation inputs."""
    if format_name!='markdown':
        return DocumentStructure(text,document_id,[],format_name)
    lines=text.splitlines(keepends=True)
    offsets=[];cursor=0
    for line in lines:
        offsets.append(cursor);cursor+=len(line)
    digest=hashlib.sha256(text.encode()).hexdigest()
    found=[];fence=None;i=0
    while i<len(lines):
        raw=lines[i].rstrip('\r\n');line_start=offsets[i]
        marker=re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$',raw)
        if fence:
            if marker and marker[1][0]==fence[0] and len(marker[1])>=fence[1] and not marker[2].strip():
                fence=None
            i+=1;continue
        if marker:
            fence=(marker[1][0],len(marker[1]));i+=1;continue
        atx=re.match(r'^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$',raw)
        title=None;kind=None;level=None;start=None;heading_end=line_start+len(raw)
        if atx:
            value=re.sub(r'[ \t]+#+[ \t]*$','',atx[2]).rstrip()
            title=value.lstrip();start=line_start+atx.start(2)+len(value)-len(value.lstrip())
            level=len(atx[1]);kind='markdown_atx'
        elif i+1<len(lines) and re.fullmatch(r' {0,3}(?:={3,}|-{3,})[ \t]*',lines[i+1].rstrip('\r\n')):
            # Avoid treating list items, blockquotes or code indentation as titles.
            if re.match(r'^ {0,3}(?![#>*+\-]|\d+[.)]\s)\S',raw):
                title=raw.strip();start=line_start+len(raw)-len(raw.lstrip())
                level=1 if lines[i+1].lstrip().startswith('=') else 2
                kind='markdown_setext';heading_end=offsets[i+1]+len(lines[i+1].rstrip('\r\n'));i+=1
        elif bold_headings:
            bold=re.fullmatch(r' {0,3}(?:\*\*(.+?)\*\*|__(.+?)__)[ \t]*',raw)
            if bold:
                group=1 if bold[1] is not None else 2
                value=bold[group];title=value.strip();start=line_start+bold.start(group)+len(value)-len(value.lstrip())
                level=7;kind='standalone_bold_heuristic'
        if title:
            sid='sec-'+hashlib.sha256(f'{document_id}\0{digest}\0{line_start}'.encode()).hexdigest()[:16]
            found.append(Section(sid,level,title,start,start+len(title),line_start,heading_end,len(text),len(text),
                                 bisect_right(offsets,line_start),[],kind))
        i+=1
    stack=[]
    for index,s in enumerate(found):
        if index:
            found[index-1].block_end=s.start
        while stack and stack[-1].level>=s.level:
            stack.pop().end=s.start
        s.path=[p.id for p in stack]+[s.id]
        stack.append(s)
    return DocumentStructure(text,document_id,found,format_name)


def locate_chunks(text, chunk_texts):
    """Locate ordered splitter output literally, allowing overlaps and repeats.

    Raises if a splitter transformed the source; never fabricates a location.
    Callers must supply chunks in their original document order.
    """
    cursor=0;spans=[]
    for chunk in chunk_texts:
        if not chunk:
            raise ValueError('Empty chunk cannot be located')
        start=text.find(chunk,cursor)
        if start<0:
            raise ValueError('Chunk is not an ordered literal source substring')
        spans.append((start,start+len(chunk)));cursor=start+1
    return spans
