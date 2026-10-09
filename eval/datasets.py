"""Ground-truth loading for the evaluation framework.

WeKnora defines ground truth in five Parquet files whose join key is
``query_id``; a sample carries ``question``, ``relevant_passage_ids`` and a
reference ``answer``, which lets retrieval errors and generation errors be
scored separately. This project keeps that split and reads it from the JSONL
splits already used by the existing experiments:

- ``experiments/rag-baseline/questions.jsonl``            smart-home
- ``experiments/rag-baseline/accounting/questions*.jsonl`` accounting

Passage-level labels are optional and graded. The accounting splits ship
reference excerpts without passage ids (relevance is expressed by which
article the question names), so a question may legitimately carry no labels;
metrics must report how many questions were actually scored instead of
treating missing labels as misses.

The loader is a pure file reader: it never calls a model and never touches
the vector store.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

QUESTION_ID_FIELDS = ('id', 'query_id', 'question_id')
QUESTION_TEXT_FIELDS = ('question', 'query')
# ``evidence_ids`` marks relevance strictly; ``reference_excerpts`` is the
# weaker same-source evidence used by the accounting split.
RELEVANCE_FIELDS = ('relevant_passage_ids', 'evidence_ids')

# Fields the dataset layer consumes itself; anything else is kept in
# ``metadata`` so a new question-generation pass cannot silently drop columns.
# Deliberately narrow: only fields whose value the loader has already turned
# into a first-class attribute are listed here.
RESERVED_FIELDS = set(QUESTION_ID_FIELDS + QUESTION_TEXT_FIELDS + RELEVANCE_FIELDS + (
    'answer', 'reference_answer', 'split', 'knowledge_base',
    'reference_excerpts', 'relevance'))


class DatasetError(Exception):
    """The dataset file is malformed."""


@dataclass
class Question:
    """One graded question plus the passage labels it should retrieve."""
    id: str
    question: str
    relevant_passage_ids: list = field(default_factory=list)
    answer: str = ''
    split: str = ''
    knowledge_base: str = ''
    metadata: dict = field(default_factory=dict)

    @property
    def graded(self):
        """True when this question carries retrieval labels at all."""
        return bool(self.relevant_passage_ids)

    def relevance(self, passage_id):
        """1.0 when the passage is labelled relevant, else 0.0."""
        return 1.0 if passage_id in self.relevant_passage_ids else 0.0


@dataclass
class Dataset:
    """A named set of questions, optionally filtered to one split."""
    name: str
    questions: list = field(default_factory=list)

    def __len__(self):
        return len(self.questions)

    @property
    def graded(self):
        return [question for question in self.questions if question.graded]

    def splits(self):
        return sorted({question.split for question in self.questions if question.split})

    def summary(self):
        return {'name': self.name, 'questions': len(self.questions),
                'graded': len(self.graded), 'splits': self.splits()}


def _first(record, fields):
    for name in fields:
        if name in record and record[name] not in (None, ''):
            return record[name]
    return None


def passage_ids_from_excerpts(record):
    """Passage ids carried by ``reference_excerpts``.

    The accounting split expresses evidence as excerpts whose ``article_id``
    is the same identifier used by the retrieval index (``articles.jsonl``),
    so those questions are graded without guessing: an excerpt without an id
    is dropped instead of being matched by text.
    """
    ids = []
    for excerpt in record.get('reference_excerpts') or ():
        if isinstance(excerpt, dict):
            value = excerpt.get('article_id') or excerpt.get('passage_id')
            if value:
                ids.append(str(value))
    return ids


def parse_question(record, path, line_number):
    """Validate one JSONL record into a Question.

    Duplicate ids are checked by the caller, which owns the whole file.
    """
    if not isinstance(record, dict):
        raise DatasetError(f'{path}:{line_number} 不是 JSON 对象')
    identifier = _first(record, QUESTION_ID_FIELDS)
    if not identifier:
        raise DatasetError(f'{path}:{line_number} 缺少题目 ID（{" / ".join(QUESTION_ID_FIELDS)}）')
    text = _first(record, QUESTION_TEXT_FIELDS)
    if not text or not str(text).strip():
        raise DatasetError(f'{path}:{line_number} 缺少题目文本')
    labels = []
    for name in RELEVANCE_FIELDS:
        value = record.get(name)
        if isinstance(value, list):
            labels.extend(str(item) for item in value)
    labels.extend(passage_ids_from_excerpts(record))
    labels = list(dict.fromkeys(label for label in labels if label))
    answer = _first(record, ('answer', 'reference_answer')) or ''
    metadata = {key: value for key, value in record.items() if key not in RESERVED_FIELDS}
    return Question(id=str(identifier), question=str(text).strip(), relevant_passage_ids=labels,
                    answer=str(answer), split=str(record.get('split') or ''),
                    knowledge_base=str(record.get('knowledge_base') or ''), metadata=metadata)


def load_dataset(path, split=None, name=None):
    """Read a JSONL ground-truth file.

    ``split`` filters on the ``split`` field when given. Raises DatasetError
    on duplicate ids or malformed lines instead of skipping them, because a
    silently shortened evaluation set makes every later metric wrong.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f'数据集文件不存在：{path}')
    questions, seen = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f'{path}:{number} 不是合法 JSON：{exc.msg}') from exc
        question = parse_question(record, path, number)
        if question.id in seen:
            raise DatasetError(f'{path}:{number} 题目 ID 重复：{question.id}')
        seen.add(question.id)
        if split and question.split != split:
            continue
        questions.append(question)
    return Dataset(name=name or path.stem, questions=questions)


def load_knowledge_base_dataset(root, knowledge_base, split=None, version='v1'):
    """Load the question file of one knowledge base and dataset version.

    ``root`` is ``experiments/rag-baseline``. Two sets exist and they must not
    be confused:

    - ``v1`` (default) is the delivered set used by the recorded experiments:
      201 questions = smart-home 60 + accounting 141.
    - ``v2`` lives under ``evaluation/v2`` and is the larger review-candidate
      set: 225 questions = smart-home 72 + accounting 153. Its annotations are
      still ``draft_needs_domain_review``, so it is not signed-off gold.

    The accounting base keeps its questions in its own directory and the
    smart-home base at the root; a base directory wins, so the lookup is a
    rule rather than a guess.
    """
    root = Path(root)
    if version not in ('v1', 'v2'):
        raise DatasetError(f'未知数据集版本：{version}（支持 v1 / v2）')
    base = root if version == 'v1' else root / 'evaluation' / version
    candidates = (base / knowledge_base / 'questions.jsonl', base / 'questions.jsonl')
    for candidate in candidates:
        if candidate.is_file():
            return load_dataset(candidate, split=split,
                                name=f'{knowledge_base}:{version}:{candidate.stem}')
    raise DatasetError(f'未找到 {knowledge_base} 的 {version} 题目文件，查找过：'
                       + '、'.join(str(item) for item in candidates))


def load_all(baseline_root, split=None, version='v1'):
    """Load both knowledge bases; metrics are always reported per base."""
    return {name: load_knowledge_base_dataset(baseline_root, name, split=split, version=version)
            for name in ('smart-home', 'accounting')}

