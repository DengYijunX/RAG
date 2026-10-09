"""Assistant first-pass review of generated answers, per QUALITY_RUBRIC.md.

The rubric fixes three dimensions scored 0/1/2 plus one explicit outcome, and a
closed set of error labels. This module asks the configured chat model to apply
that rubric to one answer, then **validates the reply like any other untrusted
input**:

- every dimension must be present with a value in ``{0, 1, 2, "na"}``;
- error labels must come from the rubric's closed list;
- a reply that fails validation is recorded as ``invalid`` and carries **no
  score** — it is never coerced to zero, because "the judge did not answer" and
  "the answer was wrong" are different findings.

This produces an **assistant first pass**, not human review. The rubric is
explicit that assistant scores must not be reported as human accuracy, so the
records keep ``human_*`` fields empty and every artefact says so.
"""
from dataclasses import dataclass, field
import json
import re

DIMENSIONS = ('key_points', 'citation_faithfulness', 'behavior_match')
ERROR_LABELS = ('retrieval_missing', 'unsupported_claim', 'wrong_scope',
                'missing_clarification', 'fabricated_action_or_data', 'annotation_issue')
SEVERITIES = ('high', 'medium', 'low')
VALID_SCORES = {0, 1, 2, 'na'}

SYSTEM_PROMPT = (
    '你是严格的答案审阅员，按给定评分标准逐项打分。'
    '只输出一个 JSON 对象，不要解释、不要 Markdown 代码块、不要多余文字。'
    '评分必须依据提供的证据片段与参考答案，不得凭你自己的知识判断对错。'
    '若证据不足以判断某项，就给 "na" 而不是猜测。'
)

RUBRIC = '''评分标准（严格照此执行）：
- key_points 要点完整性：2=正确覆盖全部必需要点与条件；1=部分正确但有关键遗漏；0=未回答核心或核心错误；"na"=预期澄清或拒答题。
- citation_faithfulness 引用忠实性：2=全部实质结论由所标注片段支持且型号版本匹配；1=有支持但部分结论缺引用、支持不充分或范围扩大；0=核心结论无依据、伪造来源或错版；"na"=没有实质事实主张的纯澄清。
- behavior_match 行为匹配：2=正确回答／澄清／说明不能回答；1=基本谨慎但澄清不足或先作无根据判断；0=应澄清却直接断言、应拒答却编造，或可回答题完全拒答；"na"=无。
- 无引用编号却输出实质操作步骤，引用忠实性不可给 2。
- error_labels 只能从这些里选（可多选，可为空）：
  retrieval_missing（语料有答案但检索漏掉必要证据）、unsupported_claim（片段不足却给实质结论）、
  wrong_scope（型号/年份/一般与特殊规则错用）、missing_clarification（缺关键前提却不问）、
  fabricated_action_or_data（虚构已执行操作、订单或企业数据、电话号码等）、annotation_issue（题目或 gold 有歧义）。
- severity 只能是 high/medium/low；error_labels 为空时 severity 用 "none"。'''


def build_prompt(question, answer, citations, reference_points, expected_action):
    parts = [f'问题：{question}', f'预期行为：{expected_action or "未标注"}', '']
    if reference_points:
        parts.append('参考要点（来自数据集标注，仅供核对要点是否覆盖）：')
        parts.extend(f'- {point}' for point in reference_points)
        parts.append('')
    parts.append('本次检索到的片段（回答只能依据这些片段）：')
    for position, citation in enumerate(citations, 1):
        text = (citation.get('content') or '')[:600]
        parts.append(f'[{position}] 来源={citation.get("title", "")} 位置={citation.get("start")}-{citation.get("end")}')
        parts.append(text)
    if not citations:
        parts.append('（本次没有检索到任何片段）')
    parts.append('')
    parts.append(f'模型回答：\n{answer}')
    parts.append('')
    parts.append('输出 JSON：{"key_points": <0|1|2|"na">, "citation_faithfulness": <0|1|2|"na">, '
                 '"behavior_match": <0|1|2|"na">, "error_labels": [...], "severity": "...", '
                 '"justification": "一句话，引用回答原句"}')
    return '\n'.join(parts)


def extract_json(text):
    """Pull the first JSON object out of a reply that may carry stray prose."""
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


@dataclass
class ReviewOutcome:
    """One judged answer. ``valid`` is False when the reply could not be used."""
    scores: dict = field(default_factory=dict)
    error_labels: list = field(default_factory=list)
    severity: str = ''
    justification: str = ''
    valid: bool = False
    reason: str = ''
    raw: str = ''


def validate_reply(payload):
    """Validate a judge reply into a ReviewOutcome, or an invalid one with a reason."""
    if not isinstance(payload, dict):
        return ReviewOutcome(valid=False, reason='reply_not_an_object')
    scores = {}
    for dimension in DIMENSIONS:
        if dimension not in payload:
            return ReviewOutcome(valid=False, reason=f'missing_dimension:{dimension}')
        value = payload[dimension]
        if isinstance(value, str) and value.strip().lower() in {'na', 'n/a'}:
            value = 'na'
        if value not in VALID_SCORES:
            return ReviewOutcome(valid=False, reason=f'invalid_score:{dimension}={value!r}')
        scores[dimension] = value
    labels = payload.get('error_labels', [])
    if labels in (None, ''):
        labels = []
    if not isinstance(labels, list) or any(label not in ERROR_LABELS for label in labels):
        return ReviewOutcome(valid=False, reason=f'invalid_error_labels:{labels!r}')
    severity = payload.get('severity', '')
    if labels and severity not in SEVERITIES:
        return ReviewOutcome(valid=False, reason=f'invalid_severity:{severity!r}')
    if not labels:
        severity = 'none'
    justification = payload.get('justification', '')
    if not isinstance(justification, str):
        return ReviewOutcome(valid=False, reason='invalid_justification')
    return ReviewOutcome(scores=scores, error_labels=sorted(set(labels)), severity=severity,
                         justification=justification[:500], valid=True, reason='ok')


class AnswerJudge:
    """Applies the rubric with the configured chat model.

    ``client_factory`` is injected so the validation path is testable offline.
    """

    def __init__(self, config, client_factory=None):
        self.config = config
        self._client_factory = client_factory or self._default_client
        self.calls = 0
        self.invalid = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_seconds = 0.0

    def _default_client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.config.llm_base_url, api_key=self.config.llm_api_key,
                      timeout=60, max_retries=1)

    def review(self, question, answer, citations=(), reference_points=(), expected_action=''):
        prompt = build_prompt(question, answer, list(citations), list(reference_points),
                              expected_action)
        self.calls += 1
        started = _now()
        try:
            with self._client_factory() as client:
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                              {'role': 'user', 'content': RUBRIC + '\n\n' + prompt}])
            raw = response.choices[0].message.content or ''
            usage = getattr(response, 'usage', None)
        except Exception as exc:
            self.total_seconds += _now() - started
            self.invalid += 1
            return ReviewOutcome(valid=False, reason=f'call_failed:{type(exc).__name__}')
        self.total_seconds += _now() - started
        if usage is not None:
            self.total_prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self.total_completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
        outcome = validate_reply(extract_json(raw))
        outcome.raw = raw[:400]
        if not outcome.valid:
            self.invalid += 1
        return outcome

    def stats(self):
        return {'calls': self.calls, 'invalid': self.invalid,
                'invalid_rate': self.invalid / self.calls if self.calls else None,
                'prompt_tokens': self.total_prompt_tokens,
                'completion_tokens': self.total_completion_tokens,
                'seconds': round(self.total_seconds, 2)}


def _now():
    import time

    return time.time()
