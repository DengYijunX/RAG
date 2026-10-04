"""Run a predeclared dev-only sample against the actual application service."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import baseline as b
from rag.config import load_config
from rag.knowledge import KnowledgeService, normalize

EVAL = Path(__file__).resolve().parent


def select_questions(kb, sample, questions):
    if sample['split'] != 'dev':
        raise ValueError('Only dev diagnosis is allowed')
    requested = sample['knowledge_bases'][kb]
    if len(requested) != len(set(requested)):
        raise ValueError('Duplicate sample IDs')
    lookup = {q['id']: q for q in questions}
    selected = [lookup[key] for key in requested]
    if any(q['split'] != 'dev' for q in selected):
        raise ValueError('Sample may not contain held-out questions')
    return selected


def citation_checks(response, corpus):
    citations = response['citations']
    ids = [c['citation'] for c in citations]
    references = [int(n) for n in re.findall(r'\[(\d+)\]', response['answer'])]
    checks = {'unique_citation_ids': len(ids) == len(set(ids)),
              'referenced_ids_exist': all(n in ids for n in references),
              'all_sources_in_selected_kb': all(c['source'] in corpus.sources for c in citations),
              'all_snippets_in_source': all(c['source'] in corpus.sources and normalize(c['content']) in corpus.sources[c['source']]['normalized'] for c in citations)}
    return {'checks': checks, 'passed': all(checks.values()),
            'used_citation_ids': sorted(set(references)), 'semantic_faithfulness_measured': False}


class ObservedService(KnowledgeService):
    def __init__(self, config):
        super().__init__(config)
        self.embedding_calls = self.generation_calls = 0

    def vectors(self, texts):
        self.embedding_calls += 1
        return super().vectors(texts)

    def generate(self, question, citations):
        self.generation_calls += 1
        return super().generate(question, citations)


def run(kb):
    sample_path = EVAL/'quality_sample.json'
    sample = json.loads(sample_path.read_text())
    question_path = EVAL/'v2'/kb/'questions.jsonl'
    questions = select_questions(kb, sample, b.read(question_path))
    config = load_config()
    service = ObservedService(config)
    corpus = service.corpus(kb)
    active = service.active(kb)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = EVAL/'quality_runs'/kb/run_id
    out.mkdir(parents=True)
    application = ['rag/knowledge.py', 'rag/knowledge_routing.py', 'rag/config.py']
    manifest = {'status': 'running', 'split': 'dev', 'knowledge_base': kb, 'sample_ids': [q['id'] for q in questions],
                'selection_bias': True, 'overall_accuracy_estimated': False, 'human_review_status': 'pending',
                'sample_sha256': b.sha(sample_path), 'rubric_sha256': b.sha(EVAL/'QUALITY_RUBRIC.md'),
                'questions_sha256': b.sha(question_path), 'script_sha256': b.sha(Path(__file__)),
                'application_sha256': {name: b.sha(b.REPO/name) for name in application},
                'repository_commit': subprocess.check_output(['git', '-C', str(b.REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                'index': active, 'llm_model': config.llm_model, 'llm_url': config.llm_base_url}
    b.dump(out/'manifest.json', manifest)
    completed = 0
    try:
        for q in questions:
            start = time.perf_counter()
            before = (service.embedding_calls, service.generation_calls)
            # Gold and question ID never enter the actual application call.
            response = service.query(kb, q['question'])
            assert response['answer']
            row = {'question_id': q['id'], 'question': q['question'], 'response': response,
                   'seconds': time.perf_counter()-start,
                   'embedding_calls': service.embedding_calls-before[0], 'generation_calls': service.generation_calls-before[1],
                   'automatic_checks': citation_checks(response, corpus)}
            with (out/'results.jsonl').open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            review = {'question_id': q['id'], 'question': q['question'], 'expected_action': q['expected_action'],
                      'expected_points': q['expected_points'], 'expected_evidence': [corpus.parents[key] for key in q['evidence_ids']],
                      'response': response, 'annotation_review': None, 'assistant_scores': None,
                      'human_scores': None, 'human_reviewer': None}
            with (out/'review_packet.jsonl').open('a') as handle:
                handle.write(json.dumps(review, ensure_ascii=False)+'\n')
            completed += 1
            print(kb, q['id'], response['status'], 'saved', flush=True)
        manifest.update(status='completed', completed_questions=completed,
                        embedding_calls=service.embedding_calls, generation_calls=service.generation_calls,
                        results_sha256=b.sha(out/'results.jsonl'), review_packet_sha256=b.sha(out/'review_packet.jsonl'))
        b.dump(out/'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', completed_questions=completed, error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        b.dump(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', choices=list(b.CONFIG), required=True)
    args = parser.parse_args()
    try:
        run(args.kb)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}), file=sys.stderr)
        sys.exit(2)
