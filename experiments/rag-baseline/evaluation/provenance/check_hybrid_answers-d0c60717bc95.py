"""Predeclared diagnostic sample only; no overall answer-accuracy estimate."""
import argparse
import json
from pathlib import Path
import sys
import time
import baseline as b

EVAL = Path(__file__).resolve().parent
SAMPLES = {
    'accounting': [(qid, 'exact_then_dense') for qid in
                   ['ACC-Q041', 'ACC-Q059', 'ACC-Q098', 'ACC-Q114', 'ACC-Q085', 'ACC-Q116', 'ACC-Q118']]
                  + [('ACC-Q085', 'routed_rrf')],
    'smart-home': [('Q039', 'diverse_rrf'), ('Q040', 'diverse_rrf')],
}


def run(kb):
    candidates = []
    for path in sorted((EVAL/'hybrid_runs'/kb).glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest['status'] == 'completed' and manifest.get('revision'):
            candidates.append((path, manifest))
    path, source = candidates[-1]
    rows = {r['question_id']: r for r in b.read(path.with_name('results.jsonl'))}
    settings = b.settings(True)
    assert settings == source['model_config'] and source['split'] == 'dev'
    baseline_path = b.ROOT/source['baseline_manifest']
    baseline_rows = {r['question_id']: r for r in b.read(baseline_path.with_name('results.jsonl'))}
    output = path.with_name('diagnostic_answers.jsonl')
    if output.exists():
        raise ValueError('Existing diagnostic answers; refusing duplicate paid calls')
    manifest = {'status': 'running', 'knowledge_base': kb, 'split': 'dev', 'selection': SAMPLES[kb],
                'selection_bias': True, 'overall_accuracy_estimated': False,
                'model_config': settings, 'script_sha256': b.sha(Path(__file__)),
                'retrieval_manifest_sha256': b.sha(path), 'retrieval_results_sha256': b.sha(path.with_name('results.jsonl')),
                'selection_reason': 'known fixes, new regressions and unresolved multi-evidence cases in dev'}
    manifest_path = path.with_name('diagnostic_manifest.json')
    b.dump(manifest_path, manifest)
    client = b.client(settings['llm_url'], 'LLM_API_KEY')
    completed = 0
    try:
        for qid, variant in SAMPLES[kb]:
            row = rows[qid]
            hits = row[variant]['hits']
            context = '\n\n---\n\n'.join(f'[来源: {h["source"]}]\n{h["content"]}' for h in hits)
            start = time.perf_counter()
            response = client.chat.completions.create(model=settings['llm_model'], messages=[
                {'role': 'system', 'content': '你是一个基于文档回答问题的助手。请根据提供的文档内容回答问题。如果文档中没有相关信息，请如实告知。'},
                {'role': 'user', 'content': f'参考文档：\n\n{context}\n\n问题：{row["question"]}'}])
            answer = response.choices[0].message.content
            assert answer
            record = {'question_id': qid, 'question': row['question'], 'variant': variant, 'answer': answer,
                      'baseline_answer': baseline_rows[qid]['answer'], 'usage': response.usage.model_dump() if response.usage else None,
                      'generation_seconds': time.perf_counter()-start, 'context_chars': row[variant]['context_chars'],
                      'human_domain_review': 'pending', 'answer_correctness_score': None}
            with output.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            completed += 1
            print(kb, qid, variant, 'saved', flush=True)
        manifest.update(status='completed', actual_generation_calls=completed, answers_sha256=b.sha(output))
        b.dump(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status='failed', actual_generation_calls=completed, error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        b.dump(manifest_path, manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', choices=list(SAMPLES), required=True)
    args = parser.parse_args()
    try:
        run(args.kb)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}), file=sys.stderr)
        sys.exit(2)
