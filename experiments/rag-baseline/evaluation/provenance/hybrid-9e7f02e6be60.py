"""Dev-only BM25/RRF and query-derived metadata routing; no gold in retriever.

Reuses original Chroma vectors and independently cached query embeddings. The
runner validates provenance before scoring; rerunning retrieval makes no API calls.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

import baseline as b
import structure_experiment as s

EVAL = Path(__file__).resolve().parent
PARAMS = {'bm25_k1': 1.5, 'bm25_b': .75, 'rrf_k': 60,
          'candidate_depth': 20, 'max_hits': 4, 'body_char_budget': 2000}


def number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip('零〇一二三四五六七八九', [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    total = current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in '十百千':
            total += (current or 1) * {'十': 10, '百': 100, '千': 1000}[char]
            current = 0
        else:
            raise ValueError('Unsupported numeral')
    return total + current


def tokens(text):
    """No segmentation dependency: Latin tokens and Chinese character bigrams."""
    output = []
    for part in re.findall(r'[a-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]+', text.lower()):
        if '\u4e00' <= part[0] <= '\u9fff':
            output.extend(part[i:i+2] for i in range(len(part)-1))
            if len(part) == 1:
                output.append(part)
        else:
            output.append(part)
    return output


class BM25:
    def __init__(self, texts):
        self.counts = {key: Counter(tokens(value)) for key, value in texts.items()}
        self.lengths = {key: sum(value.values()) for key, value in self.counts.items()}
        self.average = sum(self.lengths.values()) / max(1, len(texts))
        frequencies = Counter(term for counts in self.counts.values() for term in counts)
        self.idf = {term: math.log(1 + (len(texts)-freq+.5)/(freq+.5))
                    for term, freq in frequencies.items()}

    def rank(self, query, allowed):
        scores = {}
        for key in allowed:
            norm = PARAMS['bm25_k1'] * (1-PARAMS['bm25_b'] + PARAMS['bm25_b'] * self.lengths[key]/max(self.average, 1))
            score = 0
            for term in set(tokens(query)):
                freq = self.counts[key].get(term, 0)
                score += self.idf.get(term, 0) * freq * (PARAMS['bm25_k1']+1)/(freq+norm)
            if score > 0:
                scores[key] = score
        return sorted(scores, key=lambda key: (-scores[key], key))[:PARAMS['candidate_depth']]


def fuse(*rankings):
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] += 1/(PARAMS['rrf_k']+rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def pack(candidates):
    """Whole source spans, bounded context, no duplicate/mostly overlapping hits."""
    selected = []
    for hit in candidates:
        overlaps = [(max(hit['start'], old['start']), min(hit['end'], old['end']))
                    for old in selected if old['source'] == hit['source']]
        overlap = b.union_size([(a, z) for a, z in overlaps if z > a])
        if overlap >= .8 * (hit['end']-hit['start']):
            continue
        if sum(len(old['content']) for old in selected)+len(hit['content']) > PARAMS['body_char_budget']:
            continue
        selected.append(hit)
        if len(selected) == PARAMS['max_hits']:
            break
    return selected


def route_accounting(query, documents):
    numerals = r'[0-9零〇一二三四五六七八九十百千]+'
    families = {str(number(n)) for n in re.findall(r'第('+numerals+r')号', query)}
    years = {int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', query)}
    articles = {number(n) for n in re.findall(r'第('+numerals+r')条', query)}
    # Derive aliases exclusively from corpus titles, not evaluation labels.
    aliases = {}
    for doc in documents:
        alias = re.sub(r'^企业会计准则(?:第\d+号)?[—－-]+', '', doc['title'])
        alias = re.sub(r'[（(]\d{4}[）)]$', '', alias)
        aliases[alias] = doc['standard_id'].lstrip('0') or '0'
    matches = [alias for alias in aliases if alias in query]
    if not families and matches:
        longest = max(map(len, matches))
        families = {aliases[alias] for alias in matches if len(alias) == longest}
    docs = [d for d in documents if (d['standard_id'].lstrip('0') or '0') in families
            and (not years or d['version_year'] in years)]
    # Multiple standards/clauses can have ambiguous pairings; do not cross-product them.
    exact = bool(len(families) == 1 and years and len(articles) == 1)
    return {'families': sorted(families), 'years': sorted(years), 'articles': sorted(articles),
            'sources': [d['id']+'.md' for d in docs] if families else None,
            'exact_enabled': exact, 'version_unspecified': bool(families and not years),
            'unmatched_explicit_scope': bool(families and not docs)}


class Retriever:
    def __init__(self, kb, chunks, parents, records, documents, collection):
        self.kb = kb
        self.lookup = {c['id']: c for c in chunks}
        self.parents = parents
        self.records = records
        self.documents = documents
        self.collection = collection
        self.sources = sorted({c['source'] for c in chunks})
        titles = {d['id']+'.md': d['title'] for d in documents} if kb == 'accounting' else {}
        self.lexical = BM25({c['id']: titles.get(c['source'], Path(c['source']).stem)+'\n'+c['content'] for c in chunks})

    def dense(self, vector, sources=None, depth=20):
        allowed = [key for key, c in self.lookup.items() if sources is None or c['source'] in sources]
        if not allowed:
            return []
        kwargs = {'where': {'source': {'$in': sources}}} if sources is not None else {}
        result = self.collection.query(query_embeddings=[vector], n_results=min(depth, len(allowed)), include=['distances'], **kwargs)
        return result['ids'][0]

    def retrieve(self, query, vector):
        dense = self.dense(vector)
        lexical = self.lexical.rank(query, self.lookup)
        fused = fuse(dense, lexical)
        replay = self.dense(vector, depth=4)
        routing = {'sources': None, 'exact_enabled': False}
        mandatory = []
        if self.kb == 'accounting':
            routing = route_accounting(query, self.documents)
            if routing['exact_enabled']:
                mandatory = [self.parents[r['id']] for r in self.records
                             if r['document_id']+'.md' in routing['sources'] and r['article_number'] in routing['articles']]
        else:
            products = sorted(set(re.findall(r'(?<![A-Z0-9])[A-Z]\d+(?![A-Z0-9])', query.upper())))
            groups = [source for source in self.sources if any(re.search(r'(?<![A-Z0-9])'+re.escape(p)+r'(?![A-Z0-9])', source.upper()) for p in products)]
            policy = [source for source in self.sources if '售后' in source]
            routing = {'products': products, 'sources': groups+policy if products else None,
                       'diversity_sources': groups, 'exact_enabled': False}
        scope = routing['sources']
        allowed = [key for key, hit in self.lookup.items() if scope is None or hit['source'] in scope]
        scoped = fuse(self.dense(vector, scope), self.lexical.rank(query, allowed))
        routed_hits = pack(mandatory+[self.lookup[key] for key in scoped])
        diverse_sources = routing.get('diversity_sources', [])
        if self.kb == 'accounting' and len(routing.get('years', [])) > 1:
            diverse_sources = scope or []
        diverse = []
        if len(diverse_sources) > 1:
            for source in diverse_sources:
                first = next((self.lookup[key] for key in scoped if self.lookup[key]['source'] == source), None)
                if first is not None:
                    diverse.append(first)
        return {'dense_replay': [self.lookup[key] for key in replay],
                'exact_then_dense': pack(mandatory+[self.lookup[key] for key in replay]) if mandatory else [self.lookup[key] for key in replay],
                'rrf': pack([self.lookup[key] for key in fused]),
                'routed_rrf': routed_hits,
                'diverse_rrf': pack(mandatory+diverse+[self.lookup[key] for key in scoped])}, routing


def run(kb, cache_root):
    import chromadb
    from chromadb.config import Settings
    import numpy as np
    docs, chunks, index = b.prepare(kb)
    _, parents, _, _ = s.units(kb)
    questions = [q for q in b.validate_questions(kb, index) if q['split'] == 'dev']
    baseline_path, baseline_manifest = s.selected_baseline(kb)
    old = {r['question_id']: r for r in b.read(baseline_path.with_name('results.jsonl'))}
    cache = cache_root/kb/baseline_path.parent.name
    assert b.sha(cache/'manifest.json') == b.sha(baseline_path)
    for source, digest in baseline_manifest['source_files'].items():
        assert b.sha(b.ROOT/source) == digest
    assert b.sha(b.ROOT/'evaluation/v2'/kb/'questions.jsonl') == baseline_manifest['question_sha256']
    prior_path, prior = next((p, json.loads(p.read_text())) for p in sorted((EVAL/'experiments'/kb).glob('*/manifest.json'))
                            if json.loads(p.read_text()).get('contextual_embedding') and json.loads(p.read_text())['status'] == 'completed')
    assert prior['questions_sha256'] == baseline_manifest['question_sha256']
    assert prior['model_config'] == baseline_manifest['config']
    prior_rows = b.read(prior_path.with_name('results.jsonl'))
    assert [r['question_id'] for r in prior_rows] == [q['id'] for q in questions]
    vector_path = prior_path.parent/'chroma/query_vectors.npy'
    vectors = np.load(vector_path)
    assert len(vectors) == len(questions) and np.isfinite(vectors).all()
    client = chromadb.PersistentClient(path=str(cache/'chroma'), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection('rag_documents', embedding_function=None)
    stored = collection.get(include=['documents', 'metadatas'])
    lookup = {c['id']: c for c in chunks}
    assert set(stored['ids']) == set(lookup)
    for key, text, meta in zip(stored['ids'], stored['documents'], stored['metadatas']):
        assert lookup[key]['content'] == text and lookup[key]['source'] == meta['source']
    retriever = Retriever(kb, chunks, parents, b.read(b.ROOT/kb/b.CONFIG[kb][1]), b.read(b.ROOT/kb/'documents.jsonl'), collection)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out = EVAL/'hybrid_runs'/kb/run_id
    out.mkdir(parents=True)
    manifest = {'status': 'running', 'kind': 'hybrid_ablation', 'knowledge_base': kb, 'split': 'dev',
                'revision': 'v2_add_exact_only_fallback_after_v1_dev_regressions',
                'parameters': PARAMS, 'model_config': prior['model_config'], 'question_count': len(questions),
                'questions_sha256': prior['questions_sha256'], 'source_files': baseline_manifest['source_files'],
                'baseline_manifest': str(baseline_path.relative_to(b.ROOT)), 'baseline_manifest_sha256': b.sha(baseline_path),
                'query_cache_manifest': str(prior_path.relative_to(b.ROOT)), 'query_vectors_sha256': b.sha(vector_path),
                'script_sha256': b.sha(Path(__file__)), 'baseline_helper_sha256': b.sha(Path(b.__file__)),
                'structure_helper_sha256': b.sha(Path(s.__file__)),
                'retrieval_api_calls': 0, 'gold_used_for_retrieval': False,
                'latency_scope': 'local replay; excludes embedding/network and is not production latency'}
    b.dump(out/'manifest.json', manifest)
    rows = []
    try:
        for q, vector in zip(questions, vectors):
            start = time.perf_counter()
            # Only question text + cached embedding enter retrieval, never labels/category/IDs.
            variants, routing = retriever.retrieve(q['question'], vector.tolist())
            elapsed = time.perf_counter()-start
            variants['baseline'] = old[q['id']]['hits']
            row = {'question_id': q['id'], 'question': q['question'], 'category': q['category'],
                   'expected_action': q['expected_action'], 'routing': routing, 'local_retrieval_seconds': elapsed}
            for name, hits in variants.items():
                metrics = b.score(q['evidence_ids'], hits, index) if q['expected_action'] in ('answer', 'flag_conflict') else None
                row[name] = {'hits': hits, 'metrics': metrics, 'context_chars': sum(len(h['content']) for h in hits)}
            assert row['baseline']['metrics'] == old[q['id']]['metrics']
            rows.append(row)
        summary = {name: s.aggregate(rows, name) for name in variants}
        summary['dense_replay_hit_order_changes'] = [r['question_id'] for r in rows if [h['id'] for h in r['baseline']['hits']] != [h['id'] for h in r['dense_replay']['hits']]]
        summary['changes_vs_baseline'] = {name: {change: [r['question_id'] for r in rows if r[name]['metrics'] is not None and compare(r[name]['metrics']['coverage_recall_at_k'], r['baseline']['metrics']['coverage_recall_at_k'])]
                                               for change, compare in [('improved', lambda x,y:x>y), ('regressed', lambda x,y:x<y)]}
                                          for name in variants if name != 'baseline'}
        with (out/'results.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        b.dump(out/'summary.json', summary)
        manifest.update(status='completed', completed_questions=len(rows))
        b.dump(out/'manifest.json', manifest)
        print(json.dumps({'output': str(out), 'summary': summary}, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        b.dump(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', choices=list(b.CONFIG), required=True)
    parser.add_argument('--baseline-cache-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.kb, args.baseline_cache_root)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        sys.exit(2)
