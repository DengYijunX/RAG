"""Find the working rerank endpoint path for the configured provider.

compatible-mode exposes chat and embeddings, but reranking is often only on the
provider's native API. This sends one minimal request per candidate path and
reports the status, so a 404 (wrong path) can be told apart from a 400/422
(path exists, body or model name wrong) without guessing.

    .venv/Scripts/python experiments/rag-baseline/evaluation/rerank_path_probe.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import load_config

BODY = {'model': None, 'query': '测试问题', 'documents': ['候选一', '候选二'], 'top_n': 2}


def post(url, api_key, body, timeout=30):
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode('utf-8', 'replace')
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def main():
    config = load_config()
    base = config.rerank_base_url.rstrip('/')
    host = base.split('/compatible-mode')[0]
    model = config.rerank_model

    candidates = []
    for prefix, label in ((base, 'compatible-mode'), (host, 'host-root')):
        for path in ('/rerank', '/v1/rerank', '/text-rerank',
                     '/api/v1/services/rerank/text-rerank/text-rerank'):
            candidates.append((f'{prefix}{path}', f'{label}{path}'))

    print(f'model = {model}')
    print(f'key   = {"已配置" if config.rerank_api_key else "缺失"}')
    print()
    for url, label in candidates:
        body = dict(BODY, model=model)
        status, text = post(url, config.rerank_api_key, body)
        summary = text[:220].replace('\n', ' ')
        print(f'  [{status}] {label}')
        print(f'        {summary}')
        if status == 200:
            print('        ^^^ 可用')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
