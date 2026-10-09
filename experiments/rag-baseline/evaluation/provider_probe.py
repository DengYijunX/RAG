"""Check that the configured embedding and rerank endpoints actually work.

Run this first after changing providers. It makes exactly one call to each
endpoint and prints the shapes that matter:

- embedding: vector dimension, so an index can be rebuilt against it;
- rerank: whether the response matches the expected ``results[].index`` /
  ``relevance_score`` shape, because a different shape would make every rerank
  fall back silently.

    .venv/Scripts/python experiments/rag-baseline/evaluation/provider_probe.py

It never prints credentials, only whether they are present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import load_config
from specialized_rerank import HttpRerankClient, parse_rerank_response


def looks_like_a_placeholder(value):
    """True when a key still carries the template text instead of a real key.

    ``.env.example`` ships ``sk-xxx`` and the local template ships
    ``sk-在此填入``; either will fail deep inside the HTTP stack with an
    encoding error, which reads like a network problem. Catch it here instead.
    """
    if not value:
        return True
    if not value.isascii():
        return True
    return value.rstrip('/') in {'sk-xxx', 'sk-在此填入', 'xxx', 'your-key'}


def probe_embedding(config):
    from openai import OpenAI

    print('--- Embedding ---')
    print(f'  base_url = {config.embedding_base_url}')
    print(f'  model    = {config.embedding_model}')
    print(f'  api_key  = {"已配置" if config.embedding_api_key else "缺失"}')
    if looks_like_a_placeholder(config.embedding_api_key):
        print('  未填真实密钥（仍是 .env 里的占位文本），跳过调用')
        return False, None
    try:
        with OpenAI(base_url=config.embedding_base_url, api_key=config.embedding_api_key,
                    timeout=60, max_retries=1) as client:
            response = client.embeddings.create(model=config.embedding_model, input=['连通性测试'])
        vector = response.data[0].embedding
        print(f'  OK 维度 = {len(vector)}，返回条数 = {len(response.data)}')
        return True, len(vector)
    except Exception as exc:
        print(f'  失败：{type(exc).__name__}: {str(exc)[:200]}')
        return False, None


def probe_rerank(config):
    print('--- Rerank ---')
    print(f'  base_url = {config.rerank_base_url or "缺失"}')
    print(f'  model    = {config.rerank_model or "缺失"}')
    print(f'  api_key  = {"已配置" if config.rerank_api_key else "缺失"}')
    if not config.rerank_configured or looks_like_a_placeholder(config.rerank_api_key):
        print('  未配置完整或仍是占位密钥，跳过调用')
        return False
    documents = ['第五条 业务分部，是指企业内可区分的组成部分。',
                 '第十一条 属于在某一时段内履行履约义务。',
                 '与问题无关的一段文字。']
    try:
        client = HttpRerankClient(config.rerank_base_url, config.rerank_api_key,
                                  config.rerank_model)
        print(f'  POST {client.url}')
        payload = client.rerank('分部报告的业务分部指什么？', documents)
        print(f'  原始响应键：{sorted(payload) if isinstance(payload, dict) else type(payload).__name__}')
        print(f'  原始响应：{json.dumps(payload, ensure_ascii=False)[:400]}')
        try:
            parsed = parse_rerank_response(payload, len(documents))
        except ValueError as exc:
            print(f'  !! 响应形态与预期不符：{exc}')
            print('     期望 results: [{"index": 0-based, "relevance_score": number}, ...]')
            return False
        for index, score in sorted(parsed, key=lambda item: -item[1]):
            print(f'    index={index} score={score:.4f}  {documents[index][:24]}')
        return True
    except Exception as exc:
        print(f'  失败：{type(exc).__name__}: {str(exc)[:300]}')
        return False


def main():
    config = load_config()
    print(f'配置来源：{REPO / ".env"}')
    print(f'LLM      = {config.llm_base_url} / {config.llm_model}\n')
    embedding_ok, dimension = probe_embedding(config)
    print()
    rerank_ok = probe_rerank(config)
    print()
    print('=== 结论 ===')
    print(f'  Embedding {"可用" if embedding_ok else "不可用"}'
          + (f'（维度 {dimension}）' if dimension else ''))
    print(f'  Rerank    {"可用" if rerank_ok else "不可用或未配置"}')
    if embedding_ok and dimension and dimension != 1024:
        print(f'  注意：维度 {dimension} 与既有索引不同，必须重建索引后重跑基线')
    return 0 if (embedding_ok and rerank_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
