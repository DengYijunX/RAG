import argparse
from rag.config import load_config


def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG CLI - 文档索引与检索问答",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="索引文档目录")
    index_parser.add_argument("path", help="文档目录路径")
    index_parser.add_argument("--glob", default="*", help="文件匹配模式，默认 *")

    query_parser = subparsers.add_parser("query", help="检索并回答问题")
    query_parser.add_argument("question", help="要查询的问题")
    query_parser.add_argument("--top-k", type=int, help="检索数量，默认从 .env 读取")

    args = parser.parse_args()

    if args.command == "index":
        _cmd_index(args, config)
    elif args.command == "query":
        _cmd_query(args, config)
    else:
        parser.print_help()


def _cmd_index(args, config):
    from rag.loader import load_from_dir
    from rag.splitter import split
    from rag.embedder import embed
    from rag.store import reset_collection, add

    print(f"正在索引目录: {args.path}")
    docs = load_from_dir(args.path, args.glob)
    print(f"已加载 {len(docs)} 个文档")

    if not docs:
        print("未找到支持的文档文件（.md / .txt / .pdf）")
        return

    chunks = split(docs, config.chunk_size, config.chunk_overlap)
    print(f"已切分为 {len(chunks)} 个 chunk")

    print("正在进行向量化...")
    batch_size = 20
    all_vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embed([c.content for c in batch], config)
        all_vectors.extend(vectors)
        print(f"  向量化 {min(i + batch_size, len(chunks))}/{len(chunks)}")
    print("向量化完成")

    reset_collection(config)
    add(chunks, all_vectors, config)
    print(f"索引完成，共 {len(chunks)} 个 chunk 已写入向量库")


def _cmd_query(args, config):
    from rag.retriever import retrieve
    from rag.generator import generate

    print(f"正在检索: {args.question}")
    chunks = retrieve(args.question, config, top_k=args.top_k)

    print(f"已检索到 {len(chunks)} 个相关片段，正在生成回答...\n")
    answer = generate(args.question, chunks, config)
    print(answer)
    print("\n--- 引用来源 ---")
    seen = set()
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        if source not in seen:
            print(f"  - {source}")
            seen.add(source)
