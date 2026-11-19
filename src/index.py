from collections import Counter
from pathlib import Path

import chromadb

from src import config
from src.chunking import chunk_documents
from src.embeddings import count_tokens, embed, get_model

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHECK_QUERY = "What is the lodging limit in high-cost cities?"


def build_index(persist_dir=config.CHROMA_DIR):
    """Rebuild the collection from docs/. Returns (collection, chunks)."""
    chunks = chunk_documents(DOCS_DIR)

    client = chromadb.PersistentClient(path=str(persist_dir))
    if config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": config.EMBEDDING_MODEL},
        embedding_function=None,
    )

    collection.add(
        ids=[c["metadata"]["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
    )
    return collection, chunks


def search(collection, query, k=3):
    """Return the top-k results as (distance, metadata, text) tuples."""
    res = collection.query(query_embeddings=embed([query]), n_results=k)
    return list(zip(res["distances"][0], res["metadatas"][0], res["documents"][0]))


if __name__ == "__main__":
    collection, chunks = build_index()

    print("Chunks per doc:")
    for doc, n in sorted(Counter(c["metadata"]["doc_name"] for c in chunks).items()):
        print(f"  {doc}: {n}")
    print(f"  TOTAL: {len(chunks)} (collection count = {collection.count()})")

    tokens = [count_tokens(c["text"]) for c in chunks]
    limit = get_model().max_seq_length
    print(f"\nTokens per chunk: min={min(tokens)} avg={sum(tokens) / len(tokens):.1f} "
          f"max={max(tokens)}  (model limit = {limit})")
    print(f"Chunks over limit: {sum(t > limit for t in tokens)}")

    print("\nChunks containing a table:")
    for c in chunks:
        if "|---" in c["text"]:
            print(f"  {c['metadata']['chunk_id']}  ({c['metadata']['section']})")

    print(f"\nFULL-SCALE RETRIEVAL CHECK: {CHECK_QUERY!r}")
    results = search(collection, CHECK_QUERY)
    for rank, (dist, meta, _) in enumerate(results, start=1):
        print(f"  {rank}. distance={dist:.4f}  {meta['doc_name']}  {meta['section']}")
    print("PASS" if results[0][1]["section"].startswith("EXP-204") else "FAIL")
