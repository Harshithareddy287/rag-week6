import re
import sys

import chromadb

from src import config
from src.index import build_index
from src.llm import ask
from src.rerank import fmt_score, rerank
from src.retrieve import dense_search, hybrid_search

SYSTEM = (
    "You answer questions about company policy. "
    "Answer ONLY from the context blocks below. "
    "If the context does not contain the answer, say so. "
    "Cite the block numbers you used, like [1][3]. "
    "Answer in plain text, no markdown."
)


def get_collection():
    """Open the existing collection, or build it if missing."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        return client.get_collection(config.COLLECTION_NAME, embedding_function=None)
    collection, _ = build_index()
    return collection


def build_prompt(question, retrieved):
    # Only the chunk text goes in the prompt -- no doc_name, version, or date.
    blocks = "\n\n".join(f"[{i}] {text}" for i, (_, _, text) in enumerate(retrieved, start=1))
    return f"{SYSTEM}\n\nContext:\n{blocks}\n\nQuestion: {question}\nAnswer:"


def retrieve(collection, question, mode="rerank"):
    """Top-k chunks for a question.
    mode: "rerank" (hybrid top 20 -> cross-encoder -> top k),
          "hybrid" (vector + keyword, RRF) or "dense" (vector only)."""
    if mode == "rerank":
        return rerank(question, hybrid_search(collection, question, 20), config.TOP_K)
    if mode == "hybrid":
        return hybrid_search(collection, question, config.TOP_K)
    if mode == "dense":
        return dense_search(collection, question, config.TOP_K)
    raise ValueError(f"unknown mode: {mode!r}")


def answer(question, mode="rerank", collection=None):
    """Return (answer_text, sources, retrieved, bad_citations)."""
    retrieved = retrieve(collection or get_collection(), question, mode)
    text = ask(build_prompt(question, retrieved))

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources, bad = [], []
    for n in cited:
        if 1 <= n <= len(retrieved):
            meta = retrieved[n - 1][1]
            sources.append({"n": n, "doc_name": meta["doc_name"], "section": meta["section"]})
        else:
            bad.append(n)
    return text, sources, retrieved, bad


def main(argv=None):
    """CLI: python -m src.rag [--mode rerank|hybrid|dense] "question"."""
    args, mode = sys.argv[1:] if argv is None else argv, "rerank"
    if args[:1] == ["--mode"]:
        mode, args = args[1], args[2:]
    question = " ".join(args) or "What is the lodging limit in high-cost cities?"
    text, sources, retrieved, bad = answer(question, mode)

    print(f"Question: {question}  (mode={mode})\n")
    print(f"Answer:\n{text}\n")
    print("Sources:")
    for s in sources:
        print(f"  [{s['n']}] {s['doc_name']} § {s['section']}")
    if not sources:
        print("  (none cited)")
    for n in bad:
        print(f"  WARNING: cited [{n}] but context only has [1]..[{len(retrieved)}]")

    print("\nRetrieved:")
    for i, (score, meta, _) in enumerate(retrieved, start=1):
        shown = {"dense": f"distance={score:.4f}", "hybrid": f"rrf={score:.4f}",
                 "rerank": fmt_score(score)}[mode]
        print(f"  [{i}] {shown}  {meta['chunk_id']} § {meta['section']}")


if __name__ == "__main__":
    main()
