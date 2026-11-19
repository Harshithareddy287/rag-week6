import re
import sys

import chromadb

from src import config
from src.index import build_index, search
from src.llm import ask

SYSTEM = (
    "You answer questions about company policy. "
    "Answer ONLY from the context blocks below. "
    "If the context does not contain the answer, say so. "
    "Cite the block numbers you used, like [1][3]."
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


def answer(question):
    """Return (answer_text, sources, retrieved, bad_citations)."""
    retrieved = search(get_collection(), question, k=config.TOP_K)
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


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the lodging limit in high-cost cities?"
    text, sources, retrieved, bad = answer(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{text}\n")
    print("Sources:")
    for s in sources:
        print(f"  [{s['n']}] {s['doc_name']} § {s['section']}")
    if not sources:
        print("  (none cited)")
    for n in bad:
        print(f"  WARNING: cited [{n}] but context only has [1]..[{len(retrieved)}]")

    print("\nRetrieved:")
    for i, (dist, meta, _) in enumerate(retrieved, start=1):
        print(f"  [{i}] distance={dist:.4f}  {meta['doc_name']} § {meta['section']}")
