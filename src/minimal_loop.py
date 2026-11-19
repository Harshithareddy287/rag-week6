import chromadb

from src.embeddings import embed

PTO_TEXT = "Employees earn 20 days of paid time off per year."
HOTEL_TEXT = "Hotel stays are reimbursed up to $250 per night."

# (query, expected top text)
QUERIES = [
    ("How much vacation do I get each year?", PTO_TEXT),
    ("What is the hotel limit per night?", HOTEL_TEXT),
]


def run_minimal_loop():
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="minimal_loop",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )

    # STEP 1: embed and store one text
    pto_emb = embed([PTO_TEXT])
    collection.add(ids=["pto"], documents=[PTO_TEXT], embeddings=pto_emb)
    print(f"STEP 1: embedding size = {len(pto_emb[0])}, count = {collection.count()}")

    # STEP 2: embed and store a second text
    hotel_emb = embed([HOTEL_TEXT])
    collection.add(ids=["hotel"], documents=[HOTEL_TEXT], embeddings=hotel_emb)
    print(f"STEP 2: count = {collection.count()}")

    # STEP 3: run each query with the same model
    all_ranked = []
    for label, (query, expected) in zip("AB", QUERIES):
        query_emb = embed([query])
        results = collection.query(query_embeddings=query_emb, n_results=2)
        texts = results["documents"][0]
        distances = results["distances"][0]

        print(f"STEP 3{label}: query = {query!r}")
        for rank, (text, dist) in enumerate(zip(texts, distances), start=1):
            print(f"  {rank}. distance={dist:.4f}  {text}")
        print(f"  gap = {distances[1] - distances[0]:.4f}")
        print(f"  {'PASS' if texts[0] == expected else 'FAIL'}")

        all_ranked.append(texts)

    return all_ranked


if __name__ == "__main__":
    ranked = run_minimal_loop()
    ok = all(r[0] == expected for r, (_, expected) in zip(ranked, QUERIES))
    print("ALL PASS" if ok else "FAIL")
