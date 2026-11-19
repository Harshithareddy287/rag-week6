from src.index import build_index
from src.retrieve import dense_search, hybrid_search

# (query, code whose section is the expected chunk)
QUERIES = [
    ("What does EXP-207 require?", "EXP-207"),
    ("What is EXP-209?", "EXP-209"),
    ("What does EXP-210 say?", "EXP-210"),
    ("What is RW-103?", "RW-103"),
    ("What does EXP-205 cover?", "EXP-205"),
]
K = 5


def expected_rank(results, code):
    """1-based rank of the first chunk whose section contains the code, else None."""
    for rank, (_, meta, _) in enumerate(results, start=1):
        if code in meta["section"]:
            return rank
    return None


def _row(results, i):
    if i >= len(results):
        return ""
    meta = results[i][1]
    return f"{meta['chunk_id']} § {meta['section']}"[:54]


def main():
    collection, _ = build_index()
    summary = []

    for query, code in QUERIES:
        dense = dense_search(collection, query, K)
        hybrid = hybrid_search(collection, query, K)
        d_rank, h_rank = expected_rank(dense, code), expected_rank(hybrid, code)
        summary.append((query, d_rank, h_rank))

        print(f"\nQUERY: {query!r}  (expected: {code})")
        print(f"  {'#':<3}{'VECTOR-ONLY':<56}{'HYBRID'}")
        for i in range(K):
            print(f"  {i + 1:<3}{_row(dense, i):<56}{_row(hybrid, i)}")
        print(f"  expected rank: dense={d_rank or 'miss'}  hybrid={h_rank or 'miss'}")

    print("\nSUMMARY")
    print(f"  {'query':<30}| {'dense rank':<11}| {'hybrid rank':<12}| result")
    for query, d, h in summary:
        win = h is not None and (d is None or h < d)
        result = "HYBRID WINS" if win else ("tie" if d == h else "dense better")
        print(f"  {query:<30}| {str(d or 'miss'):<11}| {str(h or 'miss'):<12}| {result}")


if __name__ == "__main__":
    main()
