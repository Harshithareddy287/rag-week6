from src.index import build_index
from src.rerank import fmt_score, rerank
from src.retrieve import hybrid_search

# (query, substring of the most relevant section)
QUERIES = [
    ("How many PTO days do I get per year?", "PTO Accrual"),
    ("How quickly must I report a lost laptop?", "RW-104"),
    ("Who must approve a fully remote arrangement?", "RW-102"),
    ("Can I expense my spouse's travel?", "Non-Reimbursable"),
    ("What is the mileage reimbursement rate?", "EXP-205"),
    # Extra: queries where hybrid ranks the right section below #1.
    ("What happens if I submit a fake expense?", "Misuse"),
    ("Who pays to fix my broken monitor?", "Repairs"),
    ("Do I get paid overtime when working from home?", "Time Tracking"),
]
K = 5
POOL = 20


def target_rank(results, target):
    for rank, (_, meta, _) in enumerate(results, start=1):
        if target in meta["section"]:
            return rank
    return None


def _cell(results, i):
    score, meta, _ = results[i]
    return f"{meta['chunk_id']} § {meta['section']}"[:46], score


def main():
    collection, _ = build_index()
    summary = []

    for query, target in QUERIES:
        pool = hybrid_search(collection, query, POOL)
        before = pool[:K]
        after = rerank(query, pool, K)
        b_rank, a_rank = target_rank(before, target), target_rank(after, target)
        summary.append((query, b_rank, a_rank))

        print(f"\nQUERY: {query!r}  (most relevant: {target})")
        print(f"  {'#':<3}{'BEFORE (hybrid)':<46} {'rrf':>7}   {'AFTER (reranked)':<46} {'cross-encoder':>13}")
        for i in range(K):
            b, bs = _cell(before, i)
            a, as_ = _cell(after, i)
            print(f"  {i + 1:<3}{b:<46} {bs:>7.4f}   {a:<46} {fmt_score(as_):>13}")
        print(f"  most relevant rank: before={b_rank or 'miss'}  after={a_rank or 'miss'}")

        before_ids = [m["chunk_id"] for _, m, _ in before]
        after_ids = [m["chunk_id"] for _, m, _ in after]
        dropped = [(m["chunk_id"], m["section"]) for _, m, _ in before if m["chunk_id"] not in after_ids]
        added = [(m["chunk_id"], m["section"]) for _, m, _ in after if m["chunk_id"] not in before_ids]
        for cid, sec in dropped:
            print(f"  dropped out: {cid} § {sec}")
        for cid, sec in added:
            print(f"  moved in:    {cid} § {sec}")

    print("\nSUMMARY")
    print(f"  {'query':<46}| {'before':<7}| {'after':<6}| result")
    for query, b, a in summary:
        if a is not None and (b is None or a < b):
            result = "RERANK IMPROVES"
        elif a == b:
            result = "same"
        else:
            result = "worse"
        print(f"  {query:<46}| {str(b or 'miss'):<7}| {str(a or 'miss'):<6}| {result}")


if __name__ == "__main__":
    main()
