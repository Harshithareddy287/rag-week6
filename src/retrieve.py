import re

from src import config
from src.index import search

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "do", "does", "did", "i", "me", "my", "we", "our", "you", "your",
    "it", "its", "this", "that", "these", "those", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "about", "and", "or", "but", "if",
    "can", "could", "should", "would", "will", "may", "say", "says",
    "get", "there", "any", "much", "many",
}
CODE_RE = re.compile(r"\b[A-Z]{2,3}-\d{3}\b")
CODE_BONUS = 10


def _terms(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
            if t not in STOPWORDS]


def dense_search(collection, query, k):
    """Vector-only search. Returns (distance, metadata, text) tuples."""
    return search(collection, query, k=k)


def keyword_search(collection, query, k):
    """ Keyword search. Returns (score, metadata, text) tuples, best first. """
    terms = set(_terms(query))
    codes = set(CODE_RE.findall(query.upper()))
    if not terms and not codes:
        return []

    data = collection.get(include=["documents", "metadatas"])
    scored = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        code_hits = len(codes & set(CODE_RE.findall(text)))
        # If the query names a policy code, only chunks with that code count:
        # RRF ignores score size, so a weak word match must not compete with it.
        if codes and not code_hits:
            continue
        score = len(terms & set(_terms(text))) + CODE_BONUS * code_hits
        if score > 0:
            scored.append((score, meta, text))
    scored.sort(key=lambda r: r[0], reverse=True)
    return scored[:k]


def rrf(ranked_lists, k=60):
    """ Reciprocal Rank Fusion. Each list holds ids, best first.
    Returns [(id, score), ...] sorted by fused score. """
    scores = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(collection, query, k=config.TOP_K):
    """ Dense top 20 + keyword top 20, fused with RRF.
    Returns (rrf_score, metadata, text) tuples, same shape as search(). """
    dense = dense_search(collection, query, 20)
    keyword = keyword_search(collection, query, 20)

    by_id = {}
    for _, meta, text in dense + keyword:
        by_id[meta["chunk_id"]] = (meta, text)

    fused = rrf([[m["chunk_id"] for _, m, _ in dense],
                 [m["chunk_id"] for _, m, _ in keyword]])
    return [(score, *by_id[cid]) for cid, score in fused[:k]]
