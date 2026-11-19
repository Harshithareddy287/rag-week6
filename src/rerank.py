from functools import lru_cache

from sentence_transformers import CrossEncoder

from src import config


@lru_cache(maxsize=1)
def get_reranker():
    """ The only place the cross-encoder is loaded. Cached after first call. """
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query, candidates, top_n=config.TOP_K):
    """ Re-score (score, metadata, text) candidates with the cross-encoder.
    Returns the top_n as (rerank_score, metadata, text), best first. """
    if not candidates:
        return []
    scores = get_reranker().predict([(query, text) for _, _, text in candidates])
    rescored = [(float(s), meta, text) for s, (_, meta, text) in zip(scores, candidates)]
    rescored.sort(key=lambda r: r[0], reverse=True)
    return rescored[:top_n]


def fmt_score(score):
    """Display only: the raw cross-encoder score."""
    return f"ce={score:.2f}"
