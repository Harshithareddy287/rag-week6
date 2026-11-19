import pytest

from src.index import build_index
from src.rerank import rerank
from src.retrieve import hybrid_search


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    coll, _ = build_index(persist_dir=tmp_path_factory.mktemp("chroma"))
    return coll


@pytest.fixture(scope="module")
def pool(collection):
    query = "How quickly must I report a lost laptop?"
    return query, hybrid_search(collection, query, 20)


def test_rerank_sorted_best_first(pool):
    query, candidates = pool
    scores = [s for s, _, _ in rerank(query, candidates, 10)]
    assert scores == sorted(scores, reverse=True)


def test_rerank_top_n_from_candidates(pool):
    query, candidates = pool
    results = rerank(query, candidates, 3)
    assert len(results) == 3
    input_ids = {m["chunk_id"] for _, m, _ in candidates}
    assert all(m["chunk_id"] in input_ids for _, m, _ in results)


# Observed in `python -m src.rerank_demo`.
@pytest.mark.parametrize("query,target", [
    ("How quickly must I report a lost laptop?", "RW-104"),
    ("What happens if I submit a fake expense?", "Misuse"),  # hybrid #5 -> #1
])
def test_rerank_puts_target_first(collection, query, target):
    _, meta, _ = rerank(query, hybrid_search(collection, query, 20))[0]
    assert target in meta["section"]
