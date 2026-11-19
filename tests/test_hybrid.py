import pytest

from src.hybrid_demo import QUERIES
from src.index import build_index
from src.retrieve import hybrid_search, keyword_search, rrf


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    coll, _ = build_index(persist_dir=tmp_path_factory.mktemp("chroma"))
    return coll


def test_rrf_order():
    # "b" is 2nd and 1st -> best; "a" is 1st and 3rd; "c" is in one list only.
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]])
    assert [item for item, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_keyword_search_finds_exp_207_first(collection):
    _, meta, _ = keyword_search(collection, "EXP-207", 5)[0]
    assert meta["section"].startswith("EXP-207")


@pytest.mark.parametrize("query,code", QUERIES)
def test_hybrid_ranks_expected_chunk_first(collection, query, code):
    _, meta, _ = hybrid_search(collection, query)[0]
    assert code in meta["section"]


def test_stopwords_only_returns_nothing(collection):
    assert keyword_search(collection, "what is the", 5) == []
