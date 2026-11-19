import pytest

from src import config
from src.index import CHECK_QUERY, build_index, search


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_index(persist_dir=tmp_path_factory.mktemp("chroma"))


def test_count_matches_chunks(built):
    collection, chunks = built
    assert collection.count() == len(chunks)


def test_stored_embed_model(built):
    collection, _ = built
    assert collection.metadata["embed_model"] == config.EMBEDDING_MODEL


def test_lodging_query_returns_exp_204_first(built):
    collection, _ = built
    _, meta, _ = search(collection, CHECK_QUERY)[0]
    assert meta["section"].startswith("EXP-204")


def test_rebuild_replaces_existing_collection(tmp_path):
    # Second build finds the existing collection, deletes it, and recreates it.
    first, chunks = build_index(persist_dir=tmp_path)
    second, _ = build_index(persist_dir=tmp_path)
    assert second.count() == len(chunks)
