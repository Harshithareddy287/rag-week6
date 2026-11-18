import pytest

from src import config


def test_config_loads():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.CHUNK_SIZE == 500
    assert config.CHUNK_OVERLAP == 75
    assert config.TOP_K == 5


def test_libraries_import():
    import chromadb  # noqa: F401
    import dotenv  # noqa: F401
    import openai  # noqa: F401
    import sentence_transformers  # noqa: F401


@pytest.mark.skipif(not config.AZURE_OPENAI_API_KEY, reason="AZURE_OPENAI_API_KEY not set")
def test_llm_ask():
    from src.llm import ask

    answer = ask("Reply with the single word: pong")
    assert answer and answer.strip()
