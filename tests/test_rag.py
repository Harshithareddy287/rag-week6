import pytest

from src import config


@pytest.mark.skipif(not config.AZURE_OPENAI_API_KEY, reason="AZURE_OPENAI_API_KEY not set")
def test_lodging_answer_and_source():
    from src.rag import answer

    text, sources, _, _ = answer("What is the lodging limit in high-cost cities?")
    assert "350" in text
    assert any(s["doc_name"] == "expense_policy.md" for s in sources)
