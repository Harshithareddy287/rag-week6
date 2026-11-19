from src.chunking import _is_table, chunk_documents
from src.embeddings import count_tokens, get_model
from src.index import DOCS_DIR

CHUNKS = chunk_documents(DOCS_DIR)


def _tables():
    tables = []
    for path in DOCS_DIR.glob("*.md"):
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            if _is_table(block):
                tables.append(block.strip())
    return tables


def test_every_chunk_fits_token_limit():
    limit = get_model().max_seq_length
    assert all(count_tokens(c["text"]) <= limit for c in CHUNKS)


def test_each_table_is_whole_in_one_chunk():
    tables = _tables()
    assert tables, "expected the docs to contain tables"
    for table in tables:
        assert any(table in c["text"] for c in CHUNKS), table[:60]


def test_every_chunk_has_required_metadata():
    for c in CHUNKS:
        for key in ("doc_name", "section", "chunk_id"):
            assert c["metadata"][key]


def test_each_exp_rule_has_its_own_chunk():
    sections = {c["metadata"]["section"] for c in CHUNKS}
    for n in range(201, 211):
        assert any(s.startswith(f"EXP-{n} ") for s in sections), f"EXP-{n}"


def test_both_pto_docs_are_chunked():
    docs = {c["metadata"]["doc_name"] for c in CHUNKS}
    assert {"pto_policy.md", "pto_policy_2024.md"} <= docs
