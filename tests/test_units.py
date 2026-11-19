import pytest

from src import config, rag
from src.chunking import _is_table, _split, _tail
from src.retrieve import CODE_RE, _terms, rrf
from tests.conftest import answer_passes, is_hit

pytestmark = pytest.mark.unit

TABLE = "| Rule | Limit |\n|---|---|\n| EXP-204 | $350 |"
LONG = " ".join(f"Sentence number {i} explains one small policy detail." for i in range(40))


# ---------- chunking ----------

def test_is_table():
    assert _is_table(TABLE)
    assert not _is_table("Hotel stays are reimbursed up to $250 per night.")


def test_split_short_text_is_one_piece():
    assert _split("A short policy line.") == ["A short policy line."]


def test_split_long_text_pieces_fit():
    pieces = _split(LONG)
    assert len(pieces) > 1
    assert all(len(p) <= config.CHUNK_SIZE for p in pieces)


def test_split_never_breaks_a_table():
    big_table = "| Code | Rule |\n|---|---|\n" + "\n".join(
        f"| EXP-{n} | A rule with a fairly long description |" for n in range(200, 220))
    assert len(big_table) > config.CHUNK_SIZE
    pieces = _split(f"Intro paragraph.\n\n{big_table}\n\nClosing paragraph.")
    assert big_table in pieces


def test_split_overlap_piece_2_starts_with_tail_of_piece_1():
    first, second = _split(LONG)[:2]
    assert second.startswith(_tail(first))


def test_tail():
    tail = _tail(LONG)
    assert 0 < len(tail) <= config.CHUNK_OVERLAP
    assert LONG.endswith(tail)
    assert LONG[-len(tail) - 1] == " "  # starts right after a space
    assert _tail("Some intro text. " + TABLE) == ""


# ---------- retrieve ----------

def test_terms():
    assert _terms("What does EXP-207 require?") == ["exp-207", "require"]
    assert _terms("What is the") == []


def test_code_re():
    assert CODE_RE.findall("See EXP-207 and RW-103.") == ["EXP-207", "RW-103"]
    assert CODE_RE.findall("Effective date: 2024-01-01") == []


def test_rrf_high_in_both_lists_wins():
    fused = rrf([["both", "only_a", "x"], ["only_b", "both", "y"]])
    assert fused[0][0] == "both"


# ---------- eval helpers ----------

META = {"doc_name": "expense_policy.md", "section": "EXP-204 Lodging"}


def test_is_hit():
    retrieved = [(0.1, META, "text")]
    assert is_hit(retrieved, [{"doc": "expense_policy.md", "section": "EXP-204"}])
    assert not is_hit(retrieved, [{"doc": "pto_policy.md", "section": "EXP-204"}])


def test_answer_passes():
    item = {"must_contain": ["5 days", "5"], "must_not_contain": ["3 days"]}
    assert answer_passes("You can carry over 5 days.", item)
    assert not answer_passes("You can carry over 5 days, or 3 days.", item)
    assert not answer_passes("See the policy [5].", item)


# ---------- rag.answer with a mocked LLM ----------

def test_answer_maps_citations_to_sources(monkeypatch):
    fake = [
        (0.9, {"doc_name": "expense_policy.md", "section": "EXP-204 Lodging"}, "a"),
        (0.8, {"doc_name": "pto_policy.md", "section": "3. PTO Accrual"}, "b"),
        (0.7, {"doc_name": "remote_work_policy.md", "section": "5. Security"}, "c"),
    ]
    monkeypatch.setattr(rag, "retrieve", lambda collection, question, mode: fake)
    monkeypatch.setattr(rag, "ask", lambda prompt: "Answer text [1][3][9]")

    text, sources, retrieved, bad = rag.answer("any question", collection=object())

    assert text == "Answer text [1][3][9]"
    assert sources == [
        {"n": 1, "doc_name": "expense_policy.md", "section": "EXP-204 Lodging"},
        {"n": 3, "doc_name": "remote_work_policy.md", "section": "5. Security"},
    ]
    assert bad == [9]
    assert retrieved == fake


# ---------- coverage of edge branches ----------

def test_split_skips_empty_paragraphs():
    text = "A" * 300 + "\n\n\n\n" + "B" * 300  # "\n\n\n\n" leaves an empty piece
    first, second = _split(text)
    assert first == "A" * 300
    # piece 2 = overlap tail of piece 1 + "\n\n" + B's; the empty piece is gone
    assert second == "A" * config.CHUNK_OVERLAP + "\n\n" + "B" * 300


def test_rerank_empty_candidates():
    from src.rerank import rerank

    assert rerank("any question", []) == []


def test_fmt_score():
    from src.rerank import fmt_score

    assert fmt_score(1.059) == "ce=1.06"
    assert fmt_score(-11.18) == "ce=-11.18"


def test_retrieve_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        rag.retrieve(object(), "question", mode="nope")


def test_get_collection_builds_index_when_missing(monkeypatch):
    class FakeClient:
        def __init__(self, path):
            pass

        def list_collections(self):
            return []

    built = object()
    monkeypatch.setattr(rag.chromadb, "PersistentClient", FakeClient)
    monkeypatch.setattr(rag, "build_index", lambda: (built, []))
    assert rag.get_collection() is built


# ---------- rag CLI ----------

RETRIEVED = [
    (0.5, {"doc_name": "expense_policy.md", "section": "EXP-204 Lodging",
           "chunk_id": "expense_policy.md#5"}, "chunk text"),
]


def _fake_answer(sources, bad, calls):
    def fake(question, mode="rerank", collection=None):
        calls.append((question, mode))
        return "Fake answer [1]", sources, RETRIEVED, bad
    return fake


def test_main_prints_all_sections(monkeypatch, capsys):
    calls = []
    sources = [{"n": 1, "doc_name": "expense_policy.md", "section": "EXP-204 Lodging"}]
    monkeypatch.setattr(rag, "answer", _fake_answer(sources, [7], calls))

    rag.main(["What", "is", "the", "lodging", "limit?"])
    out = capsys.readouterr().out

    assert calls == [("What is the lodging limit?", "rerank")]
    assert "Question: What is the lodging limit?  (mode=rerank)" in out
    assert "Answer:\nFake answer [1]" in out
    assert "Sources:\n  [1] expense_policy.md § EXP-204 Lodging" in out
    assert "WARNING: cited [7]" in out
    assert "Retrieved:\n  [1] ce=0.50  expense_policy.md#5 § EXP-204 Lodging" in out


def test_main_mode_dense_and_no_sources(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rag, "answer", _fake_answer([], [], calls))

    rag.main(["--mode", "dense", "Some question"])
    out = capsys.readouterr().out

    assert calls == [("Some question", "dense")]
    assert "(mode=dense)" in out
    assert "(none cited)" in out
    assert "[1] distance=0.5000  expense_policy.md#5" in out


def test_main_default_question(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rag, "answer", _fake_answer([], [], calls))
    rag.main([])
    assert calls == [("What is the lodging limit in high-cost cities?", "rerank")]
