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
