import json
import re
from pathlib import Path

import pytest

EVAL_SET = json.loads((Path(__file__).parent / "eval_set.json").read_text())
MODES = ["dense", "hybrid", "rerank"]

# Filled in by the eval fixtures, read by the summary below.
RESULTS = {"hits": {m: {} for m in MODES}, "answers": {}}


def is_hit(retrieved, expected_sources):
    """HIT = any retrieved chunk matches an expected doc AND section prefix."""
    return any(meta["doc_name"] == e["doc"] and meta["section"].startswith(e["section"])
               for _, meta, _ in retrieved for e in expected_sources)


def answer_passes(text, item):
    """ANY must_contain appears and NO must_not_contain appears (case-insensitive).
    Citation markers like [5] are removed first so they can't match "5"."""
    text = re.sub(r"\[\d+\]", "", text).lower()
    return (any(s.lower() in text for s in item["must_contain"])
            and not any(s.lower() in text for s in item["must_not_contain"]))


@pytest.fixture(scope="session")
def eval_collection(tmp_path_factory):
    from src.index import build_index

    collection, _ = build_index(persist_dir=tmp_path_factory.mktemp("eval_chroma"))
    return collection


@pytest.fixture(scope="session")
def retrieval(eval_collection):
    """{mode: {id: retrieved top-5}} for every question, no LLM."""
    from src.rag import retrieve

    out = {}
    for mode in MODES:
        out[mode] = {}
        for item in EVAL_SET:
            retrieved = retrieve(eval_collection, item["question"], mode)
            out[mode][item["id"]] = retrieved
            RESULTS["hits"][mode][item["id"]] = is_hit(retrieved, item["expected_sources"])
    return out


@pytest.fixture(scope="session")
def get_answer(eval_collection):
    """Call answer() in rerank mode once per question, cached for the session."""
    from src.rag import answer

    cache = {}

    def _get(item):
        if item["id"] not in cache:
            text, *_ = answer(item["question"], "rerank", eval_collection)
            cache[item["id"]] = text
            RESULTS["answers"][item["id"]] = answer_passes(text, item)
        return cache[item["id"]]

    return _get


def pytest_terminal_summary(terminalreporter):
    hits, answers = RESULTS["hits"], RESULTS["answers"]
    if not any(hits.values()):
        return
    w = terminalreporter.write_line
    n = len(EVAL_SET)
    mark = lambda v: "-" if v is None else ("HIT" if v else "miss")
    verdict = lambda v: "skipped" if v is None else ("pass" if v else "FAIL")

    w("")
    w("EVAL SUMMARY")
    w("Recall@5   " + " | ".join(f"{m} {sum(hits[m].values())}/{n}" for m in MODES))

    normal = [q for q in EVAL_SET if not q["conflict"]]
    conflict = [q for q in EVAL_SET if q["conflict"]]
    if answers:
        correct = sum(bool(answers.get(q["id"])) for q in normal)
        w(f"Answer accuracy (non-conflict): {correct}/{len(normal)}")
        parts = ", ".join(f"{q['id']} {verdict(answers.get(q['id']))}" for q in conflict)
        w(f"Conflict questions: {parts} (expected — planted data issue)")
    else:
        w("Answer accuracy: skipped (no AZURE_OPENAI_API_KEY)")

    w("")
    w(f"{'id':<5}| {'dense':<6}{'hybrid':<7}{'rerank':<7}| answer")
    for q in EVAL_SET:
        i = q["id"]
        row = "".join(f"{mark(hits[m].get(i)):<{6 if m == 'dense' else 7}}" for m in MODES)
        tag = "  (conflict)" if q["conflict"] else ""
        w(f"{i:<5}| {row}| {verdict(answers.get(i))}{tag}")
