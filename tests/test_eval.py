import pytest

from src import config
from tests.conftest import EVAL_SET, answer_passes, is_hit

# Retrieval is deterministic, so recall thresholds equal the observed values.
# Answer accuracy (LLM) allows one miss: observed 9/9.
RECALL_MIN = {"dense": 10, "hybrid": 11, "rerank": 11}
ACCURACY_MIN = 8

NORMAL = [q for q in EVAL_SET if not q["conflict"]]
CONFLICT = [q for q in EVAL_SET if q["conflict"]]
BY_ID = {q["id"]: q for q in EVAL_SET}
needs_llm = pytest.mark.skipif(not config.AZURE_OPENAI_API_KEY,reason="AZURE_OPENAI_API_KEY not set")


# ---------- Retrieval (no LLM) ----------

@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank"])
def test_recall_at_5(retrieval, mode):
    hits = sum(is_hit(retrieval[mode][q["id"]], q["expected_sources"]) for q in EVAL_SET)
    print(f"\nRecall@5 {mode}: {hits}/{len(EVAL_SET)}")
    assert hits >= RECALL_MIN[mode]


def test_conflict_both_pto_files_retrieved(retrieval):
    docs = {meta["doc_name"] for _, meta, _ in retrieval["rerank"]["q1"]}
    assert {"pto_policy.md", "pto_policy_2024.md"} <= docs


# ---------- Answer accuracy (LLM) ----------

@needs_llm
def test_answer_accuracy_non_conflict(get_answer):
    correct = sum(answer_passes(get_answer(q), q) for q in NORMAL)
    print(f"\nAnswer accuracy (non-conflict): {correct}/{len(NORMAL)}")
    assert correct >= ACCURACY_MIN

@needs_llm
@pytest.mark.parametrize("qid", [
    pytest.param(q["id"], marks=pytest.mark.xfail(strict=False,reason="planted outdated PTO duplicate"))
    for q in CONFLICT
])
def test_conflict_answer(get_answer, qid):
    item = BY_ID[qid]
    assert answer_passes(get_answer(item), item)
