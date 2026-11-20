# Data Quality Diagnosis

## Planted issue (answer key)
`docs/pto_policy.md` is an outdated duplicate of `docs/pto_policy_2024.md`, with no version or effective date.

| Detail | pto_policy.md (old) | pto_policy_2024.md (current) |
|---|---|---|
| Version / date | Missing | 2.0 / 2024-01-01 |
| PTO days per year | 15 | 20 |
| Monthly accrual | 1.25 | 1.67 |
| Carryover | 3 days | 5 days |

## Symptom
**Q:** "How many PTO days do I get per year?"
**A:** "The policy blocks conflict: one says full-time employees receive 20 PTO days per year, while another says 15. The provided context does not establish which figure applies. [1][2]"
**Sources:** [1] pto_policy_2024.md § 3. PTO Accrual, [2] pto_policy.md § 3. PTO Accrual
The correct answer is 20 days; the system can't tell.

## Q1 — Was it retrieval? **No.**
Both Accrual chunks are retrieved in every mode:

| Mode | pto_policy_2024.md | pto_policy.md |
|---|---|---|
| dense | #2 | #3 |
| hybrid | #2 | #1 |
| rerank | #1 (ce=7.59) | #2 (ce=7.40) |

The eval agrees: q1/q2 are **HIT in all modes**, yet the answers fail.
Contrast: q4 (EXP-207) was a real retrieval miss — hybrid fixed it. Better retrieval can't fix q1/q2.

## Generation check — **worked.**
The LLM cited both blocks, reported both numbers exactly, and invented nothing. The prompt has no file name, version or date, so it had no way to prefer the 2024 file.

## Q2 — Was it the source data? **Yes.**
- `diff` shows the files differ **only** in the version/date header, 15 vs 20 days, 1.25 vs 1.67, and 3 vs 5 carryover.
- The old file has **no version or date**, so nothing marks it as outdated.
- **Control:** q9 (bereavement, identical in both files) passes — the failure follows the conflicting data, not the pipeline.
- Side effect: 14 of 66 chunks are stale; 2 of the 5 retrieved slots were wasted on them.

## Confirmation
Same pipeline, temporary index **without** `pto_policy.md` (no code, prompt or model changed):
**A:** "Full-time employees earn 20 PTO days per year. [1]" → **Sources:** pto_policy_2024.md § 3. PTO Accrual (ce=7.59, unchanged).
Only the data changed, and the answer became correct.

## Root cause
**Source data (data lineage):** an outdated, undated duplicate was indexed next to the current policy. Not a retrieval or generation bug.

## Recommended fix (not implemented)
- Require version / effective-date metadata at ingestion
- Exclude undated or superseded documents from the index
- Detect duplicate documents at ingestion
- Keep only the latest version of each policy