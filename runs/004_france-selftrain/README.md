# 004 — France self-training (negative result)

Run 003's model + `--pseudo`: run 003 scored all test pairs (`tools/test_probs.py`); test pairs of
countries absent from training (France) with p ≥ 0.97 (label 1) or p ≤ 0.03 (label 0) were added to the
training set — 1,932,866 pairs, 37.9% positive. Features and blocking identical to 003.

| Metric | 003 | 004 |
|---|---|---|
| Validation F0.5 (US/India) | 0.9712 | 0.9704 |
| France matches / S1, % empty | 3.11 / 6.9% | 3.12 / 6.8% |
| France accepted pairs with disagreeing house number | 1.86% | 1.79% |

France predictions barely moved: self-training on the model's own confident labels mostly reinforces
what it already believes. **Not submitted** (5 submissions left at the time). Label-free France check:
`tools/france_diag.py`; France rejections with agreeing numbers are largely same-address businesses with a
swapped category word (`Mind Pharmacie` vs `Mind Sportive`), i.e. correct rejections — `tools/france_rejects.py`.
