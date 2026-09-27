# 014 — within-record tie-break features

Run 013 (same blocking, no synthetic decoys) + features comparing a record's candidates *against each other*.

## Why (`tools/decompose.py`, `tools/sibling_check.py`, `tools/segment_threshold.py`, `tools/tiebreak_check.py`)

013's remaining loss (dense validation 0.9701): never retrieved +1.49, retrieved-but-rejected +1.01,
false merges +0.54. Records with **no address** are 47% of the missed true pairs; they are 97.7% true records
(decoys almost always carry an address) but ~92% of their names are shared by several S1s, so the model
cannot choose. A lower threshold for them hurts (0.7 stays optimal); same-name sibling votes are only 55%
accurate. What *does* break ties is the raw text, compared across the record's candidates:
no-address ties — raw-name best = true S1 52% (73% when unique) vs 32% chance;
address ties (1.59M records) — raw-address best 97.3%, raw-name best 66.5% vs 48% chance.

## New features (13, label-free)

`raw_n`, `raw_a` (rapidfuzz ratio on raw lowercase name / address); for `raw_n`, `raw_a`, `n_ratio`, `a_tset`:
gap to the record's best candidate (`*_qgap`) and is-best flag (`*_qbest`); ties for best (`raw_n_qties`,
`raw_a_qties`). First version used `rank().over("q")` — hours on ~10M groups; replaced by gap/is-best (2 s).
The first launch was also preempted by Modal (log: `modal_preempted_slow.log`).

## Results

| Metric | 013 | **014** |
|---|---|---|
| Validation F0.5 | 0.9794 | **0.9806** |
| **Dense validation (≈ leaderboard)** | 0.9701 | **0.9717** |
| Threshold | 0.70 | 0.75 |
| Blocking recall / oracle | 0.9706 / 0.9899 | same |
| Leaderboard | 0.967 | _pending_ (expected ≈ 0.969) |

New features take 7.8% of gain, mostly `a_tset_qgap` 5.1% and `a_tset_qbest` 1.1%. 449 trees.
