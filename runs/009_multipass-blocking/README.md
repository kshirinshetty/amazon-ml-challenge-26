# 009 — multi-pass blocking (exact-key passes)

Run 006's pipeline (no synthetic decoys, no stage 2) + two exact-key blocking passes in `src/block.py`.

## Why (`tools/decompose.py`, run 006 on validation)

Fixing one error type perfectly would gain (dense validation ≈ leaderboard, base 0.9637):
**never-retrieved true pairs +2.14**, retrieved-but-rejected +0.93, false merges +0.63.
TF-IDF drops n-grams present in >0.5% of S1s, so a name made only of common words (`Coimbatore Foundation`)
can lose even an S1 with an identical normalized name. `tools/pass_coverage.py`: exact keys recover
139,726 of 325,318 missed train pairs (43%) — house number + street 79,990, sorted name words 61,748.

## Change

- key 1: normalized name words sorted (order-free); key 2: first house number + the word after it.
- Skip keys shared by >50 S1s; per record and key keep the 2 S1s closest on the *other* field (rapidfuzz ratio).
- Key pairs get TF-IDF cosines like any other pair, a `via_key` bitmask feature, and skip the REL/RMAX prune.

## Results

| Metric | 006 | **009** |
|---|---|---|
| Blocking recall (validation) | 0.9581 | **0.9673** |
| Oracle F0.5 | 0.9852 | **0.9887** |
| Validation F0.5 | 0.9747 | **0.9781** |
| **Dense validation (≈ leaderboard)** | 0.9638 | **0.9684** |
| Test candidates / S1 | 9.4 | 11.9 (France 13.6, India 12.6, US 10.2) |
| Leaderboard | 0.961 | _pending_ |

Prune grid with key passes (train recall / candidates per S1): RMAX 10 REL 0.8 cap 30 → 0.9674 / 9.39 (used);
RMAX 5 → 0.9663 / 8.61; RMAX 3 → 0.9642 / 7.99 — options for a smaller final candidate set.
Top gains: `top2` 21.4%, `top1` 15.5%, `rank` 15.1%, `a_tset` 11.1%, `dec_max` 7.6%.
