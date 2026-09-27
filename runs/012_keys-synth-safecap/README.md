# 012 — extra keys + synthetic decoys + best-candidate-safe cap (best single run)

Run 011 (TF-IDF top-10 + 5 exact-key passes, synthetic decoys) with the prune fix from `tools/prune_loss.py`:
the 30-per-S1 cap now trims **runner-up candidates only**, so a record's own best S1 is never cut at hub S1s.

| Metric | 010 | 011 | **012** |
|---|---|---|---|
| Blocking recall (after prune) | 0.9662 | 0.9675 | **0.9702** (0.9759 before prune) |
| Oracle F0.5 | 0.9883 | 0.9888 | **0.9897** |
| Own validation (with synthetic decoys) | 0.9727 | 0.9729 | **0.9741** (dense 0.9614) |
| **Real validation records, test-density context** (`tools/compare_real.py`) | 0.9759 | — | **0.9774** |
| same, dense | 0.9650 | — | **0.9670** |
| Test candidates / S1 | 11.9 | 13.3 | 13.5 (France 15.1, India 14.3, US 11.8) |
| Leaderboard | _pending_ | — | see root README |

Threshold 0.70, 489 trees. Top gains: `a_tset` 42.5%, `dec_max` 16.2%, `rank` 10.8%, `num_jac` 4.5%.
Superseded as the final package by run 013 (no synthetic decoys), which scored 0.967 on the leaderboard.
