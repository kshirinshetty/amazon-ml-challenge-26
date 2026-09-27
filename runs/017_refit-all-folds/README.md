# 017 — run 016 refit on all training folds

Same candidates and features as [016](../016_address-pass/); the one 1023-leaf LightGBM is retrained on **all**
training S1s (016 held out fold 0 = 20% for early stopping and the threshold) with 1.25× 016's early-stopped
rounds (573 → 716), keeping 016's threshold 0.75: `match.py refit runs/017_refit-all-folds runs/016_address-pass`
(`modal_app.py --start fit --refit runs/016_address-pass`).

## Why

25% more training pairs usually help a GBDT a little; approved as the last use of the Modal credit.

## Results

No validation is possible (every labelled S1 is now in training).

| | 016 | 017 |
|---|---|---|
| Matched pairs (test) | 5,728,341 | 5,729,781 |
| Pairs only in this run | 33,734 | 35,174 |
| S1s whose match list differs | | 65,468 (3.8%) |
| Test matches / S1 (France, India, US) | 3.218, 3.309, 3.337 | 3.219, 3.312, 3.336 |
| Candidates | identical | identical |

**Leaderboard: 0.971 vs 016's 0.974 — the refit hurt** (likely the fixed 1.25× rounds / reused threshold no longer fit the more confident model). Training took 526 s. Submit as a second upload next to 016: the leaderboard decides which goes in the zip.
