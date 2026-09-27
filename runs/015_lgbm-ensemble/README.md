# 015 — LightGBM ensemble (fit-only)

Run 014's features and candidates, unchanged; only the model differs.

## Why

HANDOFF §9 ranked a seed/size ensemble as the cheapest likely-positive step (fit-only, ≈ $0.5).

## What

`match.py fit --reuse runs/014_relative-tiebreak/model.txt`: 014's model plus two new members trained on the
same 80% of S1s, probabilities averaged, threshold picked on dense validation.

| Member | Params | Trees | Dense val alone |
|---|---|---|---|
| 0 | 014's model (511 leaves) | 449 | 0.9717 |
| 1 | 014 params, `seed=1` | 502 | 0.9718 |
| 2 | `num_leaves 1023, min_data_in_leaf 100, seed=2` | 546 | **0.9722** |

## Results

| Metric | 014 | **015 (mean of 3)** |
|---|---|---|
| Validation F0.5 | 0.9806 | 0.9809 |
| **Dense validation** | 0.9717 | **0.9721** |
| Threshold | 0.75 | 0.70 |
| Test candidates / S1 | 13.5 | 13.5 |

Averaging adds nothing over the 1023-leaf member alone (0.9721 vs 0.9722), so later runs train only that one
model. Not submitted: +0.0004 is inside the leaderboard noise. The first launch was preempted by Modal after
one member and restarted (≈ $0.2 lost).
