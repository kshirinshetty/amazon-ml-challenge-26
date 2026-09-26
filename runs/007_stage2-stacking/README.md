# 007 — stage-2 stacking + test-like validation

Stage 1 = run 006's setup (blocking K=10/REL 0.8/cap 30, 51 features, LightGBM lr 0.05 / 511 leaves),
retrained here (reproduces 006: normal 0.9747). New: `src/stack.py` and the test-like "dense" validation.

## Why: the test is decoy-denser than train (`tools/shift.py`)

| | records / S1 | implied decoy share (label-free) | records pointing at each S1 |
|---|---|---|---|
| train (US) | 4.67 | 26% (truth 26.0%) | 5.2 |
| test (US / India / France) | 5.76 / 5.82 / 5.53 | 43% / 41% / ≥32% | 6.3 |

Validation with 50% of true records removed (`sim_dense` in `src/match.py`) reproduces the leaderboard:
run 006 dense **0.9638** vs leaderboard **0.961** (normal val 0.9747). Thresholds are now chosen on it.
Density-matched training barely helps (`tools/exp_density.py`: dense 0.9638 base, 0.9640 thinned train,
**0.9644 decoy-weight ×2**, 0.9641 ×3) → the lever is discrimination, not calibration.

## Stage 2 (`src/stack.py`)

1. Out-of-fold stage-1 probabilities for all train pairs (4 models on 3 of the 4 training folds each;
   validation fold from the folds-1..4 model).
2. Context features from them: record level `q_pmax`, `q_p2`, `q_gap`, `q_runner`, `q_n`, `q_nhi`;
   S1 level `s_sum`, `s_nhi`, `s_rank`, `s_n`; the S1's most confident other record (`sib_p`,
   `sib_n_ratio`, `sib_a_ratio`, `sib_num_eq`).
3. LightGBM (same params) on stage-1 features + p + context → 191 trees; threshold 0.70 (dense-optimal).

## Results

| Metric | stage 1 (= 006) | **stage 2** |
|---|---|---|
| Validation F0.5 (normal) | 0.9747 | **0.9770** |
| **Validation F0.5 (dense, ≈ leaderboard)** | 0.9637 | **0.9674** |
| Test candidates / S1 | 9.38 | 9.38 |
| Leaderboard | 0.961 (run 006) | _pending_ |

Stage-2 gain by feature: `p` 84.8%, `q_p2` 7.7%, `q_pmax` 4.1%, `q_gap` 2.2% — it mostly resolves records
torn between two look-alike S1s; sibling-agreement features contribute ~0.1%. The record-level context does
not depend on decoy density, so the dense estimate should transfer.
