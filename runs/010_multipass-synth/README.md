# 010 — multi-pass blocking (009) + synthetic decoys (008)

Run 009's blocking (TF-IDF top-10 + 2 exact-key passes) with run 008's 2.6M synthetic decoys in train.
Stage-1 LightGBM only. Ran on Modal.

## Results

Own validation contains synthetic decoys, so it is not comparable with 009's; the fair comparison scores
both models on the **same real validation records**, with each S1's group context at test-like decoy
density (synthetic decoys present), `tools/compare_real.py`:

| Model | real val F0.5 | real val, dense |
|---|---|---|
| 009 | 0.9730 | 0.9612 |
| **010** | **0.9759** | **0.9650** |
| blend 0.3·009 + 0.7·010 (`tools/blend_eval.py`) | 0.9764 | 0.9658 |

(009 on its own features without synthetic context: 0.9781 / dense 0.9684 — which of the two contexts the
real test resembles is exactly the open question, so both are submission candidates.)

| Metric | 010 |
|---|---|
| Own validation (with synthetic decoys) | 0.9727 (dense 0.9596), rule expected-F |
| Blocking recall / oracle | 0.9662 / 0.9883 |
| Test candidates / S1 | 11.9 (France 13.6, India 12.6, US 10.2) |
| Test matches / S1, % empty | France 3.10 / 6.0%, India 3.23 / 5.9%, US 3.18 / 5.9% |
| Trees | 705 |
| Leaderboard | _pending_ |
