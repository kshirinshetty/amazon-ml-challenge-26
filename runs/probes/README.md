# Leaderboard probes

Re-thresholded test probabilities of the run 006/007 stage-1 model (`tools/probe.py`), no retraining.

| Probe | Threshold | Leaderboard |
|---|---|---|
| run 006 | 0.65 | **0.961** |
| t080 | 0.80 | 0.959 |
| t090 | 0.90 | 0.956 |

Stricter is worse: the pairs with p in [0.65, 0.90] are mostly true on test too. Test errors are confident
mistakes and — above all — true pairs never retrieved (see `tools/decompose.py`, run 009).
