# 024 — cross-encoder on a wider hard-pair band (stopped)

022 (e5-small) with the hard-pair band widened to best p in [0.002, 0.9995] or second-best > 0.005, p ≥ 0.002
(train 2.91M hard pairs / 2.52M records, test 3.09M): 1.5M training pairs, 253 s on one H100. Validation hard pairs:
mean p_ce 0.93 (true) vs 0.26 (false), accuracy@0.5 0.869 vs LightGBM 0.936.

Scoring the 3.1M test pairs crashed the container with SIGSEGV three times (Modal retried; most likely tokenizing all
3.1M pairs in one call), so the run was stopped at 23:35 IST before the blend. Fix for next time: tokenize and
score the test set in chunks of ~500k pairs in `ce_gpu.py`'s `score`.
