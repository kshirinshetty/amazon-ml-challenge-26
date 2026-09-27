# 020 — 019 features, LightGBM learning rate 0.03

Fit-only on 019's features, run in parallel with 019/021: `num_leaves 1023, min_data_in_leaf 100, lr 0.03`.
1125 trees; validation 0.9846, **dense 0.9770** (019: 0.9769), threshold 0.75. LightGBM tuning has plateaued
(019/020/021 within 0.0002). `model_0.txt` (127 MB) is over GitHub's limit: kept locally and in the Modal volume.
