# 023 — larger cross-encoder (multilingual-e5-base)

022 with `intfloat/multilingual-e5-base` (MIT, 278M params) instead of e5-small: same hard pairs (0.74M train,
0.39M validation, 1.41M test), 1 epoch, batch 256, lr 4e-5, bf16, 290 s on one H100.

| | 022 (e5-small) | **023 (e5-base)** |
|---|---|---|
| CE accuracy@0.5 on validation hard pairs (LightGBM 0.876) | 0.755 | 0.759 |
| Blend dense validation, w = 0.35 / **0.5** | 0.9804 / 0.9806 | 0.9806 / **0.9809** |
| Validation F0.5 | 0.9870 | **0.9871** |
| Threshold | 0.6 | 0.6 |

Test: 5.76M matches (3.33 per S1); candidates = 019's. Final package built from this run.
