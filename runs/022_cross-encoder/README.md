# 022 — cross-encoder on the hard pairs, blended with LightGBM

## Why

LightGBM variants plateaued at dense 0.977 (019–021). Its features reduce the text to a few similarity scores;
a transformer reading both raw records can see spelling, transliteration and formatting directly.

## What (`src/ce_prep.py`, `src/ce_gpu.py`, `src/ce_blend.py`)

1. **ce_prep** (CPU): 019's LightGBM scores every train/test pair. *Hard* records: best p in [0.02, 0.995] or
   second-best p > 0.02; their candidates with p ≥ 0.005 (top 5). Train: 1.13M hard pairs / 0.81M records;
   test: 1.41M / 1.18M. CE training set = hard pairs of records with no fold-0 candidate (0.74M, 66% positive);
   CE validation = hard pairs of records with a fold-0 candidate (0.39M).
2. **ce_gpu** (1× H100): `intfloat/multilingual-e5-small` (MIT, 118M) + a linear head, input
   `"name | address"` (raw) of record vs S1, 1 epoch, batch 512, lr 8e-5, bf16, max 128 tokens. 2 min training.
   On the validation hard pairs: mean p_ce 0.79 (true) vs 0.32 (false); accuracy@0.5 0.755 vs LightGBM 0.876.
3. **ce_blend** (CPU): p = (1−w)·p_LightGBM + w·p_CE on hard pairs; w and threshold on dense validation.

## Results

| w | 0 | 0.2 | 0.35 | **0.5** | 0.65 | 0.8 |
|---|---|---|---|---|---|---|
| Dense validation | 0.9769 | 0.9791 | 0.9804 | **0.9806** | 0.9803 | 0.9790 |
| Validation | 0.9845 | 0.9857 | 0.9867 | **0.9870** | 0.9867 | 0.9858 |

**Dense 0.9806 vs 0.9769** (+0.0037, the largest single gain of the project) at threshold 0.6. w = 0 reproduces
019 exactly. Test: 5.77M matches (3.33 per S1), candidates = 019's (15.3 per S1). Final package built from this run.
