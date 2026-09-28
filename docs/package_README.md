# Business Entity Resolution — reproduction guide

End-to-end pipeline that regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the challenge data. No external data, APIs or pretrained models are used; the only model is a
LightGBM classifier (MIT license) trained from scratch on the provided training labels.

```
raw TSVs ─▶ normalize.py ─▶ block.py ─▶ block.py --prune ─▶ features.py ─▶ match.py fit/predict ─▶ output/*.tsv
            text cleanup +   TF-IDF top-10  drop weak /       pair           LightGBM, F0.5 threshold,
            learned maps     + exact keys   cap per S1        similarities   best-S1 assignment
                             + address top-1
```

## Setup

- Python 3.12, then `pip install -r requirements.txt` (versions pinned).
- Unzip the challenge's `student_resource.zip` so the data sits at
  `data/student_resource/dataset/{train,test}/*.tsv` **relative to this folder**.
- Resources: we ran on 32 CPU cores / 64 GB RAM (~25 min end to end). 16 cores work but take
  ~1–1.5 h; ≥32 GB RAM is recommended for the feature and training steps (~30M candidate pairs).

## Run (from this folder)

```bash
python src/normalize.py             # raw TSVs -> work/{train,test}.parquet, work/train_gt.parquet
python src/block.py train           # retrieval: TF-IDF top-10 + key + address passes -> work/train_cands_full.parquet
python src/block.py test            # -> work/test_cands_full.parquet
python src/block.py train --prune   # candidate pruning -> work/train_cands.parquet (prints recall vs size)
python src/block.py test --prune    # -> work/test_cands.parquet (= candidate_pairs.tsv)
python src/features.py train        # pair features + labels -> work/train_feats.parquet
python src/features.py test         # -> work/test_feats.parquet
python src/match.py fit final       # LightGBM + threshold tuning -> final/model_0.txt, final/metrics.json
python src/match.py predict final   # -> final/output/matching_results.tsv, final/output/candidate_pairs.tsv
# second stage: cross-encoder on the uncertain pairs (needs a GPU; we ran ce_gpu.py on Modal, one H100, ~5 min)
python src/ce_prep.py final         # LightGBM probabilities + hard pairs with raw text -> work/ce_{train,val,test}.parquet
modal run src/ce_gpu.py             # fine-tune multilingual-e5-base, score -> work/ce_pred_{val,test}.parquet
python src/ce_blend.py final        # blend weight + threshold on dense validation -> final/output/*.tsv
```
`ce_gpu.py` reads/writes `/vol/work` (the Modal volume); to run it on a local GPU, call `run.local()` with the
paths changed to `work/`. The pretrained model is downloaded from Hugging Face (MIT license).

`final/metrics.json` holds the validation F0.5 (20% of training S1 entities held out), blocking
recall, the F0.5-vs-threshold curve, feature importances and per-country test statistics;
`final/errors.tsv` holds a sample of validation false merges and missed matches.

## Files

| File | Role |
|---|---|
| `src/normalize.py` | Transliteration, alias/legal-suffix removal, spelling maps learned from training pairs, per-country filler-word detection |
| `src/block.py` | Candidate generation: each S2/S3 record retrieves its top-10 S1 records (same country) by sparse TF-IDF cosine, plus exact-key passes (sorted name words, name skeleton, spacing-free name, first two words, house number + street) and an address-only top-1 pass; `--prune` drops weak runner-ups and caps each S1's list |
| `src/features.py` | Pairwise string/number/context features for every candidate pair |
| `src/match.py` | LightGBM training, macro-F0.5 threshold/rule search on test-density validation, one-S1-per-record assignment, output writing |
