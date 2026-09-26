# 000 — blocking v0 (char-3gram TF-IDF, heavy pruning)

Blocking-only experiment, run locally (16 threads). No model, no submission.

**Config** (differs from 001 only in blocking):
- Name: TF-IDF on character **3-grams** (`char_wb`) of the normalized name with spaces removed
- Address: TF-IDF on word **unigrams**
- `max_df = 0.002` (drop any n-gram present in >0.2% of S1 records) — needed for speed
- Combined score `0.6 * name_cos + 0.4 * addr_cos`, top-K = 3 S1 per S2/S3 record, within country

**Result** (train, all 7.64M true pairs) — see `block_train.log`:

| K | recall | avg candidates / S1 |
|---|---|---|
| 1 | 0.8151 | 4.65 |
| 2 | 0.8397 | 9.29 |
| 3 | 0.8512 | 13.93 |

**Why it failed:** in 1M+ business names almost every useful 3-gram ("tit", "tan", "imp") and address word
("red", "hill", "highland") is above the 0.2% cut, so vectors were left with rare, accidental n-grams.
Example: `Dr TITAN IMPORT` ranked `Indrtech Infra` first (shared rare gram "drt"), not `Titan Import`.
Empty-address queries: 41% recall; Indic-script names: 76%.

**Fix → run 001:** rarer-by-construction features (char 4-grams, address word bigrams) survive pruning.
