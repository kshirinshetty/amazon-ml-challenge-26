# 001 — baseline: TF-IDF reverse blocking + LightGBM

First full pipeline and first leaderboard submission. Ran on Modal (32 cores / 64 GB).

## Architecture

```
normalize ─▶ block (per country) ─▶ features (29) ─▶ LightGBM ─▶ best-S1 assignment + threshold
```

1. **Normalize** — unidecode transliteration, lowercase; names: drop legal forms/fillers, phones,
   domains, collapse repeated letters; addresses: canonical abbreviations + state codes, strip zero padding.
2. **Block** — for each S2/S3 record, top-3 S1 records of the same country by
   `0.6·cos(name char-4gram TF-IDF) + 0.4·cos(address word 1–2gram TF-IDF)`, n-grams with df > 0.5% dropped.
3. **Features** — rapidfuzz name/address similarities, house-number overlap, lengths, transliteration
   flag, retrieval rank/score/gap/margin, how many records point at the S1.
4. **Model** — LightGBM binary on pairs of 40% of train S1 entities, early stopping on a 20% held-out fold.
5. **Assign** — each record → its highest-probability S1 if p ≥ t; t tuned for macro F0.5 on the held-out fold.

## Hyperparameters

| Stage | Setting |
|---|---|
| block | K=3, W_NAME=0.6, name `char_wb` (4,4), address words (1,2), `max_df=0.005` |
| fit | `objective=binary, learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1`, early stopping 30 → 219 trees |
| split | S1 hash % 5: fold 0 = validation, folds 1–2 = training (12.4M pairs) |
| assign | threshold t = 0.60 |

Full code: [`src/`](src/). All numbers: [`metrics.json`](metrics.json). Mistakes: [`errors.tsv`](errors.tsv).

## Results

| Metric | Value |
|---|---|
| **Validation F0.5** (macro, held-out S1s) | **0.949** |
| Blocking recall (validation fold) | 0.953 (recall@1 0.932, @2 0.946) |
| Oracle F0.5 (perfect matcher on these candidates) | 0.984 |
| Test candidates / S1 | 17.3 (France 16.6, India 17.5, US 17.3) |
| Test matches / S1, % empty | France 3.37 / 5.8%, India 3.24 / 6.5%, US 3.28 / 6.2% |
| **Leaderboard (public)** | **0.933** (submitted 2026-09-26 21:32 IST; #1 at the time: 0.9907) |

F0.5 vs threshold: 0.30 → 0.935, 0.40 → 0.943, 0.50 → 0.948, **0.60 → 0.949**, 0.70 → 0.948, 0.80 → 0.946, 0.90 → 0.938.

Top features by LightGBM gain: `gap` 71.5% (distance to the record's best retrieval score), `a_tset` 5.4%,
`n_jw` 3.5%, `rank` 2.9%, `top2` 2.1%, `len_aq` 1.7%, `a_partial` 1.5%, `n_ratio` 1.5%, `s1_n` 1.4%, `name_cos` 1.3%.

## Error analysis (`errors.tsv`)

- **False merges**: near-duplicates at the same/adjacent address — different house number
  (`9101` vs `9104`), extra name token (`... Technology Overseas`), different first word.
- **Misses** (300 sampled): 46% not in candidates (empty address + noisy name, Indic-script names),
  42% below threshold (typos / changed numbers), 12% assigned to another S1 (bad transliteration).

## Next

- **Hub S1 entities** (found after the run): transliterated legal words are not in the stop list —
  प्राइवेट → `praivet`, Tamil → `piraivet`, Bengali → `praibhet`, `limitet` — so Indic-script records share
  the rare 4-gram `raiv` with S1s like *Rai Ventures Private Limited*, which collects **181,535** candidates.
  22k S1s have >100 candidates (17.5% of all pairs; median is 11, p99 118). Fix the stop list and cap
  candidates per S1 → better Indic recall and a much smaller candidate set.
- Candidate set: 17.3/S1 is large; rank-1 alone gives 93.2% recall at 4.7/S1 → prune weak rank-2/3 pairs.
- Train on all non-validation folds; features for Indic names (consonant skeleton), DBA splits, number ranges.
- Blocking: a name-only retrieval path for address-less records.
