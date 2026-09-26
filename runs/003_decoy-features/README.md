# 003 — label-free decoy features

Builds on [002](../002_learned-cleaning/): same normalization, blocking and pruning (candidate file is
byte-identical to 002); only features changed. Ran on Modal (32 cores / 64 GB).

## Why

Error analysis of 002: **74% of false merges were decoys** — S2/S3 records that match no S1 (26% of all
S2/S3). The generator makes them as near-copies of a real business:

```
S1  QD Midwest Babcock Center          | 925 Brookfield Drive, Unit 1, Mansfield, OH
 +  QD Center Babcock Midwest          | 925-A Brookfield Dr, # 1, Mansfielld, Ohio   (true: typos, reorder, same number)
 -  Qd Midwest Babcock Center Llc      | Unit 1, 929 Brookfield Drive ...            (decoy: number 925 → 929)
S1  Anthony Canary Inc.                | 1610 Clear Springs Court ...
 -  Anthony Canary Midtown Inc.        | ... 2621 CLEAR SPRINGS CT                   (decoy: extra word + number)
```

- True records keep the S1's first house number 84% of the time, decoys 9%.
- Decoys add a marker word; 002's filler removal deleted exactly those words (`midtown` 100% decoys,
  `holdings` 87%, US `ltd` 72%, India `overseas` 95%) and legal forms, making decoys identical to their S1.

## New features (all label-free, so identical on train/test and on France)

| Feature | Meaning |
|---|---|
| `n_xq`, `n_xs` | words in the record's **full** name (legal forms + filler kept) missing from the S1's, and vice versa |
| `dec_max`, `dec_sum` | decoy score of those extra words: per (split, country, word), among S2/S3 records whose best S1 lacks the word, the share whose first house number disagrees with that S1 (shrunk to the country mean, ≥200 records). Correlates 0.82 (India) / 0.88 (US) with the true decoy share |
| `s_num_mindiff`, `s_num_in_q` | distance from the S1's first house number to the closest number in the record; exact hit |
| `g_same_num`, `g_same_num_frac` | how many *other* records pointing at the S1 share this record's first number |
| `g_uniq_x` | this record's extra words that no other record of the S1 group has |

Learned decoy words (test, no labels): France `holding participations distribution international snc`
≈0.985; US `uptown lakeside northside riverside greater holdings midtown` ≈0.99. Stored per run in
`work/{split}_decoy_words.parquet` on the volume.

## Results

| Metric | 002 | **003** |
|---|---|---|
| **Validation F0.5** | 0.9568 | **0.9712** (threshold 0.70; expected-F rule 0.9711) |
| Blocking recall (validation) / oracle F0.5 | 0.9524 / 0.9834 | same |
| Test candidates / S1 | 7.8 | 7.8 |
| Test matches / S1, % empty | France 3.24/5.8%, India 3.18/5.9%, US 3.28/5.6% | France 3.11/6.9%, India 3.19/6.6%, US 3.27/6.0% |
| Leaderboard | 0.945 | _pending_ |

Top features by gain: `num_jac` 38.1%, **`dec_max` 13.8%**, `a_tset` 10.4%, `len_nq` 7.0%, `top2` 5.1%,
`rank` 3.6%, **`g_same_num_frac` 3.4%**, `s1_rank` 2.8%, `s1_n` 2.7%. LightGBM stopped at 245 trees.

France now predicts slightly fewer matches (3.11 vs 3.24) and more empties (6.9%) — consistent with
decoys there being rejected by the new features.

## Next

- Self-training on France (`--pseudo`): confident test pairs of unseen countries as extra labels.
- Blocking recall (4.8% of true pairs never reach the model) is now the biggest remaining gap.
