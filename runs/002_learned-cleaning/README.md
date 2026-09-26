# 002 — learned cleaning + candidate pruning + expected-F0.5 decisions

Builds on [001](../001_tfidf-lgbm-baseline/). Ran on Modal (32 cores / 64 GB).

## What changed vs 001

| Stage | Change |
|---|---|
| normalize | **Alias rule**: `X formerly/fka/aka/dba/doing business as Y` → `Y` (≈30k train names; the S1 name is always the part after the marker) |
| normalize | **Spelling maps learned from matched train pairs** (1,468 name + 4,707 address tokens): `praivet/piraivet/praibhet → private`, `limitet → limited`, `eksports → exports`, `sacrmento → sacramento`… Only S2/S3-side spellings are remapped; lossy maps (run-together words, 1-letter or function-word targets) are skipped |
| normalize | **Injected filler words detected per split + country** (no labels needed, so it works on France): tokens ≥3× (names) / ≥10× (addresses) over-represented in S2/S3 vs S1 — e.g. France `participations holding groupe distribution et`, `nord gironde atlantique`; US `midtown northside eastgate`; India `shri smt hardware stores` |
| normalize | French `n°` (`ndeg`) stripped |
| block | New **prune** step: keep a record's rank-2/3 S1 only if score ≥ 0.7 × its best score; cap 30 candidates per S1 |
| features | +4: `num_first_diff` (house-number distance), `num_q_in_s`, `n_extra_q`, `n_extra_s` (name words only on one side) → 33 features |
| fit | trained on 80% of S1s (folds 1–4) instead of 40% |
| decide | **per-S1 expected-F0.5 rule**: keep the top-k assigned records maximizing `1.25·Σp_top-k / (0.25·Σp + k)`, or none when `Π(1−p)` is higher; chosen over the global threshold because it scored higher on validation |

Hyperparameters otherwise as 001 (LightGBM `lr=0.1, num_leaves=255, min_data_in_leaf=100`, early stopping → 402 trees).

## Results

| Metric | 001 | **002** |
|---|---|---|
| **Validation F0.5** | 0.9490 | **0.9568** (expected-F rule; threshold rule 0.9555 @ t=0.65) |
| Blocking recall, before prune (train) | 0.9528 | 0.9550 |
| Blocking recall, after prune (validation) | 0.9527 | 0.9524 |
| Oracle F0.5 | 0.9836 | 0.9834 |
| Test candidates / S1 | 17.3 | **7.8** (France 8.3, India 7.8, US 7.6) |
| Largest candidate list (test) | 181,535 | **30** |
| `candidate_pairs.tsv` | 408 MB | **196 MB** |
| Test matches / S1, % empty | France 3.37/5.8%, India 3.24/6.5%, US 3.28/6.2% | France 3.24/5.8%, India 3.18/5.9%, US 3.28/5.6% |
| Leaderboard | 0.933 | _pending_ |

Prune grid on train (recall vs candidates/S1): no prune 0.9550 / 14.0 · rel 0.5 cap 30 0.9525 / 8.9 ·
**rel 0.7 cap 30 0.9524 / 6.3** · rel 0.8 cap 30 0.9517 / 5.6 · top-1 only 0.9375 / 4.7.

Top features by gain: `num_jac` 43.1%, `a_tset` 11.2%, `s1_n` 7.1%, `len_aq` 6.9%, `score` 4.2%,
`num_first_diff` 4.1% (new), `rank` 2.9%, `name_cos` 2.8%. `gap` dropped from 71.5% to near zero —
pruning already removes the large-gap pairs it used to reject.

Files: [`metrics.json`](metrics.json), [`errors.tsv`](errors.tsv), [`src/`](src/), logs
(`modal_normalize.log` first cleaning attempt, `modal_prep.log` normalize → block → prune grid, `modal.log` prune → predict).

## Notes / next

- Cleaning barely moved blocking recall (+0.2pt); most of the gain is in the matcher (cleaner names,
  number features, 2× training data, expected-F decisions).
- Absolute length features (`len_aq`, `len_as`, `len_nq`) carry ~12% of gain — a France risk if French
  address lengths differ; worth testing without them.
- Remaining gap to the oracle (0.957 → 0.983) is the matcher; gap to 1.0 beyond that is blocking recall.
