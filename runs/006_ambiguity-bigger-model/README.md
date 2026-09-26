# 006 — ambiguity features + larger LightGBM

Builds on [005](../005_deeper-retrieval/) blocking (top-10, REL 0.8, cap 30) and [003](../003_decoy-features/) features.
Ran on Modal (32 cores / 64 GB).

## Why

Remaining errors in 005 (`errors.tsv`): 164 of 300 sampled false merges belong to a *different* S1 —
mostly records with **no address** whose name is shared by several S1s (`Continental Equity | None`), and
**made-up trade names** at an S1's exact address (`Umbraveramira | H-103 RIICO ...`) — while address-less
records with a unique name were missed at p≈0.3–0.6. Nothing told the model how ambiguous a pair is.

## New features (label-free)

| Feature | Meaning |
|---|---|
| `amb_s_name`, `amb_q_name` | S1s in the country with exactly the candidate's / the record's normalized name |
| `amb_s_addr` | S1s in the country sharing the candidate's exact normalized address |
| `name_eq`, `q_name_ties` | record name equals the candidate's; how many of the record's candidates it equals |
| `q_known_frac` | share of the record's name words seen in ≥2 S1 names of the country (≈0 for made-up names) |

Model: LightGBM `learning_rate 0.05` (was 0.1), `num_leaves 511` (255), `min_data_in_leaf 50` (100),
`feature_fraction 0.8` (0.9), up to 3000 rounds, early stopping 60 → 326 trees.

## Results

| Metric | 003 | 005 | **006** |
|---|---|---|---|
| **Validation F0.5** | 0.9712 | 0.9711 | **0.9747** (threshold 0.65) |
| Blocking recall / oracle | 0.9524 / 0.9834 | 0.9581 / 0.9852 | 0.9581 / 0.9852 |
| Test candidates / S1 | 7.8 | 9.4 | 9.4 |
| France matches / S1, % empty | 3.11 / 6.9% | 3.16 / 6.5% | 3.18 / 6.5% |
| Leaderboard | 0.957 | — | **0.961** |

Top gains: `num_jac` 39.0%, `a_tset` 22.1%, `dec_max` 7.4%, `s_num_mindiff` 5.4%, `rank` 5.1%,
`g_same_num_frac` 4.1%; new ambiguity features ~1.6% combined.

`output_unseen0.5/`: same model with threshold 0.5 for countries absent from training (France) — adds ~15k
France matches, 16% of which have disagreeing house numbers (vs 2% of France's accepted pairs). Judged too
risky for a submission.
