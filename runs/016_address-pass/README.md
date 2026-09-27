# 016 — address-only retrieval pass

014's pipeline plus one blocking pass, and 015's best single model (`num_leaves 1023, min_data_in_leaf 100`).

## Why

Never-retrieved true pairs were the largest loss (+1.49 dense F0.5 if fixed; HANDOFF §8). In 014's error sample,
many misses carry the S1's exact address but a name that loses to look-alike S1 names in the combined score
`0.6·name + 0.4·address` ("Ram Private Limited Center" at its S1's address; made-up trade names; Indic-script
names). A local simulation (full S1 set of a country, 8–10% of its S2/S3 records) of adding each record's
**top-1 S1 by address alone**:

| Country | Recall after prune, 014 | + address top-1 | Candidates / S1 |
|---|---|---|---|
| India | 0.9523 | **0.9700** | +1.1 |
| US | 0.9829 | 0.9863 | +1.1 |

Top-2/3 or a cosine floor added candidates for ≤ +0.1 pt. A name-only top-k pass on top of this added nothing
(India 0.9699 → 0.9700 at +1.7 candidates/S1), so it was dropped.

## What

- `block.py`: `addr_pairs` — address TF-IDF (same word 1–2-gram vectorizer) top-1 S1 per record, flagged
  `via_key` bit 32, so prune keeps it like the key-pass pairs. Part of `block_country` for full runs;
  `block.py SPLIT --addr` adds it to an existing `cands_full` (this run: `modal_app.py --start addr`, ≈ 8 min,
  instead of redoing retrieval).
- Features unchanged (61; `via_key` carries the new bit). Model: one LightGBM, 1023 leaves.

## Results

| Metric | 014 | **016** |
|---|---|---|
| Blocking recall (train, after prune) | 0.9706 | **0.9785** |
| Oracle F0.5 | 0.9899 | **0.9928** |
| Validation F0.5 | 0.9806 | **0.9838** |
| **Dense validation (≈ leaderboard)** | 0.9717 | **0.9759** |
| Threshold | 0.75 | 0.75 |
| **Leaderboard** | 0.970 | **0.974** |
| Candidates / S1 (train / test) | 10.75 / 13.5 | 11.63 / **14.55** |
| Test matches / S1 (France, India, US) | 3.22, 3.26, 3.33 | 3.22, 3.31, 3.34 |

573 trees. Address pass on test: 1.0M–1.1M new pairs per large country. Remaining misses in the error sample are
mostly Indic-script names whose address shares only generic tokens with the S1 ("No. 245, Bengaluru, KA"),
and empty-address records with names shared by several S1s.
