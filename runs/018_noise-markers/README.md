# 018 — formatting-noise features

016 (address pass, 1023-leaf LightGBM) + 17 label-free per-record features describing the raw text's formatting.

## Why

A local probe on the 10.3M training S2/S3 records: true records carry more *source noise* than decoys (decoys are
near-copies of clean S1 text), and every existing feature lowercases or strips it.

| Pattern (record side) | Share | Decoy rate (base ≈ 26%) |
|---|---|---|
| name is a URL / domain | 4% | ≈ 4% |
| name all lowercase (US) | 7% | 13–14% |
| accents, digit inside a word, leading punctuation | 1–7% | 16–18% |
| `#` / `door no` / `h.no` in address (India) | 11% | 31–32% |

## What

`features.py`: `noise_feats` (`nz_*`: URL name, lower/upper case name, upper-case address, accents, digit in a word,
leading/trailing punctuation, double space, brackets, `(ID:` tags, title prefixes, `#`, `door no`/`h.no`, `PMB`,
`NULL`, zero-padded numbers). Part of `build`; `features.py SPLIT --noise` appends them to an existing feature file
(this run: `modal_app.py --start noise`, ≈ 2 min, instead of recomputing all features). Same model as 016.
A local A/B was planned but the downloaded feature file was corrupted (Modal volume download), so the A/B ran on
Modal against 016 as the baseline.

## Results

| Metric | 016 | **018** |
|---|---|---|
| Validation F0.5 | 0.9838 | **0.9843** |
| **Dense validation** | 0.9759 | **0.9766** |
| Threshold | 0.75 | 0.75 |
| Trees | 573 | 606 |
| Candidates | same | same (14.55 / S1) |

The noise group takes 0.26% of gain (top: `nz_url`, `nz_title`, `nz_digit_word`). Small, consistent gain.
