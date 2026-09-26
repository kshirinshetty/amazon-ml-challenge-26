# Amazon ML Challenge 2026 — Business Entity Resolution

Find, for every Source 1 business, all Source 2 / Source 3 records of the same real-world business
(noisy names/addresses; US + India in train, France appears only in test). Scored by macro F0.5 per
Source 1 entity; smaller candidate sets rank higher in the final review.

- Problem statement: [`docs/problem_statement.md`](docs/problem_statement.md)
- Official README, documentation template, validator: [`docs/student_resource/`](docs/student_resource/)
- Our methodology write-up (goes in the zip as `Documentation_template.md`): [`docs/methodology.md`](docs/methodology.md)

## Runs

| Run | What changed | Blocking recall (val) | Candidates / S1 (test) | Val F0.5 | Leaderboard |
|---|---|---|---|---|---|
| [000](runs/000_blocking-char3-v0/) | blocking v0: char-3gram name + address-word TF-IDF, `max_df=0.002` | 0.851 (train) | — | — | — |
| [001](runs/001_tfidf-lgbm-baseline/) | **baseline**: char-4gram name + address uni/bigram TF-IDF blocking, LightGBM on 29 features | 0.953 | 17.3 | 0.949 | _pending_ |

Every `runs/<NNN_name>/` holds a `README.md` (architecture, hyperparameters, results, notes), the
`src/` snapshot that produced it, `model.txt`, `metrics.json` (validation curve, feature gains,
per-country test stats), `errors.tsv` (sample of false merges / misses) and `modal.log`.
`output/` (the two submission TSVs, ~500 MB) stays out of git; it is kept locally and in the Modal volume.

## Pipeline (`src/`)

| Stage | Script | What it does |
|---|---|---|
| normalize | `normalize.py` | transliterate (unidecode), strip legal suffixes/phones/domains, canonicalise address abbreviations + states |
| block | `block.py` | each S2/S3 record retrieves its top-3 S1 (same country) by sparse TF-IDF cosine → candidate pairs |
| features | `features.py` | 29 pair features: rapidfuzz name/address similarities, number overlap, rank/score context |
| fit | `match.py fit RUN_DIR` | LightGBM, threshold tuned for macro F0.5 on 20% held-out S1 entities |
| predict | `match.py predict RUN_DIR` | each record → its best S1 if p ≥ threshold; writes both submission files |

## How to run

```bash
uv sync                                                      # local env (Python 3.12)
R=002_my-change; mkdir -p runs/$R
modal run modal_app.py --run $R 2>&1 | tee runs/$R/modal.log              # full pipeline on Modal
modal run modal_app.py --run $R --start fit 2>&1 | tee runs/$R/modal.log  # reuse cached blocking + features
./package.sh runs/$R wsg                                     # -> submission/wsg_submission.zip (+ validator)
```

Modal keeps data and intermediates (`work/`) in the `amazon-ml` volume and syncs `runs/$R/` back when
the run finishes. To run locally instead: `uv run python src/<script>.py ...` from the repo root with the
dataset unzipped under `data/` (see [`docs/package_README.md`](docs/package_README.md)).

## Layout

```
src/            current pipeline code          runs/        one folder per experiment (tracked)
modal_app.py    Modal runner                    docs/        problem statement, official resources, methodology
package.sh      builds the submission zip       data/ work/ submission/   dataset, caches, built zips (gitignored)
```
