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
| [001](runs/001_tfidf-lgbm-baseline/) | **baseline**: char-4gram name + address uni/bigram TF-IDF blocking, LightGBM on 29 features | 0.953 | 17.3 | 0.949 | **0.933** |
| [002](runs/002_learned-cleaning/) | learned spelling maps + alias/filler removal, prune (rel 0.7, cap 30), +4 features, 80% train, expected-F0.5 per-S1 decisions | 0.952 | **7.8** | **0.957** | **0.945** |
| [003](runs/003_decoy-features/) | label-free decoy features: full-name extra words + decoy-word score, S1 house number in record, sibling-record agreement | 0.952 | **7.8** | **0.971** | **0.957** |
| [004](runs/004_france-selftrain/) | 003 + self-training on confident France test pairs | 0.952 | 7.8 | 0.970 | not submitted (France unchanged) |
| [005](runs/005_deeper-retrieval/) | 003 + top-10 retrieval, prune REL 0.8 / cap 30 | 0.958 | 9.4 | 0.971 | not submitted (no val gain) |
| [006](runs/006_ambiguity-bigger-model/) | 005 + name/address ambiguity + made-up-name features, larger LightGBM (lr 0.05, 511 leaves) | 0.958 | 9.4 | **0.975** | **0.961** |
| [007](runs/007_stage2-stacking/) | 006 + stage-2 stacking on out-of-fold stage-1 probabilities; thresholds from test-like (dense) validation | 0.958 | 9.4 | **0.977** (dense **0.967**) | _pending_ |

Since 007, "dense" = validation with half the true records removed, matching the test's ~42% decoy share;
it tracks the leaderboard (006: dense 0.9638, leaderboard 0.961).

Every `runs/<NNN_name>/` holds a `README.md` (architecture, hyperparameters, results, notes), the
`src/` snapshot that produced it, `model.txt`, `metrics.json` (validation curve, feature gains,
per-country test stats), `errors.tsv` (sample of false merges / misses) and `modal.log`.
`output/` (the two submission TSVs, ~500 MB) stays out of git; it is kept locally and in the Modal volume.

## Pipeline (`src/`)

| Stage | Script | What it does |
|---|---|---|
| normalize | `normalize.py` | transliterate, alias/legal-suffix removal, spelling maps learned from train pairs, per-country filler-word removal |
| block | `block.py` | each S2/S3 record retrieves its top-3 S1 (same country) by sparse TF-IDF cosine; `--prune` drops weak rank-2/3 pairs, caps 30 per S1 |
| features | `features.py` | 42 pair features: rapidfuzz similarities, house-number overlap/distance, label-free decoy scores of extra words, sibling-record agreement, rank/score context |
| fit | `match.py fit RUN_DIR` | LightGBM on 80% of S1s; picks the better of a global threshold and a per-S1 expected-F0.5 rule on the 20% held out |
| predict | `match.py predict RUN_DIR` | each record → its best S1, then the chosen decision rule; writes both submission files |

## How to run

```bash
uv sync                                                      # local env (Python 3.12)
R=002_my-change; mkdir -p runs/$R
modal run modal_app.py --run $R 2>&1 | tee runs/$R/modal.log              # full pipeline on Modal
modal run modal_app.py --run $R --start fit 2>&1 | tee runs/$R/modal.log  # reuse cached blocking + features
modal run modal_app.py --script tools/decoys.py                           # full-data analysis without local RAM
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
