# HANDOFF — Amazon ML Challenge 2026 (team wsg)

> **Working notes (historical).** Written mid-competition on 2026-09-27 ~17:00 IST, when the best leaderboard score
> was 0.970. Final results, the complete method and the reproduction guide are in the [README](../../README.md) and
> [REPRODUCING.md](../REPRODUCING.md).

Everything a new session needs to continue: goal, setup, pipeline, every experiment and its lesson, where the
remaining points are, and what to try next. Written 2026-09-27 ~17:00 IST at the end of a long session.

---

## 0. TL;DR

- **Task:** entity resolution — for each Source 1 (S1) business find all matching Source 2/3 (S2/S3) records.
  Metric: macro F0.5 per S1 (precision-heavy, singletons count). Full statement: [`docs/problem_statement.md`](../problem_statement.md).
- **Best public leaderboard: 0.970 (run 014).** #1 on the board: **0.9908**. History: 0.933 → 0.945 → 0.957 → 0.961 → 0.967 → 0.970.
- **Final package is built:** `submission/wsg_submission.zip` (170 MB, from run 014, validator PASS). Rebuild any time with
  `./package.sh runs/<run> wsg`.
- **Offline proxy that predicts the leaderboard:** "dense" validation (`val_f05_dense` in each run's `metrics.json`).
  Predicted 0.961 / 0.967 / 0.969 → actual 0.961 / 0.967 / 0.970. **Optimize dense val.**
- **Modal budget is almost gone:** $25.83 of the $30/month Starter credit used (≈ $4 left ≈ 2 short runs).
  Full-data work cannot run on the laptop (15 GB RAM → OOM). See §3.
- **Deadline:** the Unstop page timer read "01 day 03 h 12 min" on 2026-09-26 ~20:00 IST → ≈ **2026-09-27 ~23:00 IST**.
  Verify on the portal. The final zip must be submitted separately from leaderboard uploads.

---

## 1. The challenge (short)

- Data: `data/student_resource/dataset/{train,test}/*_source{1,2,3}.tsv` (tab-separated; `entity_id, business_name,
  business_address, country`) + `train_ground_truth.tsv` (`source1_entity_id`, comma-separated `matched_entity_ids`).
- Countries: train US + India; **test adds France (15% of test S1s, no labels)**. `country` must stay an open label.
- Outputs (tab-separated, header required, one row per test S1, empty list allowed):
  `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) — the only file scored;
  `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`) — the candidate set fed to the model;
  **smaller candidate sets rank higher in the final review**.
- Final zip layout: `output/{matching_results,candidate_pairs}.tsv`, `code/business_entity_resolution/{src/,README.md,
  requirements.txt}`, `Documentation_template.md` (our filled version is `docs/methodology.md`) — `package.sh` does it.
- Rules: **no external data / APIs / geocoding (disqualification)**; final model MIT/Apache and ≤ 8B params
  (ours: LightGBM, MIT). Synthetic data derived from the provided data is fine.
- Validator: `python3 docs/student_resource/validate_submission.py --matching X --candidate Y --test-dir data/student_resource/dataset/test --check-ids`.
- Team: **wsg** — Siddartha Aralakuppe Yogesha, Kshirin Shetty.

## 2. Key facts about the data (measured)

| Fact | Value | Consequence |
|---|---|---|
| Train size | S1 2.21M, S2 5.03M, S3 5.29M, 7.64M true pairs (3.46 per S1) | full-data work only on Modal |
| Matches per S1 | 0 → 5.6%, 1 → 5.4%, 2–6 → ~85%, max 11 | |
| S2/S3 record → S1 | **at most one S1**, always | retrieve from S2/S3 side; assign each record to its best S1 |
| Cross-country matches | never | block per country |
| Decoys (S2/S3 matching no S1) | 26% train; **≈42% test** (label-free estimate, 5.75 vs 4.67 records per S1) | test is decoy-denser → dense validation |
| How decoys are made | near-copy of a real S1 + marker word (US `midtown eastgate greater northside…` ~100% decoy, `holdings` 87%, `ltd` 72%; India `overseas infratech exports…`; France `holding participations distribution international snc`) and/or shifted house number | true records keep the S1's first house number 84%, decoys 9% |
| Empty-address records | 97.7% are true records, but ~92% of their names are shared by several S1s | mostly unresolvable ties; half are missed |
| S1 names shared by ≥2 S1s | ~50% of best candidates | ambiguity features, raw-text tie-breaks |
| France | 11.7% of France S1s share an exact address with another S1 (US/India 4–5%) | many same-building businesses |
| Indic scripts | ~13% of India S2 names, ~14% of addresses | unidecode + learned spelling maps |

## 3. Environment and setup

### Machines
- Laptop: Linux, 16 threads, **15 GB RAM**, RTX 2080 Super 8 GB. **Never run full-data stages locally** — the kernel
  OOM-killed `normalize.py` (and the Claude session) once. Local work = small samples, reading small artifacts.
- Python 3.12 via **uv** (`pyproject.toml`, `uv.lock`; `uv sync`). Deps: polars, pyarrow, scikit-learn,
  sparse-dot-topn, rapidfuzz, unidecode, lightgbm.

### Git
- Repo: `/home/sidd/Desktop/amazon-ml` → `https://github.com/kshirinshetty/amazon-ml-challenge-26` (branch `main`,
  pushed directly; `git -c http.postBuffer=524288000 push origin main` for large model files).
- `.gitignore`: `data/ .venv/ work/ output/ logs/ runs/*/output*/ submission/ runs/probes/*/ .agents/`. Submission
  TSVs (~100–400 MB) are never committed; model files are (≤ 70 MB; GitHub hard limit 100 MB).
- Commit convention: one commit per run, **no Claude co-author trailer** (user's request).

### Modal (all full-data compute)
- Profile `siddarthaa`, **Starter plan $30/month credit — $25.83 used on 2026-09-27**. `modal billing summary` shows spend.
  Resets monthly; if more compute is needed, add credit/card or use another workspace (user decision).
- Cost: container = 32 cores + 64 GB ≈ **$2/hour**. Full pipeline run ≈ $2–3; features→predict ≈ $1–1.5;
  analysis script ≈ $0.2–0.5.
- Volume **`amazon-ml`** holds `data/` (downloaded from the CDN by the pipeline), `work/` (shared caches: normalized
  parquet, candidates, features — **overwritten by every run**), `runs/<name>/` (per-run outputs).
- `modal_app.py` (run from repo root):
  ```bash
  R=015_name; mkdir -p runs/$R
  modal run modal_app.py --run $R 2>&1 | tee runs/$R/modal.log                   # full pipeline
  modal run modal_app.py --run $R --start features 2>&1 | tee runs/$R/modal.log  # reuse cached blocking
  modal run modal_app.py --run $R --start fit                                   # reuse cached features
  modal run modal_app.py --script tools/decompose.py                            # any analysis script on full data
  ```
  Stages: `download → normalize → synth → block → prune → features → fit → predict → stack`.
  Flags: `--start/--stop`, `--no-synth` (synth stage strips synthetic rows — **keep synthetic decoys OFF**: when running
  from `synth`/`normalize` pass `--no-synth`; runs from `block` onward use whatever `work/train.parquet` holds — currently
  no synthetic rows), `--pseudo` (France self-training), `--unseen-t X` (separate threshold for unseen countries).
  The local entrypoint snapshots `src/` into `runs/$R/src/` and downloads `runs/$R` at the end, re-fetching each TSV
  alone and asserting no NUL bytes (directory downloads corrupted big files twice).
- Gotchas: warm containers need `vol.reload()` (done in `step`); a **preempted** container restarts its step from scratch
  (happened in run 014); `modal app stop -y <app-id>` to kill; `modal app list` shows running apps;
  `modal app logs <id> --timestamps` for timing; a script with two `@local_entrypoint`s makes `modal run file.py` ambiguous.

## 4. Repository layout

```
src/normalize.py   raw TSVs -> work/{train,test}.parquet (+ train_gt, maps.json, {split}_noise.json)
src/synth.py       synthetic decoys (NOT used in the best run; --remove strips them)
src/block.py       TF-IDF reverse retrieval (top-10) + exact-key passes -> *_cands_full; --prune -> *_cands
src/features.py    61 pair features -> work/{split}_feats.parquet (+ label on train)
src/match.py       LightGBM fit (80% of S1s), threshold on dense validation, predict -> runs/<run>/output/
src/stack.py       stage-2 stacking (rejected; kept for reference)
modal_app.py       Modal runner; package.sh builds the zip (runs the validator)
tools/             analysis scripts (all run via `modal run modal_app.py --script tools/<x>.py`)
runs/NNN_*/        per-run README (why / what / results), src snapshot, model.txt, metrics.json, errors.tsv, logs
docs/              problem statement, official resources, methodology.md (goes in the zip), package_README.md, HANDOFF.md
```

## 5. Current best pipeline (run 014, leaderboard 0.970)

1. **normalize.py** — alias strip (`X formerly/fka/aka/dba Y → Y`), unidecode, drop domains/phones/punctuation,
   collapse repeated letters, legal-form stop list; **spelling maps learned from matched train pairs** (1,468 name +
   4,707 address tokens; only S2/S3-side spellings; lossy compound maps skipped); **filler words per split+country
   without labels** (≥3× over-represented in S2/S3 for names, ≥10× for addresses) removed for blocking only;
   address abbreviations, US/India state names → codes, French `n°`.
2. **block.py** — per country: TF-IDF on name char-4-grams (spaces removed) + address word 1–2-grams, `max_df=0.005`,
   score `0.6·cos_name + 0.4·cos_addr`, each S2/S3 record retrieves **top-10** S1s (`sparse_dot_topn`). **Exact-key
   passes** (skip keys shared by >50 S1s; keep 2 per record by similarity of the other field): sorted name words,
   consonant skeleton, spacing-free name, first two words, house number + first real street word. `via_key` bitmask.
   **Prune**: keep rank 0, key pairs, and rank<10 with score ≥ 0.8×best; cap 30 per S1 on runner-ups only.
   Recall 0.9759 before prune → **0.9706 after**; test 13.5 candidates per S1.
3. **features.py** (61): rapidfuzz name/address similarities; TF-IDF cosines; retrieval rank/score/top1/top2/gap/margin;
   records pointing at the S1; house numbers (overlap, first-number eq/diff, S1 number in record, min distance);
   full-name extra words + **label-free decoy score** of extra words; sibling agreement; ambiguity (S1s sharing
   name/address, share of known words); **within-record tie-breaks** (raw-text similarity, gap to the record's best
   candidate, is-best, ties). No country features.
4. **match.py** — LightGBM `lr 0.05, num_leaves 511, min_data_in_leaf 50, feature_fraction 0.8, bagging 0.8`, early
   stopping (patience 60), trained on folds 1–4 of `hash(s1) % 5` (fold 0 = validation); each record → its best S1,
   kept if p ≥ t; **t chosen on dense validation** (0.75 for 014).

Numbers (014): val 0.9806, **dense 0.9717**, blocking recall 0.9706, oracle 0.9899, 449 trees.

## 6. Validation — what to trust

- **Normal val**: F0.5 on fold-0 S1s. **Dense val** (`sim_dense` in `match.py`): same, with 50% of true records removed
  so decoys are ~42% like test. **Dense val ≈ leaderboard** (006: 0.9638 → 0.961; 013: 0.9701 → 0.967; 014: 0.9717 → 0.970).
- **Do not trust**: validation with synthetic decoys present (008/010/012 looked +0.3–0.4 better there and scored
  −0.6 on the leaderboard); stage-2 gains on dense val (007: +0.37 dense, −0.1 leaderboard, because dense val removes
  true records but does not add decoys per S1).
- Validation covers US/India only; France is visible only on the leaderboard (now ≈ US/India level: LB ≈ dense val).

## 7. Experiment log (leaderboard where submitted)

| Run | Change | Dense val | LB | Lesson |
|---|---|---|---|---|
| 000 | char-3gram TF-IDF, max_df 0.002 | — | — | pruning common n-grams destroyed recall (85%) |
| 001 | char-4gram name + address bigrams, top-3, LGBM 29 feats | — | 0.933 | baseline |
| 002 | learned spelling maps, alias/filler removal, prune, 80% train, expected-F rule | — | 0.945 | |
| 003 | label-free decoy features (full-name extra words, decoy score, house numbers, siblings) | — | 0.957 | filler removal had erased decoy markers |
| 004 | France self-training (pseudo-labels) | — | not sub. | no change |
| 005 | top-10 retrieval | — | not sub. | recall +0.6 but no F0.5 gain alone |
| 006 | ambiguity features + bigger LGBM | 0.9638 | **0.961** | |
| 007 | stage-2 stacking (OOF probs) | 0.9674 | 0.960 | added decoys on test; shelved |
| probes | 006 model at t=0.80 / 0.90 | — | 0.959 / 0.956 | stricter is worse |
| 008 | 2.6M synthetic decoys | (synth ctx) | not sub. | |
| 009 | exact-key blocking passes | 0.9684 | not sub. | biggest single blocking gain |
| 010 | 009 + synthetic decoys | (synth ctx) | not sub. | |
| blend | 0.3·009 + 0.7·010 | — | 0.961 | synthetic decoys don't transfer |
| 011 | more keys | — | not sub. | prune cap dropped true pairs at hub S1s |
| 012 | more keys + synth + safe cap | (synth ctx) | 0.961 | synth −0.6 vs 013 |
| 013 | 012 without synth | 0.9701 | **0.967** | |
| **014** | 013 + within-record tie-break features | **0.9717** | **0.970** | |

Each `runs/<run>/README.md` has details; `metrics.json` has curves, feature gains, per-country test stats.

## 8. Where the remaining points are (013, dense val 0.9701; 014 is similar)

Fixing one error type perfectly would add (`tools/decompose.py`):
- **Never-retrieved true pairs: +1.49** (22.5k pairs in dense val: normal 10.7k, empty-address 7.7k, Indic 4.1k).
  Train recall 0.9759 before prune vs 0.9706 after: the prune (hub S1 cap + REL) costs ~0.5 pt recall.
- **Retrieved but rejected: +1.01** (15.3k: empty-address 10.0k — mostly name ties that are genuinely ambiguous).
- **False merges: +0.54** (almost all decoys).
Oracle F0.5 on the candidates: 0.9899 → even a perfect matcher on today's candidates tops out ≈ 0.99 on normal val.

## 9. Ideas to push further (ranked by expected gain / cost; budget is the constraint)

1. **Cheap, likely positive (fit-only, ~$0.5–1 each):**
   - LightGBM ensemble: 3–5 seeds / `num_leaves` 1023 / lr 0.03 with more rounds; average probabilities.
   - Add 014's error sample review (`runs/014_relative-tiebreak/errors.tsv`) → targeted features for the top FP/FN patterns.
2. **Blocking recall (+1.5 available):**
   - Prune loses 0.5 pt: raise the cap for key-pass pairs / exempt pairs whose raw-name or raw-address is the record's
     best; try `CAP=50` with RMAX 5 to keep size in check (grid printed by `block.py train --prune`).
   - Second TF-IDF pass **without max_df pruning** only for records whose best score is low (names made of common
     words) — or a word-level name index with IDF weighting.
   - Empty-address records: keep *all* S1s sharing the exact normalized name when there are ≤ 5 (key passes keep 2,
     picked by address similarity, which is meaningless without an address).
   - Better Indic transliteration: lower `MAP_MIN` (20 → 5) for more learned maps; consonant-skeleton TF-IDF.
3. **Rejected pairs (+1.0):** mostly ties; more tie-break signal (raw legal-form match, source S2 vs S3 behaviour,
   exact raw-name equality flags).
4. **Bigger model (only with more budget):** fine-tune a small MIT/Apache cross-encoder (e.g. multilingual-e5-small /
   MiniLM) on hard pairs on a Modal GPU and blend with LightGBM.
- **Already tried, don't repeat:** France self-training, stricter thresholds, stage-2 stacking, synthetic decoys,
  separate threshold for empty-address records, same-name sibling voting (55% accurate), deeper top-10 alone.

## 10. Cookbook

```bash
cd /home/sidd/Desktop/amazon-ml
modal billing summary                                   # spend so far
modal app list; modal app stop -y <app-id>              # running apps / kill
R=015_x; mkdir -p runs/$R
modal run modal_app.py --run $R --start fit 2>&1 | tee runs/$R/modal.log     # retrain on cached features
python3 -c "import json; m=json.load(open('runs/$R/metrics.json')); print(m['val_f05'], m['val_f05_dense'], m['threshold'])"
python3 docs/student_resource/validate_submission.py --matching runs/$R/output/matching_results.tsv \
    --candidate runs/$R/output/candidate_pairs.tsv --test-dir data/student_resource/dataset/test --check-ids
./package.sh runs/$R wsg                                # -> submission/wsg_submission.zip
git add runs/$R README.md src tools && git commit -m "Run $R: ..." && git -c http.postBuffer=524288000 push origin main
```
- Check each run's README table and the root README "Runs" table; add a row per run.
- Before uploading: `grep -c -P '\x00'` must be 0 and line count 1,732,545 for both TSVs.
- `work/` on the volume holds only the latest run's caches — a `--start features` run uses whatever blocking ran last
  (currently run 013/014's blocking, no synthetic rows).
