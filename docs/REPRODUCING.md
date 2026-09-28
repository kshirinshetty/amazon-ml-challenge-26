# Reproducing the final submission

This guide rebuilds `matching_results.tsv` and `candidate_pairs.tsv` of the final submission (run
[023](../runs/023_cross-encoder-base/)) from the raw challenge data. It covers the recommended path on
[Modal](https://modal.com), a local path for machines with enough memory, validation, packaging, and the problems
we hit along the way.

**Contents:**
[Requirements](#1-requirements) ·
[Setup](#2-setup) ·
[Run on Modal](#3-run-on-modal-recommended) ·
[Run locally](#4-run-locally) ·
[Validate and package](#5-validate-and-package) ·
[Expected numbers](#6-expected-numbers) ·
[Troubleshooting](#7-troubleshooting)

---

## 1. Requirements

| Stage | Hardware | Time (Modal) | Cost (Modal) |
|---|---|---|---|
| normalize → blocking → features → LightGBM | 32 CPU cores, 64 GB RAM | ≈ 50 min | ≈ $3 |
| cross-encoder (fine-tune + score) | 1 GPU with ≥ 24 GB (we used an H100) | ≈ 10 min incl. image build | ≈ $1 |
| hard-pair prep + blend | 32 CPU cores, 64 GB RAM | ≈ 10 min | ≈ $0.5 |

- **Memory is the hard constraint.** The full-data stages hold 25M+ candidate pairs with 78 features in memory.
  A 16 GB laptop is killed by the OOM killer during normalization; plan for ≥ 64 GB or use Modal.
- Python **3.12** managed by [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock` pin every version).
- The cross-encoder needs PyTorch 2.4 + Transformers 4.44; on Modal they are installed in the GPU image by
  `tools/ce_gpu.py`, so nothing extra is needed locally.
- Disk: ≈ 20 GB for data and intermediates (`data/`, `work/`).

## 2. Setup

```bash
git clone https://github.com/kshirinshetty/amazon-ml-challenge-26.git
cd amazon-ml-challenge-26
uv sync                      # creates .venv with the pinned dependencies
```

**Data.** On Modal the first pipeline stage downloads and unzips the official dataset by itself. For a local run:

```bash
mkdir -p data && cd data
curl -sSL https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip -o sr.zip
unzip -q sr.zip -x '__MACOSX/*' && rm sr.zip && cd ..
ls data/student_resource/dataset/{train,test}      # *_source{1,2,3}.tsv, train_ground_truth.tsv
```

**Modal** (for section 3):

```bash
uv tool install modal        # or: pip install modal
modal setup                  # opens a browser to authenticate
```

All data and intermediates live in a Modal Volume named `amazon-ml`, created on first use.

## 3. Run on Modal (recommended)

`modal_app.py` runs each pipeline stage as a function on a 32-CPU / 64 GB container, with `/vol` (the volume) as
the working directory. Four commands rebuild the submission:

```bash
# 1. LightGBM pipeline: download → normalize → synth --remove → block → prune → features → fit → predict
#    (~50 min). Writes runs/final/ on the volume (model_0.txt, metrics.json, output/) and downloads it at the end.
modal run modal_app.py --run final --no-synth

# 2. LightGBM probabilities for every pair + the uncertain ("hard") pairs with raw text (~5 min)
modal run modal_app.py --script "tools/ce_prep.py runs/final"

# 3. Fine-tune multilingual-e5-base on the hard pairs and score validation + test (one H100, ~10 min)
modal run tools/ce_gpu.py

# 4. Blend with LightGBM, pick weight + threshold on dense validation, write the submission (~5 min)
modal run modal_app.py --script "tools/ce_blend.py runs/final"
```

Step 4 overwrites `runs/final/output/` **on the volume** with the blended submission. Fetch the files one at a
time (see [troubleshooting](#7-troubleshooting) for why) and check them:

```bash
mkdir -p runs/final/output
modal volume get --force amazon-ml runs/final/output/matching_results.tsv runs/final/output/
modal volume get --force amazon-ml runs/final/output/candidate_pairs.tsv runs/final/output/
modal volume get --force amazon-ml runs/final/metrics.json runs/final/
```

**Useful flags of `modal_app.py`:** `--start STAGE` / `--stop STAGE` rerun part of the pipeline on the cached
intermediates (stages: `download normalize synth block addr prune features noise fit predict`); `--script "PATH
ARGS"` runs any script on the full data. `modal app list` shows running jobs; `modal app stop -y <id>` stops one.

> `work/` on the volume holds only the **latest** run's intermediates. Run experiments that share it one after
> another, or only in parallel once the stages they read are finished.

## 4. Run locally

On a machine with ≥ 64 GB of RAM (and a GPU for step 7), from the repository root:

```bash
uv run python src/normalize.py                 # data/ → work/{train,test}.parquet, work/train_gt.parquet, maps
uv run python src/block.py train               # TF-IDF + exact-key + address retrieval → work/train_cands_full.parquet
uv run python src/block.py test
uv run python src/block.py train --prune       # → work/train_cands.parquet (prints the recall / size grid)
uv run python src/block.py test --prune        # → work/test_cands.parquet  (= candidate_pairs)
uv run python src/features.py train            # 78 pair features (+ labels) → work/train_feats.parquet
uv run python src/features.py test
uv run python src/match.py fit runs/final      # LightGBM + dense validation → runs/final/{model_0.txt,metrics.json}
uv run python src/match.py predict runs/final  # LightGBM-only submission → runs/final/output/
uv run python tools/ce_prep.py runs/final      # hard pairs → work/ce_{train,val,test}.parquet
```

**Step 7, cross-encoder.** `tools/ce_gpu.py` is written as a Modal function that reads and writes `/vol/work`. To
run it on a local GPU: `uv pip install torch==2.4.1 transformers==4.44.2 sentencepiece`, replace the two
`/vol/work` prefixes with `work`, drop the `vol.reload()` / `vol.commit()` calls, and call `run.local()`. It
writes `work/ce_pred_{val,test}.parquet`. Then:

```bash
uv run python tools/ce_blend.py runs/final     # blended submission → runs/final/output/
```

## 5. Validate and package

```bash
R=runs/final/output
python3 docs/student_resource/validate_submission.py \
    --matching $R/matching_results.tsv --candidate $R/candidate_pairs.tsv \
    --test-dir data/student_resource/dataset/test --check-ids     # must print PASS
wc -l $R/*.tsv                                                    # 1,732,545 lines each (header + 1,732,544 S1)
grep -c -P '\x00' $R/*.tsv                                        # 0 NUL bytes in each

./package.sh runs/final wsg      # → submission/wsg_submission.zip (output/, code/, Documentation_template.md)
```

`package.sh` copies `runs/<run>/src` into the zip. For a run built with the commands above, copy the pipeline and
the three cross-encoder scripts there first (`mkdir -p runs/final/src && cp src/*.py tools/ce_*.py
runs/final/src/`); runs launched through `modal_app.py --run` snapshot `src/` automatically.

## 6. Expected numbers

`runs/final/metrics.json` after steps 1 and 4:

| Metric | Run 023 (final) | Tolerance |
|---|---|---|
| `blocking_recall` (train, validation fold) | 0.9790 | ± 0.0005 |
| `oracle_f05` (perfect matcher on the candidates) | 0.9930 | ± 0.0005 |
| LightGBM `val_f05` / `val_f05_dense` | 0.9845 / 0.9769 | ± 0.0005 |
| Blend `val_f05` / `val_f05_dense` (w = 0.5, t = 0.6) | 0.9871 / 0.9809 | ± 0.001 |
| Test candidates per S1 | 15.3 | ± 0.1 |
| Test matches | 5.76M (3.33 per S1) | ± 0.5% |
| Leaderboard | final package ≈ 0.985; the e5-small variant (run 022) scored 0.981 publicly | |

LightGBM is seeded (`seed=2`, fixed data folds `hash(s1) % 5`), so steps 1–2 reproduce closely. The cross-encoder
fine-tune is not bit-for-bit deterministic on GPU (random head initialization, bf16 kernels): expect ± 0.001 on
the blended dense F0.5. The dense-validation curve in `metrics.json` shows the threshold choice.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Process killed during `normalize.py` / `features.py` | out of memory | ≥ 64 GB RAM, or run on Modal |
| `ComputeError: parquet ... Data corruption detected`, or NUL bytes in a TSV after download | `modal volume get` on large files or whole directories occasionally writes NUL-filled blocks | fetch files one at a time, check `grep -c -P '\x00'`; `modal_app.py` re-fetches and checks outputs automatically |
| A stage restarts from scratch halfway | Modal preempted the container | nothing to do: the step reruns; budget ~10% extra time |
| `Runner segmentation fault` while scoring the cross-encoder | tokenizing millions of pairs in one call (run 024) | fixed: `score()` now works in chunks of 500k pairs |
| Scores differ between two parallel experiments | both wrote to the shared `work/` on the volume | run experiments that write `work/` one after another |
| `modal run file.py` asks which entrypoint | a script with two `@local_entrypoint`s | use one entrypoint per file |

For the story behind each design choice, see the per-run READMEs in [`runs/`](../runs/) and the methodology
write-up in [`methodology.md`](methodology.md).
