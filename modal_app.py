"""Run the pipeline on Modal. Data + intermediates (work/) live in the `amazon-ml` Volume; code ships from src/.
Each run gets runs/<name>/ (code snapshot, model, metrics, errors, submission files), synced back locally.

  R=002_my-change; mkdir -p runs/$R
  modal run modal_app.py --run $R 2>&1 | tee runs/$R/modal.log                 # full pipeline
  modal run modal_app.py --run $R --start fit 2>&1 | tee runs/$R/modal.log     # reuse cached blocking + features
  modal run modal_app.py --run $R --start fit --pseudo                         # + self-training on France
  modal run modal_app.py --script tools/decoys.py                             # ad-hoc analysis script
Stages: download -> normalize -> block -> prune -> features -> fit -> predict (--start/--stop pick a range).
"""
import os
import shutil
import subprocess
import sys

import modal

app = modal.App("amazon-ml")
vol = modal.Volume.from_name("amazon-ml", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "unzip")
         .uv_sync().add_local_dir("src", "/root/src").add_local_dir("tools", "/root/tools"))
DOWNLOAD = ("mkdir -p data && curl -sSL https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"
            " -o data/sr.zip && cd data && unzip -qo sr.zip -x '__MACOSX/*' && rm sr.zip")
STAGES = ["download", "normalize", "block", "prune", "features", "fit", "predict"]


@app.function(image=image, volumes={"/vol": vol}, cpu=32, memory=65536, timeout=3 * 3600)
def step(args: list[str]):
    vol.reload()  # warm containers are reused across stages; see files committed by earlier stages
    if args[0].endswith(".py"):  # src/ scripts by name, analysis scripts as tools/<name>.py
        args = [sys.executable, f"/root/{args[0]}" if "/" in args[0] else f"/root/src/{args[0]}", *args[1:]]
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/vol")
    vol.commit()


@app.local_entrypoint()
def main(run: str = "", start: str = "download", stop: str = "predict", script: str = "", pseudo: bool = False,
         unseen_t: float = -1.0):
    if script:  # analysis on the full data without touching the laptop's RAM: --script tools/x.py
        return step.remote([script])
    run_dir = f"runs/{run}"
    shutil.copytree("src", f"{run_dir}/src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for stage in STAGES[STAGES.index(start):STAGES.index(stop) + 1]:
        if stage == "download":
            step.remote(["sh", "-c", DOWNLOAD])
        elif stage == "normalize":
            step.remote(["normalize.py"])
        elif stage in ("block", "prune", "features"):  # train and test in parallel containers
            py, extra = ("block.py", ["--prune"]) if stage == "prune" else (f"{stage}.py", [])
            list(step.map([[py, split, *extra] for split in ("train", "test")]))
        else:  # --pseudo: self-train on the previous run's confident predictions for unseen countries
            extra = (["--pseudo"] if pseudo and stage == "fit" else []) + (
                ["--unseen-t", str(unseen_t)] if unseen_t >= 0 and stage == "predict" else [])
            step.remote(["match.py", stage, run_dir] + extra)
    if STAGES.index(stop) >= STAGES.index("fit"):
        get = lambda src, dst: subprocess.run(["modal", "volume", "get", "--force", "amazon-ml", src, dst], check=True)
        get(run_dir, "runs/")
        # directory downloads have twice left NUL-filled blocks in the big TSVs: re-fetch each alone and check
        for out in [d for d in os.listdir(run_dir) if d.startswith("output")]:
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                path = f"{run_dir}/{out}/{name}"
                if not os.path.exists(path):
                    continue
                get(path, f"{run_dir}/{out}/")
                with open(path, "rb") as fh:
                    assert all(b"\0" not in b for b in iter(lambda: fh.read(1 << 24), b"")), f"corrupted: {path}"

