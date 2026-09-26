"""Run the pipeline on Modal. Data + intermediates (work/) live in the `amazon-ml` Volume; code ships from src/.
Each run gets runs/<name>/ (code snapshot, model, metrics, errors, submission files), synced back locally.

  R=002_my-change; mkdir -p runs/$R
  modal run modal_app.py --run $R 2>&1 | tee runs/$R/modal.log                 # full pipeline
  modal run modal_app.py --run $R --start fit 2>&1 | tee runs/$R/modal.log     # reuse cached blocking + features
Stages: download -> normalize -> block -> prune -> features -> fit -> predict (--start/--stop pick a range).
"""
import shutil
import subprocess
import sys

import modal

app = modal.App("amazon-ml")
vol = modal.Volume.from_name("amazon-ml", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "unzip")
         .uv_sync().add_local_dir("src", "/root/src"))
DOWNLOAD = ("mkdir -p data && curl -sSL https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"
            " -o data/sr.zip && cd data && unzip -qo sr.zip -x '__MACOSX/*' && rm sr.zip")
STAGES = ["download", "normalize", "block", "prune", "features", "fit", "predict"]


@app.function(image=image, volumes={"/vol": vol}, cpu=32, memory=65536, timeout=3 * 3600)
def step(args: list[str]):
    vol.reload()  # warm containers are reused across stages; see files committed by earlier stages
    if args[0].endswith(".py"):
        args = [sys.executable, f"/root/src/{args[0]}", *args[1:]]
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd="/vol")
    vol.commit()


@app.local_entrypoint()
def main(run: str, start: str = "download", stop: str = "predict"):
    run_dir = f"runs/{run}"
    shutil.copytree("src", f"{run_dir}/src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for stage in STAGES[STAGES.index(start):STAGES.index(stop) + 1]:
        if stage == "download":
            step.remote(["sh", "-c", DOWNLOAD])
        elif stage == "normalize":
            step.remote(["normalize.py"])
        elif stage in ("block", "prune", "features"):  # train and test in parallel containers
            script, extra = ("block.py", ["--prune"]) if stage == "prune" else (f"{stage}.py", [])
            list(step.map([[script, split, *extra] for split in ("train", "test")]))
        else:
            step.remote(["match.py", stage, run_dir])
    if STAGES.index(stop) >= STAGES.index("fit"):
        subprocess.run(["modal", "volume", "get", "--force", "amazon-ml", run_dir, "runs/"], check=True)
