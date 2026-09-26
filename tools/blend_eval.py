"""Evaluate an average of several runs' stage-1 probabilities on real validation records (current features).
Usage: --script tools/blend_eval.py"""
import json
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "/root/src")
from match import WORK, predict_pairs, sim_dense  # noqa: E402

RUNS = ["runs/009_multipass-blocking", "runs/010_multipass-synth"]
f = pl.read_parquet(f"{WORK}/train_feats.parquet").filter(~pl.col("q").str.contains("-SYN"))
ps = []
for run in RUNS:
    m = json.load(open(f"{run}/metrics.json"))
    ps.append(predict_pairs(lgb.Booster(model_file=f"{run}/model.txt"), f, m["feats"]))
for w in (0.3, 0.5, 0.7):  # weight of the first run
    blend = ps[0].with_columns(p=w * pl.col("p") + (1 - w) * ps[1]["p"])
    d = sim_dense(blend)
    print(f"blend w({RUNS[0][-25:]})={w}: normal {d['normal']:.4f} @ {d['t_normal']}  dense {d['dense']:.4f} @ {d['t_dense']}", flush=True)
