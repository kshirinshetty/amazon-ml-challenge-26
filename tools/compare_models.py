"""Score several runs' stage-1 models on the *current* validation features (e.g. with synthetic decoys at test
density) so they are compared on the same data. Usage: --script tools/compare_models.py (edit RUNS)."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD, WORK, assign, f05, predict_pairs, sim_dense  # noqa: E402

RUNS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["runs/006_ambiguity-bigger-model", "runs/008_synthetic-decoys"]
f = pl.read_parquet(f"{WORK}/train_feats.parquet")
if "--real" in sys.argv:  # only real records: synthetic decoys removed from validation
    f = f.filter(~pl.col("q").str.contains("-SYN"))
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
gt_v = gt.filter(FOLD == VALID_FOLD)
n_syn = f.filter(pl.col("q").str.contains("-SYN")).height
print(f"validation features: {len(f)} pairs, {n_syn} involving synthetic decoys")
for run in RUNS:
    m = json.load(open(f"{run}/metrics.json"))
    best = assign(predict_pairs(lgb.Booster(model_file=f"{run}/model.txt"), f, m["feats"]), 0.0).filter(FOLD == VALID_FOLD)
    curve = {round(float(t), 2): f05(best.filter(pl.col("p") >= t), gt_v, s1_ids) for t in np.arange(0.4, 0.96, 0.05)}
    t = max(curve, key=curve.get)
    d = sim_dense(predict_pairs(lgb.Booster(model_file=f"{run}/model.txt"), f, m["feats"]))
    print(f"{run}: best F0.5 {curve[t]:.4f} @ t={t}  | dense {d['dense']:.4f} @ {d['t_dense']}")
