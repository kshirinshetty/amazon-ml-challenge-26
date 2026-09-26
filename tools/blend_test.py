"""Write a test submission from a weighted average of runs' stage-1 probabilities (runs must share the current
test features). Usage: --script tools/blend_test.py   -> runs/blend_009_010/output/"""
import json
import os
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "/root/src")
from match import WORK, assign, predict_pairs, write  # noqa: E402

RUNS, WEIGHTS, T, OUT = ["runs/009_multipass-blocking", "runs/010_multipass-synth"], [0.3, 0.7], 0.6, "runs/blend_009_010"
f = pl.read_parquet(f"{WORK}/test_feats.parquet")
p = None
for run, w in zip(RUNS, WEIGHTS):
    m = json.load(open(f"{run}/metrics.json"))
    pr = predict_pairs(lgb.Booster(model_file=f"{run}/model.txt"), f, m["feats"])
    p = pr.with_columns(p=w * pl.col("p")) if p is None else p.with_columns(p=pl.col("p") + w * pr["p"])
s1 = pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1).select(s1="entity_id")
m = assign(p, 0.0).filter(pl.col("p") >= T)
os.makedirs(f"{OUT}/output", exist_ok=True)
write(s1, m, "matched_entity_ids", f"{OUT}/output/matching_results.tsv")
write(s1, p, "candidate_entity_ids", f"{OUT}/output/candidate_pairs.tsv")
json.dump({"runs": RUNS, "weights": WEIGHTS, "threshold": T, "n_matches": len(m),
           "val_real_testdensity": {"normal": 0.9764, "dense": 0.9658}}, open(f"{OUT}/metrics.json", "w"), indent=2)
print(f"blend written: {len(m)} matches, {len(p) / len(s1):.2f} candidates per S1")
