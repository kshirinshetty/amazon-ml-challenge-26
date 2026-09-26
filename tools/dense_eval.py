"""Evaluate a run's model on a test-like validation: drop a share of true S2/S3 records of validation S1s so
the decoy share rises from ~26% (train) to ~42% (test). Prints F0.5 by threshold, normal vs dense."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD, WORK, assign, expected_f, f05, predict_pairs  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/006_ambiguity-bigger-model"
DROP = 0.5  # share of true records removed: 0.26 / (0.26 + 0.74 * 0.5) ~ 0.41 decoy share
m = json.load(open(f"{RUN}/metrics.json"))
f = pl.read_parquet(f"{WORK}/train_feats.parquet")
pairs = predict_pairs(lgb.Booster(model_file=f"{RUN}/model.txt"), f, m["feats"])
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
gt_v = gt.filter(FOLD == VALID_FOLD)
dropped = gt_v.sample(fraction=DROP, seed=0).select("q")
for name, P, G in [("normal", pairs, gt_v),
                   ("dense ", pairs.join(dropped, on="q", how="anti"), gt_v.join(dropped, on="q", how="anti"))]:
    best = assign(P, 0.0).filter(FOLD == VALID_FOLD)
    curve = {round(t, 2): f05(best.filter(pl.col("p") >= t), G, s1_ids) for t in np.arange(0.5, 0.96, 0.05)}
    ef = f05(expected_f(best), G, s1_ids)
    t_best = max(curve, key=curve.get)
    print(f"{name}: " + " ".join(f"t{t}={v:.4f}" for t, v in curve.items()) + f" | expected-F {ef:.4f} | best t={t_best}")
