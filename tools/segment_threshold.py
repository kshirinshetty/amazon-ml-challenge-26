"""Would a separate threshold for empty-address records help? And what do their missed true pairs look like?"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD, WORK, assign, f05, predict_pairs  # noqa: E402

RUN = "runs/013_keys-safecap-nosynth"
m = json.load(open(f"{RUN}/metrics.json"))
f = pl.read_parquet(f"{WORK}/train_feats.parquet")
pairs = predict_pairs(lgb.Booster(model_file=f"{RUN}/model.txt"), f, m["feats"]).join(
    f.select("q", "s1", "len_aq", "amb_s_name", "name_eq", "q_name_ties"), on=["q", "s1"])
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
gt_v = gt.filter(FOLD == VALID_FOLD)
dropped = gt_v.sample(fraction=0.5, seed=0).select("q")
T = m["threshold"]
for name, P, G in (("normal", pairs, gt_v), ("dense", pairs.join(dropped, on="q", how="anti"), gt_v.join(dropped, on="q", how="anti"))):
    best = assign(P, 0.0).filter(FOLD == VALID_FOLD)
    row = []
    for te in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = best.filter(pl.col("p") >= pl.when(pl.col("len_aq") == 0).then(te).otherwise(T))
        row.append(f"t_empty={te}: {f05(pred, G, s1_ids):.4f}")
    print(name, " | ".join(row), flush=True)
# missed empty-address true pairs on validation
best = assign(pairs, 0.0)
t = (gt_v.join(pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "ad"]).filter(pl.col("ad") == "")
               .select(q="entity_id"), on="q")
     .join(pairs.select("q", "s1", p_true="p", amb="amb_s_name", eq="name_eq"), on=["q", "s1"], how="left")
     .join(best.select("q", s_best="s1", p_best="p"), on="q", how="left"))
missed = t.filter(~((pl.col("s_best") == pl.col("s1")) & (pl.col("p_best") >= T)).fill_null(True))
print(f"\nvalidation empty-address true pairs {len(t)}, missed {len(missed)}")
print("  not retrieved:", missed["p_true"].is_null().sum())
ret = missed.filter(pl.col("p_true").is_not_null())
print("  retrieved, best S1 is another S1:", (ret["s_best"] != ret["s1"]).sum(), "| retrieved, best is right but p<t:", (ret["s_best"] == ret["s1"]).sum())
print("  retrieved: name unique among S1s (amb==1):", (ret["amb"] == 1).sum(), " exact name:", (ret["eq"] == 1).sum())
print("  p of right pair (retrieved):", ret["p_true"].describe(percentiles=[0.25, 0.5, 0.75]).rows())
