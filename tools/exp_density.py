"""Does training for test-like decoy density help? Compare on normal and dense validation:
  base    : train folds as-is (= run 006 setup)
  drop    : train folds with 50% of true records removed (test-like density)
  w2 / w3 : all data, pairs of decoy records (records matching no S1) weighted x2 / x3"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, PARAMS, TRAIN_FOLDS, VALID_FOLD, WORK, assign, f05  # noqa: E402

f = pl.read_parquet(f"{WORK}/train_feats.parquet").with_columns(fold=FOLD)
feats = [c for c in f.columns if c not in ("q", "s1", "label", "fold")]
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
decoy_q = f.select("q").unique().join(gt.select("q"), on="q", how="anti").with_columns(decoy=pl.lit(True))
f = f.join(decoy_q, on="q", how="left").with_columns(pl.col("decoy").fill_null(False))
s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
gt_v = gt.filter(FOLD == VALID_FOLD)
drop_v = gt_v.sample(fraction=0.5, seed=0).select("q")
drop_t = gt.filter(FOLD != VALID_FOLD).sample(fraction=0.5, seed=1).select("q")
tr_all = f.filter(pl.col("fold").is_in(TRAIN_FOLDS))
va = f.filter(pl.col("fold") == VALID_FOLD).sample(fraction=0.25, seed=0)
X = lambda d: d.select(feats).to_numpy()


def evaluate(model, name):
    p = np.concatenate([model.predict(X(f[lo:lo + 5_000_000])) for lo in range(0, len(f), 5_000_000)])
    pairs = f.select("q", "s1").with_columns(p=p)
    out = []
    for P, G in [(pairs, gt_v), (pairs.join(drop_v, on="q", how="anti"), gt_v.join(drop_v, on="q", how="anti"))]:
        best = assign(P, 0.0).filter(FOLD == VALID_FOLD)
        curve = {round(t, 2): f05(best.filter(pl.col("p") >= t), G, s1_ids) for t in np.arange(0.4, 0.96, 0.05)}
        t = max(curve, key=curve.get)
        out.append(f"{curve[t]:.4f}@{t}")
    print(f"{name:5s} normal {out[0]}  dense {out[1]}", flush=True)


for name in ("base", "drop", "w2", "w3"):
    tr = tr_all.join(drop_t, on="q", how="anti") if name == "drop" else tr_all
    w = None if name in ("base", "drop") else np.where(tr["decoy"].to_numpy(), float(name[1]), 1.0)
    t0 = time.time()
    model = lgb.train(PARAMS, lgb.Dataset(X(tr), tr["label"].to_numpy(), weight=w), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X(va), va["label"].to_numpy())],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
    print(f"{name}: {len(tr)} pairs, {model.best_iteration} trees, {time.time() - t0:.0f}s", flush=True)
    evaluate(model, name)
