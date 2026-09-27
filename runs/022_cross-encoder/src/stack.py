"""Stage 2: re-score candidate pairs using stage-1 probabilities of the *other* pairs around them.

A decoy looks fine in isolation; next to its S1's confident true records it stands out (different house
number, extra word), and a record torn between two S1s shows it in its runner-up probability. Stage 1 is
match.py's model; its train-set probabilities are out-of-fold so stage 2 learns from honest scores.

Usage: uv run python src/stack.py fit RUN_DIR       # needs RUN_DIR/model.txt + metrics.json from match.py fit
       uv run python src/stack.py predict RUN_DIR   # -> RUN_DIR/output/*.tsv (replaces stage-1 decisions)
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from match import (FOLD, PARAMS, PATIENCE, ROUNDS, TRAIN_FOLDS, VALID_FOLD, WORK, assign, f05, predict_pairs,
                   sim_dense, write)
from features import sim

DENSE_DROP = 0.5  # validation mimicking test decoy density: half of the true records removed


def context(pairs, rec):
    """pairs: q, s1, p. Adds stage-2 context features."""
    q_ctx = pairs.group_by("q").agg(q_pmax=pl.col("p").max(), q_p2=pl.col("p").top_k(2).min(), q_n=pl.len(),
                                    q_nhi=(pl.col("p") > 0.5).sum())
    d = (pairs.join(q_ctx, on="q")
         .with_columns(q_gap=pl.col("q_pmax") - pl.col("p"),
                       q_runner=pl.when(pl.col("p") == pl.col("q_pmax")).then(pl.col("q_p2")).otherwise(pl.col("q_pmax")),
                       s_sum=pl.col("p").sum().over("s1"), s_nhi=(pl.col("p") > 0.5).sum().over("s1"),
                       s_rank=pl.col("p").rank("ordinal", descending=True).over("s1"), s_n=pl.len().over("s1")))
    # the S1's most confident *other* record: agreement with it separates true siblings from decoys
    top2 = (pairs.sort("p", descending=True).group_by("s1", maintain_order=True).head(2)
            .with_columns(r=pl.int_range(pl.len()).over("s1")))
    first, second = top2.filter(pl.col("r") == 0), top2.filter(pl.col("r") == 1)
    ref = (d.select("q", "s1").join(first.select("s1", q1="q", p1="p"), on="s1", how="left")
           .join(second.select("s1", q2="q", p2="p"), on="s1", how="left")
           .with_columns(sib=pl.when(pl.col("q1") == pl.col("q")).then(pl.col("q2")).otherwise(pl.col("q1")),
                         sib_p=pl.when(pl.col("q1") == pl.col("q")).then(pl.col("p2")).otherwise(pl.col("p1")))
           .select("q", "s1", "sib", "sib_p"))
    r = rec.select(q="entity_id", nm="nm", ad="ad", num1=pl.col("ad").str.extract(r"\b(\d+)\b"))
    ref = (ref.join(r, on="q", how="left")
           .join(r.rename({"q": "sib", "nm": "sib_nm", "ad": "sib_ad", "num1": "sib_num1"}), on="sib", how="left")
           .with_columns(pl.col("nm", "ad", "sib_nm", "sib_ad").fill_null("")))
    ref = ref.with_columns(sib_n_ratio=sim(ref["nm"], ref["sib_nm"], fuzz.ratio),
                           sib_a_ratio=sim(ref["ad"], ref["sib_ad"], fuzz.ratio),
                           sib_num_eq=(pl.col("num1") == pl.col("sib_num1")).cast(pl.Int8))
    return d.join(ref.select("q", "s1", "sib_p", "sib_n_ratio", "sib_a_ratio", "sib_num_eq"), on=["q", "s1"], how="left")


def oof_probs(f, feats):
    """Out-of-fold stage-1 probabilities for training folds; fold 0 (validation) from a folds-1..4 model."""
    va = f.filter(pl.col("fold") == VALID_FOLD).sample(fraction=0.25, seed=0)
    out = []
    for k in TRAIN_FOLDS + [VALID_FOLD]:
        tr = f.filter(pl.col("fold").is_in([x for x in TRAIN_FOLDS if x != k]))
        t0 = time.time()
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(feats).to_numpy(), tr["label"].to_numpy()), num_boost_round=ROUNDS,
                      valid_sets=[lgb.Dataset(va.select(feats).to_numpy(), va["label"].to_numpy())],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        out.append(predict_pairs(m, f.filter(pl.col("fold") == k), feats))
        print(f"  oof fold {k}: {m.best_iteration} trees, {time.time() - t0:.0f}s", flush=True)
    return pl.concat(out)


def fit(run_dir):
    m1 = json.load(open(f"{run_dir}/metrics.json"))
    feats = m1["feats"]
    f = pl.read_parquet(f"{WORK}/train_feats.parquet").with_columns(fold=FOLD)
    rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "nm", "ad"])
    d = f.join(context(oof_probs(f, feats), rec), on=["q", "s1"])
    feats2 = feats + [c for c in d.columns if c not in f.columns]
    tr, va = d.filter(pl.col("fold").is_in(TRAIN_FOLDS)), d.filter(pl.col("fold") == VALID_FOLD)
    t0 = time.time()
    model = lgb.train(PARAMS, lgb.Dataset(tr.select(feats2).to_numpy(), tr["label"].to_numpy()), num_boost_round=ROUNDS,
                      valid_sets=[lgb.Dataset(va.sample(fraction=0.25, seed=0).select(feats2).to_numpy(),
                                              va.sample(fraction=0.25, seed=0)["label"].to_numpy())],
                      callbacks=[lgb.early_stopping(PATIENCE), lgb.log_evaluation(100)])
    print(f"stage 2: {model.best_iteration} trees, {time.time() - t0:.0f}s")
    pairs = d.select("q", "s1", p1="p").with_columns(p=np.concatenate(
        [model.predict(d[lo:lo + 5_000_000].select(feats2).to_numpy()) for lo in range(0, len(d), 5_000_000)]))
    res = sim_dense(pairs.select("q", "s1", "p"), DENSE_DROP)
    res1 = sim_dense(pairs.select("q", "s1", p="p1"), DENSE_DROP)
    for name, r in (("stage1", res1), ("stage2", res)):
        print(f"{name}: normal {r['normal']:.4f}@{r['t_normal']}  dense {r['dense']:.4f}@{r['t_dense']}")
    model.save_model(f"{run_dir}/model_stage2.txt")
    m1.update(stage2={"feats": feats2, "threshold": res["t_dense"], "val_f05": res["normal"], "val_f05_dense": res["dense"],
                      "stage1_val_f05_dense": res1["dense"], "best_iter": model.best_iteration,
                      "feature_gain": dict(sorted(zip(feats2, model.feature_importance("gain").round().tolist()),
                                                  key=lambda kv: -kv[1])[:25])})
    json.dump(m1, open(f"{run_dir}/metrics.json", "w"), indent=2)


def predict(run_dir):
    m = json.load(open(f"{run_dir}/metrics.json"))
    f = pl.read_parquet(f"{WORK}/test_feats.parquet")
    rec = pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "nm", "ad"])
    p1 = predict_pairs(lgb.Booster(model_file=f"{run_dir}/model.txt"), f, m["feats"])
    d = f.join(context(p1, rec), on=["q", "s1"])
    s2 = lgb.Booster(model_file=f"{run_dir}/model_stage2.txt")
    pairs = d.select("q", "s1").with_columns(p=np.concatenate(
        [s2.predict(d[lo:lo + 5_000_000].select(m["stage2"]["feats"]).to_numpy()) for lo in range(0, len(d), 5_000_000)]))
    s1 = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
          .select(s1="entity_id"))
    matches = assign(pairs, 0.0).filter(pl.col("p") >= m["stage2"]["threshold"])
    os.makedirs(f"{run_dir}/output", exist_ok=True)
    write(s1, matches, "matched_entity_ids", f"{run_dir}/output/matching_results.tsv")
    write(s1, pairs, "candidate_entity_ids", f"{run_dir}/output/candidate_pairs.tsv")
    print(f"stage-2 test: {len(matches)} matches, {len(pairs) / len(s1):.2f} candidates per S1")


if __name__ == "__main__":
    {"fit": fit, "predict": predict}[sys.argv[1]](sys.argv[2])
