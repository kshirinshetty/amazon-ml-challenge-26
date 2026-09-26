"""Where do the F0.5 points go? For run 006's model on validation (normal and dense), fix one error type at a
time perfectly and measure the gain; break each down by record type."""
import json
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD, WORK, assign, f05, predict_pairs  # noqa: E402

RUN = "runs/006_ambiguity-bigger-model"
m = json.load(open(f"{RUN}/metrics.json"))
T = m["threshold"]
f = pl.read_parquet(f"{WORK}/train_feats.parquet").filter(~pl.col("q").str.contains("-SYN"))
pairs = predict_pairs(lgb.Booster(model_file=f"{RUN}/model.txt"), f, m["feats"])
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src", "business_name", "ad", "nm"]).filter(
    ~pl.col("entity_id").str.contains("-SYN"))
s1_ids = rec.filter(pl.col("src") == 1).select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"]
gt_v = gt.filter(FOLD == VALID_FOLD)
kind = rec.filter(pl.col("src") != 1).select(
    q="entity_id",
    kind=pl.when(pl.col("ad") == "").then(pl.lit("empty_addr"))
    .when(pl.col("business_name").str.contains(r"[^\x00-\x7F]") & ~pl.col("business_name").str.contains(r"[À-ſ]"))
    .then(pl.lit("indic_script")).otherwise(pl.lit("normal")))
dropped = gt_v.sample(fraction=0.5, seed=0).select("q")
for name, P, G in (("normal", pairs, gt_v), ("dense", pairs.join(dropped, on="q", how="anti"), gt_v.join(dropped, on="q", how="anti"))):
    best = assign(P, 0.0).filter(FOLD == VALID_FOLD)
    pred = best.filter(pl.col("p") >= T).select("q", "s1")
    base = f05(pred, G, s1_ids)
    tp = pred.join(G, on=["q", "s1"])
    fp = pred.join(G, on=["q", "s1"], how="anti")
    fn = G.join(pred, on=["q", "s1"], how="anti").select("q", "s1")
    in_cands = P.select("q", "s1").unique()
    fn_retrieved = fn.join(in_cands, on=["q", "s1"])
    fn_missed = fn.join(in_cands, on=["q", "s1"], how="anti")
    print(f"\n[{name}] F0.5 {base:.4f}  |  pairs: TP {len(tp)}  FP {len(fp)}  FN(retrieved) {len(fn_retrieved)}  FN(not retrieved) {len(fn_missed)}")
    fp_other = fp.join(G.select("q"), on="q")  # record belongs to another S1
    fixes = {
        "remove all false merges": pred.join(fp, on=["q", "s1"], how="anti"),
        "  ...only decoy records": pred.join(fp.join(G.select("q"), on="q", how="anti"), on=["q", "s1"], how="anti"),
        "  ...only records of another S1": pred.join(fp_other, on=["q", "s1"], how="anti"),
        "add retrieved-but-rejected true pairs": pl.concat([pred.join(fp_other, on=["q", "s1"], how="anti"), fn_retrieved]),
        "add never-retrieved true pairs": pl.concat([pred.join(fp_other, on=["q", "s1"], how="anti"), fn_missed]),
    }
    for k, v in fixes.items():
        print(f"  {k:40s} -> {f05(v.unique(), G, s1_ids):.4f}  (+{f05(v.unique(), G, s1_ids) - base:.4f})")
    for lbl, df in (("false merges", fp), ("FN retrieved", fn_retrieved), ("FN not retrieved", fn_missed)):
        print(f"  {lbl:17s} by record type:", df.join(kind, on="q").group_by("kind").len().sort("len", descending=True).rows())
