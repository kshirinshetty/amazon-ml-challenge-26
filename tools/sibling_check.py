"""Can empty-address records borrow their S1 from same-name sibling records the model matched confidently?"""
import json
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD, WORK, assign, predict_pairs  # noqa: E402

RUN = "runs/013_keys-safecap-nosynth"
m = json.load(open(f"{RUN}/metrics.json"))
f = pl.read_parquet(f"{WORK}/train_feats.parquet")
best = assign(predict_pairs(lgb.Booster(model_file=f"{RUN}/model.txt"), f, m["feats"]), 0.0)
rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src", "country", "nm", "ad"]).filter(pl.col("src") != 1)
gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
r = (rec.select(q="entity_id", country="country", nm="nm", empty=pl.col("ad") == "")
     .join(best.select("q", s_best="s1", p="p"), on="q", how="left").join(gt.select("q", s_true="s1"), on="q", how="left"))
conf = r.filter(pl.col("p") >= 0.9).select("country", "nm", sib="q", sib_s1="s_best")
votes = (r.filter(pl.col("empty")).select("q", "country", "nm", "s_true", "s_best", "p")
         .join(conf, on=["country", "nm"]).filter(pl.col("sib") != pl.col("q"))
         .group_by("q", "sib_s1").agg(n=pl.len()).sort("n", descending=True)
         .group_by("q").agg(top_s1=pl.col("sib_s1").first(), top_n=pl.col("n").first(), n_s1=pl.len(), total=pl.col("n").sum()))
e = r.filter(pl.col("empty")).join(votes, on="q", how="left").with_columns(
    predicted=(pl.col("p") >= m["threshold"]) & (pl.col("s_best") == pl.col("s_true")),
    vote_right=pl.col("top_s1") == pl.col("s_true"), unanimous=pl.col("n_s1") == 1)
for name, df in (("true empty-address records", e.filter(pl.col("s_true").is_not_null())),
                 ("decoy empty-address records", e.filter(pl.col("s_true").is_null()))):
    print(f"\n{name}: {len(df)}")
    print("  have >=1 confident same-name sibling:", df["top_s1"].is_not_null().mean())
    if name.startswith("true"):
        miss = df.filter(~pl.col("predicted"))
        print("  currently missed:", len(miss), "| of those, sibling vote points to the right S1:",
              miss["vote_right"].fill_null(False).sum(), "| unanimous & right:",
              (miss["vote_right"].fill_null(False) & miss["unanimous"].fill_null(False)).sum())
        print("  vote accuracy when a vote exists:", df.filter(pl.col("top_s1").is_not_null())["vote_right"].mean())
    else:
        print("  would be linked by a unanimous vote:", df["unanimous"].fill_null(False).mean())
