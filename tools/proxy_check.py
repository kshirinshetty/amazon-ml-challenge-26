"""Does a label-free 'house-number mismatch rate' per extra name word track the true decoy share?"""
import json
import sys

import polars as pl

sys.path.insert(0, "/root/src")
from normalize import name_tokens  # noqa: E402

W = "work"
maps = json.load(open(f"{W}/maps.json"))["name"]
rec = pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "business_name", "ad"])
rec = rec.with_columns(full=pl.col("business_name").map_elements(name_tokens, return_dtype=pl.String)
                       .str.split(" ").list.eval(pl.element().replace(maps)).list.unique(),
                       num=pl.col("ad").str.extract(r"\b(\d+)\b"))
gt = pl.read_parquet(f"{W}/train_gt.parquet")
top = pl.read_parquet(f"{W}/train_cands_full.parquet", columns=["q", "s1", "rank"]).filter(pl.col("rank") == 0)
s1 = rec.filter(pl.col("src") == 1).select(top_s1="entity_id", full_s="full", num_s="num")
q = (rec.filter(pl.col("src") != 1).join(top.select(entity_id="q", top_s1="s1"), on="entity_id")
     .join(s1, on="top_s1").join(gt.select(entity_id="q", s1="s1"), on="entity_id", how="left")
     .with_columns(extra=pl.col("full").list.set_difference("full_s"), decoy=pl.col("s1").is_null(),
                   mismatch=(pl.col("num") != pl.col("num_s"))))  # null when either side lacks a number
t = (q.select("country", "extra", "decoy", "mismatch").explode("extra").filter(pl.col("extra").is_not_null())
     .group_by("country", "extra").agg(n=pl.len(), decoy_share=pl.col("decoy").mean(), proxy=pl.col("mismatch").mean())
     .filter(pl.col("n") >= 500))
print("tokens:", len(t), " pearson(proxy, decoy_share) per country:",
      t.group_by("country").agg(pl.corr("proxy", "decoy_share")).rows())
print(t.sort("n", descending=True).head(40).select("country", "extra", "n", "decoy_share", "proxy").rows())
print("base mismatch rate: decoys", q.filter("decoy")["mismatch"].mean(), " true", q.filter(~pl.col("decoy"))["mismatch"].mean())
