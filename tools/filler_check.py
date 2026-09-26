"""Are the 'filler' words removed in run 002 really noise, or markers of decoy records?"""
import json

import polars as pl

W = "work"
rec = pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "business_name"]).filter(pl.col("src") != 1)
gt = pl.read_parquet(f"{W}/train_gt.parquet")
rec = rec.join(gt.select(entity_id="q", s1="s1"), on="entity_id", how="left").with_columns(decoy=pl.col("s1").is_null())
base = rec.group_by("country").agg(base=pl.col("decoy").mean())
print(base.rows())
noise = json.load(open(f"{W}/train_noise.json"))
low = pl.col("business_name").str.to_lowercase()
for country, d in noise.items():
    r = rec.filter(pl.col("country") == country)
    rows = []
    for tok in d["name"]:
        m = r.filter(low.str.contains(rf"\b{tok}\b"))
        if len(m) >= 500:
            rows.append((tok, len(m), round(m["decoy"].mean(), 3)))
    rows.sort(key=lambda x: -x[2])
    print(country, "token, #records, decoy share (base ~0.26):")
    for x in rows:
        print("   ", x)
