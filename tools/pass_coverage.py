"""Which exact-key blocking passes would recover the true pairs TF-IDF retrieval misses, and at what cost?"""
import polars as pl

W = "work"
gt = pl.read_parquet(f"{W}/train_gt.parquet")
rec = (pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "nm", "ad"])
       .filter(~pl.col("entity_id").str.contains("-SYN")))
cands = pl.read_parquet(f"{W}/train_cands.parquet", columns=["q", "s1"])
num = pl.col("ad").str.extract(r"\b(\d+)\b")
street = pl.col("ad").str.extract(r"\b\d+\s+([a-z]{3,})")  # first word after the house number
keys = {
    "name": pl.col("nm"),
    "name_sorted": pl.col("nm").str.split(" ").list.sort().list.join(" "),
    "num_street": pl.when(num.is_not_null() & street.is_not_null()).then(pl.concat_str([num, street], separator=" ")),
    "name_nospace": pl.col("nm").str.replace_all(" ", ""),
}
rec = rec.with_columns(**{k: v for k, v in keys.items()}).with_columns(
    [pl.when(pl.col(k) != "").then(pl.col(k)) for k in keys])
s1 = rec.filter(pl.col("src") == 1)
q = rec.filter(pl.col("src") != 1)
miss = gt.join(cands, on=["q", "s1"], how="anti")
print(f"true pairs {len(gt)}, missed by current blocking {len(miss)} ({len(miss) / len(gt):.2%})")
m = (miss.join(q.select(pl.col("entity_id").alias("q"), *[pl.col(k).alias(f"q_{k}") for k in keys]), on="q")
     .join(s1.select(pl.col("entity_id").alias("s1"), *[pl.col(k).alias(f"s_{k}") for k in keys]), on="s1"))
covered = pl.lit(False)
for k in keys:
    hit = pl.col(f"q_{k}") == pl.col(f"s_{k}")
    covered = covered | hit.fill_null(False)
    # cost: S1s sharing the key with a record (per record, capped view)
    fan = s1.filter(pl.col(k).is_not_null()).group_by("country", k).len("n")
    cost = q.filter(pl.col(k).is_not_null()).join(fan, on=["country", k]).select(pl.col("n"))["n"]
    print(f"  key {k:12s}: recovers {m.select(hit.fill_null(False).sum()).item():>7} missed pairs; "
          f"records with a key hit {len(cost):>9}, S1s per hit mean {cost.mean():.2f} p99 {cost.quantile(0.99)}")
print(f"  any key recovers {m.select(covered.sum()).item()} of {len(miss)} missed pairs")
