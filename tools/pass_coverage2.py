"""Candidate extra blocking keys: how many of the pairs still missed (after run 009-style key passes) would each
recover, and at what fan-out? Usage: --script tools/pass_coverage2.py (after block+prune for train)."""
import polars as pl

W = "work"
gt = pl.read_parquet(f"{W}/train_gt.parquet")
rec = (pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "nm", "ad"])
       .filter(~pl.col("entity_id").str.contains("-SYN")))
cands = pl.read_parquet(f"{W}/train_cands.parquet", columns=["q", "s1"])
skel = pl.col("nm").str.replace_all(r"(\B)[aeiou]", "")  # drop vowels except word-initial
keys = {
    "num_street_loose": pl.concat_str([pl.col("ad").str.extract(r"\b(\d+)\b"),
                                       pl.col("ad").str.extract(r"\b\d+\b[^a-z]*?(?:\b[a-z]{1,2}\b\s+)*\b([a-z]{3,})")], separator=" "),
    "skeleton_sorted": skel.str.split(" ").list.sort().list.join(" "),
    "first2_words": pl.col("nm").str.split(" ").list.head(2).list.join(" "),
    "first_word_city": pl.concat_str([pl.col("nm").str.split(" ").list.first(), pl.col("ad").str.split(" ").list.last()], separator=" "),
    "name_nospace": pl.col("nm").str.replace_all(" ", ""),
}
rec = rec.with_columns(**keys).with_columns([pl.when(pl.col(k) != "").then(pl.col(k)) for k in keys])
s1, q = rec.filter(pl.col("src") == 1), rec.filter(pl.col("src") != 1)
miss = gt.join(cands, on=["q", "s1"], how="anti")
print(f"true pairs {len(gt)}, still missed {len(miss)} ({len(miss) / len(gt):.2%})")
m = (miss.join(q.select(pl.col("entity_id").alias("q"), *[pl.col(k).alias(f"q_{k}") for k in keys]), on="q")
     .join(s1.select(pl.col("entity_id").alias("s1"), *[pl.col(k).alias(f"s_{k}") for k in keys]), on="s1"))
covered = pl.lit(False)
for k in keys:
    hit = (pl.col(f"q_{k}") == pl.col(f"s_{k}")).fill_null(False)
    covered = covered | hit
    fan = s1.filter(pl.col(k).is_not_null()).group_by("country", k).len("n")
    cost = q.filter(pl.col(k).is_not_null()).join(fan, on=["country", k])["n"]
    small = (cost <= 50).mean()
    print(f"  {k:17s}: recovers {m.select(hit.sum()).item():>7}; S1s per hit mean {cost.mean():.1f} p99 {cost.quantile(0.99)}, "
          f"hits with <=50 S1s {small:.0%}")
print(f"  any: {m.select(covered.sum()).item()} of {len(miss)}")
