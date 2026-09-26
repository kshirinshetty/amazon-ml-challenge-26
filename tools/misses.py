"""Why do true pairs never reach the model? Break blocking misses down by cause and record type."""
import polars as pl

pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(40)
W = "work"
gt = pl.read_parquet(f"{W}/train_gt.parquet")
full = pl.read_parquet(f"{W}/train_cands_full.parquet", columns=["q", "s1", "rank", "score"])
pruned = pl.read_parquet(f"{W}/train_cands.parquet", columns=["q", "s1"]).with_columns(kept=pl.lit(True))
rec = pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "nm", "ad"])
has_c = full.group_by("q").agg(top1=pl.col("score").max())
g = (gt.join(full.select("q", "s1", "rank"), on=["q", "s1"], how="left").join(pruned, on=["q", "s1"], how="left")
     .join(has_c, on="q", how="left")
     .with_columns(cause=pl.when(pl.col("kept")).then(pl.lit("found"))
                   .when(pl.col("rank").is_not_null()).then(pl.lit("pruned"))
                   .when(pl.col("top1").is_not_null()).then(pl.lit("not_in_top3"))
                   .otherwise(pl.lit("no_candidates"))))
q = rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", q_name="business_name", q_addr="business_address",
                                         q_nm="nm", q_ad="ad")
s = rec.filter(pl.col("src") == 1).select(s1="entity_id", s_name="business_name", s_addr="business_address", s_nm="nm", s_ad="ad")
g = g.join(q, on="q").join(s, on="s1").with_columns(
    empty_addr=pl.col("q_ad") == "", translit=pl.col("q_name").str.contains(r"[^\x00-\x7F]"),
    empty_name=pl.col("q_nm") == "", name_exact=pl.col("q_nm") == pl.col("s_nm"))
print(g.group_by("cause").len().with_columns(share=pl.col("len") / len(g)).sort("len", descending=True))
miss = g.filter(pl.col("cause") != "found")
for col in ["country", "empty_addr", "translit", "empty_name", "name_exact"]:
    print(g.group_by(col).agg(n=pl.len(), miss_rate=(pl.col("cause") != "found").mean(),
                              share_of_misses=(pl.col("cause") != "found").sum() / len(miss)).sort(col))
print(miss.group_by("cause", "empty_addr", "translit").len().sort("len", descending=True).head(12))
for cause in ["not_in_top3", "no_candidates"]:
    print(f"\n=== {cause}")
    for r in miss.filter(pl.col("cause") == cause).sample(12, seed=5).iter_rows(named=True):
        print(f"  Q  {r['q_name']} | {r['q_addr']}   [{r['q_nm']} | {r['q_ad']}]")
        print(f"  S1 {r['s_name']} | {r['s_addr']}   [{r['s_nm']} | {r['s_ad']}]")
