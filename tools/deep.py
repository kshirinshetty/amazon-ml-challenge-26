"""Deep look: country structure (train vs test) and what stage 2 added on test."""
import polars as pl

pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(30)
W = "work"
for split in ("train", "test"):
    f = pl.read_parquet(f"{W}/{split}_feats.parquet", columns=["q", "s1", "rank", "score", "amb_s_name", "amb_s_addr",
                                                               "s_num_in_q", "n_xq", "len_aq"])
    rec = pl.read_parquet(f"{W}/{split}.parquet", columns=["entity_id", "src", "country"])
    f = f.join(rec.select(q="entity_id", country="country"), on="q")
    top = f.filter(pl.col("rank") == 0)
    print(split, "per country (best candidate of each record):")
    print(top.group_by("country").agg(
        records=pl.len(), s1_addr_shared=(pl.col("amb_s_addr") > 1).mean(), s1_name_shared=(pl.col("amb_s_name") > 1).mean(),
        num_in_q=pl.col("s_num_in_q").mean(), extra_words=pl.col("n_xq").mean(), empty_addr=(pl.col("len_aq") == 0).mean(),
        best_score=pl.col("score").mean(), cands_per_rec=pl.lit(0)).sort("country"))
    s1 = rec.filter(pl.col("src") == 1)
    ad = pl.read_parquet(f"{W}/{split}.parquet", columns=["entity_id", "src", "country", "ad"]).filter(pl.col("src") == 1)
    print(split, "S1s sharing an exact address with another S1:",
          ad.group_by("country").agg(shared=pl.col("ad").is_duplicated().mean()).sort("country").rows())

# what did stage 2 add on test?
rec = pl.read_parquet(f"{W}/test.parquet", columns=["entity_id", "src", "country", "business_name", "business_address"])
load = lambda r: (pl.read_csv(f"runs/{r}/output/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
                  .select(s1="source1_entity_id", q=pl.col("matched_entity_ids").str.split(",")).explode("q")
                  .filter(pl.col("q").is_not_null() & (pl.col("q") != "")))
added = load("007_stage2-stacking").join(load("006_ambiguity-bigger-model"), on=["s1", "q"], how="anti")
txt = rec.select("entity_id", "country", "business_name", "business_address")
added = (added.join(txt.rename({"entity_id": "s1", "business_name": "s_name", "business_address": "s_addr"}), on="s1")
         .join(txt.drop("country").rename({"entity_id": "q", "business_name": "q_name", "business_address": "q_addr"}), on="q"))
for country in ("France", "US", "India"):
    print(f"\n=== added by stage 2, {country}")
    for r in added.filter(pl.col("country") == country).sample(12, seed=2).iter_rows(named=True):
        print(f"  S1 {r['s_name']} | {r['s_addr']}\n  Q  {r['q_name']} | {r['q_addr']}")
