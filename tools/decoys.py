"""What are decoys (S2/S3 records matching no S1) and how do they differ from true matches?"""
import polars as pl

pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
W = "work"
rec = pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "nm", "ad"])
gt = pl.read_parquet(f"{W}/train_gt.parquet")
top = pl.read_parquet(f"{W}/train_cands_full.parquet", columns=["q", "s1", "rank", "score"]).filter(pl.col("rank") == 0)
q = (rec.filter(pl.col("src") != 1).rename({"entity_id": "q"}).join(gt, on="q", how="left")
     .join(top.select("q", top_s1="s1", top_score="score"), on="q", how="left")
     .with_columns(decoy=pl.col("s1").is_null()))
dec = q.filter("decoy")
print(f"records {len(q)}, decoys {len(dec)} ({len(dec) / len(q):.1%})")
print("decoys by country/source:", dec.group_by("country", pl.col("q").str.slice(0, 2).alias("src")).len().sort("country", "src").rows())

# Do decoys come in clusters of their own (several records of one hidden entity)?
s1 = rec.filter(pl.col("src") == 1).select(top_s1="entity_id", s1_nm="nm", s1_ad="ad")
d = dec.join(s1, on="top_s1", how="left")
num = lambda c: pl.col(c).str.extract(r"\b(\d+)\b")
d = d.with_columns(same_name=pl.col("nm") == pl.col("s1_nm"), same_num=num("ad") == num("s1_ad"))
print("decoy vs its top S1: exact same normalized name:", d["same_name"].mean(), " same first number:", d["same_num"].mean())
t = q.filter(~pl.col("decoy")).join(rec.filter(pl.col("src") == 1).select(s1="entity_id", s1_nm="nm", s1_ad="ad"), on="s1")
t = t.with_columns(same_name=pl.col("nm") == pl.col("s1_nm"), same_num=num("ad") == num("s1_ad"))
print("true record vs its S1:      exact same normalized name:", t["same_name"].mean(), " same first number:", t["same_num"].mean())
g = dec.group_by("top_s1").agg(n=pl.len(), same_key=pl.struct("nm", num("ad").alias("n1")).n_unique())
print("decoys per top-S1 (how many decoys point at the same S1):", g["n"].value_counts().sort("n").head(8).rows())
print("groups of >=2 decoys on one S1 sharing (name, first number) among themselves:",
      g.filter(pl.col("n") >= 2).select((pl.col("same_key") < pl.col("n")).mean()).item())

# Concrete groups: an S1, its true records (+), and decoys whose best S1 it is (-)
ex = dec.filter(pl.col("top_score") > 0.6).select("top_s1").unique().sample(8, seed=11)["top_s1"].to_list()
show = rec.select("entity_id", "business_name", "business_address")
for s in ex:
    r = show.filter(pl.col("entity_id") == s).row(0)
    print(f"\nS1 {r[1]} | {r[2]}")
    for tag, ids in (("+", gt.filter(pl.col("s1") == s)["q"]), ("-", dec.filter(pl.col("top_s1") == s)["q"])):
        for x in show.filter(pl.col("entity_id").is_in(ids.implode())).rows():
            print(f"  {tag} {x[1]} | {x[2]}")
