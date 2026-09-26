"""Is test distributed like train? Records per S1, implied decoy share, and per-S1 context features.

Implied decoy share (label-free): among records whose best candidate and the record both have a house
number, mismatch rate r = d*0.91 + (1-d)*0.19 on train (decoys vs true), so d = (r - 0.19) / 0.72."""
import polars as pl

W = "work"
for split in ("train", "test"):
    rec = pl.read_parquet(f"{W}/{split}.parquet", columns=["entity_id", "src", "country", "ad"]).with_columns(
        num1=pl.col("ad").str.extract(r"\b(\d+)\b"), empty_ad=pl.col("ad") == "")
    s = rec.filter(pl.col("src") == 1).select(s1="entity_id", num_s="num1")
    top = pl.read_parquet(f"{W}/{split}_cands_full.parquet", columns=["q", "s1", "rank", "score"]).filter(pl.col("rank") == 0)
    q = (rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", num_q="num1", empty_ad="empty_ad")
         .join(top.select("q", "s1", "score"), on="q", how="left").join(s, on="s1", how="left")
         .with_columns(mm=(pl.col("num_q") != pl.col("num_s"))))
    n1 = rec.filter(pl.col("src") == 1).group_by("country").len("n_s1")
    feats = pl.read_parquet(f"{W}/{split}_feats.parquet", columns=["q", "s1", "s1_n", "s1_n0", "g_same_num_frac", "top1", "margin"])
    ctx = feats.join(q.select("q", "country"), on="q").group_by("country").agg(
        s1_n=pl.col("s1_n").mean(), s1_n0=pl.col("s1_n0").mean(), g_same_num_frac=pl.col("g_same_num_frac").mean(),
        top1=pl.col("top1").mean(), margin=pl.col("margin").mean())
    out = (q.group_by("country").agg(n_rec=pl.len(), mismatch=pl.col("mm").mean(), empty_addr=pl.col("empty_ad").mean(),
                                     best_score=pl.col("score").mean())
           .join(n1, on="country").with_columns(rec_per_s1=pl.col("n_rec") / pl.col("n_s1"),
                                                implied_decoy=(pl.col("mismatch") - 0.19) / 0.72)
           .join(ctx, on="country").sort("country"))
    print(split)
    print(out.select("country", "n_s1", "rec_per_s1", "mismatch", "implied_decoy", "empty_addr", "best_score",
                     "s1_n", "s1_n0", "g_same_num_frac", "top1", "margin"))
gt = pl.read_parquet(f"{W}/train_gt.parquet")
print("train truth: decoy share", 1 - gt["q"].n_unique() / pl.read_parquet(f"{W}/train.parquet", columns=["src"]).filter(pl.col("src") != 1).height)
