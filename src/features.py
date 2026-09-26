"""Pairwise features for blocked candidates -> work/{split}_feats.parquet (+ label on train).

Usage: uv run python src/features.py train|test
"""
import sys

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

WORK = "work"
CHUNK = 2_000_000
NUM = r"\d+"


def sim(a, b, scorer):
    return cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def string_feats(c):
    nq, ns, aq, as_ = c["nm_q"], c["nm_s"], c["ad_q"], c["ad_s"]
    kq, ks = nq.str.replace_all(" ", ""), ns.str.replace_all(" ", "")
    return c.with_columns(
        n_ratio=sim(nq, ns, fuzz.ratio),
        n_tset=sim(nq, ns, fuzz.token_set_ratio),
        n_tsort=sim(nq, ns, fuzz.token_sort_ratio),
        n_partial=sim(nq, ns, fuzz.partial_ratio),
        n_jw=sim(kq, ks, distance.JaroWinkler.normalized_similarity),
        a_ratio=sim(aq, as_, fuzz.ratio),
        a_tset=sim(aq, as_, fuzz.token_set_ratio),
        a_partial=sim(aq, as_, fuzz.partial_ratio),
    ).with_columns(
        num_q=pl.col("ad_q").str.extract_all(NUM).list.unique(),
        num_s=pl.col("ad_s").str.extract_all(NUM).list.unique(),
    ).with_columns(
        num_inter=pl.col("num_q").list.set_intersection("num_s").list.len(),
        num_union=pl.col("num_q").list.set_union("num_s").list.len(),
        num_first_eq=(pl.col("num_q").list.first() == pl.col("num_s").list.first()).cast(pl.Int8),
        len_nq=pl.col("nm_q").str.len_chars(), len_ns=pl.col("nm_s").str.len_chars(),
        len_aq=pl.col("ad_q").str.len_chars(), len_as=pl.col("ad_s").str.len_chars(),
    ).with_columns(
        num_jac=pl.col("num_inter") / pl.col("num_union"),
    ).drop("nm_q", "nm_s", "ad_q", "ad_s", "num_q", "num_s")


def build(split):
    rec = pl.read_parquet(f"{WORK}/{split}.parquet").select(
        "entity_id", "src", "nm", "ad",
        translit=pl.col("business_name").str.contains(r"[^\x00-\x7F]").cast(pl.Int8),
    )
    c = pl.read_parquet(f"{WORK}/{split}_cands.parquet").with_columns(
        # query-level context: how clear-cut is this record's best S1?
        top1=pl.col("score").max().over("q"),
        top2=pl.col("score").top_k(2).min().over("q"),
        # S1-level context: how many records point at this S1, and where does this one rank?
        s1_n=pl.len().over("s1"),
        s1_n0=(pl.col("rank") == 0).sum().over("s1"),
        s1_rank=pl.col("score").rank("ordinal", descending=True).over("s1"),
    ).with_columns(gap=pl.col("top1") - pl.col("score"), margin=pl.col("top1") - pl.col("top2"))
    q = rec.filter(pl.col("src") != 1).rename({"entity_id": "q", "nm": "nm_q", "ad": "ad_q"})
    s = rec.filter(pl.col("src") == 1).select(s1="entity_id", nm_s="nm", ad_s="ad")
    out = []
    for lo in range(0, len(c), CHUNK):
        chunk = c[lo:lo + CHUNK].join(q, on="q", how="left").join(s, on="s1", how="left")
        out.append(string_feats(chunk))
        print(f"  {lo + len(chunk)}/{len(c)}", flush=True)
    f = pl.concat(out)
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet").with_columns(label=pl.lit(1, pl.Int8))
        f = f.join(gt, on=["q", "s1"], how="left").with_columns(pl.col("label").fill_null(0))
    f.write_parquet(f"{WORK}/{split}_feats.parquet")
    print(f.describe())


if __name__ == "__main__":
    build(sys.argv[1])
