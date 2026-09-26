"""Blocking: every S2/S3 record looks up its top-K S1 records (same country) by TF-IDF cosine.

Each S2/S3 record matches at most one S1, so searching in this direction keeps the per-S1
candidate set small (~K * #S2S3 / #S1) while top-K recall stays high.
Retrieval writes work/{split}_cands_full.parquet (q, s1, rank, score, name_cos, addr_cos); the
prune step then keeps a record's rank-2/3 S1s only when they score close to its best, caps every
S1's list, and writes work/{split}_cands.parquet: the candidate set the model runs on.

Usage: uv run python src/block.py train|test            # retrieval
       uv run python src/block.py train|test --prune    # prune (on train: prints the recall/size grid)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

WORK = "work"
K = 10  # retrieval depth; prune keeps what earns its place (4.5% of true pairs sat at rank 4+ with K=3)
W_NAME = 0.6  # combined score = W_NAME * name_cos + (1 - W_NAME) * addr_cos
CHUNK = 250_000
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())  # Modal sets this from the cpu request
MAX_DF = 0.005  # drop n-grams in >0.5% of S1 docs: long posting lists dominate matmul cost
REL = 0.8  # keep rank>0 pairs with score >= REL * the record's best score (K=10: recall 0.9582 at 7.3/S1 on train)
CAP = 30  # max candidates per S1, best scores first (true matches per S1 <= 11)
RMAX = 10  # keep ranks < RMAX
# Exact-key passes: TF-IDF drops n-grams common across S1s, so a name made only of common words
# ("Coimbatore Foundation") can lose its own identical S1. Keys recover 43% of those misses on train.
KEY_MAX_S1 = 50  # skip keys shared by more S1s ("sri ganesh traders")
KEY_TOP = 2  # per record and key: the S1s closest on the *other* field
NUM1 = pl.col("ad").str.extract(r"\b(\d+)\b")
KEYS = {  # bit -> (key expression, field used to rank S1s sharing the key); tools/pass_coverage*.py
    1: (pl.col("nm").str.split(" ").list.sort().list.join(" "), "ad"),  # name words, any order
    2: (pl.concat_str([NUM1, pl.col("ad").str.extract(  # house number + first real word after it ("109/110 durga")
        r"\b\d+\b[^a-z]*?(?:\b[a-z]{1,2}\b\s+)*\b([a-z]{3,})")], separator=" "), "nm"),
    4: (pl.col("nm").str.replace_all(r"(\B)[aeiou]", "").str.split(" ").list.sort().list.join(" "), "ad"),  # skeleton
    8: (pl.col("nm").str.replace_all(" ", ""), "ad"),  # spacing-free name ("moon light" = "moonlight")
    16: (pl.col("nm").str.split(" ").list.head(2).list.join(" "), "ad"),  # first two name words
}


def rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def block_country(s1, q):
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), dtype=np.float32, max_df=MAX_DF)
    va = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), dtype=np.float32, max_df=MAX_DF)
    key = lambda df: df["nm"].str.replace_all(" ", "").to_list()
    Sn, Sa = vn.fit_transform(key(s1)), va.fit_transform(s1["ad"].to_list())
    S_T = sp.hstack([np.sqrt(W_NAME) * Sn, np.sqrt(1 - W_NAME) * Sa]).T.tocsr()
    Sn, Sa = Sn.tocsr(), Sa.tocsr()
    s1_ids, out = s1["entity_id"].to_numpy(), []
    for lo in range(0, len(q), CHUNK):
        t = time.time()
        c = q[lo:lo + CHUNK]
        Qn, Qa = vn.transform(key(c)), va.transform(c["ad"].to_list())
        Q = sp.hstack([np.sqrt(W_NAME) * Qn, np.sqrt(1 - W_NAME) * Qa]).tocsr()
        C = sp_matmul_topn(Q, S_T, top_n=K, sort=True, n_threads=THREADS)
        counts = np.diff(C.indptr)
        rows, cols = np.repeat(np.arange(len(c)), counts), C.indices
        out.append(pl.DataFrame({
            "q": c["entity_id"].to_numpy()[rows],
            "s1": s1_ids[cols],
            "rank": (np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts)).astype(np.int8),
            "score": C.data,
            "name_cos": rowdot(Qn[rows], Sn[cols]),
            "addr_cos": rowdot(Qa[rows], Sa[cols]),
        }))
        print(f"  {lo + len(c):>9}/{len(q)}  {time.time() - t:.1f}s", flush=True)
    tf = pl.concat(out)
    kp = key_pairs(s1, q)
    new = kp.join(tf.select("q", "s1"), on=["q", "s1"], how="anti")
    if len(new):  # TF-IDF cosines for the pairs only the keys found
        new = (new.join(q.select(q="entity_id").with_row_index("qi"), on="q")
               .join(s1.select(s1="entity_id").with_row_index("si"), on="s1"))
        uq = np.unique(new["qi"].to_numpy())
        sub = q[pl.Series(uq)]
        Qn, Qa = vn.transform(key(sub)), va.transform(sub["ad"].to_list())
        r, c = np.searchsorted(uq, new["qi"].to_numpy()), new["si"].to_numpy()
        nc, ac = rowdot(Qn[r], Sn[c]), rowdot(Qa[r], Sa[c])
        new = new.select("q", "s1", "via_key").with_columns(
            name_cos=pl.Series(nc), addr_cos=pl.Series(ac), score=pl.Series(W_NAME * nc + (1 - W_NAME) * ac))
    allc = pl.concat([tf.join(kp, on=["q", "s1"], how="left").with_columns(pl.col("via_key").fill_null(0)),
                      new.with_columns(rank=pl.lit(0, pl.Int8))], how="diagonal_relaxed")
    print(f"  key passes: {len(kp)} key pairs, {len(new)} not found by TF-IDF", flush=True)
    return allc.with_columns(rank=(pl.col("score").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16))


def key_pairs(s1, q):
    """Exact-key candidates: (q, s1, via_key bitmask), top KEY_TOP per record and key by the other field."""
    parts = []
    for bit, (expr, other) in KEYS.items():
        sk = s1.select(s1="entity_id", key=expr, o_s=other).filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
        sk = sk.filter(pl.len().over("key") <= KEY_MAX_S1)
        pr = (q.select(q="entity_id", key=expr, o_q=other).filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
              .join(sk, on="key"))
        if not len(pr):
            continue
        pr = pr.with_columns(sim=cpdist(pr["o_q"].to_list(), pr["o_s"].to_list(), scorer=fuzz.ratio, workers=-1))
        parts.append(pr.sort("sim", descending=True).group_by("q", maintain_order=True).head(KEY_TOP)
                     .select("q", "s1", via_key=pl.lit(bit, pl.Int8)))
    if not parts:
        return pl.DataFrame(schema={"q": pl.String, "s1": pl.String, "via_key": pl.Int8})
    return pl.concat(parts).group_by("q", "s1").agg(pl.col("via_key").sum().cast(pl.Int8))


def prune(c, rel=REL, cap=CAP, rmax=RMAX):
    keyed = (pl.col("via_key") > 0) if "via_key" in c.columns else pl.lit(False)  # key-pass pairs skip REL/RMAX
    return (c.filter((pl.col("rank") == 0) | keyed | ((pl.col("rank") < rmax) & (pl.col("score") >= rel * pl.col("score").max().over("q"))))
            .filter(pl.col("score").rank("ordinal", descending=True).over("s1") <= cap))


if __name__ == "__main__":
    split = sys.argv[1]
    df = pl.read_parquet(f"{WORK}/{split}.parquet", columns=["entity_id", "src", "country", "nm", "ad"])
    n_s1 = df.filter(pl.col("src") == 1).height
    if "--prune" not in sys.argv:
        parts = []
        for country in df["country"].unique().sort():
            s1 = df.filter((pl.col("src") == 1) & (pl.col("country") == country))
            q = df.filter((pl.col("src") != 1) & (pl.col("country") == country))
            print(f"{country}: {len(s1)} S1, {len(q)} queries", flush=True)
            parts.append(block_country(s1, q))
        pl.concat(parts).write_parquet(f"{WORK}/{split}_cands_full.parquet")
        sys.exit()
    full = pl.read_parquet(f"{WORK}/{split}_cands_full.parquet")
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
        lab = full.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["q", "s1"], how="left").fill_null(0)
        print(f"{'rmax':>5} {'rel':>5} {'cap':>6} {'recall':>7} {'cands/S1':>9}")
        for rmax in (3, 5, 10):
            for rel in (0.7, 0.8, 0.85, 0.9):
                for cap in (30, 20):
                    p = prune(lab, rel, cap, rmax)
                    print(f"{rmax:>5} {rel:>5} {cap:>6} {p['y'].sum() / len(gt):>7.4f} {len(p) / n_s1:>9.2f}", flush=True)
    cands = prune(full)
    cands.write_parquet(f"{WORK}/{split}_cands.parquet")
    per = cands.group_by("s1").len()["len"]
    print(f"RMAX={RMAX} REL={REL} CAP={CAP}: pairs={len(cands)}  candidates per S1: avg {len(cands) / n_s1:.2f}, "
          f"median {per.median()}, p99 {per.quantile(0.99)}, max {per.max()}")
