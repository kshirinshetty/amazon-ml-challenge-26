"""Blocking: every S2/S3 record looks up its top-K S1 records (same country) by TF-IDF cosine.

Each S2/S3 record matches at most one S1, so searching in this direction keeps the per-S1
candidate set small (~K * #S2S3 / #S1) while top-K recall stays high.
Output: work/{split}_cands.parquet with q, s1, rank, score, name_cos, addr_cos.

Usage: uv run python src/block.py train|test
"""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

WORK = "work"
K = 3
W_NAME = 0.6  # combined score = W_NAME * name_cos + (1 - W_NAME) * addr_cos
CHUNK = 250_000
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())  # Modal sets this from the cpu request
MAX_DF = 0.005  # drop n-grams in >0.5% of S1 docs: long posting lists dominate matmul cost


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
    return pl.concat(out)


if __name__ == "__main__":
    split = sys.argv[1]
    df = pl.read_parquet(f"{WORK}/{split}.parquet", columns=["entity_id", "src", "country", "nm", "ad"])
    parts = []
    for country in df["country"].unique().sort():
        s1 = df.filter((pl.col("src") == 1) & (pl.col("country") == country))
        q = df.filter((pl.col("src") != 1) & (pl.col("country") == country))
        print(f"{country}: {len(s1)} S1, {len(q)} queries", flush=True)
        parts.append(block_country(s1, q))
    cands = pl.concat(parts)
    cands.write_parquet(f"{WORK}/{split}_cands.parquet")
    n_s1 = df.filter(pl.col("src") == 1).height
    print(f"pairs={len(cands)}  avg candidates per S1={len(cands) / n_s1:.2f}")
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
        hit = gt.join(cands, on=["q", "s1"], how="left")
        for k in range(K):
            print(f"recall@{k + 1} = {(hit['rank'] <= k).sum() / len(gt):.4f}  "
                  f"avg cands/S1 = {(cands['rank'] <= k).sum() / n_s1:.2f}")
