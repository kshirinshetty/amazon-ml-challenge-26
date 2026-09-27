"""Pairwise features for blocked candidates -> work/{split}_feats.parquet (+ label on train).

Decoys (26% of S2/S3, matching no S1) are near-copies of a real business with an extra word
("Midtown", "Overseas", a different legal form) and a shifted house number. The decoy features use the
*full* name (no stopword/filler removal) and are label-free, so they work the same on France.

Usage: uv run python src/features.py train|test
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from normalize import name_tokens

WORK = "work"
CHUNK = 2_000_000
NUM = r"\d+"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
PROXY_MIN, PROXY_M = 200, 50  # a word's decoy score needs >= 200 records; shrunk toward the country mean


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
        # raw lowercase text keeps legal forms / spelling that cleaning removes: it breaks ties between
        # S1s sharing a normalized name (raw-name best = true S1 in 52% of no-address ties vs 32% by chance)
        raw_n=sim(c["rq_n"], c["rs_n"], fuzz.ratio),
        raw_a=sim(c["rq_a"], c["rs_a"], fuzz.ratio),
    ).with_columns(
        num_q=pl.col("ad_q").str.extract_all(NUM).list.unique(),
        num_s=pl.col("ad_s").str.extract_all(NUM).list.unique(),
        tok_q=pl.col("nm_q").str.split(" ").list.unique(),
        tok_s=pl.col("nm_s").str.split(" ").list.unique(),
    ).with_columns(
        num_inter=pl.col("num_q").list.set_intersection("num_s").list.len(),
        num_union=pl.col("num_q").list.set_union("num_s").list.len(),
        num_first_eq=(pl.col("num_q").list.first() == pl.col("num_s").list.first()).cast(pl.Int8),
        # near-duplicate businesses differ by a few house-number units ("9101" vs "9104")
        num_first_diff=(pl.col("num_q").list.first().cast(pl.Int64, strict=False)
                        - pl.col("num_s").list.first().cast(pl.Int64, strict=False)).abs().clip(upper_bound=10**6),
        # ... or by a name word one side has and the other lacks ("Technology" vs "Technology Overseas")
        n_extra_q=pl.col("tok_q").list.set_difference("tok_s").list.len(),
        n_extra_s=pl.col("tok_s").list.set_difference("tok_q").list.len(),
        len_nq=pl.col("nm_q").str.len_chars(), len_ns=pl.col("nm_s").str.len_chars(),
        len_aq=pl.col("ad_q").str.len_chars(), len_as=pl.col("ad_s").str.len_chars(),
    ).with_columns(
        num_jac=pl.col("num_inter") / pl.col("num_union"),
        num_q_in_s=(pl.col("num_inter") == pl.col("num_q").list.len()).cast(pl.Int8),
    ).drop("nm_q", "nm_s", "ad_q", "ad_s", "num_q", "num_s", "tok_q", "tok_s", "rq_n", "rs_n", "rq_a", "rs_a")


def full_names(names):
    """Name word sets with legal forms and filler kept: exactly the words that mark decoys."""
    maps = json.load(open(f"{WORK}/maps.json"))["name"]
    with Pool(THREADS) as p:
        toks = p.map(name_tokens, names.to_list(), chunksize=20_000)
    return (pl.Series(toks).str.split(" ").list.eval(pl.element().filter(pl.element() != "").replace(maps))
            .list.unique())


def decoy_scores(rec, split):
    """Label-free decoy score per (country, word): among S2/S3 records whose best S1 lacks the word, how
    often the first house number disagrees with that S1's (train: decoys 91%, true records 19%)."""
    top = pl.read_parquet(f"{WORK}/{split}_cands_full.parquet", columns=["q", "s1", "rank"]).filter(pl.col("rank") == 0)
    s = rec.filter(pl.col("src") == 1).select(s1="entity_id", full_s="full", num1_s="num1")
    t = (rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", full="full", num1="num1")
         .join(top.select("q", "s1"), on="q").join(s, on="s1")
         .select("country", x=pl.col("full").list.set_difference("full_s"),
                 mm=(pl.col("num1") != pl.col("num1_s")).cast(pl.Float64))
         .filter(pl.col("mm").is_not_null()).explode("x").filter(pl.col("x").is_not_null()))
    prior = t.group_by("country").agg(prior=pl.col("mm").mean())
    return (t.group_by("country", "x").agg(n=pl.len(), mm=pl.col("mm").sum()).filter(pl.col("n") >= PROXY_MIN)
            .join(prior, on="country")
            .select("country", tok="x", dec=(pl.col("mm") + PROXY_M * pl.col("prior")) / (pl.col("n") + PROXY_M)))


def decoy_feats(c, rec, dec):
    """Per pair: extra words vs the S1 and how decoy-like they are; does the S1's house number appear
    in the record; do the other records pointing at this S1 agree with this one's number and words?"""
    q = rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", full_q="full", num1_q="num1",
                                             nums_q="nums")
    s = rec.filter(pl.col("src") == 1).select(s1="entity_id", full_s="full", num1_s="num1")
    d = (c.select("q", "s1").join(q, on="q").join(s, on="s1")
         .with_columns(xq=pl.col("full_q").list.set_difference("full_s"),
                       xs=pl.col("full_s").list.set_difference("full_q")))
    key = ["q", "s1"]
    xdec = (d.select(*key, "country", tok="xq").explode("tok").join(dec, on=["country", "tok"])
            .group_by(key).agg(dec_max=pl.col("dec").max(), dec_sum=pl.col("dec").sum()))
    numd = (d.select(*key, n="nums_q", t="num1_s").explode("n")
            .with_columns(diff=(pl.col("n").cast(pl.Int64, strict=False) - pl.col("t").cast(pl.Int64, strict=False)).abs())
            .group_by(key).agg(s_num_mindiff=pl.col("diff").min().clip(upper_bound=10**6)))
    tok_df = d.select("s1", tok="full_q").explode("tok").group_by("s1", "tok").len("df")
    uniq = (d.select(*key, tok="xq").explode("tok").join(tok_df, on=["s1", "tok"])
            .group_by(key).agg(g_uniq_x=(pl.col("df") == 1).sum()))
    same_num = d.filter(pl.col("num1_q").is_not_null()).group_by("s1", "num1_q").len("g_same_num")
    return (d.join(same_num, on=["s1", "num1_q"], how="left")
            .select(*key, n_xq=pl.col("xq").list.len(), n_xs=pl.col("xs").list.len(),
                    # other records of this S1 sharing this record's first number (itself excluded)
                    g_same_num=pl.col("g_same_num") - 1,
                    g_same_num_frac=(pl.col("g_same_num") - 1) / (pl.len().over("s1") - 1).clip(lower_bound=1))
            .join(xdec, on=key, how="left").join(numd, on=key, how="left").join(uniq, on=key, how="left")
            .with_columns(s_num_in_q=(pl.col("s_num_mindiff") == 0).cast(pl.Int8)))


def ambiguity_feats(c, rec):
    """Can the pair be told apart from look-alikes? Records with no address or a made-up trade name
    ("Umbraveramira" at the S1's exact address) are only safe when no other S1 shares the name/address."""
    s1 = rec.filter(pl.col("src") == 1)
    same_nm = s1.group_by("country", "nm").len("n")
    same_ad = s1.filter(pl.col("ad") != "").group_by("country", "ad").len("n")
    vocab = (s1.select("country", tok=pl.col("nm").str.split(" ")).explode("tok").filter(pl.col("tok") != "")
             .group_by("country", "tok").len("df").filter(pl.col("df") >= 2).select("country", "tok"))
    q = rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", q_nm="nm")
    known = (q.select("q", "country", tok=pl.col("q_nm").str.split(" ")).explode("tok").filter(pl.col("tok") != "")
             .join(vocab.with_columns(k=pl.lit(1)), on=["country", "tok"], how="left")
             .group_by("q").agg(q_known_frac=pl.col("k").fill_null(0).mean()))
    s = s1.select(s1="entity_id", country="country", s_nm="nm", s_ad="ad")
    d = (c.select("q", "s1").join(q, on="q").join(s.drop("country"), on="s1")
         .join(same_nm.rename({"nm": "s_nm", "n": "amb_s_name"}), on=["country", "s_nm"], how="left")
         .join(same_nm.rename({"nm": "q_nm", "n": "amb_q_name"}), on=["country", "q_nm"], how="left")
         .join(same_ad.rename({"ad": "s_ad", "n": "amb_s_addr"}), on=["country", "s_ad"], how="left")
         .with_columns(name_eq=(pl.col("q_nm") == pl.col("s_nm")).cast(pl.Int8)))
    return (d.with_columns(q_name_ties=pl.col("name_eq").sum().over("q"))
            .join(known, on="q", how="left")
            .select("q", "s1", "amb_s_name", pl.col("amb_q_name").fill_null(0), "amb_s_addr", "name_eq", "q_name_ties",
                    "q_known_frac"))


def noise_feats(split):
    """Per-record formatting noise, label-free. Decoys are near-copies of clean S1 text; true records carry more
    source noise (train: domain-only names 4% decoys vs 26% overall; lowercase names, accents, digits inside words
    13-18%; '#' / 'door no' / 'h.no' addresses 31%). Normalization and the raw-text features lowercase all of it."""
    r = (pl.read_parquet(f"{WORK}/{split}.parquet", columns=["entity_id", "src", "business_name", "business_address"])
         .filter(pl.col("src") != 1))
    n, a = pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
    flags = {
        "nz_url": n.str.contains(r"(?i)www\.|\.(com|in|fr|net|org|co)\b"),
        "nz_lower": (n == n.str.to_lowercase()) & n.str.contains(r"[a-z]"),
        "nz_upper": (n == n.str.to_uppercase()) & n.str.contains(r"[A-Z]"),
        "nz_addr_upper": (a == a.str.to_uppercase()) & a.str.contains(r"[A-Z]"),
        "nz_digit_word": n.str.contains(r"[A-Za-z][0-9][A-Za-z]"),
        "nz_lead_punct": n.str.contains(r"^[\W_]"),
        "nz_trail_punct": n.str.contains(r"[\.,;:\-]$"),
        "nz_dblspace": n.str.contains("  "),
        "nz_bracket": n.str.contains(r"[\[\(]"),
        "nz_idtag": n.str.contains(r"(?i)\(id:"),
        "nz_title": n.str.contains(r"(?i)^(mr|mrs|ms|dr|smt|shri|sri|m/s)\b"),
        "nz_addr_hash": a.str.contains("#"),
        "nz_addr_door": a.str.contains(r"(?i)door no|h\.?\s?no"),
        "nz_addr_pmb": a.str.contains(r"(?i)\bpmb\b"),
        "nz_addr_null": a.str.contains(r"(?i)null"),
        "nz_addr_zero": a.str.contains(r"\b0\d+"),
    }
    return r.select(q="entity_id", nz_accent=n.str.count_matches(r"[À-ÿ]").cast(pl.Int16),
                    **{k: v.cast(pl.Int8) for k, v in flags.items()})


def build(split):
    rec = pl.read_parquet(f"{WORK}/{split}.parquet").select(
        "entity_id", "src", "country", "nm", "ad", "business_name",
        translit=pl.col("business_name").str.contains(r"[^\x00-\x7F]").cast(pl.Int8),
        raw_n=pl.col("business_name").fill_null("").str.to_lowercase(),
        raw_a=pl.col("business_address").fill_null("").str.to_lowercase(),
        nums=pl.col("ad").str.extract_all(NUM).list.unique(),
        num1=pl.col("ad").str.extract(r"\b(\d+)\b"),
    )
    rec = rec.with_columns(full=full_names(rec["business_name"])).drop("business_name")
    dec = decoy_scores(rec, split)
    dec.write_parquet(f"{WORK}/{split}_decoy_words.parquet")
    print(dec.sort("dec", descending=True).group_by("country").head(15).sort("country", "dec", descending=[False, True]).rows())
    c = pl.read_parquet(f"{WORK}/{split}_cands.parquet").with_columns(
        # query-level context: how clear-cut is this record's best S1?
        top1=pl.col("score").max().over("q"),
        top2=pl.col("score").top_k(2).min().over("q"),
        # S1-level context: how many records point at this S1, and where does this one rank?
        s1_n=pl.len().over("s1"),
        s1_n0=(pl.col("rank") == 0).sum().over("s1"),
        s1_rank=pl.col("score").rank("ordinal", descending=True).over("s1"),
    ).with_columns(gap=pl.col("top1") - pl.col("score"), margin=pl.col("top1") - pl.col("top2"))
    c = c.join(decoy_feats(c, rec, dec), on=["q", "s1"], how="left").join(ambiguity_feats(c, rec), on=["q", "s1"], how="left")
    q = rec.filter(pl.col("src") != 1).select(q="entity_id", src="src", nm_q="nm", ad_q="ad", translit="translit",
                                             rq_n="raw_n", rq_a="raw_a")
    s = rec.filter(pl.col("src") == 1).select(s1="entity_id", nm_s="nm", ad_s="ad", rs_n="raw_n", rs_a="raw_a")
    out = []
    for lo in range(0, len(c), CHUNK):
        chunk = c[lo:lo + CHUNK].join(q, on="q", how="left").join(s, on="s1", how="left")
        out.append(string_feats(chunk))
        print(f"  {lo + len(chunk)}/{len(c)}", flush=True)
    f = pl.concat(out)
    # within-record comparison: is this candidate the closest of the record's candidates on each signal?
    # (gap / is-best / ties instead of rank().over("q"): ranking inside ~10M small groups took hours)
    for c in ("raw_n", "raw_a", "n_ratio", "a_tset"):
        f = f.with_columns((pl.col(c).max().over("q") - pl.col(c)).alias(f"{c}_qgap"))
        f = f.with_columns((pl.col(f"{c}_qgap") == 0).cast(pl.Int8).alias(f"{c}_qbest"))
    f = f.with_columns(raw_n_qties=pl.col("raw_n_qbest").sum().over("q").cast(pl.Int16),
                       raw_a_qties=pl.col("raw_a_qbest").sum().over("q").cast(pl.Int16))
    f = f.join(noise_feats(split), on="q", how="left")
    if split == "train":
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet").with_columns(label=pl.lit(1, pl.Int8))
        f = f.join(gt, on=["q", "s1"], how="left").with_columns(pl.col("label").fill_null(0))
    f.write_parquet(f"{WORK}/{split}_feats.parquet")
    print(f.describe())


if __name__ == "__main__":
    if "--noise" in sys.argv:  # add the noise features to an existing work/{split}_feats.parquet
        path = f"{WORK}/{sys.argv[1]}_feats.parquet"
        pl.read_parquet(path).join(noise_feats(sys.argv[1]), on="q", how="left").write_parquet(path)
    else:
        build(sys.argv[1])
