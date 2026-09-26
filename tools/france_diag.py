"""Label-free per-country quality check of a run's test predictions.

On train, true pairs share the S1's first house number far more often than decoys (84% vs 9%), so:
  accepted pairs whose numbers disagree ~ likely false merges (lower is better)
  rejected best candidates whose numbers agree ~ likely missed matches (lower is better)
Compare countries within a run, and the same country across runs."""
import sys

import polars as pl

W = "work"
OUTS = sys.argv[1:] or ["runs/003_decoy-features/output", "runs/004_france-selftrain/output"]  # output dirs
rec = pl.read_parquet(f"{W}/test.parquet", columns=["entity_id", "src", "country", "ad"]).with_columns(
    num1=pl.col("ad").str.extract(r"\b(\d+)\b"))
q = rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", num_q="num1")
s = rec.filter(pl.col("src") == 1).select(s1="entity_id", num_s="num1")
top = pl.read_parquet(f"{W}/test_cands.parquet", columns=["q", "s1", "rank"]).filter(pl.col("rank") == 0).select("q", "s1")
agree = lambda df: (df.join(q, on="q").join(s, on="s1")
                    .with_columns(agree=(pl.col("num_q") == pl.col("num_s"))))  # null if a number is missing
for run in OUTS:
    m = (pl.read_csv(f"{run}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
         .select(s1="source1_entity_id", q=pl.col("matched_entity_ids").str.split(",")).explode("q")
         .filter(pl.col("q").is_not_null() & (pl.col("q") != "")))
    acc = agree(m).group_by("country").agg(accepted=pl.len(), acc_disagree=(~pl.col("agree")).mean())
    rej = (agree(top.join(m, on=["q", "s1"], how="anti")).group_by("country")
           .agg(rejected_best=pl.len(), rej_agree=pl.col("agree").mean()))
    print(run)
    print(acc.join(rej, on="country").sort("country"))
