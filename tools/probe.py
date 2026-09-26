"""Leaderboard probes: re-threshold stored test probabilities without retraining.
Usage: modal run modal_app.py --script tools/probe.py   (edit PROBES; files -> runs/probes/<name>/)
work/test_probs.parquet = stage-1 test probabilities of the last match.py predict (run 007 stage 1 == run 006 model)."""
import os
import sys

import polars as pl

sys.path.insert(0, "/root/src")
from match import WORK, assign, write  # noqa: E402

PROBES = {"t080": {"all": 0.80}, "t090": {"all": 0.90}}
p = pl.read_parquet(f"{WORK}/test_probs.parquet")
s1 = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src", "country"]).filter(pl.col("src") == 1)
      .select(s1="entity_id", country="country"))
best = assign(p, 0.0).join(s1, on="s1")
for name, th in PROBES.items():
    t = pl.col("country").replace_strict(th, default=th["all"], return_dtype=pl.Float64)
    m = best.filter(pl.col("p") >= t)
    os.makedirs(f"runs/probes/{name}", exist_ok=True)
    write(s1.select("s1"), m, "matched_entity_ids", f"runs/probes/{name}/matching_results.tsv")
    print(name, th, m.group_by("country").len().sort("country").rows())
