"""How many true pairs does the prune step throw away, and are they key-pass pairs?"""
import polars as pl

W = "work"
gt = pl.read_parquet(f"{W}/train_gt.parquet")
full = pl.read_parquet(f"{W}/train_cands_full.parquet", columns=["q", "s1", "rank", "score", "via_key"])
pruned = pl.read_parquet(f"{W}/train_cands.parquet", columns=["q", "s1"])
t_full = gt.join(full, on=["q", "s1"])
t_kept = gt.join(pruned, on=["q", "s1"])
print(f"recall before prune {len(t_full) / len(gt):.4f}, after {len(t_kept) / len(gt):.4f}")
lost = t_full.join(pruned, on=["q", "s1"], how="anti")
print("lost true pairs:", len(lost), "| via key pass:", (lost["via_key"] > 0).sum())
s1_n = full.group_by("s1").len("n_full")
print("S1 group size of lost pairs:", lost.join(s1_n, on="s1")["n_full"].describe().rows())
print("candidates per S1 before prune:", s1_n["n_full"].mean(), " true pairs only via keys:", (t_full["via_key"] > 0).sum())
