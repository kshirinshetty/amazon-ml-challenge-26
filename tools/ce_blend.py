"""Cross-encoder step 3 (CPU, on Modal: --script tools/ce_blend.py): blend p = (1-w)*p_lgb + w*p_ce on the hard pairs,
pick w and the threshold on dense validation (same sim_dense as match.py), write the submission for the best w.
Reads work/lgb_{train,test}_probs.parquet and work/ce_pred_{val,test}.parquet; writes runs/$CE_RUN/."""
import json
import os
import sys

import polars as pl

sys.path.insert(0, "/root/src")
from match import assign, sim_dense, write  # noqa: E402

RUN = os.environ.get("CE_RUN", "runs/023_cross-encoder-base")


def blend(probs, ce, w):
    return (probs.join(ce, on=["q", "s1"], how="left")
            .with_columns(p=pl.when(pl.col("p_ce").is_not_null()).then((1 - w) * pl.col("p") + w * pl.col("p_ce"))
                          .otherwise(pl.col("p"))).select("q", "s1", "p"))


tr = pl.read_parquet("work/lgb_train_probs.parquet").select("q", "s1", "p")
cv = pl.read_parquet("work/ce_pred_val.parquet")
res = {}
for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8):
    d = sim_dense(blend(tr, cv, w))
    res[w] = {"dense": d["dense"], "t": float(d["t_dense"]), "normal": d["normal"], "t_normal": float(d["t_normal"])}
    print(f"w={w}: dense {d['dense']:.4f} @ {d['t_dense']}  normal {d['normal']:.4f}", flush=True)
w = max(res, key=lambda k: res[k]["dense"])
print(f"best w={w}: {res[w]}", flush=True)
te = pl.read_parquet("work/lgb_test_probs.parquet").select("q", "s1", "p")
pairs = blend(te, pl.read_parquet("work/ce_pred_test.parquet"), w)
m = assign(pairs, 0.0).filter(pl.col("p") >= res[w]["t"])
s1 = (pl.read_parquet("work/test.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
      .select(s1="entity_id"))
os.makedirs(f"{RUN}/output", exist_ok=True)
write(s1, m, "matched_entity_ids", f"{RUN}/output/matching_results.tsv")
write(s1, pairs, "candidate_entity_ids", f"{RUN}/output/candidate_pairs.tsv")
json.dump({"blend": {str(k): v for k, v in res.items()}, "w": w, "threshold": res[w]["t"],
           "val_f05_dense": res[w]["dense"], "val_f05": res[w]["normal"], "n_matches": len(m)},
          open(f"{RUN}/metrics.json", "w"), indent=2)
print(f"test matches {len(m)} ({len(m) / len(s1):.2f} per S1)")
