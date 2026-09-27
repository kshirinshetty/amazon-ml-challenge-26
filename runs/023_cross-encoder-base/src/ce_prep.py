"""Cross-encoder step 1 (CPU, on Modal: --script tools/ce_prep.py): LightGBM probabilities for every train/test
candidate pair from runs/$CE_BASE's model, then the *hard* pairs a text model could flip, with raw texts.

Hard record: its best p is in [0.02, 0.995] or its second-best p > 0.02 (a tie). Its candidates with p >= 0.005
(top 5 by p) go to the cross-encoder. Validation = every hard pair of records with a fold-0 candidate (the fold
match.py validates on); training = hard pairs of all other records.
Writes work/lgb_{train,test}_probs.parquet and work/ce_{train,val,test}.parquet."""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/root/src")
from match import FOLD, VALID_FOLD  # noqa: E402

BASE = os.environ.get("CE_BASE", "runs/019_scratch-addr2")
MAX_TRAIN = 3_000_000
m = json.load(open(f"{BASE}/metrics.json"))
model = lgb.Booster(model_file=f"{BASE}/{m.get('models', ['model.txt'])[0]}")


def probs(split):
    f = pl.read_parquet(f"work/{split}_feats.parquet")
    p = np.concatenate([model.predict(f[lo:lo + 4_000_000].select(m["feats"]).to_numpy())
                        for lo in range(0, len(f), 4_000_000)])
    out = f.select("q", "s1", *(["label"] if split == "train" else [])).with_columns(p=p)
    out.write_parquet(f"work/lgb_{split}_probs.parquet")
    return out


def texts(split):
    r = pl.read_parquet(f"work/{split}.parquet", columns=["entity_id", "business_name", "business_address"])
    return r.select("entity_id", t=pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "),
                                                   pl.col("business_address").fill_null("")]))


def hard(pr):
    qs = (pr.group_by("q").agg(pmax=pl.col("p").max(), p2=pl.col("p").top_k(2).min(), n=pl.len())
          .with_columns(p2=pl.when(pl.col("n") > 1).then(pl.col("p2")).otherwise(0.0))
          .filter(pl.col("pmax").is_between(0.02, 0.995) | (pl.col("p2") > 0.02)).select("q"))
    return (pr.join(qs, on="q").filter(pl.col("p") >= 0.005)
            .sort("p", descending=True).group_by("q", maintain_order=True).head(5))


def with_text(df, split):
    t = texts(split)
    return (df.join(t.rename({"entity_id": "q", "t": "text_a"}), on="q")
            .join(t.rename({"entity_id": "s1", "t": "text_b"}), on="s1"))


tr = probs("train")
h = hard(tr)
val_q = h.filter(FOLD == VALID_FOLD).select("q").unique()
val, trn = h.join(val_q, on="q"), h.join(val_q, on="q", how="anti")
trn = trn.sample(min(MAX_TRAIN, len(trn)), seed=0)
print(f"train pairs {len(tr)}; hard {len(h)} ({h['q'].n_unique()} records); ce_train {len(trn)} "
      f"(pos {trn['label'].mean():.3f}); ce_val {len(val)}", flush=True)
with_text(trn, "train").write_parquet("work/ce_train.parquet")
with_text(val, "train").write_parquet("work/ce_val.parquet")
del tr, h, val, trn
te = probs("test")
ht = hard(te)
print(f"test pairs {len(te)}; hard {len(ht)} ({ht['q'].n_unique()} records)", flush=True)
with_text(ht, "test").write_parquet("work/ce_test.parquet")
