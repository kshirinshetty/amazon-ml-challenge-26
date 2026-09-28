"""Cross-encoder step 1 (CPU): LightGBM probabilities for every train/test candidate pair from RUN_DIR's model,
then the *hard* pairs a text model could flip, with their raw texts.

  python tools/ce_prep.py RUN_DIR                                   # locally (>= 64 GB RAM)
  modal run modal_app.py --script "tools/ce_prep.py RUN_DIR"        # on Modal

Hard record: its best p is in [LO, HI] or its second-best p > P2 (a tie). Its candidates with p >= PMIN
(top 5 by p) go to the cross-encoder. Validation = every hard pair of records with a fold-0 candidate (the fold
match.py validates on); training = hard pairs of all other records.
Writes work/lgb_{train,test}_probs.parquet and work/ce_{train,val,test}.parquet."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path[:0] = ["/root/src", "src"]  # Modal container / repo root
from match import FOLD, VALID_FOLD  # noqa: E402

BASE = sys.argv[1]  # run whose LightGBM model scores the pairs, e.g. runs/019_scratch-addr2
MAX_TRAIN = 3_000_000
# ~4% of candidate pairs (runs 022/023). The wider band of run 024 (0.002, 0.9995, 0.005, 0.002) was not finished.
LO, HI, P2, PMIN = 0.02, 0.995, 0.02, 0.005
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
          .filter(pl.col("pmax").is_between(LO, HI) | (pl.col("p2") > P2)).select("q"))
    return (pr.join(qs, on="q").filter(pl.col("p") >= PMIN)
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
