"""Score test pairs with an existing run's model -> work/test_probs.parquet (input for match.py fit --pseudo).
Usage: modal run modal_app.py --script tools/test_probs.py   (RUN below picks the model)"""
import json
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "/root/src")
from match import WORK, predict_pairs  # noqa: E402

RUN = "runs/003_decoy-features"
m = json.load(open(f"{RUN}/metrics.json"))
p = predict_pairs(lgb.Booster(model_file=f"{RUN}/model.txt"), pl.read_parquet(f"{WORK}/test_feats.parquet"), m["feats"])
p.write_parquet(f"{WORK}/test_probs.parquet")
print(f"{RUN}: {len(p)} test pairs scored; p>=0.97: {(p['p'] >= 0.97).sum()}, p<=0.03: {(p['p'] <= 0.03).sum()}")
