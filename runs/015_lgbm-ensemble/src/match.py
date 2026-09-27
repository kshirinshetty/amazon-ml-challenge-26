"""Train LightGBM on candidate-pair features, tune the match threshold for macro F0.5 on
held-out S1 entities, and write the submission files. Everything a run produces goes in RUN_DIR.

Usage: uv run python src/match.py fit RUN_DIR [--reuse a.txt,b.txt]  # -> model_*.txt, metrics.json, errors.tsv
       uv run python src/match.py predict RUN_DIR   # -> output/matching_results.tsv, output/candidate_pairs.tsv
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

WORK = "work"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())  # Modal sets this from the cpu request
FOLD = pl.col("s1").hash(seed=42) % 5
VALID_FOLD, TRAIN_FOLDS = 0, [1, 2, 3, 4]  # 20% validation, 80% training
PSEUDO_HI, PSEUDO_LO = 0.97, 0.03  # --pseudo: confident test pairs of unseen countries become labels
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=511, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=THREADS, verbose=-1)
ROUNDS, PATIENCE = 3000, 60
# ensemble: members trained by fit (overrides of PARAMS), averaged with any --reuse model files
VARIANTS = [dict(seed=1), dict(seed=2, num_leaves=1023, min_data_in_leaf=100)]


def assign(pairs, t):
    """Each S2/S3 record goes to its single best S1 (it matches at most one), if p >= t."""
    return pairs.sort("p", descending=True).unique("q", keep="first").filter(pl.col("p") >= t)


def expected_f(best):
    """Per-S1 decision: of the records assigned to an S1 (each at its argmax), keep the top-k by p that
    maximize the plug-in expected F0.5 = 1.25*sum(p_1..p_k) / (0.25*sum(p) + k); keep none when
    P(no true match) = prod(1 - p) is higher. best: (q, s1, p) with p >= 0 for every record."""
    b = (best.sort(["s1", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1"), cum=pl.col("p").cum_sum().over("s1"),
                       tot=pl.col("p").sum().over("s1"), none=(1 - pl.col("p")).log().sum().over("s1").exp())
         .with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("k"))))
    keep = (b.group_by("s1").agg(best_k=pl.col("k").get(pl.col("ef").arg_max()), best_ef=pl.col("ef").max(),
                                 none=pl.col("none").first())
            .filter(pl.col("best_ef") > pl.col("none")))
    return b.join(keep, on="s1").filter(pl.col("k") <= pl.col("best_k")).select("q", "s1", "p")


def decide(pairs, rule, t):
    best = assign(pairs, 0.0)
    return expected_f(best) if rule == "expected_f" else best.filter(pl.col("p") >= t)


def f05(pred, gt, s1_ids):
    """Leaderboard metric: F0.5 per S1 (singletons included), macro-averaged over s1_ids."""
    count = lambda df, name: df.group_by("s1").len(name)
    d = (pl.DataFrame({"s1": s1_ids})
         .join(count(pred.join(gt, on=["s1", "q"]), "tp"), on="s1", how="left")
         .join(count(pred, "n_pred"), on="s1", how="left")
         .join(count(gt, "n_true"), on="s1", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("n_pred"), pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.select(f.mean()).item()


def sim_dense(pairs, drop=0.5):
    """Validation F0.5 as-is and at test-like decoy density. Test has ~42% decoys among S2/S3 records vs
    26% in train; removing a share `drop` of true records reproduces that (run 006: dense 0.9638 vs
    leaderboard 0.961, normal 0.9747). pairs: q, s1, p over all train folds."""
    gt = pl.read_parquet(f"{WORK}/train_gt.parquet").filter(FOLD == VALID_FOLD)
    s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)
              .select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
    dropped = gt.sample(fraction=drop, seed=0).select("q")
    out = {}
    for name, P, G in (("normal", pairs, gt),
                       ("dense", pairs.join(dropped, on="q", how="anti"), gt.join(dropped, on="q", how="anti"))):
        best = assign(P, 0.0).filter(FOLD == VALID_FOLD)
        curve = {round(float(t), 2): f05(best.filter(pl.col("p") >= t), G, s1_ids) for t in np.arange(0.3, 0.96, 0.05)}
        t = max(curve, key=curve.get)
        out.update({name: curve[t], f"t_{name}": t, f"curve_{name}": curve})
    return out


def predict_pairs(models, f, feats):
    """Mean probability over the ensemble members."""
    p = np.concatenate([np.mean([m.predict(x) for m in models], axis=0)
                        for x in (f[lo:lo + 5_000_000].select(feats).to_numpy() for lo in range(0, len(f), 5_000_000))])
    return f.select("q", "s1").with_columns(p=p)


def error_sample(best, pred, gt, n=300):
    """Validation mistakes with their text, for eyeballing: wrong merges and missed matches."""
    rec = pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "business_name", "business_address"])
    txt = lambda side: rec.rename({"entity_id": side, "business_name": f"{side}_name",
                                   "business_address": f"{side}_addr"})
    fp = pred.join(gt, on=["s1", "q"], how="anti").with_columns(kind=pl.lit("false_merge"))
    fn = (gt.join(pred.select("q", "s1"), on=["s1", "q"], how="anti")
          .join(best.select("q", best_s1="s1", best_p="p"), on="q", how="left")
          .with_columns(kind=pl.when(pl.col("best_s1").is_null()).then(pl.lit("miss:not_in_candidates"))
                        .when(pl.col("best_s1") != pl.col("s1")).then(pl.lit("miss:assigned_elsewhere"))
                        .otherwise(pl.lit("miss:not_selected")), p=pl.col("best_p"))
          .drop("best_s1", "best_p"))
    return (pl.concat([fp.sample(min(n, len(fp)), seed=0), fn.sample(min(n, len(fn)), seed=0)], how="diagonal")
            .join(txt("s1"), on="s1", how="left").join(txt("q"), on="q", how="left")
            .select("kind", "p", "s1", "s1_name", "s1_addr", "q", "q_name", "q_addr"))


def pseudo_labels(feats):
    """Self-training for countries absent from train (France): the previous run's confident test pairs
    (work/test_probs.parquet, written by predict) join the training set as labels."""
    train_c = pl.read_parquet(f"{WORK}/train.parquet", columns=["country"])["country"].unique()
    country = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src", "country"])
               .filter((pl.col("src") != 1) & ~pl.col("country").is_in(train_c.implode())).select(q="entity_id"))
    probs = (pl.read_parquet(f"{WORK}/test_probs.parquet").join(country, on="q")
             .filter((pl.col("p") >= PSEUDO_HI) | (pl.col("p") <= PSEUDO_LO))
             .select("q", "s1", label=(pl.col("p") >= PSEUDO_HI).cast(pl.Int8)))
    ps = pl.read_parquet(f"{WORK}/test_feats.parquet").join(probs, on=["q", "s1"]).select(*feats, "label")
    print(f"pseudo-labels: {len(ps)} pairs, {ps['label'].mean():.3f} positive", flush=True)
    return ps


def fit(run_dir, pseudo=False, reuse=()):
    f = pl.read_parquet(f"{WORK}/train_feats.parquet").with_columns(fold=FOLD)
    feats = [c for c in f.columns if c not in ("q", "s1", "label", "fold")]
    tr = f.filter(pl.col("fold").is_in(TRAIN_FOLDS))
    if pseudo:
        tr = pl.concat([tr.select(*feats, "label"), pseudo_labels(feats)], how="vertical_relaxed")
    va = f.filter(pl.col("fold") == VALID_FOLD).sample(fraction=0.25, seed=0)
    t0 = time.time()
    models = [lgb.Booster(model_file=m) for m in reuse]
    dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["label"].to_numpy())
    dva = lgb.Dataset(va.select(feats).to_numpy(), va["label"].to_numpy(), reference=dtr)
    for v in VARIANTS:
        models.append(lgb.train({**PARAMS, **v}, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(PATIENCE), lgb.log_evaluation(100)]))
        print(f"member {v}: best iter {models[-1].best_iteration}, {time.time() - t0:.0f}s", flush=True)
    train_s, n_train = time.time() - t0, len(tr)
    model = models[-1]
    del tr, va, dtr, dva

    # Score every pair (records compete across folds), then evaluate on held-out S1s only.
    member = [predict_pairs([m], f, feats) for m in models]
    all_pairs = member[0].with_columns(p=np.mean([m["p"].to_numpy() for m in member], axis=0))
    if len(models) > 1:  # each member alone on dense validation, to see what the ensemble adds
        for i, m in enumerate(member):
            print(f"member {i} alone: dense F0.5={sim_dense(m)['dense']:.4f}", flush=True)
    del member
    best = assign(all_pairs, 0.0).filter(FOLD == VALID_FOLD)
    s1_ids = (pl.read_parquet(f"{WORK}/train.parquet", columns=["entity_id", "src"])
              .filter(pl.col("src") == 1).select(s1="entity_id").filter(FOLD == VALID_FOLD)["s1"])
    gt = pl.read_parquet(f"{WORK}/train_gt.parquet").filter(FOLD == VALID_FOLD)
    found = f.filter((pl.col("fold") == VALID_FOLD) & (pl.col("label") == 1))
    curve = {f"{t:.2f}": f05(best.filter(pl.col("p") >= t), gt, s1_ids) for t in np.arange(0.1, 1.0, 0.05)}
    t_best = max(curve, key=curve.get)
    ef = f05(expected_f(best), gt, s1_ids)
    rule = "expected_f" if ef > curve[t_best] else "threshold"
    dense = sim_dense(all_pairs)  # the leaderboard proxy; its threshold is the one used on test
    metrics = {
        "val_f05": max(ef, curve[t_best]), "rule": rule, "val_f05_expected_f": ef,
        "val_f05_threshold": curve[t_best], "threshold": float(dense["t_dense"]), "threshold_normal": float(t_best),
        "val_f05_dense": dense["dense"], "f05_dense_by_threshold": dense["curve_dense"],
        "blocking_recall": len(found) / len(gt), "oracle_f05": f05(found, gt, s1_ids),
        "f05_by_threshold": curve, "best_iter": model.best_iteration, "variants": VARIANTS, "reuse": list(reuse),
        "n_train_pairs": n_train, "train_seconds": round(train_s), "params": PARAMS, "feats": feats, "pseudo": pseudo,
        "feature_gain": dict(sorted(zip(feats, model.feature_importance("gain").round().tolist()),
                                    key=lambda kv: -kv[1])),
    }
    for t, v in curve.items():
        print(f"  t={t}  F0.5={v:.4f}")
    print(f"blocking recall={metrics['blocking_recall']:.4f}  oracle F0.5={metrics['oracle_f05']:.4f}  "
          f"best t={t_best}  F0.5={curve[t_best]:.4f}  expected-F rule F0.5={ef:.4f}  -> {rule}  "
          f"| test-like dense F0.5={dense['dense']:.4f} @ t={dense['t_dense']}")
    os.makedirs(run_dir, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f"{run_dir}/model_{i}.txt")
    metrics["models"] = [f"model_{i}.txt" for i in range(len(models))]
    json.dump(metrics, open(f"{run_dir}/metrics.json", "w"), indent=2)
    pred = expected_f(best) if rule == "expected_f" else best.filter(pl.col("p") >= float(t_best))
    error_sample(best, pred, gt).write_csv(f"{run_dir}/errors.tsv", separator="\t")


def write(s1_ids, pairs, col, path):
    agg = pairs.group_by("s1").agg(pl.col("q").sort().str.join(",").alias(col))
    (s1_ids.join(agg, on="s1", how="left", maintain_order="left").with_columns(pl.col(col).fill_null(""))
     .rename({"s1": "source1_entity_id"}).write_csv(path, separator="\t", quote_style="never"))


def predict(run_dir, unseen_t=None):
    """unseen_t: separate threshold for countries absent from training (France); output goes to
    output_unseen{t}/ so it can be compared with output/ on the leaderboard."""
    metrics = json.load(open(f"{run_dir}/metrics.json"))
    models = [lgb.Booster(model_file=f"{run_dir}/{m}") for m in metrics.get("models", ["model.txt"])]
    pairs = predict_pairs(models, pl.read_parquet(f"{WORK}/test_feats.parquet"), metrics["feats"])
    pairs.write_parquet(f"{WORK}/test_probs.parquet")  # input for the next run's --pseudo
    s1 = (pl.read_parquet(f"{WORK}/test.parquet", columns=["entity_id", "src", "country"])
          .filter(pl.col("src") == 1).select(s1="entity_id", country="country"))
    m = decide(pairs, metrics.get("rule", "threshold"), metrics["threshold"])
    out = f"{run_dir}/output"
    if unseen_t is not None:
        seen = pl.read_parquet(f"{WORK}/train.parquet", columns=["country"])["country"].unique()
        unseen = s1.filter(~pl.col("country").is_in(seen.implode())).select("s1")
        m = pl.concat([m.join(unseen, on="s1", how="anti"),
                       assign(pairs, 0.0).join(unseen, on="s1").filter(pl.col("p") >= unseen_t)])
        out = f"{run_dir}/output_unseen{unseen_t}"
    os.makedirs(out, exist_ok=True)
    write(s1.select("s1"), m, "matched_entity_ids", f"{out}/matching_results.tsv")
    write(s1.select("s1"), pairs, "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
    per = lambda df, name: df.group_by("s1").len(name)
    by_country = (s1.join(per(pairs, "cands"), on="s1", how="left").join(per(m, "matches"), on="s1", how="left")
                  .fill_null(0).group_by("country").agg(
                      n_s1=pl.len(), avg_candidates=pl.col("cands").mean(), avg_matches=pl.col("matches").mean(),
                      pct_empty=(pl.col("matches") == 0).mean() * 100).sort("country"))
    metrics["test" if unseen_t is None else f"test_unseen{unseen_t}"] = {"n_s1": len(s1), "n_matches": len(m), "avg_candidates_per_s1": len(pairs) / len(s1),
                       "by_country": by_country.to_dicts()}
    json.dump(metrics, open(f"{run_dir}/metrics.json", "w"), indent=2)
    print(by_country)


if __name__ == "__main__":
    # README example: pred {47,193,812} vs truth {47,812} -> 0.714; singleton right -> 1, wrong -> 0
    P = pl.DataFrame({"s1": ["a", "a", "a", "c"], "q": ["47", "193", "812", "9"]})
    G = pl.DataFrame({"s1": ["a", "a"], "q": ["47", "812"]})
    assert abs(f05(P, G, ["a"]) - 0.7143) < 1e-3 and f05(P, G, ["b"]) == 1.0 and f05(P, G, ["c"]) == 0.0
    # expected-F: confident pair kept, weak tail dropped; an S1 with only weak candidates predicts nothing
    B = pl.DataFrame({"q": ["1", "2", "3", "4"], "s1": ["a", "a", "a", "b"], "p": [0.95, 0.9, 0.2, 0.3]})
    assert sorted(expected_f(B)["q"].to_list()) == ["1", "2"]
    if sys.argv[1] == "fit":
        fit(sys.argv[2], pseudo="--pseudo" in sys.argv,
            reuse=sys.argv[sys.argv.index("--reuse") + 1].split(",") if "--reuse" in sys.argv else ())
    else:
        t = sys.argv[sys.argv.index("--unseen-t") + 1] if "--unseen-t" in sys.argv else None
        predict(sys.argv[2], None if t is None else float(t))
