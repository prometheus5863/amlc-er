"""Step 4: train the pair classifier (LightGBM), out-of-fold, and decode.

    python -m amlc.model train      # OOF probabilities + threshold sweep + score with erharness
    python -m amlc.model predict    # score test pairs, decode -> output/*.tsv

Decoding uses one data fact from the ground truth: every Source-2/3 record is
matched to AT MOST ONE Source-1 entity. So each target is first assigned to its
single most likely S1 entity (`assign_targets`) and dropped from all others;
then per-entity thresholds decide what is kept.
"""
from __future__ import annotations

import json
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import pq as pq_path
from .features import FEATURES
from .paths import work_dir

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=0, seed=42)
ROUNDS = 500


def W(name):
    return str(work_dir() / "work" / name)


def load_gt_sets(sample_ids=None):
    gt = pq.read_table(pq_path("train_ground_truth")).to_pandas()
    if sample_ids is not None:
        gt = gt[gt.source1_entity_id.isin(sample_ids)]
    return {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}


def sample_ids(frac: float):
    """Same entity subset that features.run(sample=frac) used (DuckDB hash)."""
    con = duckdb.connect()
    q = f"SELECT source1_entity_id FROM '{pq_path('train_ground_truth')}'"
    if frac < 1:
        q += f" WHERE (hash(source1_entity_id) % 1000) < {int(frac * 1000)}"
    return set(con.execute(q).df().source1_entity_id)


def assign_targets(df: pd.DataFrame, prob_col="prob", margin: float = 0.0) -> pd.Series:
    """Each target keeps its probability only for its best S1 (ties within
    `margin` also kept); every other S1 gets 0 for that target."""
    best = df.groupby("cand_id")[prob_col].transform("max")
    return df[prob_col].where(df[prob_col] >= best - margin, 0.0)


def train(frac: float = 1.0, k: int = 3) -> None:
    from erharness import metric
    from erharness.analysis import oracles
    from erharness.decode import DecodeConfig
    from erharness.splits import kfold
    from erharness.sweep import Scorer, best_config, grid_search

    t0 = time.time()
    f = pq.read_table(W("train_feats")).to_pandas()
    ids = sample_ids(frac)
    gt = load_gt_sets(ids)
    f = f[f.s1_id.isin(gt)].reset_index(drop=True)
    true_pairs = {(s, c) for s, cs in gt.items() for c in cs}
    y = np.fromiter(((s, c) in true_pairs for s, c in zip(f.s1_id, f.cand_id)), bool, len(f))
    fold_of = kfold(gt, k)
    fold = f.s1_id.map(fold_of).to_numpy()
    print(f"{len(f):,} pairs, {y.mean():.4f} positive, {len(gt):,} S1 entities  ({time.time() - t0:.0f}s)", flush=True)

    oof = np.zeros(len(f))
    imps = []
    for i in range(k):
        tr, va = fold != i, fold == i
        dtr = lgb.Dataset(f.loc[tr, FEATURES], y[tr])
        dva = lgb.Dataset(f.loc[va, FEATURES], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(f.loc[va, FEATURES], num_iteration=m.best_iteration)
        imps.append(pd.Series(m.feature_importance("gain"), FEATURES))
        print(f"fold {i}: best_iter={m.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)

    c = f[["s1_id", "cand_id"]].copy()
    c["prob_raw"] = oof
    c["prob"] = assign_targets(c, "prob_raw")
    c.to_parquet(W("train_oof.parquet"), index=False)

    report = {}
    for name, col in [("raw", "prob_raw"), ("assigned", "prob")]:
        sc = Scorer(c.rename(columns={col: "p"})[["s1_id", "cand_id", "p"]].rename(columns={"p": "prob"}), gt)
        res = grid_search(sc, {"gate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                               "pair": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
        best = best_config(res)
        report[name] = {"config": best.as_dict(), **{k: float(v) for k, v in res.iloc[0].items()}}
        print(f"\n[{name}] best: gate={best.gate} pair={best.pair}  score={res.iloc[0].score:.5f}")
        print(oracles(sc, best).to_string(index=False))
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\ntop features:\n" + imp.head(15).round(0).to_string())

    pick = max(report, key=lambda n: report[n]["score"])
    cfg = DecodeConfig(**report[pick]["config"])
    json.dump({"prob_col": "prob" if pick == "assigned" else "prob_raw", "decode": cfg.as_dict(),
               "rounds": int(np.mean([ROUNDS])), "local": report},
              open(W("decode.json"), "w"), indent=1, default=float)
    # final model on all sampled entities, fixed rounds = median best_iter was ~ early stop; refit
    del c
    fit_final(f, y)
    print(f"\npicked '{pick}', saved model + decode.json  ({time.time() - t0:.0f}s)")


def fit_final(f=None, y=None, frac: float = 0.1, rounds: int = ROUNDS) -> None:
    """Final model on all (sampled) training pairs with a fixed number of rounds."""
    if f is None:
        f = pq.read_table(W("train_feats"), columns=["s1_id", "cand_id", *FEATURES]).to_pandas()
        gt = load_gt_sets(sample_ids(frac))
        f = f[f.s1_id.isin(gt)].reset_index(drop=True)
        tp = {(s, c) for s, cs in gt.items() for c in cs}
        y = np.fromiter(((s, c) in tp for s, c in zip(f.s1_id, f.cand_id)), bool, len(f))
    X = f[FEATURES].to_numpy(np.float32)
    del f
    m = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATURES, free_raw_data=True), num_boost_round=rounds)
    m.save_model(W("model.txt"))
    print(f"saved {W('model.txt')} ({rounds} rounds)", flush=True)


def _refit_rounds(f, y, fold):
    tr, va = fold != 0, fold == 0
    m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, FEATURES], y[tr]), ROUNDS,
                  valid_sets=[lgb.Dataset(f.loc[va, FEATURES], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return max(50, int(m.best_iteration * 1.1))


def predict() -> None:
    """Scores test pairs bucket by bucket (never all ~100M pairs in memory),
    then assigns targets and decodes."""
    import glob
    from erharness.decode import DecodeConfig, Prepared, decode
    from erharness.io import write_candidate_pairs, write_match_sets

    cfg_all = json.load(open(W("decode.json")))
    cfg = DecodeConfig(**cfg_all["decode"])
    m = lgb.Booster(model_file=W("model.txt"))
    parts = []
    for p in sorted(glob.glob(W("test_feats") + "/*.parquet")):
        f = pq.read_table(p).to_pandas()
        c = f[["s1_id", "cand_id"]].copy()
        c["prob_raw"] = m.predict(f[FEATURES]).astype(np.float32)
        parts.append(c)
        print(f"scored {p.split('/')[-1]}: {len(c):,} pairs", flush=True)
        del f
    c = pd.concat(parts, ignore_index=True)
    c["prob"] = assign_targets(c, "prob_raw")
    c.to_parquet(W("test_scores.parquet"), index=False)
    use = c[["s1_id", "cand_id", cfg_all["prob_col"]]].rename(columns={cfg_all["prob_col"]: "prob"})
    ids = pq.read_table(pq_path("test_source1"), columns=["entity_id"]).to_pandas().entity_id.tolist()
    out = work_dir().parent / "output" if (work_dir().parent / "output").exists() else work_dir() / "output"
    out.mkdir(parents=True, exist_ok=True)
    sets = decode(Prepared(use, ids), cfg)
    write_match_sets(sets, str(out / "matching_results.tsv"), order=ids)
    write_candidate_pairs(c, str(out / "candidate_pairs.tsv"), order=ids)
    n_open = sum(bool(v) for v in sets.values())
    print(f"wrote {out}/matching_results.tsv: {len(ids):,} rows, {n_open:,} non-empty, "
          f"{sum(map(len, sets.values())):,} matches")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit_final()
    elif sys.argv[1] == "train":
        train(float(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 1.0)
    else:
        predict()
