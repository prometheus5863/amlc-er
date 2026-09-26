"""Stage 2: re-score every pair using the stage-1 probabilities of its neighbours.

Stage 1 judges each (S1, candidate) pair in isolation. Stage 2 adds context:
  * within the S1 entity: how this candidate ranks, the gap to the best candidate,
    how many strong candidates there are (the expected group size);
  * within the target record: whether another S1 entity claims it more strongly
    (each S2/S3 record belongs to at most one S1 entity);
  * group consistency: does this candidate look like the other confident members
    of the same group (name / address similarity to the strongest sibling)?
  * interactions from the public 0.976 pipeline: name x address similarity,
    "exact name but target address missing".

Training uses OUT-OF-FOLD stage-1 probabilities, so stage 2 never sees scores
the stage-1 model produced on its own training rows.

    python -m amlc.stage2 train     # CV score vs stage 1, saves model2.txt + decode2.json
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .features import FEATURES
from .model import PARAMS, W, assign_targets, load_gt_sets, sample_ids

CTX = ["p1", "p1_logit", "p1_rank_s1", "p1_gap_best_s1", "p1_gap_next_s1", "s1_n_strong", "s1_sum_p",
       "s1_max_p", "t_max_p", "t_gap_best_other", "t_rank", "t_n_strong", "is_best_for_t",
       "sib_name_max", "sib_addr_max", "sib_n", "x_name_addr", "x_exact_name_noaddr", "x_name_min_addr"]
FEATURES2 = FEATURES + CTX
STRONG = 0.5


def add_context(f: pd.DataFrame, attrs: pd.DataFrame | None = None) -> pd.DataFrame:
    """f: pair table with s1_id, cand_id, p1 (stage-1 prob) and stage-1 features.
    attrs: optional entity_id -> name_core, addr_clean for sibling similarity."""
    p = f.p1.astype(np.float32)
    g1 = p.groupby(f.s1_id, observed=True)
    f["p1_logit"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).astype(np.float32)
    f["p1_rank_s1"] = g1.rank(ascending=False, method="min").astype(np.int16)
    f["s1_max_p"] = g1.transform("max").astype(np.float32)
    f["p1_gap_best_s1"] = (f.s1_max_p - p).astype(np.float32)
    # gap to the next-lower candidate (sharp drop after the true group = clean boundary)
    order = f.assign(_p=p).sort_values(["s1_id", "_p"], ascending=[True, False])
    nxt = order.groupby("s1_id", observed=True)._p.shift(-1).fillna(0)
    f["p1_gap_next_s1"] = (order._p - nxt).reindex(f.index).astype(np.float32)
    f["s1_n_strong"] = (p >= STRONG).groupby(f.s1_id, observed=True).transform("sum").astype(np.int16)
    f["s1_sum_p"] = g1.transform("sum").astype(np.float32)
    gt = p.groupby(f.cand_id, observed=True)
    f["t_max_p"] = gt.transform("max").astype(np.float32)
    # best probability among the OTHER S1 entities claiming this target
    top1 = gt.transform("max")
    cnt_top = (p == top1).groupby(f.cand_id, observed=True).transform("sum")
    second = f.assign(_p=p.where(p < top1, -1.0)).groupby("cand_id", observed=True)._p.transform("max").clip(lower=0)
    best_other = np.where((p == top1) & (cnt_top == 1), second, top1)
    f["t_gap_best_other"] = (p - best_other).astype(np.float32)
    f["t_rank"] = gt.rank(ascending=False, method="min").astype(np.int16)
    f["t_n_strong"] = (p >= STRONG).groupby(f.cand_id, observed=True).transform("sum").astype(np.int16)
    f["is_best_for_t"] = (p >= top1).astype(np.int8)
    f["x_name_addr"] = (f.n_tset * np.clip(f.a_tset, 0, 1)).astype(np.float32)
    f["x_exact_name_noaddr"] = ((f.n_concat_ratio >= 0.999) & (f.addr_empty == 1)).astype(np.int8)
    f["x_name_min_addr"] = np.minimum(f.n_tset, np.where(f.a_tset < 0, f.n_tset, f.a_tset)).astype(np.float32)
    if attrs is not None:
        _siblings(f, attrs)
    else:
        f["sib_name_max"] = f["sib_addr_max"] = np.float32(-1)
        f["sib_n"] = 0
    return f


def _siblings(f: pd.DataFrame, attrs: pd.DataFrame, max_sib: int = 4) -> None:
    """For every candidate: similarity to the strongest OTHER candidates of the same
    S1 entity (its likely group). True group members look alike; intruders don't."""
    strong = f.loc[f.p1 >= STRONG, ["s1_id", "cand_id", "p1"]]
    strong = strong.sort_values(["s1_id", "p1"], ascending=[True, False])
    strong = strong[strong.groupby("s1_id", observed=True).cumcount() < max_sib]
    live = f.loc[f.p1 >= 0.02, ["s1_id", "cand_id"]]  # near-zero pairs can't be rescued by siblings
    pairs = live.reset_index().merge(
        strong[["s1_id", "cand_id"]].rename(columns={"cand_id": "sib"}), on="s1_id")
    pairs = pairs[pairs.cand_id != pairs.sib]
    a = attrs.reindex(pairs.cand_id.astype(str).values)
    b = attrs.reindex(pairs.sib.astype(str).values)
    ns = process.cpdist(a.name_core.fillna("").tolist(), b.name_core.fillna("").tolist(),
                        scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    aa, ab = a.addr_clean.fillna("").to_numpy(), b.addr_clean.fillna("").to_numpy()
    asim = process.cpdist(aa.tolist(), ab.tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    asim = np.where((aa == "") | (ab == ""), -1, asim)
    agg = pd.DataFrame({"idx": pairs["index"].values, "n": ns, "a": asim}).groupby("idx").agg(
        sib_name_max=("n", "max"), sib_addr_max=("a", "max"), sib_n=("n", "size"))
    f["sib_name_max"] = agg.sib_name_max.reindex(f.index).fillna(-1).astype(np.float32).values
    f["sib_addr_max"] = agg.sib_addr_max.reindex(f.index).fillna(-1).astype(np.float32).values
    f["sib_n"] = agg.sib_n.reindex(f.index).fillna(0).astype(np.int16).values


def load_attrs(split: str, ids) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"entity_id": list(ids)}))
    return con.execute(f"SELECT entity_id, name_core, addr_clean FROM '{W(split + '_norm.parquet')}' "
                       f"SEMI JOIN ids USING (entity_id)").df().set_index("entity_id")


def train(frac: float = 0.1, k: int = 3) -> dict:
    from erharness.analysis import oracles
    from erharness.splits import kfold
    from erharness.sweep import Scorer, best_config, grid_search

    t0 = time.time()
    gt = load_gt_sets(sample_ids(frac))
    have = set(pq.ParquetDataset(W("train_feats")).schema.names)
    f = pq.read_table(W("train_feats"), columns=["s1_id", "cand_id", *[c for c in FEATURES if c in have]]).to_pandas()
    for c in FEATURES:          # features built by an older blocking version lack newer key flags
        if c not in f:
            f[c] = np.int8(0)
    oof = pd.read_parquet(W("train_oof.parquet"), columns=["s1_id", "cand_id", "prob_raw"])
    # train_oof rows are in the same order as the (sampled) train_feats rows it was built from
    base = f.s1_id.isin(load_gt_sets(sample_ids(0.1)) if frac < 0.1 else gt).to_numpy()
    f = f[base].reset_index(drop=True)
    assert len(f) == len(oof) and (f.cand_id.values == oof.cand_id.values).all(), "OOF/feature row mismatch"
    f["p1"] = oof.prob_raw.to_numpy(np.float32)
    del oof
    f = f[f.s1_id.isin(gt).to_numpy()].reset_index(drop=True)
    f["s1_id"] = f.s1_id.astype("category")
    f["cand_id"] = f.cand_id.astype("category")
    attrs = load_attrs("train", set(f.cand_id.cat.categories))
    f = add_context(f, attrs)
    del attrs
    tp = {(s, c) for s, cs in gt.items() for c in cs}
    y = np.fromiter(((s, c) in tp for s, c in zip(f.s1_id.astype(str), f.cand_id.astype(str))), bool, len(f))
    fold = f.s1_id.astype(str).map(kfold(gt, k, seed=7)).to_numpy()
    print(f"stage2: {len(f):,} pairs, context built ({time.time() - t0:.0f}s)", flush=True)
    oof2 = np.zeros(len(f))
    imps = []
    for i in range(k):
        tr, va = fold != i, fold == i
        m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, FEATURES2], y[tr]), 1500,
                      valid_sets=[lgb.Dataset(f.loc[va, FEATURES2], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof2[va] = m.predict(f.loc[va, FEATURES2], num_iteration=m.best_iteration)
        imps.append(pd.Series(m.feature_importance("gain"), FEATURES2))
        print(f"  fold {i}: best_iter={m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    c = f[["s1_id", "cand_id"]].astype(str)
    out = {}
    for name, pr in [("stage1", f.p1.to_numpy()), ("stage2", oof2)]:
        c["prob_raw"] = pr
        c["prob"] = assign_targets(c, "prob_raw")
        sc = Scorer(c[["s1_id", "cand_id", "prob"]], gt)
        res = grid_search(sc, {"gate": [0.1, 0.2, 0.3, 0.4, 0.5], "pair": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
        best = best_config(res)
        out[name] = {"config": best.as_dict(), "score": float(res.iloc[0].score)}
        print(f"[{name}] gate={best.gate} pair={best.pair}  macro F0.5 = {res.iloc[0].score:.5f}", flush=True)
        if name == "stage2":
            print(oracles(sc, best).to_string(index=False))
            c[["s1_id", "cand_id", "prob_raw", "prob"]].to_parquet(W("train_oof2.parquet"), index=False)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print("top stage-2 features:\n" + imp.head(12).round(0).to_string(), flush=True)
    json.dump({"prob_col": "prob", "decode": out["stage2"]["config"], "local": out},
              open(W("decode2.json"), "w"), indent=1, default=float)
    X = f[FEATURES2].to_numpy(np.float32)
    rounds = max(100, int(np.median([m.best_iteration]) * 1.1))
    lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATURES2), rounds).save_model(W("model2.txt"))
    print(f"saved model2.txt ({rounds} rounds) in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(float(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0.1)
