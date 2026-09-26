"""Word-level edit fingerprints of the data generator (v6).

The data is synthetic. When the target's name has a word the S1 name lacks, WHICH word it is
decides a lot (contested training pairs):
    added 'holdings' / 'group' / 'overseas' / 'public'  -> 0% same business (generated look-alikes)
    added 'dba' / 'doing business as' / 'nee'            -> 98-99% same business (aliases)
Each pair gets the smoothed log-odds of its added / removed words, learned from labelled
training pairs. Training rows get OUT-OF-FOLD values (stats from the other folds only), so
the model never sees a word statistic computed from its own label; test rows use statistics
from all training pairs. Same-sample A/B (5%, 3-fold): 0.96541 -> 0.97125.
"""
from __future__ import annotations

import json
import time

import duckdb
import numpy as np
import pandas as pd

TOK_FEATURES = ["tok_add_min", "tok_add_max", "tok_rem_min", "tok_n_add", "tok_n_rem"]
K = 20  # smoothing strength (pseudo-count toward the prior)


def _W(name):
    from .model import W
    return W(name)


def _names(split: str, ids) -> pd.Series:
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"entity_id": pd.unique(np.asarray(ids))}))
    d = con.execute(f"SELECT entity_id, name_clean FROM '{_W(split + '_norm.parquet')}' "
                    f"SEMI JOIN ids USING (entity_id)").df()
    return d.set_index("entity_id").name_clean


def diff_table(s1_ids, cand_ids, names: pd.Series) -> pd.DataFrame:
    """One row per (pair row r, kind k, word t): k=0 word only in the target, k=1 word only in S1."""
    na = names.reindex(np.asarray(s1_ids)).fillna("").to_numpy()
    nb = names.reindex(np.asarray(cand_ids)).fillna("").to_numpy()
    rows, toks, kind = [], [], []
    for i, (x, y) in enumerate(zip(na, nb)):
        if x == y:
            continue
        sx, sy = set(x.split()), set(y.split())
        for t in sy - sx:
            rows.append(i); toks.append(t); kind.append(0)
        for t in sx - sy:
            rows.append(i); toks.append(t); kind.append(1)
    return pd.DataFrame({"r": np.asarray(rows, np.int64), "k": np.asarray(kind, np.int8),
                         "t": pd.Categorical(toks)})


def _fit(T: pd.DataFrame, y: np.ndarray, prior: float) -> pd.Series:
    yy = y[T.r.to_numpy()]
    st = pd.DataFrame({"k": T.k.to_numpy(), "t": T.t, "y": yy}).groupby(["k", "t"], observed=True).y.agg(["size", "sum"])
    return np.log((st["sum"] + K * prior) / (st["size"] - st["sum"] + K * (1 - prior))).rename("lo")


def _apply(f: pd.DataFrame, T: pd.DataFrame, lo: pd.Series, prior: float, rows=None) -> None:
    d0 = float(np.log(prior / (1 - prior)))
    V = T if rows is None else T[np.isin(T.r.to_numpy(), rows)]
    V = V.assign(t=V.t.astype(str)).join(lo, on=["k", "t"])
    V["lo"] = V["lo"].fillna(d0)
    for col, k, agg in (("tok_add_min", 0, "min"), ("tok_add_max", 0, "max"), ("tok_rem_min", 1, "min")):
        g = V[V.k == k].groupby("r").lo.agg(agg)
        f.loc[g.index, col] = g.to_numpy()


def _init(f: pd.DataFrame, T: pd.DataFrame, prior: float) -> None:
    d0 = float(np.log(prior / (1 - prior)))
    for c in ("tok_add_min", "tok_add_max", "tok_rem_min"):
        f[c] = np.full(len(f), d0, np.float64)
    f["tok_n_add"] = np.bincount(T.r[T.k == 0], minlength=len(f)).astype(np.int16)
    f["tok_n_rem"] = np.bincount(T.r[T.k == 1], minlength=len(f)).astype(np.int16)


def add_train(f: pd.DataFrame, y: np.ndarray, fold: np.ndarray) -> None:
    """Out-of-fold word log-odds for training rows; saves all-train statistics for test."""
    t0 = time.time()
    T = diff_table(f.s1_id, f.cand_id, _names("train", np.concatenate([f.s1_id.unique(), f.cand_id.unique()])))
    prior = float(y.mean())
    _init(f, T, prior)
    T_fold = fold[T.r.to_numpy()]
    for i in np.unique(fold):
        lo = _fit(T[T_fold != i], y, prior)
        lo.index = lo.index.set_levels(lo.index.levels[1].astype(str), level=1)
        _apply(f, T, lo, prior, rows=np.flatnonzero(fold == i))
    lo_all = _fit(T, y, prior)
    x = lo_all.reset_index()
    x["t"] = x["t"].astype(str)
    x.to_parquet(_W("tokstats.parquet"), index=False)
    json.dump({"prior": prior}, open(_W("tokstats.json"), "w"))
    print(f"word-diff features: {len(T):,} diff words, {len(lo_all):,} word stats ({time.time() - t0:.0f}s)", flush=True)


def add_test(f: pd.DataFrame, split: str = "test", names: pd.Series | None = None) -> None:
    prior = json.load(open(_W("tokstats.json")))["prior"]
    lo = pd.read_parquet(_W("tokstats.parquet")).set_index(["k", "t"]).lo
    if names is None:
        names = _names(split, np.concatenate([f.s1_id.unique(), f.cand_id.unique()]))
    T = diff_table(f.s1_id, f.cand_id, names)
    _init(f, T, prior)
    _apply(f, T, lo, prior)
