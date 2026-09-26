"""Same-sample A/B of feature sets: 3-fold OOF (split by S1), decode sweep, macro F0.5 overall + per country.
    python scripts/ablate.py 0.05"""
import sys, time
sys.path[:0] = ["src", "."]
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq, duckdb
from amlc.features import FEATURES
from amlc.model import PARAMS, W, assign_targets, load_gt_sets, sample_ids
from erharness.splits import kfold
from erharness.sweep import Scorer, best_config, grid_search

NEW = ["n_phon", "nf_s1_a", "nf_s1_b", "nf_s1_lg_b", "nf_t_b", "addr_empty_s1", "addr_empty_t", "house_near",
       "k_phonhs", "k_phonat"]
frac = float(sys.argv[1])
t0 = time.time()
gt = load_gt_sets(sample_ids(frac))
f = pq.read_table(W("train_feats"), columns=["s1_id", "cand_id", *FEATURES]).to_pandas()
tp = {(s, c) for s, cs in gt.items() for c in cs}
y = np.fromiter(((s, c) in tp for s, c in zip(f.s1_id, f.cand_id)), bool, len(f))
fold = f.s1_id.map(kfold(gt, 3, seed=7)).to_numpy()
cty = duckdb.connect().execute("select entity_id, country from 'data/parquet/train_source1.parquet'").df().set_index("entity_id").country
print(f"{len(f):,} pairs, {y.mean():.4f} pos, {len(gt):,} S1 ({time.time()-t0:.0f}s)", flush=True)
arms = {"v2 features": [c for c in FEATURES if c not in NEW], "v3 features": FEATURES}
for name, cols in arms.items():
    oof = np.zeros(len(f))
    for i in range(3):
        tr, va = fold != i, fold == i
        m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, cols], y[tr]), 2000,
                      valid_sets=[lgb.Dataset(f.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(f.loc[va, cols], num_iteration=m.best_iteration)
    c = f[["s1_id", "cand_id"]].copy(); c["prob_raw"] = oof; c["prob"] = assign_targets(c, "prob_raw")
    res = grid_search(Scorer(c[["s1_id", "cand_id", "prob"]], gt),
                      {"gate": [0.1, 0.2, 0.3, 0.4, 0.5], "pair": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    b = best_config(res)
    per = {}
    for k in ("US", "India"):
        g = {s: v for s, v in gt.items() if cty.get(s) == k}
        per[k] = Scorer(c.loc[c.s1_id.isin(g.keys()), ["s1_id", "cand_id", "prob"]], g).score(b)["score"]
    print(f"[{name}] macro F0.5 {res.iloc[0].score:.5f}  US {per['US']:.5f}  India {per['India']:.5f}  "
          f"gate={b.gate} pair={b.pair} iters={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    if name == "v3 features":
        imp = pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False)
        print("new-feature gain rank:", {c: int(imp.index.get_loc(c)) + 1 for c in NEW})
