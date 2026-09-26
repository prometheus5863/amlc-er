"""Where does macro F0.5 go? Per-S1 error buckets on out-of-fold predictions + pair-level causes.
    python scripts/error_analysis.py   (uses data/work/train_oof.parquet, same decode as the pipeline)"""
import sys; sys.path[:0] = ["src", "."]
import numpy as np, pandas as pd, duckdb
from amlc.model import load_gt_sets, assign_targets

GATE, PAIR = 0.3, 0.7
o = pd.read_parquet("data/work/train_oof.parquet", columns=["s1_id", "cand_id", "prob_raw"])
gt = load_gt_sets(set(o.s1_id.unique()))
o["prob"] = assign_targets(o, "prob_raw")
mx = o.groupby("s1_id").prob.transform("max")
o["pred"] = (o.prob >= PAIR) & (mx >= GATE)
tp = {(s, c) for s, cs in gt.items() for c in cs}
o["y"] = np.fromiter(((s, c) in tp for s, c in zip(o.s1_id, o.cand_id)), bool, len(o))

# per-S1 counts
g = o.groupby("s1_id").agg(k=("pred", "sum"), h=("pred", lambda x: 0), )
k = o[o.pred].groupby("s1_id").size()
h = o[o.pred & o.y].groupby("s1_id").size()
inc = o[o.y].groupby("s1_id").size()                       # true pairs that reached the candidates
S = pd.DataFrame({"t": pd.Series({s: len(v) for s, v in gt.items()})})
S["k"] = k.reindex(S.index).fillna(0); S["h"] = h.reindex(S.index).fillna(0); S["inc"] = inc.reindex(S.index).fillna(0)
P = np.where(S.k > 0, S.h / S.k.clip(lower=1), 1.0); R = np.where(S.t > 0, S.h / S.t.clip(lower=1), 1.0)
F = np.where((S.t == 0) & (S.k == 0), 1.0, np.where(S.h == 0, 0.0, 1.25 * P * R / (0.25 * P + R)))
S["f"] = F; S["loss"] = 1 - F
fp = S.k - S.h; fn_block = S.t - S.inc; fn_model = S.inc - S.h
def bucket(r, fpv, fbv, fmv):
    if r.loss == 0: return "perfect"
    if r.t == 0: return "singleton but predicted matches"
    if r.k == 0: return "matched business, predicted EMPTY"
    parts = []
    if fbv: parts.append("blocking miss")
    if fmv: parts.append("rejected in candidates")
    if fpv: parts.append("false positive")
    return " + ".join(parts)
S["bucket"] = [bucket(r, a, b, c) for r, a, b, c in zip(S.itertuples(), fp, fn_block, fn_model)]
N = len(S)
tab = S.groupby("bucket").agg(businesses=("f", "size"), loss=("loss", "sum")).sort_values("loss", ascending=False)
tab["share_of_businesses"] = (tab.businesses / N).round(4); tab["points_lost"] = (tab.loss / N).round(5)
print(f"{N:,} businesses, macro F0.5 = {S.f.mean():.5f}  (points lost {1 - S.f.mean():.5f})\n")
print(tab[["businesses", "share_of_businesses", "points_lost"]].to_string())

# pair-level: characterize FNs (rejected) and FPs
con = duckdb.connect()
ids = pd.unique(np.concatenate([o.s1_id[o.y ^ o.pred], o.cand_id[o.y ^ o.pred]]))
con.register("ids", pd.DataFrame({"entity_id": ids}))
a = con.execute("""select n.entity_id, n.country, n.name_concat, n.addr_clean, n.house_no, n.is_translit, n.legal
                   from 'data/work/train_norm.parquet' n semi join ids using (entity_id)""").df().set_index("entity_id")
nf = con.execute("""select country, name_concat, count(*) filter (where src='S1') s1n from 'data/work/train_norm.parquet' group by 1,2""").df().set_index(["country","name_concat"]).s1n
def describe(df, label):
    A = a.reindex(df.s1_id.values).reset_index(drop=True); B = a.reindex(df.cand_id.values).reset_index(drop=True)
    d = pd.DataFrame({
        "country": A.country, "same_name": A.name_concat.values == B.name_concat.values,
        "target_addr_empty": B.addr_clean.eq("").values, "script": (A.is_translit | B.is_translit).values,
        "same_house": (A.house_no.values == B.house_no.values) & (A.house_no.values != ""),
        "name_shared_by_>1_S1": nf.reindex(list(zip(B.country, B.name_concat))).fillna(0).values > 1,
        "p": df.prob_raw.values})
    print(f"\n== {label}: {len(d):,} pairs ==")
    print(d.drop(columns=["country", "p"]).mean().round(3).to_string())
    print("country:", d.country.value_counts(normalize=True).round(3).to_dict())
    print("prob quantiles 10/50/90:", np.percentile(d.p, [10, 50, 90]).round(3).tolist())
    return d
fnp = describe(o[o.y & ~o.pred], "TRUE pairs we rejected")
fpp = describe(o[~o.y & o.pred], "FALSE pairs we kept")
# a few examples of each, raw text
raw = con.execute("""select entity_id, business_name, business_address from (
   select * from 'data/parquet/train_source1.parquet' union all select * from 'data/parquet/train_source2.parquet'
   union all select * from 'data/parquet/train_source3.parquet') semi join ids using (entity_id)""").df().set_index("entity_id")
for lab, df in [("REJECTED true", o[o.y & ~o.pred]), ("KEPT false", o[~o.y & o.pred])]:
    print(f"\n-- examples: {lab} --")
    for r in df.sample(8, random_state=3).itertuples():
        x, y = raw.loc[r.s1_id], raw.loc[r.cand_id]
        print(f"p={r.prob_raw:.2f} | {x.business_name[:40]!r} @ {x.business_address[:50]!r}\n          | {y.business_name[:40]!r} @ {y.business_address[:50]!r}")
S.to_parquet("data/work/error_buckets.parquet")
