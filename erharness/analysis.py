"""Where is the score being lost? Blocking audit, oracle ceilings, error buckets."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .decode import DecodeConfig
from .io import MatchSets, label_candidates
from .metric import f05_vec
from .sweep import Scorer


# ---------------------------------------------------------------- blocking --
def blocking_audit(cands: pd.DataFrame, gt: MatchSets, n_targets: Optional[int] = None,
                   key_col: str = "key", key_sep: str = "|") -> Dict:
    """Recall/cost of a candidate set, overall and per blocking key.

    `cands` needs s1_id, cand_id; optional `key` column listing which blocking
    key(s) produced the pair (e.g. "pin3_name|phonetic"). `n_targets` =
    |S2|+|S3|, used for the reduction ratio vs. the full cross product.
    """
    c = cands[["s1_id", "cand_id"] + ([key_col] if key_col in cands.columns else [])]
    c = c.groupby(["s1_id", "cand_id"], as_index=False).agg(
        {key_col: lambda s: key_sep.join(sorted(set(key_sep.join(s).split(key_sep))))}
    ) if key_col in c.columns else c.drop_duplicates(["s1_id", "cand_id"])
    c = label_candidates(c, gt)

    ents = list(gt)
    t = pd.Series({e: len(gt[e]) for e in ents})
    per_ent = c.groupby("s1_id").agg(n_cand=("cand_id", "size"), hits=("label", "sum"))
    per_ent = per_ent.reindex(ents, fill_value=0)
    matched = t > 0
    n_true_pairs = int(t.sum())
    found = int(c.label.sum())
    out = {
        "n_candidate_pairs": int(len(c)),
        "cands_per_entity": {
            "mean": float(per_ent.n_cand.mean()),
            "p50": float(per_ent.n_cand.quantile(0.5)),
            "p95": float(per_ent.n_cand.quantile(0.95)),
            "max": int(per_ent.n_cand.max()),
            "zero_cand_entities": int((per_ent.n_cand == 0).sum()),
        },
        "pair_recall": found / n_true_pairs if n_true_pairs else float("nan"),
        "entity_full_recall": float((per_ent.hits[matched] == t[matched]).mean()) if matched.any() else float("nan"),
        "entity_any_hit": float((per_ent.hits[matched] > 0).mean()) if matched.any() else float("nan"),
        "pair_precision": found / len(c) if len(c) else float("nan"),
        # perfect matcher on these candidates, perfect gate: the ceiling blocking allows
        "macro_f05_ceiling": float(f05_vec(per_ent.hits, per_ent.hits, t).mean()),
    }
    if n_targets:
        out["reduction_ratio"] = 1 - len(c) / (len(ents) * n_targets)

    if key_col in c.columns:
        exploded = c.assign(_k=c[key_col].str.split(key_sep)).explode("_k")
        n_keys_per_pair = exploded.groupby(["s1_id", "cand_id"])._k.transform("nunique")
        exploded["_unique"] = n_keys_per_pair == 1
        rows = []
        for k, g in exploded.groupby("_k"):
            rows.append({
                "key": k,
                "pairs": len(g),
                "pair_recall": g.label.sum() / n_true_pairs if n_true_pairs else np.nan,
                "precision": g.label.mean(),
                "unique_true_pairs": int((g.label & g._unique).sum()),
            })
        out["per_key"] = pd.DataFrame(rows).sort_values("pair_recall", ascending=False).reset_index(drop=True)
        out["greedy_key_order"] = _greedy_keys(exploded, n_true_pairs)
    true_pairs = pd.DataFrame([(e, x) for e in ents for x in gt[e]], columns=["s1_id", "cand_id"])
    missed = true_pairs.merge(c[["s1_id", "cand_id"]], how="left", indicator=True)
    out["missed_true_pairs"] = missed[missed._merge == "left_only"].drop(columns="_merge").reset_index(drop=True)
    return out


def _greedy_keys(exploded: pd.DataFrame, n_true: int) -> pd.DataFrame:
    """Forward selection: which key adds the most recall next, and at what cost."""
    pairs_by_key = {k: set(zip(g.s1_id, g.cand_id)) for k, g in exploded.groupby("_k")}
    true_by_key = {k: set(zip(g.s1_id[g.label], g.cand_id[g.label])) for k, g in exploded.groupby("_k")}
    chosen, got_true, got_pairs, rows = [], set(), set(), []
    remaining = set(pairs_by_key)
    while remaining:
        best = max(remaining, key=lambda k: (len(true_by_key[k] - got_true), -len(pairs_by_key[k] - got_pairs)))
        got_true |= true_by_key[best]
        got_pairs |= pairs_by_key[best]
        remaining.remove(best)
        chosen.append(best)
        rows.append({"add_key": best, "cum_pair_recall": len(got_true) / n_true if n_true else np.nan,
                     "cum_pairs": len(got_pairs)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- oracles --
def oracles(scorer: Scorer, cfg: DecodeConfig) -> pd.DataFrame:
    """Counterfactual ceilings. The biggest gap to `current` is where to work."""
    p = scorer.prep
    n, t = p.n_entities, scorer.t
    lab = p.label

    def counts(mask):
        k = np.bincount(p.ent[mask], minlength=n)
        h = np.bincount(p.ent[mask], weights=lab[mask].astype(float), minlength=n)
        return h, k

    h, k = counts(p.mask(cfg))
    cur = f05_vec(h, k, t)

    # 1. perfect gate: singletons empty; matched entities forced open with your
    #    set rule (gate removed), falling back to your top-1 if nothing passes
    open_cfg = DecodeConfig(**{**cfg.as_dict(), "gate": -np.inf}) if cfg.mode == "threshold" else cfg
    ho, ko = counts(p.mask(open_cfg))
    h1, k1 = counts(p.rank == 0)
    use_top1 = ko == 0
    hg = np.where(t == 0, 0, np.where(use_top1, h1, ho))
    kg = np.where(t == 0, 0, np.where(use_top1, k1, ko))
    gate_oracle = f05_vec(hg, kg, t)

    # 2. perfect sets given YOUR gate: when you open a matched entity, you output
    #    exactly its true matches that survived blocking
    hb, _ = counts(lab.astype(bool))
    opened = k > 0
    hs = np.where(opened, hb, 0)
    ks = np.where(opened & (t > 0), hb, k)       # opened singletons still pay
    set_oracle = f05_vec(hs, ks, t)

    # 3. perfect everything on these candidates = blocking ceiling
    block_oracle = f05_vec(hb, hb, t)

    rows = [
        ("current", cur.mean()),
        ("perfect gate (your matcher)", gate_oracle.mean()),
        ("perfect sets (your gate)", set_oracle.mean()),
        ("perfect gate + sets = blocking ceiling", block_oracle.mean()),
        ("perfect blocking too", 1.0),
    ]
    df = pd.DataFrame(rows, columns=["scenario", "macro_f05"])
    df["gain_vs_current"] = df.macro_f05 - cur.mean()
    return df


# ----------------------------------------------------------- error buckets --
def error_pairs(scorer: Scorer, cfg: DecodeConfig, gt: MatchSets) -> Dict[str, pd.DataFrame]:
    """Pair-level errors, split by the stage that caused them:
      blocking_miss : true pair never became a candidate  -> fix blocking keys
      model_miss    : true pair was a candidate, not kept  -> fix features/threshold/gate
      false_merge_singleton : predicted pair for an entity with no true matches
      false_merge_matched   : wrong extra pair for an entity that has matches
    Each table is sorted so the most confident mistakes come first."""
    p = scorer.prep
    df = p.df.assign(kept=p.mask(cfg), n_true=scorer.t[p.ent])
    true_pairs = pd.DataFrame([(e, x) for e in gt for x in gt[e]], columns=["s1_id", "cand_id"])
    in_c = true_pairs.merge(df[["s1_id", "cand_id"]], how="left", indicator=True)
    out = {
        "blocking_miss": in_c[in_c._merge == "left_only"].drop(columns="_merge").reset_index(drop=True),
        "model_miss": df[df.label & ~df.kept].sort_values("prob", ascending=False).reset_index(drop=True),
        "false_merge_singleton": df[~df.label & df.kept & (df.n_true == 0)].sort_values("prob", ascending=False).reset_index(drop=True),
        "false_merge_matched": df[~df.label & df.kept & (df.n_true > 0)].sort_values("prob", ascending=False).reset_index(drop=True),
    }
    return out


def calibrate_oof(cands: pd.DataFrame, fold: dict, method: str = "isotonic") -> pd.Series:
    """Cross-fitted recalibration of `prob` (needs `label`): each fold's
    calibrator is fit on the other folds, so the output is still out-of-fold.
    Apply the calibrator fit on ALL train pairs to test probabilities."""
    out = pd.Series(np.nan, index=cands.index)
    f = cands.s1_id.map(fold)
    for k in sorted(f.dropna().unique()):
        tr, va = f != k, f == k
        cal = fit_calibrator(cands.prob[tr].to_numpy(), cands.label[tr].to_numpy(), method)
        out[va] = cal(cands.prob[va].to_numpy())
    return out


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str = "isotonic"):
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return m.predict
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        m = LogisticRegression().fit(z[:, None], y)
        return lambda q: m.predict_proba(np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))[:, None])[:, 1]
    raise ValueError(method)


def calibration_table(cands: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted prob vs observed match rate per bin.
    Needed before trusting expected_f decoding or a fixed gate threshold."""
    if "label" not in cands.columns:
        raise ValueError("label candidates first (io.label_candidates)")
    b = pd.cut(cands.prob, np.linspace(0, 1, bins + 1), include_lowest=True)
    g = cands.groupby(b, observed=True).agg(n=("label", "size"), mean_prob=("prob", "mean"), match_rate=("label", "mean"))
    g["gap"] = g.match_rate - g.mean_prob
    ece = float((g.n * g.gap.abs()).sum() / g.n.sum())
    g.attrs["ece"] = ece
    return g.reset_index().rename(columns={"prob": "bin"})
