"""The competition metric, reproduced locally, plus its decomposition.

Per Source-1 entity e with true set T_e and predicted set P_e:
  * T_e empty (singleton):  score = 1 if P_e empty else 0
  * T_e non-empty:          score = F0.5(P_e, T_e)  (0 if P_e empty or no hit)
Final score = unweighted mean over all S1 entities in the ground truth.

With h = |P∩T|, k = |P|, t = |T| (k, t > 0):
  F0.5 = 1.25·P·R / (0.25·P + R) = 1.25·h / (k + 0.25·t)
That closed form lets the sweep code score thousands of configs vectorised.

Decomposition (s = singleton fraction):
  score = s · spec  +  (1 − s) · open_rate · f_open
  spec      = share of singletons predicted empty            (gate on singletons)
  open_rate = share of matched entities with a non-empty set (gate on matched)
  f_open    = mean F0.5 over matched entities that were opened (set quality)
Because the three parts are separate, `project_score` can re-weight a local
score to the singleton rate of the test set (learned from an all-empty submit).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .io import MatchSets

BETA2 = 0.25  # beta = 0.5


def f05(h: int, k: int, t: int) -> float:
    """Per-entity score from counts (hits, predicted, true)."""
    if t == 0:
        return 1.0 if k == 0 else 0.0
    if k == 0 or h == 0:
        return 0.0
    return (1 + BETA2) * h / (k + BETA2 * t)


def f05_vec(h: np.ndarray, k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorised `f05` over arrays of counts."""
    h, k, t = (np.asarray(a, dtype=float) for a in (h, k, t))
    out = np.zeros_like(h)
    single = t == 0
    out[single] = (k[single] == 0).astype(float)
    m = (~single) & (k > 0)
    out[m] = (1 + BETA2) * h[m] / (k[m] + BETA2 * t[m])
    return out


def per_entity(pred: MatchSets, gt: MatchSets, strict: bool = False) -> pd.DataFrame:
    """One row per GT entity: counts, score and an error-type tag.

    Entities missing from `pred` are treated as empty predictions (and warned
    about, or raised when strict=True — the leaderboard may reject such files).
    """
    missing = [e for e in gt if e not in pred]
    extra = [e for e in pred if e not in gt]
    if missing:
        msg = f"{len(missing)} GT entities missing from prediction (scored as empty), e.g. {missing[:3]}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    if extra:
        warnings.warn(f"{len(extra)} predicted ids not in GT were ignored, e.g. {extra[:3]}")

    ids = list(gt)
    h = np.empty(len(ids), dtype=int)
    k = np.empty(len(ids), dtype=int)
    t = np.empty(len(ids), dtype=int)
    for i, e in enumerate(ids):
        T = gt[e]
        P = pred.get(e, set())
        h[i], k[i], t[i] = len(P & T), len(P), len(T)
    df = pd.DataFrame({"s1_id": ids, "hits": h, "n_pred": k, "n_true": t})
    df["score"] = f05_vec(h, k, t)
    df["fp"] = df.n_pred - df.hits
    df["fn"] = df.n_true - df.hits
    df["outcome"] = np.select(
        [
            (df.n_true == 0) & (df.n_pred == 0),
            (df.n_true == 0) & (df.n_pred > 0),
            (df.n_true > 0) & (df.n_pred == 0),
            (df.n_true > 0) & (df.hits == df.n_true) & (df.fp == 0),
        ],
        ["singleton_ok", "singleton_false_merge", "gated_shut", "exact"],
        default="partial",
    )
    return df


@dataclass
class Report:
    score: float
    n: int
    singleton_frac: float
    spec: float                   # singletons correctly left empty
    open_rate: float              # matched entities given a non-empty set
    f_open: float                 # mean F0.5 on opened matched entities
    f_matched: float              # mean F0.5 on all matched entities
    pair_precision: float         # micro, over all predicted pairs
    pair_recall: float            # micro, over all true pairs
    exact_rate: float             # matched entities predicted perfectly
    outcomes: Dict[str, int] = field(default_factory=dict)
    by_n_true: Dict[str, float] = field(default_factory=dict)
    by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"macro F0.5        {self.score:.5f}   (n={self.n})",
            f"  singleton frac  {self.singleton_frac:.4f}",
            f"  spec (singletons left empty)     {self.spec:.4f}",
            f"  open rate (matched given a set)  {self.open_rate:.4f}",
            f"  F0.5 on opened matched           {self.f_open:.4f}",
            f"  F0.5 on all matched              {self.f_matched:.4f}",
            f"  pair precision / recall          {self.pair_precision:.4f} / {self.pair_recall:.4f}",
            f"  exact-set rate on matched        {self.exact_rate:.4f}",
            "  outcomes: " + ", ".join(f"{k}={v}" for k, v in self.outcomes.items()),
            "  by n_true: " + ", ".join(f"{k}:{v:.4f}" for k, v in self.by_n_true.items()),
        ]
        if self.by_group:
            lines.append("  by group (n, score, share of total loss):")
            for g, d in sorted(self.by_group.items(), key=lambda kv: -kv[1]["loss_share"]):
                lines.append(f"    {g:<20} n={int(d['n']):>6}  score={d['score']:.4f}  loss_share={d['loss_share']:.3f}")
        return "\n".join(lines)


def _bucket(t: int) -> str:
    return "0" if t == 0 else ("1" if t == 1 else ("2" if t == 2 else "3+"))


def report_from_table(df: pd.DataFrame, groups: Optional[Mapping[str, str]] = None) -> Report:
    single = df.n_true == 0
    matched = ~single
    opened = matched & (df.n_pred > 0)
    s = float(single.mean())
    rep = Report(
        score=float(df.score.mean()),
        n=len(df),
        singleton_frac=s,
        spec=float((df.n_pred[single] == 0).mean()) if single.any() else float("nan"),
        open_rate=float(opened.sum() / matched.sum()) if matched.any() else float("nan"),
        f_open=float(df.score[opened].mean()) if opened.any() else 0.0,
        f_matched=float(df.score[matched].mean()) if matched.any() else float("nan"),
        pair_precision=float(df.hits.sum() / df.n_pred.sum()) if df.n_pred.sum() else float("nan"),
        pair_recall=float(df.hits.sum() / df.n_true.sum()) if df.n_true.sum() else float("nan"),
        exact_rate=float((df.outcome[matched] == "exact").mean()) if matched.any() else float("nan"),
        outcomes={k: int(v) for k, v in df.outcome.value_counts().items()},
        by_n_true={b: float(g.score.mean()) for b, g in df.groupby(df.n_true.map(_bucket))},
    )
    if groups:
        g = df.s1_id.map(groups).fillna("<none>")
        total_loss = float((1 - df.score).sum()) or 1.0
        rep.by_group = {
            name: {
                "n": float(len(sub)),
                "score": float(sub.score.mean()),
                "loss_share": float((1 - sub.score).sum() / total_loss),
            }
            for name, sub in df.groupby(g)
        }
    return rep


def evaluate(pred: MatchSets, gt: MatchSets, groups: Optional[Mapping[str, str]] = None,
             strict: bool = False) -> Report:
    return report_from_table(per_entity(pred, gt, strict=strict), groups)


def project_score(rep: Report, test_singleton_frac: float) -> float:
    """Re-weight a local report to a different singleton rate (e.g. the test
    rate measured by an all-empty submission)."""
    s = test_singleton_frac
    return s * rep.spec + (1 - s) * rep.open_rate * rep.f_open
