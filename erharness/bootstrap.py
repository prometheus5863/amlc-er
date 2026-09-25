"""Is a change real or noise? Paired bootstrap over S1 entities.

Both systems are scored on the same resampled entity sets, so shared
difficulty cancels out and the test is far more sensitive than comparing two
independent confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Comparison:
    score_a: float
    score_b: float
    delta: float            # b - a
    ci_low: float           # 2.5th percentile of delta
    ci_high: float          # 97.5th percentile of delta
    win_rate: float         # share of resamples where b > a
    n_changed: int          # entities whose score differs
    n_better: int
    n_worse: int
    verdict: str

    def summary(self) -> str:
        return (
            f"A={self.score_a:.5f}  B={self.score_b:.5f}  delta={self.delta:+.5f}  "
            f"95% CI [{self.ci_low:+.5f}, {self.ci_high:+.5f}]  P(B>A)={self.win_rate:.3f}\n"
            f"entities changed={self.n_changed} (better {self.n_better}, worse {self.n_worse})  "
            f"-> {self.verdict}"
        )


def _aligned(a: pd.DataFrame, b: pd.DataFrame):
    m = a[["s1_id", "score"]].merge(b[["s1_id", "score"]], on="s1_id", suffixes=("_a", "_b"), validate="1:1")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("per-entity tables cover different entities")
    return m.score_a.to_numpy(), m.score_b.to_numpy()


def compare(table_a: pd.DataFrame, table_b: pd.DataFrame, n_boot: int = 2000,
            seed: int = 0, win_threshold: float = 0.9) -> Comparison:
    """Paired bootstrap on two `metric.per_entity` tables (B is the candidate)."""
    sa, sb = _aligned(table_a, table_b)
    d = sb - sa
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    # chunk to bound memory on large n
    chunk = max(1, int(2e7 // max(n, 1)))
    i = 0
    while i < n_boot:
        m = min(chunk, n_boot - i)
        idx = rng.integers(0, n, size=(m, n))
        boots[i:i + m] = d[idx].mean(axis=1)
        i += m
    # ties (e.g. identical systems, or a change touching few entities) count half
    win = float((boots > 0).mean() + 0.5 * (boots == 0).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if not (d != 0).any():
        verdict = "NO CHANGE (identical per-entity scores)"
    elif win >= win_threshold:
        verdict = "B WINS (keep it)"
    elif win <= 1 - win_threshold:
        verdict = "B LOSES (revert)"
    else:
        verdict = "NOISE (not distinguishable; don't spend a submission on it)"
    return Comparison(
        score_a=float(sa.mean()), score_b=float(sb.mean()), delta=float(d.mean()),
        ci_low=float(lo), ci_high=float(hi), win_rate=win,
        n_changed=int((d != 0).sum()), n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()),
        verdict=verdict,
    )


def ci(table: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple:
    """95% bootstrap CI of a single system's macro score (its noise floor)."""
    s = table.score.to_numpy()
    rng = np.random.default_rng(seed)
    boots = s[rng.integers(0, len(s), size=(n_boot, len(s)))].mean(axis=1) if len(s) * n_boot <= 5e7 else \
        np.array([s[rng.integers(0, len(s), len(s))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(s.mean()), float(lo), float(hi)
