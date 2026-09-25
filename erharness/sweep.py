"""Vectorised threshold search against the real metric.

Use OUT-OF-FOLD candidate probabilities (see splits.py): thresholds tuned on
probabilities the model was trained on are biased toward over-confidence.

Picking rule: the raw arg-max of a grid is noisy. `robust=True` picks the
config with the best *neighbourhood-average* score (itself + adjacent grid
points on every axis), which lands in the middle of a plateau and transfers
better to the hidden test set.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .decode import DecodeConfig, Prepared
from .io import MatchSets, label_candidates
from .metric import f05_vec


class Scorer:
    """Scores decode configs without materialising match sets."""

    def __init__(self, cands: pd.DataFrame, gt: MatchSets):
        if "label" not in cands.columns:
            cands = label_candidates(cands, gt)
        self.prep = Prepared(cands, gt.keys())
        self.t = np.array([len(gt[e]) for e in self.prep.entity_ids])

    def table_arrays(self, cfg: DecodeConfig):
        m = self.prep.mask(cfg)
        n = self.prep.n_entities
        k = np.bincount(self.prep.ent[m], minlength=n)
        h = np.bincount(self.prep.ent[m], weights=self.prep.label[m].astype(float), minlength=n)
        return h, k, self.t

    def score(self, cfg: DecodeConfig) -> Dict[str, float]:
        h, k, t = self.table_arrays(cfg)
        s = f05_vec(h, k, t)
        single = t == 0
        matched = ~single
        opened = matched & (k > 0)
        return {
            "score": float(s.mean()),
            "spec": float((k[single] == 0).mean()) if single.any() else np.nan,
            "open_rate": float(opened.sum() / max(matched.sum(), 1)),
            "f_open": float(s[opened].mean()) if opened.any() else 0.0,
            "avg_set_size": float(k[k > 0].mean()) if (k > 0).any() else 0.0,
        }


def grid_search(scorer: Scorer, grid: Mapping[str, Sequence], base: DecodeConfig = DecodeConfig(),
                robust: bool = True) -> pd.DataFrame:
    """Evaluate every combination in `grid` (keys are DecodeConfig fields).
    Returns a DataFrame sorted by the pick criterion; row 0 is the pick."""
    keys = list(grid)
    rows: List[dict] = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = DecodeConfig(**{**base.as_dict(), **dict(zip(keys, vals))})
        rows.append({**dict(zip(keys, vals)), **scorer.score(cfg)})
    df = pd.DataFrame(rows)
    shape = tuple(len(grid[k]) for k in keys)
    cube = df.score.to_numpy().reshape(shape)
    df["score_smooth"] = _neighbour_mean(cube).ravel()
    crit = "score_smooth" if robust else "score"
    return df.sort_values([crit, "score"], ascending=False).reset_index(drop=True)


def _neighbour_mean(a: np.ndarray) -> np.ndarray:
    tot = a.copy()
    cnt = np.ones_like(a)
    for ax in range(a.ndim):
        if a.shape[ax] < 2:
            continue
        for shift in (1, -1):
            rolled = np.roll(a, shift, axis=ax)
            valid = np.ones_like(a, dtype=bool)
            sl = [slice(None)] * a.ndim
            sl[ax] = 0 if shift == 1 else -1
            valid[tuple(sl)] = False
            tot += np.where(valid, rolled, 0)
            cnt += valid
    return tot / cnt


def edge_warnings(table: pd.DataFrame, grid: Mapping[str, Sequence]) -> List[str]:
    """Params whose picked value sits on the edge of its grid: the optimum may
    lie outside, so widen the grid in that direction."""
    out = []
    for k, vals in grid.items():
        num = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(num) < 3 or len(num) != len(vals):
            continue
        v = table.iloc[0][k]
        if v == min(num):
            out.append(f"{k}={v} is the grid minimum; extend the grid lower")
        elif v == max(num):
            out.append(f"{k}={v} is the grid maximum; extend the grid higher")
    return out


def best_config(table: pd.DataFrame, base: DecodeConfig = DecodeConfig()) -> DecodeConfig:
    fields = set(DecodeConfig.__dataclass_fields__)
    top = {k: v for k, v in table.iloc[0].to_dict().items() if k in fields}
    for k, v in top.items():  # numpy scalars -> python
        if isinstance(v, (np.integer,)):
            top[k] = int(v)
        elif isinstance(v, (np.floating,)):
            top[k] = float(v)
    return DecodeConfig(**{**base.as_dict(), **top})


DEFAULT_GRID = {
    "gate": [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)],
    "pair": [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)],
    "rel": [0.0, 0.5, 0.8],
}
