"""Entity-level splits so validation looks like the hidden test set.

Rules:
  * split by S1 entity, never by pair (pairs of one entity in both train and
    validation leak the answer);
  * stratify by match-count bucket (0 / 1 / 2 / 3+) and, if given, by region,
    so every fold has the same singleton rate and region mix;
  * train the matcher K times and keep out-of-fold probabilities for
    threshold sweeps.
"""
from __future__ import annotations

from typing import Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .io import MatchSets


def entity_frame(gt: MatchSets, groups: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    df = pd.DataFrame({"s1_id": list(gt), "n_true": [len(v) for v in gt.values()]})
    df["bucket"] = df.n_true.clip(upper=3).astype(str)
    df["group"] = df.s1_id.map(groups).fillna("<none>") if groups else "<none>"
    return df


def kfold(gt: MatchSets, k: int = 5, groups: Optional[Mapping[str, str]] = None,
          seed: int = 42) -> Dict[str, int]:
    """{s1_id: fold} stratified by (match bucket, group). Deterministic."""
    df = entity_frame(gt, groups)
    rng = np.random.default_rng(seed)
    fold = {}
    for _, g in df.groupby(["bucket", "group"], sort=True):
        ids = g.s1_id.to_numpy().copy()
        rng.shuffle(ids)
        start = rng.integers(0, k)  # avoid always overfilling fold 0 with small strata
        for i, e in enumerate(ids):
            fold[e] = int((start + i) % k)
    return fold


def iter_folds(fold: Mapping[str, int]) -> Iterator[Tuple[int, set, set]]:
    ks = sorted(set(fold.values()))
    for f in ks:
        va = {e for e, v in fold.items() if v == f}
        tr = {e for e, v in fold.items() if v != f}
        yield f, tr, va


def fold_balance(gt: MatchSets, fold: Mapping[str, int], groups: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Singleton rate / match mix / size per fold — should be near-identical."""
    df = entity_frame(gt, groups)
    df["fold"] = df.s1_id.map(fold)
    t = df.groupby("fold").agg(n=("s1_id", "size"), singleton_frac=("n_true", lambda x: (x == 0).mean()),
                               mean_matches=("n_true", "mean"))
    return t


def subset(gt: MatchSets, ids) -> MatchSets:
    ids = set(ids)
    return {e: v for e, v in gt.items() if e in ids}
