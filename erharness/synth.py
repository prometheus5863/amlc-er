"""Synthetic ground truth + scored candidates, for tests and for dry-running
the whole harness before the real data is wired in."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def make(n: int = 5000, singleton_frac: float = 0.4, blocking_recall: float = 0.95,
         seed: int = 0, out_dir: str = None):
    rng = np.random.default_rng(seed)
    regions = np.array(["north", "south", "west", "east"])
    gt_rows, cand_rows, grp_rows = [], [], []
    tgt = 0
    for i in range(n):
        e = f"S1_{i:06d}"
        region = regions[rng.integers(0, 4)]
        grp_rows.append((e, region))
        hard = region == "east"                     # one region is noisier
        if rng.random() < singleton_frac:
            n_true = 0
        else:
            n_true = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        true_ids = []
        for _ in range(n_true):
            src = "s2" if rng.random() < 0.5 else "s3"
            tgt += 1
            true_ids.append((f"{src.upper()}_{tgt:07d}", src))
        gt_rows.append((e, ",".join(t for t, _ in true_ids)))
        for t, src in true_ids:
            if rng.random() < blocking_recall:
                p = rng.beta(4 if hard else 7, 2.5)
                cand_rows.append((e, t, p, src, _keys(rng)))
        n_neg = rng.integers(3, 25)
        for _ in range(n_neg):
            src = "s2" if rng.random() < 0.5 else "s3"
            tgt += 1
            a, b = (2, 4) if (n_true == 0 and rng.random() < 0.3) or hard else (1, 7)
            cand_rows.append((e, f"{src.upper()}_{tgt:07d}", rng.beta(a, b), src, _keys(rng)))
    gt = pd.DataFrame(gt_rows, columns=["s1_id", "matches"])
    cands = pd.DataFrame(cand_rows, columns=["s1_id", "cand_id", "prob", "source", "key"])
    groups = pd.DataFrame(grp_rows, columns=["s1_id", "region"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        gt.to_csv(os.path.join(out_dir, "gt.tsv"), sep="\t", index=False)
        cands.to_csv(os.path.join(out_dir, "candidates.tsv"), sep="\t", index=False)
        groups.to_csv(os.path.join(out_dir, "groups.tsv"), sep="\t", index=False)
    return gt, cands, groups


def _keys(rng) -> str:
    keys = ["pin3_name", "phonetic_city", "tfidf_knn", "phone"]
    pick = [k for k, p in zip(keys, [0.6, 0.4, 0.7, 0.15]) if rng.random() < p] or ["tfidf_knn"]
    return "|".join(pick)
