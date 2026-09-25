"""From scored candidate pairs to match sets.

Input: a candidate table with s1_id, cand_id, prob (your matcher's calibrated
P(match)), optionally `source` ("s2"/"s3") and a per-entity gate probability
column. Decoding is a pure function of (table, DecodeConfig), and the sweep code
uses the exact same mask, so a config chosen by the sweep decodes identically.

Two modes:
  threshold   : open entity if gate_score >= gate; keep candidates with
                prob >= pair and prob >= rel * best_prob; cap by max_k /
                max_per_source.
  expected_f  : for each entity choose the top-k (k = 0..K) that maximises the
                *expected* macro F0.5 under the model's probabilities
                (k = 0 scores P(no true match among candidates)). This folds
                the gate and the set size into one decision-theoretic rule.
                Requires calibrated probabilities (check calibration first).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .io import MatchSets


@dataclass(frozen=True)
class DecodeConfig:
    mode: str = "threshold"          # "threshold" | "expected_f"
    gate: float = 0.5                # threshold mode: min gate score to open an entity
    gate_source: str = "max"         # "max" | "any" (1-prod(1-p)) | name of a per-row column
    pair: float = 0.5                # threshold mode: min prob per kept candidate
    rel: float = 0.0                 # keep only prob >= rel * best prob of the entity
    max_k: Optional[int] = None      # cap on set size
    max_per_source: Optional[int] = None  # cap per source (needs `source` column)
    min_prob: float = 0.0            # expected_f mode: candidates below are ignored
    ef_max_cands: int = 15           # expected_f mode: consider top-N only
    ef_samples: int = 400            # expected_f mode: Monte-Carlo samples
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Prepared:
    """Candidate table sorted and indexed once; reused across many configs."""

    def __init__(self, cands: pd.DataFrame, entity_ids: Iterable[str]):
        if "prob" not in cands.columns:
            raise ValueError("candidate table needs a `prob` column")
        self.entity_ids = list(dict.fromkeys(entity_ids))
        index = {e: i for i, e in enumerate(self.entity_ids)}
        df = cands[cands.s1_id.isin(index)].copy()
        dropped = len(cands) - len(df)
        if dropped:
            import warnings
            warnings.warn(f"{dropped} candidate rows reference unknown S1 ids and were dropped")
        df = df.drop_duplicates(["s1_id", "cand_id"])
        df = df.sort_values(["s1_id", "prob"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
        self.df = df
        self.ent = df.s1_id.map(index).to_numpy()
        self.prob = df.prob.to_numpy(dtype=float)
        self.rank = df.groupby("s1_id", sort=False).cumcount().to_numpy()
        self.maxp = df.groupby("s1_id", sort=False).prob.transform("max").to_numpy()
        one_minus = np.log1p(-np.clip(self.prob, 0, 1 - 1e-12))
        self.anyp = 1 - np.exp(pd.Series(one_minus).groupby(self.ent).transform("sum").to_numpy())
        self.src_rank = (
            df.groupby(["s1_id", "source"], sort=False).cumcount().to_numpy() if "source" in df.columns else None
        )
        self.label = df.label.to_numpy(dtype=bool) if "label" in df.columns else None
        self._ef_cache = {}

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    def gate_score(self, cfg: DecodeConfig) -> np.ndarray:
        if cfg.gate_source == "max":
            return self.maxp
        if cfg.gate_source == "any":
            return self.anyp
        if cfg.gate_source in self.df.columns:
            return self.df[cfg.gate_source].to_numpy(dtype=float)
        raise ValueError(f"unknown gate_source {cfg.gate_source!r}")

    def mask(self, cfg: DecodeConfig) -> np.ndarray:
        if cfg.mode == "threshold":
            m = (self.gate_score(cfg) >= cfg.gate) & (self.prob >= cfg.pair)
            if cfg.rel > 0:
                m &= self.prob >= cfg.rel * self.maxp
        elif cfg.mode == "expected_f":
            k_star = self._expected_f_k(cfg)
            m = (self.rank < k_star[self.ent]) & (self.prob >= cfg.min_prob)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        if cfg.max_k is not None:
            m &= self.rank < cfg.max_k
        if cfg.max_per_source is not None:
            if self.src_rank is None:
                raise ValueError("max_per_source needs a `source` column")
            m &= self.src_rank < cfg.max_per_source
        return m

    def _expected_f_k(self, cfg: DecodeConfig) -> np.ndarray:
        key = (cfg.min_prob, cfg.ef_max_cands, cfg.ef_samples, cfg.seed)
        if key in self._ef_cache:
            return self._ef_cache[key]
        K = cfg.ef_max_cands
        n = self.n_entities
        P = np.zeros((n, K))
        sel = (self.rank < K) & (self.prob >= cfg.min_prob)
        P[self.ent[sel], self.rank[sel]] = self.prob[sel]
        rng = np.random.default_rng(cfg.seed)
        exp_f = np.zeros((n, K + 1))
        ks = np.arange(1, K + 1)
        S = cfg.ef_samples
        step = max(1, int(2e7 // (S * K)))
        for a in range(0, n, step):
            p = P[a:a + step]                                   # (b, K)
            y = rng.random((S,) + p.shape) < p                  # (S, b, K)
            T = y.sum(-1, keepdims=True)                        # (S, b, 1)
            h = np.cumsum(y, -1)                                # hits in top-k
            f = np.where(T > 0, 1.25 * h / (ks + 0.25 * T), 0.0)
            exp_f[a:a + step, 1:] = f.mean(0)
            exp_f[a:a + step, 0] = (T[..., 0] == 0).mean(0)
        k_star = exp_f.argmax(1)
        self._ef_cache[key] = k_star
        return k_star


def decode(prep: Prepared, cfg: DecodeConfig) -> MatchSets:
    m = prep.mask(cfg)
    out: MatchSets = {e: set() for e in prep.entity_ids}
    sub = prep.df.loc[m, ["s1_id", "cand_id"]]
    for s, c in zip(sub.s1_id, sub.cand_id):
        out[s].add(c)
    return out
