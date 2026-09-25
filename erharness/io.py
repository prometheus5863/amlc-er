"""Loading ground truth, submissions and candidate pairs.

All files are tab-separated. IDs are always handled as stripped strings so that
"00123" and "123" are never silently merged or split by dtype inference.

Match-set format (ground truth and matching_results.tsv):
    <s1_id> \t <comma-separated list of S2/S3 ids, empty = no match>
If the file has more than two columns (e.g. separate S2 and S3 columns), every
column after the id column is unioned into one set. Override with `match_cols`.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional, Sequence, Set

import pandas as pd

MatchSets = Dict[str, Set[str]]


def _split_ids(cell) -> Set[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return set()
    s = s.strip("[]")
    return {t.strip().strip("'\"") for t in s.split(",") if t.strip().strip("'\"")}


def read_match_sets(
    path: str,
    id_col: Optional[str] = None,
    match_cols: Optional[Sequence[str]] = None,
) -> MatchSets:
    """Read a ground-truth or submission file into {s1_id: set(matched ids)}."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if df.shape[1] < 1:
        raise ValueError(f"{path}: no columns")
    id_col = id_col or df.columns[0]
    if match_cols is None:
        match_cols = [c for c in df.columns if c != id_col]
    out: MatchSets = {}
    dup = 0
    id_vals = df[id_col].to_numpy()
    match_vals = df[list(match_cols)].to_numpy() if match_cols else None
    for i, raw_id in enumerate(id_vals):
        rid = str(raw_id).strip()
        ids: Set[str] = set()
        if match_vals is not None:
            for cell in match_vals[i]:
                ids |= _split_ids(cell)
        if rid in out:
            dup += 1
            out[rid] |= ids
        else:
            out[rid] = ids
    if dup:
        warnings.warn(f"{path}: {dup} duplicate {id_col} rows were unioned")
    return out


def write_match_sets(
    sets: MatchSets,
    path: str,
    id_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
    order: Optional[Iterable[str]] = None,
) -> None:
    """Write {s1_id: set} as a submission TSV. Pass the header names the official
    sample submission uses; `order` keeps the S1 row order of the test file."""
    keys = list(order) if order is not None else sorted(sets)
    rows = [(k, ",".join(sorted(sets.get(k, set())))) for k in keys]
    pd.DataFrame(rows, columns=[id_col, match_col]).to_csv(path, sep="\t", index=False)


def read_candidates(
    path: str,
    s1_col: str = "s1_id",
    cand_col: str = "cand_id",
    prob_col: Optional[str] = "prob",
) -> pd.DataFrame:
    """Read a candidate-pair table. Output columns are normalised to
    s1_id, cand_id, [prob], plus any extra columns (source, key, ...)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ren = {s1_col: "s1_id", cand_col: "cand_id"}
    if prob_col and prob_col in df.columns:
        ren[prob_col] = "prob"
    df = df.rename(columns=ren)
    for c in ("s1_id", "cand_id"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c!r} (columns: {list(df.columns)})")
        df[c] = df[c].str.strip()
    if "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="raise")
    return df


def label_candidates(cands: pd.DataFrame, gt: MatchSets) -> pd.DataFrame:
    """Add a boolean `label` column: is (s1_id, cand_id) a true match?"""
    true_pairs = {(s, c) for s, cs in gt.items() for c in cs}
    out = cands.copy()
    out["label"] = [(s, c) in true_pairs for s, c in zip(out["s1_id"], out["cand_id"])]
    return out


def read_groups(path: str, id_col: Optional[str] = None, group_col: Optional[str] = None) -> Dict[str, str]:
    """Optional {s1_id: group} map (region, country, city tier ...) for breakdowns."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    id_col = id_col or df.columns[0]
    group_col = group_col or df.columns[1]
    return dict(zip(df[id_col].str.strip(), df[group_col].str.strip()))


def write_candidate_pairs(cands: pd.DataFrame, path: str, order: Iterable[str]) -> None:
    """Official candidate_pairs.tsv: one row per S1 entity (all of `order`),
    comma-separated candidate ids, empty when blocking found nothing."""
    g = cands.drop_duplicates(["s1_id", "cand_id"]).groupby("s1_id").cand_id.agg(",".join)
    keys = list(order)
    pd.DataFrame({"source1_entity_id": keys,
                  "candidate_entity_ids": [g.get(k, "") for k in keys]}).to_csv(path, sep="\t", index=False)
