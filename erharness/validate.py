"""Pre-submission checks. Run the official utils/validate_submission.py too;
this adds the semantic checks a format validator won't catch."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .io import _split_ids


def check_submission(path: str, s1_ids: Iterable[str], valid_targets: Optional[Iterable[str]] = None,
                     sample_path: Optional[str] = None) -> List[str]:
    """Return a list of problems (empty list = clean)."""
    problems: List[str] = []
    raw = open(path, encoding="utf-8").read()
    if "\t" not in raw.splitlines()[0]:
        problems.append("header line has no TAB — file is probably not tab-separated")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if sample_path:
        sample_cols = list(pd.read_csv(sample_path, sep="\t", nrows=0).columns)
        if list(df.columns) != sample_cols:
            problems.append(f"columns {list(df.columns)} != sample {sample_cols}")
    idc = df.columns[0]
    ids = df[idc].str.strip()
    want = set(map(str, s1_ids))
    dup = ids[ids.duplicated()].unique()
    if len(dup):
        problems.append(f"{len(dup)} duplicated S1 ids, e.g. {list(dup[:3])}")
    missing = want - set(ids)
    if missing:
        problems.append(f"{len(missing)} S1 ids missing, e.g. {sorted(missing)[:3]}")
    extra = set(ids) - want
    if extra:
        problems.append(f"{len(extra)} unknown S1 ids, e.g. {sorted(extra)[:3]}")
    targets = set(map(str, valid_targets)) if valid_targets is not None else None
    bad, selfref, n_pairs, n_nonempty = set(), 0, 0, 0
    for e, cell in zip(ids, df[df.columns[1]] if df.shape[1] > 1 else [""] * len(df)):
        m = _split_ids(cell)
        n_pairs += len(m)
        n_nonempty += bool(m)
        if e in m:
            selfref += 1
        if targets is not None:
            bad |= m - targets
        if " " in str(cell).strip() and len(problems) < 50:
            problems.append(f"whitespace inside match list for {e!r}: {cell!r}")
    if bad:
        problems.append(f"{len(bad)} predicted ids are not S2/S3 ids, e.g. {sorted(bad)[:3]}")
    if selfref:
        problems.append(f"{selfref} rows list their own S1 id as a match")
    frac = n_nonempty / max(len(df), 1)
    print(f"[info] rows={len(df)}  non-empty={n_nonempty} ({frac:.3f})  predicted pairs={n_pairs}  "
          f"avg set size={n_pairs / max(n_nonempty, 1):.2f}")
    return problems
