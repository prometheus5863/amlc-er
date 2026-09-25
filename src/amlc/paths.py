"""Where the data lives, on any machine.

The raw dataset is never copied into the repo. Each person points to their own
copy once, and every script and notebook finds it the same way:

  1. environment variable AMLC_RAW  (folder that contains train/ and test/)
  2. file `configs/paths.local.json`  {"raw": "...", "work": "..."}  (gitignored)
  3. auto-detect: Kaggle input folders, ./data/raw

`work` is where Parquet copies, the dev subset and pipeline outputs go
(default ./data).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _local_cfg() -> dict:
    p = REPO / "configs" / "paths.local.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _looks_like_raw(p: Path) -> bool:
    return (p / "train" / "train_source1.tsv").exists() and (p / "test" / "test_source1.tsv").exists()


def raw_dir() -> Path:
    cands = [os.environ.get("AMLC_RAW"), _local_cfg().get("raw")]
    cands += glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*") + glob.glob("/kaggle/input/*/*/*")
    cands += [str(REPO / "data" / "raw"), str(REPO / "data" / "raw" / "dataset")]
    for c in cands:
        if c and _looks_like_raw(Path(c)):
            return Path(c)
    raise FileNotFoundError(
        "Raw dataset not found. Set AMLC_RAW to the folder containing train/ and test/, "
        "or create configs/paths.local.json with {\"raw\": \"<that folder>\"}."
    )


ON_KAGGLE = Path("/kaggle/working").exists()


def work_dir() -> Path:
    """Intermediate files (normalised tables, keys, candidates, features, model).
    On Kaggle this is /tmp/amlc: large, fast, and NOT saved as notebook output."""
    w = os.environ.get("AMLC_WORK") or _local_cfg().get("work")
    if not w:
        w = "/tmp/amlc" if ON_KAGGLE else str(REPO / "data")
    p = Path(w)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parquet_dir() -> Path:
    """Parquet copies of the 7 raw tables. Uses, in order: AMLC_PARQUET, the
    config file, <work>/parquet if already converted, a Kaggle input dataset
    that contains train_source1.parquet, else <work>/parquet (convert target)."""
    for c in (os.environ.get("AMLC_PARQUET"), _local_cfg().get("parquet")):
        if c:
            return Path(c)
    local = work_dir() / "parquet"
    if (local / "train_source1.parquet").exists():
        return local
    hits = glob.glob("/kaggle/input/**/train_source1.parquet", recursive=True)
    if hits:
        return Path(hits[0]).parent
    return local


def output_dir() -> Path:
    """Where matching_results.tsv / candidate_pairs.tsv go."""
    o = os.environ.get("AMLC_OUTPUT") or ("/kaggle/working/output" if ON_KAGGLE else str(REPO / "output"))
    p = Path(o)
    p.mkdir(parents=True, exist_ok=True)
    return p


def threads() -> int:
    return int(os.environ.get("AMLC_THREADS") or os.cpu_count() or 2)


def mem() -> str:
    """DuckDB memory limit. Default: ~60% of RAM (Kaggle 30 GB -> 18GB, laptop 8 GB -> 4GB)."""
    if os.environ.get("AMLC_MEM"):
        return os.environ["AMLC_MEM"]
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        total = 8
    return f"{max(2, int(total * 0.6))}GB"


def dev_dir() -> Path:
    return work_dir() / "dev"
