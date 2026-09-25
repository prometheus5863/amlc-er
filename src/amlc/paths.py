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


def work_dir() -> Path:
    w = os.environ.get("AMLC_WORK") or _local_cfg().get("work")
    if not w:
        w = "/kaggle/working/data" if Path("/kaggle/working").exists() else str(REPO / "data")
    p = Path(w)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parquet_dir() -> Path:
    return work_dir() / "parquet"


def dev_dir() -> Path:
    return work_dir() / "dev"
