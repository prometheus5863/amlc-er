"""Step 1: normalise every source table once -> data/work/<split>_norm.parquet

    python -m amlc.prep train        # or: test | all

Runs in chunks on all CPU cores; output keeps only the columns the later
stages need, as compact Arrow strings.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import pq as pq_path
from .normalize import normalize_frame
from .paths import work_dir

KEEP = ["entity_id", "src", "country", "name_clean", "name_core", "name_concat", "name_skel", "name_phon", "legal",
        "is_domain", "is_translit", "alias_core", "addr_clean", "house_no", "addr_nums", "street", "postcode"]
CHUNK = 250_000


def out_path(split: str) -> str:
    d = work_dir() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{split}_norm.parquet")


def _job(args):
    path, start, stop = args
    t = pq.read_table(path).slice(start, stop - start).to_pandas()
    return normalize_frame(t)[KEEP]


def run(split: str, procs: int = os.cpu_count() or 1) -> None:
    t0 = time.time()
    jobs = []
    for s in ("1", "2", "3"):
        path = pq_path(f"{split}_source{s}")
        n = pq.ParquetFile(path).metadata.num_rows
        jobs += [(path, i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
    writer = None
    done = 0
    with Pool(procs) as pool:
        for df in pool.imap(_job, jobs):
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path(split), tbl.schema, compression="zstd")
            writer.write_table(tbl)
            done += len(df)
            print(f"{split}: {done:,} rows  {time.time() - t0:.0f}s", flush=True)
    writer.close()


if __name__ == "__main__":
    for sp in (["train", "test"] if sys.argv[1] == "all" else [sys.argv[1]]):
        run(sp)
