"""Raw TSV -> Parquet, a dev subset, a data profile, and one `load()` for everyone.

    python -m amlc.data convert   # all 7 TSVs -> data/parquet/*.parquet (row counts verified)
    python -m amlc.data dev       # 2% entity-consistent dev subset -> data/dev/
    python -m amlc.data profile   # data/profile.md

DuckDB streams the files, so this works on a laptop with a few GB of RAM.
Everything is read as text (IDs, pincodes and postcodes keep leading zeros) and
quote characters are NOT treated specially (a stray `"` in a business name
cannot swallow the following rows).
"""
from __future__ import annotations

import sys
import time

import duckdb  # noqa
import pandas as pd

from .paths import dev_dir, parquet_dir, raw_dir, work_dir

FILES = [
    "train/train_source1", "train/train_source2", "train/train_source3", "train/train_ground_truth",
    "test/test_source1", "test/test_source2", "test/test_source3",
]


# official row counts: a Kaggle copy that differs is not the official data
EXPECTED_ROWS = {"train_source1": 2206821, "train_source2": 5034616, "train_source3": 5285603,
                 "train_ground_truth": 2206821, "test_source1": 1732544, "test_source2": 4887273,
                 "test_source3": 5082316}


def _con(mem="1500MB"):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET preserve_insertion_order=true; SET enable_progress_bar=false;")
    return con


def _read_tsv(path) -> str:
    return (f"read_csv('{path}', delim='\t', header=true, quote='', escape='', "
            f"all_varchar=true, null_padding=true, strict_mode=false, nullstr='\x01')")


def _count_lines(path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n


def pq(name: str, dev: bool = False) -> str:
    base = dev_dir() if dev else parquet_dir()
    return str(base / f"{name.split('/')[-1]}.parquet")


def convert(force: bool = False) -> None:
    raw, out = raw_dir(), work_dir() / "parquet"
    out.mkdir(parents=True, exist_ok=True)
    con = _con()
    for name in FILES:
        src, dst = raw / f"{name}.tsv", str(out / f"{name.split('/')[-1]}.parquet")
        if not force and __import__("os").path.exists(dst):
            print(f"skip {name} (exists)")
            continue
        t = time.time()
        cols = con.execute(f"DESCRIBE SELECT * FROM {_read_tsv(src)}").fetchall()
        sel = ", ".join(f"coalesce(\"{c[0]}\", '') AS \"{c[0]}\"" for c in cols)  # empty stays '' (never NULL)
        con.execute(f"COPY (SELECT {sel} FROM {_read_tsv(src)}) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)")
        rows = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
        lines = _count_lines(src) - 1
        ok = "OK" if rows == lines else f"MISMATCH ({lines} data lines)"
        mb = __import__("os").path.getsize(dst) / 1e6
        print(f"{name:<28} rows={rows:>10,}  {mb:7.1f} MB  {time.time() - t:5.1f}s  {ok}")
        if rows != lines:
            raise SystemExit(f"{name}: parsed {rows} rows but file has {lines} lines — inspect before continuing")
        exp = EXPECTED_ROWS.get(name.split("/")[-1])
        if exp and rows != exp:
            raise SystemExit(f"{name}: {rows} rows, official data has {exp} — wrong or modified dataset")


def load(name: str, dev: bool = False, columns=None) -> pd.DataFrame:
    """load('train_source1'), load('train_ground_truth', dev=True), ..."""
    name = name if "/" in name else next(f for f in FILES if f.endswith("/" + name))
    return pd.read_parquet(pq(name, dev), columns=columns)


def make_dev(pct: int = 2) -> None:
    """Entity-consistent subset: pct% of S1 entities with ALL their matched S2/S3
    records, plus pct% of the S2/S3 records that match nothing (distractors).
    Keeps the matched:distractor ratio of the full data, so blocking and
    singleton statistics are representative."""
    out = dev_dir()
    out.mkdir(parents=True, exist_ok=True)
    con = _con()
    gt, s1, s2, s3 = (pq(f) for f in ("train_ground_truth", "train_source1", "train_source2", "train_source3"))
    mod = 100 // pct
    con.execute(f"""
        CREATE TEMP TABLE keep1 AS SELECT source1_entity_id AS id FROM '{gt}' WHERE hash(source1_entity_id) % {mod} = 0;
        CREATE TEMP TABLE pairs AS
            SELECT source1_entity_id AS s1, trim(unnest(string_split(matched_entity_ids, ','))) AS t
            FROM '{gt}' WHERE matched_entity_ids <> '';
        CREATE TEMP TABLE matched_any AS SELECT DISTINCT t FROM pairs;
        CREATE TEMP TABLE keept AS
            SELECT t AS id FROM pairs WHERE s1 IN (SELECT id FROM keep1)
            UNION
            SELECT entity_id FROM (SELECT entity_id FROM '{s2}' UNION ALL SELECT entity_id FROM '{s3}')
            WHERE entity_id NOT IN (SELECT t FROM matched_any) AND hash(entity_id) % {mod} = 0;
    """)
    for name, src, key in [("train_ground_truth", gt, "source1_entity_id"), ("train_source1", s1, "entity_id")]:
        con.execute(f"COPY (SELECT * FROM '{src}' WHERE {key} IN (SELECT id FROM keep1)) TO '{pq(name, True)}' (FORMAT parquet)")
    for name, src in [("train_source2", s2), ("train_source3", s3)]:
        con.execute(f"COPY (SELECT * FROM '{src}' WHERE entity_id IN (SELECT id FROM keept)) TO '{pq(name, True)}' (FORMAT parquet)")
    for name in ("train_ground_truth", "train_source1", "train_source2", "train_source3"):
        n = con.execute(f"SELECT count(*) FROM '{pq(name, True)}'").fetchone()[0]
        print(f"dev/{name:<22} rows={n:>9,}")
    # the harness reads TSV ground truth
    con.execute(f"COPY (SELECT * FROM '{pq('train_ground_truth', True)}') TO '{out / 'train_ground_truth.tsv'}' (HEADER, DELIMITER '\t', QUOTE '')")


def profile() -> str:
    con = _con()
    L = ["# Data profile", ""]
    L.append("| file | rows | countries | empty name | empty address | Devanagari name | median name len | median addr len |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in FILES:
        if "ground_truth" in f:
            continue
        p = pq(f)
        r = con.execute(f"""
            SELECT sum(n)::BIGINT,
                   string_agg(country || ':' || n, ' ' ORDER BY n DESC)
            FROM (SELECT country, count(*) n FROM '{p}' GROUP BY 1)""").fetchone()
        s = con.execute(f"""
            SELECT avg((coalesce(business_name,'') = '')::INT), avg((coalesce(business_address,'') = '')::INT),
                   avg(regexp_matches(business_name, '[ऀ-ॿ]')::INT),
                   median(length(business_name)), median(length(business_address))
            FROM '{p}'""").fetchone()
        s = [x if x is not None else 0 for x in s]
        L.append(f"| {f} | {r[0]:,} | {r[1]} | {s[0]:.3%} | {s[1]:.3%} | {s[2]:.2%} | {s[3]:.0f} | {s[4]:.0f} |")
    g = pq("train_ground_truth")
    dist = con.execute(f"""
        SELECT n, count(*) c FROM (
          SELECT CASE WHEN matched_entity_ids = '' THEN 0
                      ELSE len(string_split(matched_entity_ids, ',')) END AS n FROM '{g}')
        GROUP BY 1 ORDER BY 1""").fetchall()
    tot = sum(c for _, c in dist)
    L += ["", "## Matches per Source-1 entity (train)", "", "| n matches | entities | share |", "|---|---|---|"]
    L += [f"| {n} | {c:,} | {c / tot:.3%} |" for n, c in dist]
    bysrc = con.execute(f"""
        SELECT substr(t, 1, 2) src, count(*) FROM (
          SELECT trim(unnest(string_split(matched_entity_ids, ','))) t FROM '{g}' WHERE matched_entity_ids <> '')
        GROUP BY 1 ORDER BY 1""").fetchall()
    L += ["", "Matched ids by source: " + ", ".join(f"{s}: {c:,}" for s, c in bysrc)]
    bycountry = con.execute(f"""
        SELECT s.country, count(*) n, avg((g.matched_entity_ids = '')::INT) singleton
        FROM '{g}' g JOIN '{pq('train_source1')}' s ON s.entity_id = g.source1_entity_id
        GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    L += ["", "| country (S1 train) | entities | singleton share |", "|---|---|---|"]
    L += [f"| {c} | {n:,} | {s:.3%} |" for c, n, s in bycountry]
    reuse = con.execute(f"""
        SELECT count(*) - count(DISTINCT t) FROM (
          SELECT trim(unnest(string_split(matched_entity_ids, ','))) t FROM '{g}' WHERE matched_entity_ids <> '')""").fetchone()[0]
    L += ["", f"S2/S3 ids matched to more than one S1 entity (extra occurrences): {reuse:,}"]
    text = "\n".join(L) + "\n"
    (work_dir() / "profile.md").write_text(text)
    return text


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("convert", "all"):
        convert(force="--force" in sys.argv)
    if cmd in ("dev", "all"):
        make_dev()
    if cmd in ("profile", "all"):
        print(profile())
