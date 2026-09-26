"""Step 2: candidate generation (blocking) at full scale with DuckDB.

    python -m amlc.blocking train     # -> data/work/train_cands.parquet  (+ recall report)
    python -m amlc.blocking test      # -> data/work/test_cands.parquet

Each record gets several cheap keys; a Source-1 record and a Source-2/3 record
become a candidate pair when they share at least one key. Keys are always
prefixed with the country so countries never mix. Oversized blocks (very
common keys) are skipped: they cost a lot and carry little evidence.

keys
  addr   house number + first street word            "US|1303|cattle"  (street-type words skipped: STREET_STOP)
  tok    each of the 2 rarest name tokens            "US|yeager"
  pre    first 8 chars of the name without spaces    "US|yeagerst"   (domains, joined words)
  skel   first 6 chars of the consonant skeleton     "India|rmrktn"  (transliteration, typos)
  num    house number + rarest name token            "US|1303|yeager" (survives big 'addr'/'tok' blocks)
  full   whole name without spaces                   "US|pioneermedia" (common names, exact)
  tokat  rarest name token + rarest address word     "US|pioneer|burgess" (house-number noise, reordering)
  prehs  name prefix + house number                  "US|pioneer|1303" (splits big 'pre' blocks)
  skelhs skeleton prefix + house number              "India|snrs|3"
  prest  name prefix + first street word             "US|pioneer|cattle"
  skelat skeleton prefix + rarest address word       "India|snrs|thakkar" (transliterated names)
  addr2  two rarest address words, name ignored      "India|chawl|thakkar" (garbled / transliterated names)
  phonhs sound key prefix + house number             "India|dnmktr|7"
  phonat sound key prefix + rarest address word      "India|dnmk|krishna"
The output has one row per (s1_id, cand_id) with `keys` = '|'-joined key names.
"""
from __future__ import annotations

import sys
import time

import duckdb

from .paths import mem as default_mem, threads, work_dir

KEYS = ("addr", "tok", "pre", "skel", "num", "full", "tokat",
        "prehs", "skelhs", "prest", "skelat", "addr2", "phonhs", "phonat")
CAP_TARGETS = {"addr": 60, "tok": 150, "pre": 60, "skel": 40, "num": 60, "full": 200, "tokat": 80,
               "prehs": 60, "skelhs": 60, "prest": 60, "skelat": 60, "addr2": 60,
               "phonhs": 60, "phonat": 60}   # max S2+S3 per block
CAP_S1 = {"addr": 25, "tok": 60, "pre": 25, "skel": 15, "num": 25, "full": 80, "tokat": 30,
          "prehs": 25, "skelhs": 25, "prest": 25, "skelat": 25, "addr2": 25,
          "phonhs": 25, "phonat": 25}         # max S1 per block
# v2 keys (from the missed-pair breakdown): 60% of misses shared a name key whose block was too big ->
# split those blocks by house number / street word / rare address word; 40% shared no key but 90% share
# two rare address words (mostly transliterated names) -> 'addr2' ignores the name entirely.
# v3 keys: India recall was 92.6% vs US 97.6%; most India misses are English names written in an
# Indian script ('dayanamik trading') whose consonant skeleton differs -> 'phon*' use a sound key
# (+4.5k of 33.6k missed pairs on a 10% sample; a bare 'phon' prefix key added 13M pairs for 0.5k: dropped).
# street-type / filler words skipped when picking the "first street word" for keys: in France the street
# starts with its type ('11 rue alfred delattre' -> 'rue' = a giant block that gets capped) and Indian
# addresses often start with 'plot number' / 'house number' / 'shop no'.
STREET_STOP = ("rue", "avenue", "boulevard", "place", "chemin", "impasse", "allee", "route", "quai", "cours",
               "square", "passage", "residence", "lotissement", "faubourg", "bis", "ter", "de", "la", "le", "les",
               "du", "des", "plot", "house", "number", "no", "shop", "flat", "door", "office", "floor", "room",
               "unit", "suite", "block", "building", "near", "opp", "opposite", "sector", "north", "south",
               "east", "west", "the", "old", "new", "ground", "first", "second", "third", "main")
_STOP_SQL = ", ".join(f"'{w}'" for w in STREET_STOP)
TOK_DF_MAX = 5000  # tokens more frequent than this never become 'tok' keys


def _paths(split):
    w = work_dir() / "work"
    return str(w / f"{split}_norm.parquet"), str(w / f"{split}_cands"), str(w / f"{split}_keys.parquet")


def build_keys(con, norm: str, keys_out: str) -> None:
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE r AS
      SELECT entity_id, src, country, name_core, name_concat, name_skel, name_phon, house_no, street, addr_clean,
             coalesce(list_filter(string_split(street, ' '),
                                  x -> length(x) >= 3 AND NOT regexp_matches(x, '[0-9]') AND x NOT IN ({_STOP_SQL}))[1], '') AS skey,
             row_number() OVER () AS rid
      FROM '{norm}';

    -- document frequency of core-name tokens, per country
    CREATE OR REPLACE TEMP TABLE toks AS
      SELECT rid, country, tok FROM (
        SELECT rid, country, unnest(string_split(name_core, ' ')) AS tok FROM r)
      WHERE length(tok) >= 2;
    CREATE OR REPLACE TEMP TABLE df AS SELECT country, tok, count(*) AS df FROM toks GROUP BY 1, 2;
    CREATE OR REPLACE TEMP TABLE rare AS
      SELECT rid, tok, df FROM (
        SELECT t.rid, t.tok, d.df,
               row_number() OVER (PARTITION BY t.rid ORDER BY d.df, t.tok) AS k
        FROM (SELECT DISTINCT rid, country, tok FROM toks) t JOIN df d USING (country, tok))
      WHERE k <= 2 AND df <= {TOK_DF_MAX};

    -- rarest address word (letters only, >= 4 chars) per record
    CREATE OR REPLACE TEMP TABLE atoks AS
      SELECT DISTINCT rid, country, tok FROM (
        SELECT rid, country, unnest(string_split(regexp_replace(addr_clean, '[^a-z ]', ' ', 'g'), ' ')) AS tok FROM r)
      WHERE length(tok) >= 4;
    CREATE OR REPLACE TEMP TABLE adf AS SELECT country, tok, count(*) AS df FROM atoks GROUP BY 1, 2;
    CREATE OR REPLACE TEMP TABLE arare AS
      SELECT rid, tok FROM (
        SELECT a.rid, a.tok, row_number() OVER (PARTITION BY a.rid ORDER BY d.df, a.tok) AS k
        FROM atoks a JOIN adf d USING (country, tok))
      WHERE k = 1;
    CREATE OR REPLACE TEMP TABLE arare2 AS
      SELECT rid, min(tok) || '|' || max(tok) AS pair FROM (
        SELECT a.rid, a.tok, row_number() OVER (PARTITION BY a.rid ORDER BY d.df, a.tok) AS k
        FROM atoks a JOIN adf d USING (country, tok))
      WHERE k <= 2 GROUP BY rid HAVING count(*) = 2;

    COPY (
      SELECT entity_id, src, 'addr' AS kind,
             country || '|' || house_no || '|' || skey AS key
        FROM r WHERE house_no <> '' AND skey <> ''
      UNION ALL
      SELECT r.entity_id, r.src, 'tok', r.country || '|' || rare.tok
        FROM r JOIN rare USING (rid)
      UNION ALL
      SELECT entity_id, src, 'pre', country || '|' || substr(name_concat, 1, 8)
        FROM r WHERE length(name_concat) >= 5
      UNION ALL
      SELECT entity_id, src, 'skel', country || '|' || substr(name_skel, 1, 6)
        FROM r WHERE length(name_skel) >= 4
      UNION ALL
      SELECT r.entity_id, r.src, 'num', r.country || '|' || r.house_no || '|' || rare.tok
        FROM r JOIN (SELECT rid, tok FROM rare QUALIFY row_number() OVER (PARTITION BY rid ORDER BY df) = 1) rare
        USING (rid) WHERE r.house_no <> ''
      UNION ALL
      SELECT entity_id, src, 'full', country || '|' || name_concat FROM r WHERE length(name_concat) >= 4
      UNION ALL
      SELECT r.entity_id, r.src, 'tokat', r.country || '|' || rare.tok || '|' || arare.tok
        FROM r JOIN rare USING (rid) JOIN arare USING (rid)
      UNION ALL
      SELECT entity_id, src, 'prehs', country || '|' || substr(name_concat, 1, 8) || '|' || house_no
        FROM r WHERE length(name_concat) >= 5 AND house_no <> ''
      UNION ALL
      SELECT entity_id, src, 'skelhs', country || '|' || substr(name_skel, 1, 6) || '|' || house_no
        FROM r WHERE length(name_skel) >= 4 AND house_no <> ''
      UNION ALL
      SELECT entity_id, src, 'prest', country || '|' || substr(name_concat, 1, 8) || '|' || skey
        FROM r WHERE length(name_concat) >= 5 AND skey <> ''
      UNION ALL
      SELECT r.entity_id, r.src, 'skelat', r.country || '|' || substr(r.name_skel, 1, 4) || '|' || arare.tok
        FROM r JOIN arare USING (rid) WHERE length(r.name_skel) >= 3
      UNION ALL
      SELECT r.entity_id, r.src, 'addr2', r.country || '|' || arare2.pair
        FROM r JOIN arare2 USING (rid)
      UNION ALL
      SELECT entity_id, src, 'phonhs', country || '|' || substr(name_phon, 1, 6) || '|' || house_no
        FROM r WHERE length(name_phon) >= 3 AND house_no <> ''
      UNION ALL
      SELECT r.entity_id, r.src, 'phonat', r.country || '|' || substr(r.name_phon, 1, 4) || '|' || arare.tok
        FROM r JOIN arare USING (rid) WHERE length(r.name_phon) >= 3
    ) TO '{keys_out}' (FORMAT parquet);
    """)


def build_candidates(con, keys: str, out_dir: str, n_buckets: int = 8) -> dict:
    """Pairs per key kind -> disk, then merged per S1-hash bucket (bounded memory)."""
    import os, shutil
    caps = " OR ".join(
        f"(kind = '{k}' AND (n_t > {CAP_TARGETS[k]} OR n_s > {CAP_S1[k]}))" for k in KEYS)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE blk AS
      SELECT kind, key,
             count(*) FILTER (WHERE src <> 'S1') AS n_t,
             count(*) FILTER (WHERE src = 'S1') AS n_s
      FROM '{keys}' GROUP BY 1, 2;
    CREATE OR REPLACE TEMP TABLE ok AS
      SELECT kind, key FROM blk WHERE n_t > 0 AND n_s > 0 AND NOT ({caps});""")
    tmp = out_dir + "_tmp"
    for d in (tmp, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for k in KEYS:
        con.execute(f"""
        COPY (
          SELECT a.entity_id AS s1_id, b.entity_id AS cand_id, '{k}' AS kind
          FROM (SELECT entity_id, key FROM '{keys}' WHERE kind = '{k}' AND src = 'S1'
                AND key IN (SELECT key FROM ok WHERE kind = '{k}')) a
          JOIN (SELECT entity_id, key FROM '{keys}' WHERE kind = '{k}' AND src <> 'S1'
                AND key IN (SELECT key FROM ok WHERE kind = '{k}')) b USING (key)
        ) TO '{tmp}/{k}.parquet' (FORMAT parquet)""")
    for b in range(n_buckets):
        con.execute(f"""
        COPY (
          SELECT s1_id, cand_id, string_agg(DISTINCT kind, '|' ORDER BY kind) AS keys
          FROM '{tmp}/*.parquet' WHERE (hash(s1_id) % {n_buckets}) = {b}
          GROUP BY 1, 2
        ) TO '{out_dir}/part{b}.parquet' (FORMAT parquet)""")
    shutil.rmtree(tmp, ignore_errors=True)
    stats = con.execute(f"""
      SELECT kind, count(*) blocks, sum(n_t) targets,
             count(*) FILTER (WHERE n_t > 0 AND n_s > 0) usable,
             sum(CASE WHEN n_t > 0 AND n_s > 0 THEN n_t * n_s ELSE 0 END) raw_pairs
      FROM blk GROUP BY 1 ORDER BY 1""").fetchall()
    return {r[0]: dict(blocks=r[1], usable=r[3], raw_pairs=r[4]) for r in stats}


def recall_report(con, cands: str, gt: str) -> str:
    q = con.execute(f"""
      WITH truth AS (
        SELECT source1_entity_id AS s1_id, trim(unnest(string_split(matched_entity_ids, ','))) AS cand_id
        FROM '{gt}' WHERE matched_entity_ids <> ''),
      c AS (SELECT * FROM '{cands}/*.parquet'),
      hit AS (SELECT t.s1_id, t.cand_id, c.keys FROM truth t LEFT JOIN c USING (s1_id, cand_id))
      SELECT
        (SELECT count(*) FROM c) AS n_pairs,
        (SELECT count(DISTINCT s1_id) FROM c) AS n_s1_with_cands,
        (SELECT count(*) FROM truth) AS n_true,
        (SELECT count(*) FROM hit WHERE keys IS NOT NULL) AS found
    """).fetchone()
    per_key = con.execute(f"""
      WITH truth AS (
        SELECT source1_entity_id AS s1_id, trim(unnest(string_split(matched_entity_ids, ','))) AS cand_id
        FROM '{gt}' WHERE matched_entity_ids <> ''),
      c AS (SELECT s1_id, cand_id, unnest(string_split(keys, '|')) AS k, keys FROM '{cands}/*.parquet')
      SELECT k, count(*) AS pairs, count(t.s1_id) AS true_pairs,
             count(t.s1_id) FILTER (WHERE NOT contains(keys, '|')) AS only_this_key
      FROM c LEFT JOIN truth t USING (s1_id, cand_id) GROUP BY 1 ORDER BY 1""").fetchall()
    n_pairs, n_s1, n_true, found = q
    n_s1_total = con.execute(f"SELECT count(*) FROM '{gt}'").fetchone()[0]
    lines = [f"candidate pairs: {n_pairs:,}  ({n_pairs / n_s1_total:.1f} per S1 entity)",
             f"pair recall: {found / n_true:.4f}  ({found:,} / {n_true:,} true pairs)",
             "key      pairs         true pairs   precision  only-this-key"]
    lines += [f"{k:<8} {p:>12,} {t:>12,}   {t / p:8.4f}   {o:>10,}" for k, p, t, o in per_key]
    return "\n".join(lines)


def run(split: str, mem: str = None) -> None:
    mem = mem or default_mem()
    norm, cands, keys = _paths(split)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET enable_progress_bar=false; "
                f"SET temp_directory='{work_dir() / 'work' / 'duck_tmp'}'; SET preserve_insertion_order=false; SET threads={threads()};")
    t = time.time()
    build_keys(con, norm, keys)
    print(f"keys built {time.time() - t:.0f}s", flush=True)
    stats = build_candidates(con, keys, cands)
    print(f"candidates built {time.time() - t:.0f}s", flush=True)
    for k, v in stats.items():
        print(f"  {k:<5} usable blocks={v['usable']:,}  uncapped pairs={v['raw_pairs']:,}")
    if split == "train":
        from .data import pq as pq_path
        print(recall_report(con, cands, pq_path("train_ground_truth")))


if __name__ == "__main__":
    run(sys.argv[1])
