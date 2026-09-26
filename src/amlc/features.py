"""Step 3: pair features for every candidate pair.

    python -m amlc.features train [--sample 0.1]   # -> data/work/train_feats.parquet
    python -m amlc.features test

Features are country-agnostic on purpose: `country` is never a model input, so
France (absent from train) is scored with the same similarity logic.
Pairs are processed in S1-hash buckets to bound memory.
"""
from __future__ import annotations

import math
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .paths import mem as default_mem, threads, work_dir

from .blocking import KEYS

N_BUCKETS = 48  # test split has ~100M pairs: ~2M per bucket keeps pandas under ~2 GB
DROP_ONLY_KEYS = ("'tok'",)  # pairs found ONLY by these keys are skipped (cost >> recall; see blocking report)
FEATURES = [
    "n_tset", "n_tsort", "n_concat_ratio", "n_concat_partial", "n_jw", "n_skel", "n_clean_ratio", "n_alias",
    "n_idf_overlap", "n_shared_max_idf", "n_unshared_max_idf", "n_len_diff", "n_tok_a", "n_tok_b",
    "legal_same", "legal_conflict", "legal_missing", "dom_any", "translit_any",
    "a_tset", "a_ratio", "a_street", "house_eq", "nums_jacc", "post_eq", "addr_empty",
    # v3: sound-key similarity, global name frequency (is this exact name shared by other S1
    # businesses?), which side lacks an address, house-number near-misses (2827 vs 2825)
    "n_phon", "nf_s1_a", "nf_s1_b", "nf_s1_lg_b", "nf_t_b", "addr_empty_s1", "addr_empty_t", "house_near",
    "is_s3", "n_keys", *[f"k_{k}" for k in KEYS],
    "s1_ncands", "t_ncands", "rank_in_s1", "gap_to_best_s1",
]


def _paths(split):
    w = work_dir() / "work"
    return (str(w / f"{split}_norm.parquet"), str(w / f"{split}_cands/*.parquet"),
            str(w / f"{split}_feats"), str(w / f"{split}_idf.parquet"))


def _idf_table(con, norm, out):
    con.execute(f"""
      COPY (SELECT country, tok, ln((SELECT count(*) FROM '{norm}') / count(*)::DOUBLE) AS idf FROM (
              SELECT country, unnest(string_split(name_core, ' ')) AS tok FROM '{norm}')
            WHERE tok <> '' GROUP BY 1, 2) TO '{out}' (FORMAT parquet)""")


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / 100.0


_IDF_DEFAULT = 16.3  # ln(#records); set per split in run()


def _idf_feats(core_a, core_b, country, idf):
    n = len(core_a)
    ov, smax, umax = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    default = _IDF_DEFAULT
    for i in range(n):
        ta, tb = set(core_a[i].split()), set(core_b[i].split())
        if not ta or not tb:
            continue
        c = country[i]
        w = {t: idf.get((c, t), default) for t in ta | tb}
        sh = ta & tb
        ws = sum(w[t] for t in sh)
        ov[i] = ws / (sum(w.values()) or 1.0)
        smax[i] = max((w[t] for t in sh), default=0.0)
        umax[i] = max((w[t] for t in (ta ^ tb)), default=0.0)
    return ov, smax, umax


def pair_features(df: pd.DataFrame, idf: dict) -> pd.DataFrame:
    """df has s1_id, cand_id, keys and *_a / *_b attribute columns."""
    f = pd.DataFrame({"s1_id": df.s1_id.values, "cand_id": df.cand_id.values})
    ca, cb = df.name_core_a.tolist(), df.name_core_b.tolist()
    f["n_tset"] = _cp(ca, cb, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(ca, cb, fuzz.token_sort_ratio)
    xa, xb = df.name_concat_a.tolist(), df.name_concat_b.tolist()
    f["n_concat_ratio"] = _cp(xa, xb, fuzz.ratio)
    f["n_concat_partial"] = _cp(xa, xb, fuzz.partial_ratio)
    f["n_jw"] = process.cpdist(xa, xb, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    f["n_skel"] = _cp(df.name_skel_a.tolist(), df.name_skel_b.tolist(), fuzz.ratio)
    f["n_clean_ratio"] = _cp(df.name_clean_a.tolist(), df.name_clean_b.tolist(), fuzz.ratio)
    al_a, al_b = df.alias_core_a.tolist(), df.alias_core_b.tolist()
    alias = np.maximum(_cp([x or "\x00" for x in al_a], cb, fuzz.token_set_ratio),
                       _cp(ca, [x or "\x00" for x in al_b], fuzz.token_set_ratio))
    has_alias = (df.alias_core_a.values != "") | (df.alias_core_b.values != "")
    f["n_alias"] = np.where(has_alias, alias, -1).astype(np.float32)
    ov, smax, umax = _idf_feats(ca, cb, df.country_a.tolist(), idf)
    f["n_idf_overlap"], f["n_shared_max_idf"], f["n_unshared_max_idf"] = ov, smax, umax
    la, lb = df.name_core_a.str.len().values, df.name_core_b.str.len().values
    f["n_len_diff"] = (np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    f["n_tok_a"] = df.name_core_a.str.count(" ").values + 1
    f["n_tok_b"] = df.name_core_b.str.count(" ").values + 1
    lga, lgb = df.legal_a.values, df.legal_b.values
    both = (lga != "") & (lgb != "")
    inter = np.array([bool(set(a.split("|")) & set(b.split("|"))) if a and b else False for a, b in zip(lga, lgb)])
    f["legal_same"] = (both & inter).astype(np.int8)
    f["legal_conflict"] = (both & ~inter).astype(np.int8)
    f["legal_missing"] = (~both).astype(np.int8)
    f["dom_any"] = (df.is_domain_a.values | df.is_domain_b.values).astype(np.int8)
    f["translit_any"] = (df.is_translit_a.values | df.is_translit_b.values).astype(np.int8)

    aa, ab = df.addr_clean_a.tolist(), df.addr_clean_b.tolist()
    empty = (df.addr_clean_a.values == "") | (df.addr_clean_b.values == "")
    f["a_tset"] = np.where(empty, -1, _cp(aa, ab, fuzz.token_set_ratio)).astype(np.float32)
    f["a_ratio"] = np.where(empty, -1, _cp(aa, ab, fuzz.ratio)).astype(np.float32)
    f["a_street"] = np.where(empty, -1, _cp(df.street_a.tolist(), df.street_b.tolist(), fuzz.ratio)).astype(np.float32)
    ha, hb = df.house_no_a.values, df.house_no_b.values
    f["house_eq"] = np.where((ha == "") | (hb == ""), -1, (ha == hb).astype(int)).astype(np.int8)
    na, nb = df.addr_nums_a.values, df.addr_nums_b.values
    f["nums_jacc"] = np.array([(len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split())))
                               if a and b else -1 for a, b in zip(na, nb)], np.float32)
    pa_, pb_ = df.postcode_a.values, df.postcode_b.values
    f["post_eq"] = np.where((pa_ == "") | (pb_ == ""), -1, (pa_ == pb_).astype(int)).astype(np.int8)
    f["addr_empty"] = empty.astype(np.int8)

    f["n_phon"] = _cp(df.name_phon_a.tolist(), df.name_phon_b.tolist(), fuzz.ratio)
    for c in ("nf_s1_a", "nf_s1_b", "nf_s1_lg_b", "nf_t_b"):
        f[c] = df[c].fillna(0).to_numpy(np.int32)
    f["addr_empty_s1"] = (df.addr_clean_a.values == "").astype(np.int8)
    f["addr_empty_t"] = (df.addr_clean_b.values == "").astype(np.int8)
    hd = process.cpdist(list(ha), list(hb), scorer=Levenshtein.distance, workers=-1)
    near = (hd <= 1) | np.array([bool(a) and bool(b) and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a))
                                 for a, b in zip(ha, hb)])
    f["house_near"] = np.where((ha == "") | (hb == ""), -1, np.where(ha == hb, 2, near.astype(int))).astype(np.int8)

    f["is_s3"] = (df.cand_id.str[:2].values == "S3").astype(np.int8)
    keys = [set(s.split("|")) for s in df["keys"].values]
    for k in KEYS:
        f[f"k_{k}"] = np.array([k in s for s in keys], np.int8)
    f["n_keys"] = f[[f"k_{k}" for k in KEYS]].sum(axis=1).astype(np.int8)
    return f


def add_context(f: pd.DataFrame) -> pd.DataFrame:
    """S1-side context (complete within a bucket, since buckets are by S1):
    how crowded is the candidate list and where this pair ranks on a cheap
    combined similarity. `t_ncands` comes from SQL over all pairs."""
    s = (f.n_tset + f.n_concat_ratio + f.n_idf_overlap + np.clip(f.a_tset, 0, 1)).astype(np.float32)
    f["s1_ncands"] = f.groupby("s1_id").s1_id.transform("size").astype(np.int32)
    f["rank_in_s1"] = s.groupby(f.s1_id).rank(ascending=False, method="min").astype(np.int16)
    f["gap_to_best_s1"] = (s.groupby(f.s1_id).transform("max") - s).astype(np.float32)
    return f


ATTRS = ["name_clean", "name_core", "name_concat", "name_skel", "name_phon", "legal", "is_domain", "is_translit", "alias_core",
         "addr_clean", "house_no", "addr_nums", "street", "postcode", "country"]


def run(split: str, sample: float = 1.0, mem: str = None) -> str:
    """Writes <split>_feats/part<b>.parquet, one file per S1-hash bucket."""
    import os
    import shutil
    norm, cands, out, idf_path = _paths(split)
    mem = mem or os.environ.get("AMLC_FEAT_MEM", "2GB")  # python side needs the rest of the RAM
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET enable_progress_bar=false; SET preserve_insertion_order=false; SET threads={threads()};")
    _idf_table(con, norm, idf_path)
    # only tokens seen >= 2 times are stored; unseen/singleton tokens get the max idf ln(N)
    n_rec = con.execute(f"SELECT count(*) FROM '{norm}'").fetchone()[0]
    global _IDF_DEFAULT
    _IDF_DEFAULT = float(np.log(n_rec))
    idf_df = con.execute(f"SELECT country, tok, idf FROM '{idf_path}' WHERE idf < {_IDF_DEFAULT - 1e-6}").df()
    idf = dict(zip(zip(idf_df.country, idf_df.tok), idf_df.idf.astype(np.float32)))
    del idf_df
    import gc; gc.collect()
    con.execute(f"CREATE TEMP TABLE tcount AS SELECT cand_id, count(*)::INT AS t_ncands FROM '{cands}' GROUP BY 1")
    # how many S1 businesses / S2+S3 records carry exactly this name (per country)
    con.execute(f"""CREATE TEMP TABLE nf AS SELECT country, name_concat,
                      count(*) FILTER (WHERE src = 'S1')::INT AS nf_s1, count(*) FILTER (WHERE src <> 'S1')::INT AS nf_t
                    FROM '{norm}' GROUP BY 1, 2""")
    con.execute(f"""CREATE TEMP TABLE nfl AS SELECT country, name_concat, legal, count(*)::INT AS nf_s1_lg
                    FROM '{norm}' WHERE src = 'S1' GROUP BY 1, 2, 3""")
    sel_a = ", ".join(f"a.{c} AS {c}_a" for c in ATTRS)
    sel_b = ", ".join(f"b.{c} AS {c}_b" for c in ATTRS)
    samp = f"AND (hash(c.s1_id) % 1000) < {int(sample * 1000)}" if sample < 1 else ""
    t0, total = time.time(), 0
    for b in range(N_BUCKETS):
        df = con.execute(f"""
          SELECT c.s1_id, c.cand_id, c.keys, t.t_ncands, {sel_a}, {sel_b},
                 na.nf_s1 AS nf_s1_a, nb.nf_s1 AS nf_s1_b, nb.nf_t AS nf_t_b, nl.nf_s1_lg AS nf_s1_lg_b
          FROM '{cands}' c
          JOIN tcount t USING (cand_id)
          JOIN '{norm}' a ON a.entity_id = c.s1_id
          JOIN '{norm}' b ON b.entity_id = c.cand_id
          LEFT JOIN nf na ON na.country = a.country AND na.name_concat = a.name_concat
          LEFT JOIN nf nb ON nb.country = b.country AND nb.name_concat = b.name_concat
          LEFT JOIN nfl nl ON nl.country = b.country AND nl.name_concat = b.name_concat AND nl.legal = b.legal
          WHERE (hash(c.s1_id) % {N_BUCKETS}) = {b} {samp}
            AND c.keys NOT IN ({", ".join(DROP_ONLY_KEYS)})""").df()
        if len(df):
            f = pair_features(df, idf)
            f["t_ncands"] = df.t_ncands.values
            f = add_context(f)
            pq.write_table(pa.Table.from_pandas(f, preserve_index=False), f"{out}/part{b:02d}.parquet",
                           compression="zstd")
            total += len(f)
        print(f"{split} bucket {b + 1}/{N_BUCKETS}: {total:,} pairs  {time.time() - t0:.0f}s", flush=True)
        del df
    return out


if __name__ == "__main__":
    s = float(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 1.0
    run(sys.argv[1], s)
