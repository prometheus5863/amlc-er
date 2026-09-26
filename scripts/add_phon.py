"""One-off: add name_phon to an existing <split>_norm.parquet (made before v3) without re-normalising."""
import sys, os, time
sys.path.insert(0, "src")
import duckdb, pandas as pd
from amlc.normalize import phonetic
from amlc.paths import work_dir

split = sys.argv[1]
p = str(work_dir() / "work" / f"{split}_norm.parquet")
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET preserve_insertion_order=true")
if "name_phon" in con.execute(f"DESCRIBE SELECT * FROM '{p}'").df().column_name.tolist():
    sys.exit("already has name_phon")
t = time.time()
u = con.execute(f"SELECT DISTINCT name_concat FROM '{p}'").df()
u["name_phon"] = [phonetic(x) for x in u.name_concat.to_numpy()]
con.register("u", u)
con.execute(f"COPY (SELECT n.*, u.name_phon FROM '{p}' n JOIN u USING (name_concat)) TO '{p}.tmp' (FORMAT parquet, COMPRESSION zstd)")
os.replace(p + ".tmp", p)
print(f"{split}: added name_phon for {len(u):,} distinct names in {time.time()-t:.0f}s")
