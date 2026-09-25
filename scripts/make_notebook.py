"""Builds notebooks/00_start_here.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# 00 · Start here — look at the data

Run the cells top to bottom (Shift+Enter). Works the same on your laptop, on Kaggle and on Colab.

* **Dev subset** (2% of Source-1 entities + all their matches + 2% of distractors) is used for exploring — fast, fits in RAM.
* The full data is only loaded by the pipeline, via DuckDB/Parquet.
* Scores are always computed with `erharness` so everyone's numbers are comparable."""),
    code("""# --- setup: make `amlc` and `erharness` importable from anywhere in the repo ---
import sys, pathlib
REPO = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src" / "amlc").exists())
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
pd.set_option("display.max_colwidth", 80, "display.width", 200)
from amlc import paths, data
print("raw data :", paths.raw_dir())
print("work dir :", paths.work_dir())"""),
    md("## 1 · Convert to Parquet + build the dev subset (first run only, ~5–15 min on a laptop)"),
    code("""import os
if not os.path.exists(data.pq("test_source3")):
    data.convert()
if not os.path.exists(data.pq("train_source1", dev=True)):
    data.make_dev()
print(open(paths.work_dir() / "profile.md").read() if (paths.work_dir() / "profile.md").exists() else data.profile())"""),
    md("## 2 · Load the dev subset"),
    code("""gt = data.load("train_ground_truth", dev=True)
s1 = data.load("train_source1", dev=True)
s2 = data.load("train_source2", dev=True)
s3 = data.load("train_source3", dev=True)
recs = pd.concat([s1, s2, s3]).set_index("entity_id")
gt["matches"] = gt.matched_entity_ids.map(lambda x: [t for t in x.split(",") if t])
gt["n"] = gt.matches.map(len)
print({k: len(v) for k, v in dict(gt=gt, s1=s1, s2=s2, s3=s3).items()})
gt.n.value_counts().sort_index()"""),
    md("## 3 · Read real match groups — this is where the normalisation rules come from"),
    code("""def show(s1_id):
    row = gt.loc[gt.source1_entity_id == s1_id].iloc[0]
    ids = [s1_id] + row.matches
    return recs.loc[[i for i in ids if i in recs.index], ["business_name", "business_address", "country"]]

country = s1.set_index("entity_id").country
for c in ["US", "India"]:
    pool = gt[(gt.source1_entity_id.map(country) == c) & (gt.n > 0)]
    for sid in pool.sample(min(3, len(pool)), random_state=1).source1_entity_id:
        display(show(sid))"""),
    md("## 4 · Singletons (score 1.0 only if we predict an empty list)"),
    code("""sing = gt[gt.n == 0].source1_entity_id
recs.loc[sing.sample(min(10, len(sing)), random_state=0), ["business_name", "business_address", "country"]]"""),
    md("## 5 · France — only in the test set, never in train"),
    code("""t1 = pd.read_parquet(data.pq("test_source1"), columns=["business_name", "business_address", "country"])
print(t1.country.value_counts())
fr = t1[t1.country == "France"]
fr.sample(min(15, len(fr)), random_state=0)"""),
    md("## 6 · Scoring sanity check with the harness\nAll-empty prediction scores exactly the singleton share."),
    code("""from erharness import metric
gt_sets = {r.source1_entity_id: set(r.matches) for r in gt.itertuples()}
rep = metric.evaluate({k: set() for k in gt_sets}, gt_sets)
print(rep.summary())"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/00_start_here.ipynb")
print("wrote notebooks/00_start_here.ipynb")
