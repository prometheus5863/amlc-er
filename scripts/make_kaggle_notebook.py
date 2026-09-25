"""Builds notebooks/kaggle_pipeline.ipynb: the one notebook the team runs on Kaggle."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# AMLC 2026 · entity resolution — full pipeline

Runs the whole pipeline from the team repo and writes a **validated** `matching_results.tsv` + `candidate_pairs.tsv`.

**Before running** (right-hand panel):
1. **Add Input** → the team's private dataset (the 7 Parquet files, or the official TSV folder — both work).
2. **Settings → Internet: On** (needed to clone the repo).
3. **Add-ons → Secrets** → `GITHUB_TOKEN` = a GitHub token with read access to the repo (only needed while the repo is private).

Then **Save Version → Save & Run All (Commit)**. It runs in the background (~1.5–2 h on CPU) and the files appear under **Output**.
To try your own changes: push a branch and set `BRANCH` below."""),
    code("""BRANCH = "baseline-v1"        # <- change to your branch
SAMPLE = 0.1                   # share of Source-1 entities used for training
REPO = "prometheus5863/amlc-er\""""),
    code("""# clone into /tmp so the token never ends up in the saved notebook output
import os, subprocess, sys
token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    print("no GITHUB_TOKEN secret: cloning without auth (works only if the repo is public)")
url = f"https://{token + '@' if token else ''}github.com/{REPO}.git"
subprocess.run(["rm", "-rf", "/tmp/amlc-er"])
r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "/tmp/amlc-er"], capture_output=True, text=True)
print(r.stderr.replace(token, "***") if token else r.stderr)
print(subprocess.run(["git", "-C", "/tmp/amlc-er", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)"""),
    code("""!pip install -q duckdb rapidfuzz lightgbm 2>&1 | tail -1"""),
    code("""# run the pipeline (every step prints progress; finished steps are skipped on re-runs)
env = {**os.environ, "PYTHONPATH": "/tmp/amlc-er/src:/tmp/amlc-er"}
p = subprocess.Popen([sys.executable, "-u", "-m", "amlc.pipeline", "all", "--sample", str(SAMPLE)],
                     cwd="/tmp/amlc-er", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if " rows  " not in line:          # keep the log readable
        print(line, end="")
p.wait()
assert p.returncode == 0, "pipeline failed — see log above\""""),
    code("""# local score of this run (out-of-fold on the training sample) + what was written
import json, pathlib
out = pathlib.Path("/kaggle/working/output")
d = json.load(open(out / "decode.json"))
for k, v in d["local"].items():
    print(f"{k:>9}: macro F0.5 = {v['score']:.5f}   gate={v['gate']} pair={v['pair']}")
for f in sorted(out.iterdir()):
    print(f"{f.name:<28} {f.stat().st_size / 1e6:8.1f} MB" if f.is_file() else f"{f.name}/")"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/kaggle_pipeline.ipynb")
print("wrote notebooks/kaggle_pipeline.ipynb")
