"""Builds notebooks/kaggle_rerank.ipynb: the GPU re-ranker, run on top of a finished pipeline run."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# AMLC 2026 · multilingual cross-encoder re-ranker (GPU)

Re-scores the pairs the LightGBM pipeline is unsure about with a fine-tuned multilingual transformer,
measures the gain on **unseen businesses**, and writes a new `matching_results.tsv` **only if it wins**.

**Before running** (right-hand panel):
1. **Accelerator: GPU T4 x2** (or P100).
2. **Add Input** → the `amazon-ml-challenge-2026` dataset by **satwiksps** (same as the pipeline notebook).
3. **Add Input → Your Work → Notebooks** → the finished **pipeline** notebook version (v4). Its `output/rerank/` folder is the input here.
4. **Internet: On** and **Secrets → `GITHUB_TOKEN`** ticked (same as the pipeline notebook).

Then **Save Version → Save & Run All**. ~1.5–2.5 h. Result: `output/rerank_result.json` (stage 1 vs blended on unseen
businesses) and, if blended wins, `output/matching_results.tsv` + `candidate_pairs.tsv` ready to submit."""),
    code("""BRANCH = "v3"
REPO = "prometheus5863/amlc-er"
N_TRAIN = 1_500_000      # cross-encoder fine-tuning pairs (uncertain band only)
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # Apache-2.0, 50+ languages"""),
    code("""import os, subprocess, sys
token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    print("no GITHUB_TOKEN secret: cloning without auth")
url = f"https://{token + '@' if token else ''}github.com/{REPO}.git"
subprocess.run(["rm", "-rf", "/tmp/amlc-er"])
r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "/tmp/amlc-er"], capture_output=True, text=True)
err = r.stderr.replace(token, "***") if token else r.stderr
if r.returncode != 0:
    raise RuntimeError("git clone failed: " + err + " | check Internet ON, GITHUB_TOKEN secret ticked, branch exists")
print(subprocess.run(["git", "-C", "/tmp/amlc-er", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - turn on the GPU accelerator!")"""),
    code("""!pip install -q duckdb rapidfuzz lightgbm 2>&1 | tail -1"""),
    code("""import glob
hits = glob.glob("/kaggle/input/**/rerank/train_pairs.parquet", recursive=True)
assert hits, "pipeline output not found: Add Input -> Your Work -> the finished pipeline notebook (needs output/rerank/)"
IN = os.path.dirname(hits[0]); print("re-ranker input:", IN, os.listdir(IN))
env = {**os.environ, "PYTHONPATH": "/tmp/amlc-er/src:/tmp/amlc-er", "AMLC_CE_MODEL": BASE, "AMLC_CE_NTRAIN": str(N_TRAIN)}
# raw TSV -> parquet (texts + test S1 order), ~1-2 min
subprocess.run([sys.executable, "-c", "from amlc import data; data.convert()"], cwd="/tmp/amlc-er", env=env, check=True)
p = subprocess.Popen([sys.executable, "-u", "-m", "amlc.rerank", "run", IN, "/kaggle/working/output"],
                     cwd="/tmp/amlc-er", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
assert p.returncode == 0, "re-ranker failed - see log above\""""),
    code("""# verdict + files to submit
import json, shutil, pathlib
out = pathlib.Path("/kaggle/working/output")
r = json.load(open(out / "rerank_result.json")); print(json.dumps(r, indent=1))
cands = glob.glob("/kaggle/input/**/output/candidate_pairs.tsv", recursive=True)
if cands:
    shutil.copy(cands[0], out / "candidate_pairs.tsv")
if r["gain"] > 0:
    print(f"\\nSUBMIT output/matching_results.tsv  (unseen-business gain {r['gain']:+.5f})")
else:
    print("\\nNo gain: keep the pipeline notebook's matching_results.tsv")
for f in sorted(out.iterdir()):
    print(f.name)"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/kaggle_rerank.ipynb")
print("wrote notebooks/kaggle_rerank.ipynb")
