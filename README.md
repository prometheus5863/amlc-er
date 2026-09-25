# amlc-er — Amazon ML Challenge 2026: Business Entity Resolution

For every Source 1 business, find all matching records in Source 2 and Source 3.
The score is **macro F0.5 per Source 1 entity**, so precision counts twice as much as recall.

## Facts from the data (train)

| | |
|---|---|
| Source 1 entities | 2,206,821 (test: 1,732,544) |
| Source 2 / Source 3 records | 5.03M / 5.29M (test: 4.89M / 5.08M) |
| Singletons (no match) | **5.6%**: the empty-list decision is small here |
| Matches per entity | 1: 5.4% · 2: 17% · 3: 24% · 4: 22% · 5+: 26%, so **many-to-many, never 1:1** |
| Countries | US, India in train; **France appears only in test** |
| Noise | Devanagari names (राम मार्केटिंग प्राइवेट लिमिटेड), junk prefixes (`--`, `<<`), website-style names, reordered address parts, missing addresses |
| Model rule | final model must be MIT/Apache-2.0 and ≤ 8B parameters; no external data or APIs |

## Folder layout

```
amlc-er/
├── notebooks/00_start_here.ipynb   ← open this first
├── src/amlc/                       ← pipeline code (paths, data, later: normalize, blocking, features, models)
├── erharness/                      ← scoring harness (metric, bootstrap, sweeps, oracles, experiment log)
├── tests/                          ← run: python -m pytest -q
├── experiments/runs/               ← one file per scored run (commit these)
├── output/                         ← matching_results.tsv, candidate_pairs.tsv
├── configs/paths.local.json        ← where YOUR copy of the raw data is (not committed)
└── data/                           ← Parquet, dev subset, profile (not committed)
```

The raw TSVs are **never** copied into the repo. Everyone points to their own copy in `configs/paths.local.json`:

```json
{"raw": "C:/Users/<you>/Downloads/.../student_resource/dataset"}
```

On Kaggle nothing needs to be set: the code finds `/kaggle/input/...` by itself.

## Run it on your laptop

1. Install Python 3.10+ and VS Code (with the *Python* and *Jupyter* extensions).
2. In a terminal inside this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Open `notebooks/00_start_here.ipynb` in VS Code, pick the `.venv` kernel, and run all cells.
   The first run converts the data to Parquet and builds the dev subset (takes a few minutes).

## Run it on Kaggle (for teammates, and for GPU later)

1. Kaggle → **Create → New Notebook**.
2. Right panel → **Add Input** → search for the challenge dataset → **Add**.
3. Right panel → Settings → **Internet on** (needs a phone-verified account).
4. First cell:
   ```
   !git clone https://github.com/<you>/amlc-er.git
   %cd amlc-er
   !pip install -q duckdb
   ```
   For a private repo, save a GitHub token under **Add-ons → Secrets** and clone with it.
5. Upload `notebooks/00_start_here.ipynb` (File → Import notebook) or copy in its cells.

## Team rules

1. Everyone uses the same folds (`folds.tsv`, committed once) and the same scorer (`erharness`).
2. Every scored run is logged: `python -m erharness score ... --log "<what changed>"`.
3. Only changes that `python -m erharness compare <old> <new>` calls **WINS** go to the leaderboard. One person submits.
4. Explore in your own notebook (`notebooks/NN_topic_yourname.ipynb`). Code that works moves into `src/amlc/`.

## Commands

```
python -m amlc.data convert     # TSV → Parquet (row counts verified against line counts)
python -m amlc.data dev         # 2% dev subset, keeps every match group whole
python -m amlc.data profile     # data/profile.md
python -m erharness -h          # scoring, sweeps, oracles, blocking audit, experiment log
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir <raw>/test
```
