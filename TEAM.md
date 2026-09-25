# Team guide

Everything the team shares lives in three places:

| What | Where | Who sets it up |
|---|---|---|
| Code, notebooks, experiment log | GitHub `prometheus5863/amlc-er` (private) | Harsh adds each teammate as a collaborator |
| Data (7 Parquet files, 0.96 GB) | a **private Kaggle dataset** | Harsh uploads once, shares with teammates' Kaggle usernames |
| Heavy runs (full pipeline, ~1.5–2 h) | Kaggle notebook `notebooks/kaggle_pipeline.ipynb` | anyone, in their own Kaggle account |

## One-time setup (Harsh)

1. **GitHub** → repo → Settings → Collaborators → *Add people* → each teammate's GitHub username.
2. **Kaggle dataset** → kaggle.com → *Create → New Dataset* → drag in the 7 files from
   `projects\amlc-er\data\parquet\` → name it `amlc-2026-parquet` → keep **Private** → *Create*.
   Then on the dataset page → *Share* (or ⋯ → Sharing) → add teammates' Kaggle usernames.
3. **Kaggle notebook** → *Create → New Notebook* → *File → Import Notebook* → upload
   `notebooks/kaggle_pipeline.ipynb` → *Share* → add teammates as collaborators.

## One-time setup (each teammate)

1. Accept the GitHub invite (email).
2. Make a GitHub token: GitHub → Settings → Developer settings → *Fine-grained tokens* → *Generate* →
   Repository access: only `amlc-er` → Permissions: *Contents: Read* → copy the token.
3. In the Kaggle notebook: *Add-ons → Secrets → Add* → name `GITHUB_TOKEN`, value = the token.
   (Kaggle secrets are per person and never shared with collaborators.)
4. Kaggle needs a **phone-verified** account for *Internet: On*.

## Running

- Open the notebook → check the dataset is attached (*Add Input* if not) → *Internet: On* →
  *Save Version → Save & Run All (Commit)*. You can close the tab; it runs in the background.
- Results appear under the version's **Output**: `output/matching_results.tsv`, `candidate_pairs.tsv`,
  `model.txt`, `decode.json` and the local score.

## Trying an idea

1. `git checkout -b yourname/idea` → change code in `src/amlc/` → `python -m pytest -q` → push.
2. In the notebook set `BRANCH = "yourname/idea"` → run.
3. Compare against the current best with `python -m erharness compare <old_run> <new_run>`.
   Only a **WINS** verdict goes to the leaderboard, and only the submission owner uploads.

## Rules that protect the score

- Never edit `erharness/`, the folds, or the ground truth to "improve" a number.
- No external data or APIs (competition rule — disqualification). Any new package must be checked:
  no downloaded databases (e.g. geocoders, postcode lists).
- Final model must be MIT/Apache-2.0 and ≤ 8B parameters.
- Don't commit data. `.gitignore` already excludes `data/` and outputs.
