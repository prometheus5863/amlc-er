"""One command for the whole pipeline, with each step skipped when its output exists.

    python -m amlc.pipeline all              # everything: data -> matching_results.tsv (validated)
    python -m amlc.pipeline train            # up to the trained model + local score
    python -m amlc.pipeline predict          # test side only (needs a trained model)
    python -m amlc.pipeline all --force      # recompute every step
    python -m amlc.pipeline all --sample 0.2 # train on 20% of Source-1 entities (default 10%)

Same code on a laptop, in this repo's cloud workspace, or in a Kaggle notebook
(paths.py finds the data and picks memory/threads). Finished artifacts
(model.txt, decode.json, the run log) are copied next to the outputs so a
Kaggle "Save Version" keeps them.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time

from . import blocking, features, model, prep
from .data import pq as pq_path
from .paths import REPO, output_dir, parquet_dir, raw_dir, work_dir

W = lambda name: str(work_dir() / "work" / name)  # noqa: E731


def _done(path_glob: str) -> bool:
    return bool(glob.glob(path_glob))


def step(name, output_glob, fn, force):
    if not force and _done(output_glob):
        print(f"[skip] {name} (found {output_glob})", flush=True)
        return
    t = time.time()
    print(f"[run ] {name}", flush=True)
    fn()
    print(f"[done] {name} in {time.time() - t:.0f}s", flush=True)


def ensure_parquet(force=False):
    if (parquet_dir() / "test_source3.parquet").exists() and not force:
        print(f"[skip] parquet (using {parquet_dir()})", flush=True)
        return
    from . import data
    data.convert(force=force)


def validate() -> bool:
    """Official validator. Needs test TSVs: uses the raw folder if present,
    otherwise writes them from Parquet into a temp folder."""
    out = output_dir()
    try:
        test_dir = str(raw_dir() / "test")
    except FileNotFoundError:
        import duckdb
        test_dir = str(work_dir() / "test_tsv")
        os.makedirs(test_dir, exist_ok=True)
        con = duckdb.connect()
        for s in "123":
            dst = f"{test_dir}/test_source{s}.tsv"
            if not os.path.exists(dst):
                con.execute(f"COPY (SELECT * FROM '{pq_path(f'test_source{s}')}') TO '{dst}' "
                            f"(HEADER, DELIMITER '\t', QUOTE '')")
    r = subprocess.run([sys.executable, str(REPO / "utils" / "validate_submission.py"),
                        "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"),
                        "--test-dir", test_dir, "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    return r.returncode == 0


def save_artifacts():
    out = output_dir()
    for f in ("model.txt", "decode.json"):
        if os.path.exists(W(f)):
            shutil.copy(W(f), out / f)
    runs = REPO / "experiments" / "runs"
    if runs.exists():
        shutil.copytree(runs, out / "experiment_runs", dirs_exist_ok=True)
    print(f"artifacts copied to {out}", flush=True)


def main(argv):
    what = argv[0] if argv else "all"
    force = "--force" in argv
    frac = float(argv[argv.index("--sample") + 1]) if "--sample" in argv else 0.1
    t0 = time.time()
    ensure_parquet(force)
    if what in ("all", "train"):
        step("normalise train", W("train_norm.parquet"), lambda: prep.run("train"), force)
        step("blocking train", W("train_cands/*.parquet"), lambda: blocking.run("train"), force)
        step(f"features train (sample {frac})", W("train_feats/*.parquet"), lambda: features.run("train", frac), force)
        step("train model (OOF + decode sweep)", W("decode.json"), lambda: model.train(frac), force)
        step("fit final model", W("model.txt"), lambda: model.fit_final(frac=frac), force)
    if what in ("all", "predict"):
        step("normalise test", W("test_norm.parquet"), lambda: prep.run("test"), force)
        step("blocking test", W("test_cands/*.parquet"), lambda: blocking.run("test"), force)
        step("features test", W("test_feats/*.parquet"), lambda: features.run("test"), force)
        step("predict + decode", str(output_dir() / "matching_results.tsv"), model.predict, force)
        ok = validate()
        print("VALIDATOR:", "PASS" if ok else "FAILED — see messages above", flush=True)
    save_artifacts()
    print(f"pipeline finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
