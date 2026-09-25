"""Experiment log: every local run and every leaderboard submission, in one place.

  experiments/runs/<id>.json     one file per run (id = MMDD-HHMMSS-user; no git conflicts)
  experiments/runs/<id>.tsv.gz   per-entity scores (enables bootstrap between any two runs)
  experiments/results.md         auto-rendered table for the README

The leaderboard is treated as a measuring instrument: `alignment()` tracks the
offset between local and LB scores. A stable offset means local validation
predicts the LB, so you can rank ideas offline; a drifting offset means your
validation split does not look like test (fix that before anything else).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metric import Report

ROOT = os.environ.get("ERH_EXPERIMENTS", "experiments")


def _paths(root: str):
    return os.path.join(root, "runs"), os.path.join(root, "results.md")


def _user() -> str:
    u = os.environ.get("ERH_USER") or os.environ.get("USER") or os.environ.get("USERNAME") or "anon"
    return "".join(ch for ch in u.lower() if ch.isalnum())[:12] or "anon"


def _git() -> Optional[str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if sha.returncode:
            return None
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def load(root: str = ROOT) -> List[Dict]:
    """All runs, oldest first. One JSON file per run, so teammates never
    conflict when committing results."""
    runs_dir, _ = _paths(root)
    if not os.path.isdir(runs_dir):
        return []
    runs = []
    for fn in os.listdir(runs_dir):
        if fn.endswith(".json"):
            with open(os.path.join(runs_dir, fn)) as f:
                runs.append(json.load(f))
    return sorted(runs, key=lambda r: (r["time"], r["id"]))


def _save_one(run: Dict, root: str) -> None:
    runs_dir, _ = _paths(root)
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run['id']}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(run, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def log_run(name: str, report: Report, table: pd.DataFrame, config: Optional[Dict] = None,
            notes: str = "", ci: Optional[tuple] = None, root: str = ROOT) -> str:
    runs_dir, _ = _paths(root)
    os.makedirs(runs_dir, exist_ok=True)
    now = dt.datetime.now()
    rid = f"{now:%m%d-%H%M%S}-{_user()}"
    while os.path.exists(os.path.join(runs_dir, f"{rid}.json")):
        rid += "x"
    cfg = config or {}
    table[["s1_id", "score", "hits", "n_pred", "n_true", "outcome"]].to_csv(
        os.path.join(runs_dir, f"{rid}.tsv.gz"), sep="\t", index=False)
    _save_one({
        "id": rid,
        "name": name,
        "user": _user(),
        "time": now.isoformat(timespec="seconds"),
        "git": _git(),
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8],
        "config": cfg,
        "local": report.score,
        "local_ci": list(ci) if ci else None,
        "spec": report.spec,
        "open_rate": report.open_rate,
        "f_open": report.f_open,
        "singleton_frac": report.singleton_frac,
        "n_entities": report.n,
        "lb": None,
        "notes": notes,
    }, root)
    render(root)
    return rid


def set_lb(run_id: str, lb: float, notes: str = "", root: str = ROOT) -> Dict:
    for r in load(root):
        if r["id"] == run_id:
            r["lb"] = float(lb)
            r["lb_time"] = dt.datetime.now().isoformat(timespec="seconds")
            if notes:
                r["notes"] = (r.get("notes", "") + " | " + notes).strip(" |")
            _save_one(r, root)
            render(root)
            return r
    raise KeyError(run_id)


def run_table(run_id: str, root: str = ROOT) -> pd.DataFrame:
    return pd.read_csv(os.path.join(_paths(root)[0], f"{run_id}.tsv.gz"), sep="\t", dtype={"s1_id": str})


def alignment(root: str = ROOT, drift_tol: float = 0.02) -> Dict:
    submitted = [r for r in load(root) if r.get("lb") is not None]
    # the all-empty probe measures the test singleton rate; it says nothing about model transfer
    runs = [r for r in submitted if r.get("config", {}).get("kind") != "all_empty"]
    all_empty = [r for r in submitted if r.get("config", {}).get("kind") == "all_empty"]
    if not runs:
        out = {"n_submitted": len(submitted)}
        if all_empty:
            out["test_singleton_frac"] = all_empty[-1]["lb"]
        return out
    gaps = np.array([r["lb"] - r["local"] for r in runs])
    out = {
        "n_submitted": len(submitted),
        "mean_gap_lb_minus_local": float(gaps.mean()),
        "gap_std": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
        "drifting_runs": [r["id"] for r, g in zip(runs, gaps) if abs(g - gaps.mean()) > drift_tol],
    }
    if len(runs) >= 3:
        loc = pd.Series([r["local"] for r in runs])
        lb = pd.Series([r["lb"] for r in runs])
        out["rank_corr_local_vs_lb"] = float(loc.rank().corr(lb.rank()))
    if all_empty:
        out["test_singleton_frac"] = all_empty[-1]["lb"]
    return out


def render(root: str = ROOT, budget: Optional[int] = None) -> str:
    runs = load(root)
    _, md = _paths(root)
    hdr = "| id | who | name | local | 95% CI | spec | open | F_open | LB | LB−local | git | notes |\n" \
          "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = ""
    for r in runs:
        ci = r.get("local_ci")
        ci_s = f"[{ci[1]:.4f}, {ci[2]:.4f}]" if ci else ""
        lb = r.get("lb")
        lb_s = "" if lb is None else f"{lb:.4f}"
        gap_s = "" if lb is None else f"{lb - r['local']:+.4f}"
        body += (f"| {r['id']} | {r.get('user', '')} | {r['name']} | {r['local']:.4f} | {ci_s} | {r['spec']:.3f} | "
                 f"{r['open_rate']:.3f} | {r['f_open']:.3f} | {lb_s} | {gap_s} | "
                 f"{r.get('git') or ''} | {r.get('notes', '')} |\n")
    al = alignment(root)
    foot = f"\nSubmissions used: {al['n_submitted']}" + (f" / {budget}" if budget else "") + "\n"
    if "mean_gap_lb_minus_local" in al:
        foot += f"Mean LB−local gap (model runs): {al['mean_gap_lb_minus_local']:+.4f}"
        if al.get("gap_std") is not None:
            foot += f" (std {al['gap_std']:.4f})"
        if al.get("rank_corr_local_vs_lb") is not None:
            foot += f"; rank corr local↔LB: {al['rank_corr_local_vs_lb']:.2f}"
        if al["drifting_runs"]:
            foot += f"\n⚠ runs whose gap drifts >0.02 from the mean: {', '.join(al['drifting_runs'])}"
        foot += "\n"
    if "test_singleton_frac" in al:
        foot += f"Test singleton fraction (from all-empty submission): {al['test_singleton_frac']:.4f}\n"
    text = "# Experiment log\n\n" + hdr + body + foot
    os.makedirs(root, exist_ok=True)
    with open(md, "w") as f:
        f.write(text)
    return text
