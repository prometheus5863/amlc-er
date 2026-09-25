"""Command-line entry point:  python -m erharness <command> ...

Run `python -m erharness -h` or `python -m erharness <command> -h` for flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import pandas as pd

from . import analysis, bootstrap, metric, splits, tracker
from .decode import DecodeConfig, Prepared, decode
from .io import label_candidates, read_candidates, read_groups, read_match_sets, write_match_sets
from .sweep import DEFAULT_GRID, Scorer, best_config, edge_warnings, grid_search
from .validate import check_submission


def _cands(a):
    return read_candidates(a.cands, a.s1_col, a.cand_col, a.prob_col)


def _cfg(path):
    if not path:
        return DecodeConfig()
    with open(path) as f:
        return DecodeConfig(**json.load(f))


def _table_for(ref, gt):
    """A per-entity table from either a run id (exp007) or a prediction file."""
    if not os.path.exists(ref):  # a run id
        return tracker.run_table(ref)
    return metric.per_entity(read_match_sets(ref), gt)


def _ids_from(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df[df.columns[0]].str.strip().tolist()


# ------------------------------------------------------------------ commands --
def cmd_score(a):
    gt = read_match_sets(a.gt)
    pred = read_match_sets(a.pred)
    groups = read_groups(a.groups) if a.groups else None
    table = metric.per_entity(pred, gt, strict=a.strict)
    rep = metric.report_from_table(table, groups)
    mean, lo, hi = bootstrap.ci(table)
    print(rep.summary())
    print(f"  95% bootstrap CI  [{lo:.5f}, {hi:.5f}]  (half-width {(hi - lo) / 2:.5f} = your noise floor)")
    if a.test_s is not None:
        print(f"  projected at test singleton frac {a.test_s:.4f}: {metric.project_score(rep, a.test_s):.5f}")
    if a.dump:
        table.to_csv(a.dump, sep="\t", index=False)
    if a.log:
        cfg = json.load(open(a.config)) if a.config else {}
        rid = tracker.log_run(a.log, rep, table, cfg, a.notes, (mean, lo, hi))
        print(f"logged as {rid}")


def cmd_compare(a):
    gt = read_match_sets(a.gt) if a.gt else None
    ta, tb = _table_for(a.a, gt), _table_for(a.b, gt)
    print(bootstrap.compare(ta, tb, n_boot=a.n_boot).summary())


def cmd_sweep(a):
    gt = read_match_sets(a.gt)
    scorer = Scorer(_cands(a), gt)
    grid = json.load(open(a.grid)) if a.grid else DEFAULT_GRID
    base = _cfg(a.base)
    res = grid_search(scorer, grid, base, robust=not a.argmax)
    pd.set_option("display.width", 160)
    print(res.head(a.top).to_string(index=False))
    best = best_config(res, base)
    print("\npicked:", json.dumps(best.as_dict()))
    for w in edge_warnings(res, grid):
        print("⚠", w)
    print("oracle ceilings for the pick:")
    print(analysis.oracles(scorer, best).to_string(index=False))
    if a.out:
        res.to_csv(a.out, sep="\t", index=False)
    if a.save_config:
        json.dump(best.as_dict(), open(a.save_config, "w"), indent=2)
        print(f"saved -> {a.save_config}")


def cmd_decode(a):
    ids = _ids_from(a.ids)
    prep = Prepared(_cands(a), ids)
    sets = decode(prep, _cfg(a.config))
    write_match_sets(sets, a.out, a.id_header, a.match_header, order=ids)
    n_open = sum(bool(v) for v in sets.values())
    print(f"wrote {a.out}: {len(ids)} rows, {n_open} non-empty ({n_open / max(len(ids), 1):.3f})")


def cmd_oracles(a):
    gt = read_match_sets(a.gt)
    print(analysis.oracles(Scorer(_cands(a), gt), _cfg(a.config)).to_string(index=False))


def cmd_blocking(a):
    gt = read_match_sets(a.gt)
    c = read_candidates(a.cands, a.s1_col, a.cand_col, prob_col=None)
    res = analysis.blocking_audit(c, gt, a.n_targets)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            continue
        print(f"{k:<22} {v}")
    if "per_key" in res:
        print("\nper key:\n" + res["per_key"].to_string(index=False))
        print("\ngreedy key order (marginal recall per key added):\n" + res["greedy_key_order"].to_string(index=False))
    print(f"\nmissed true pairs: {len(res['missed_true_pairs'])}")
    if a.missed:
        res["missed_true_pairs"].to_csv(a.missed, sep="\t", index=False)
        print(f"  -> {a.missed} (join with raw records and eyeball them)")


def cmd_errors(a):
    gt = read_match_sets(a.gt)
    buckets = analysis.error_pairs(Scorer(_cands(a), gt), _cfg(a.config), gt)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, df in buckets.items():
        df.to_csv(os.path.join(a.out_dir, f"{name}.tsv"), sep="\t", index=False)
        print(f"{name:<24} {len(df):>7} pairs -> {a.out_dir}/{name}.tsv")


def cmd_calib(a):
    gt = read_match_sets(a.gt)
    c = label_candidates(_cands(a), gt)
    t = analysis.calibration_table(c, a.bins)
    print(t.to_string(index=False))
    print(f"ECE = {t.attrs['ece']:.4f}  (>0.03: calibrate before expected_f decoding)")
    if a.folds and a.out:
        f = pd.read_csv(a.folds, sep="\t", dtype={"s1_id": str})
        c["prob_raw"] = c["prob"]
        c["prob"] = analysis.calibrate_oof(c, dict(zip(f.s1_id, f.fold)), a.method).to_numpy()
        t2 = analysis.calibration_table(c, a.bins)
        print(f"after {a.method} (cross-fitted): ECE = {t2.attrs['ece']:.4f}")
        c.drop(columns="label").to_csv(a.out, sep="\t", index=False)
        print(f"-> {a.out} (prob = calibrated, prob_raw kept)")


def cmd_split(a):
    gt = read_match_sets(a.gt)
    groups = read_groups(a.groups) if a.groups else None
    fold = splits.kfold(gt, a.k, groups, a.seed)
    pd.DataFrame(sorted(fold.items()), columns=["s1_id", "fold"]).to_csv(a.out, sep="\t", index=False)
    print(splits.fold_balance(gt, fold, groups).to_string())
    print(f"-> {a.out}")


def cmd_empty(a):
    ids = _ids_from(a.ids)
    write_match_sets({e: set() for e in ids}, a.out, a.id_header, a.match_header, order=ids)
    print(f"wrote all-empty submission {a.out} ({len(ids)} rows). Its LB score = test singleton fraction.")
    if a.gt:
        gt = read_match_sets(a.gt)
        table = metric.per_entity({e: set() for e in gt}, gt)
        rid = tracker.log_run("all-empty", metric.report_from_table(table), table, {"kind": "all_empty"},
                              "submit this; then `lb <id> <score>`")
        print(f"logged as {rid} (local = train singleton frac {table.score.mean():.4f})")


def cmd_lb(a):
    r = tracker.set_lb(a.run_id, a.score, a.notes)
    print(f"{r['id']}: local {r['local']:.4f}  LB {r['lb']:.4f}  gap {r['lb'] - r['local']:+.4f}")
    print(json.dumps(tracker.alignment(), indent=2))


def cmd_report(a):
    print(tracker.render(budget=a.budget))


def cmd_validate(a):
    targets = None
    if a.targets:
        targets = [t for p in a.targets for t in _ids_from(p)]
    probs = check_submission(a.pred, _ids_from(a.ids), targets, a.sample)
    if probs:
        print("PROBLEMS:\n  " + "\n  ".join(probs))
        sys.exit(1)
    print("OK")


def cmd_demo(a):
    from . import synth
    d = a.out_dir
    synth.make(n=a.n, out_dir=d)
    print(f"synthetic data -> {d}/ (gt.tsv, candidates.tsv, groups.tsv)\n")
    gt = read_match_sets(f"{d}/gt.tsv")
    cands = read_candidates(f"{d}/candidates.tsv")
    groups = read_groups(f"{d}/groups.tsv")
    print("== blocking audit ==")
    b = analysis.blocking_audit(cands, gt)
    print({k: v for k, v in b.items() if not isinstance(v, pd.DataFrame)})
    scorer = Scorer(cands, gt)
    print("\n== sweep (threshold mode) ==")
    res = grid_search(scorer, DEFAULT_GRID)
    print(res.head(5).to_string(index=False))
    best = best_config(res)
    print("\n== expected-F decoding (no thresholds) ==")
    print("raw probs      :", {k: round(v, 4) for k, v in scorer.score(DecodeConfig(mode="expected_f")).items()})
    lab = label_candidates(cands, gt)
    print(f"raw ECE        : {analysis.calibration_table(lab).attrs['ece']:.4f}")
    fold = splits.kfold(gt, 5, groups)
    lab["prob"] = analysis.calibrate_oof(lab, fold).to_numpy()
    print(f"calibrated ECE : {analysis.calibration_table(lab).attrs['ece']:.4f}")
    cal_scorer = Scorer(lab, gt)
    print("calibrated     :", {k: round(v, 4) for k, v in cal_scorer.score(DecodeConfig(mode="expected_f")).items()})
    print("\n== oracles for the pick ==")
    print(analysis.oracles(scorer, best).to_string(index=False))
    naive = DecodeConfig(gate=0.5, pair=0.5)
    t_naive = metric.per_entity(decode(scorer.prep, naive), gt)
    t_best = metric.per_entity(decode(scorer.prep, best), gt)
    print("\n== naive 0.5/0.5 vs tuned: paired bootstrap ==")
    print(bootstrap.compare(t_naive, t_best).summary())
    print("\n== tuned report ==")
    print(metric.report_from_table(t_best, groups).summary())


# ---------------------------------------------------------------------- main --
def main(argv=None):
    warnings.simplefilter("always", UserWarning)
    p = argparse.ArgumentParser(prog="erharness", description="Local scoring harness for macro-F0.5 entity resolution")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cand_args(sp):
        sp.add_argument("--cands", required=True, help="scored candidate pairs TSV")
        sp.add_argument("--s1-col", default="s1_id")
        sp.add_argument("--cand-col", default="cand_id")
        sp.add_argument("--prob-col", default="prob")

    def hdr_args(sp):
        sp.add_argument("--id-header", default="source1_entity_id", help="first column header (copy the official sample)")
        sp.add_argument("--match-header", default="matched_entity_ids", help="second column header (copy the official sample)")

    s = sub.add_parser("score", help="macro F0.5 + decomposition + CI for a prediction file")
    s.add_argument("--gt", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--groups"); s.add_argument("--strict", action="store_true")
    s.add_argument("--test-s", type=float, help="project to this test singleton fraction")
    s.add_argument("--dump", help="write per-entity table")
    s.add_argument("--log", metavar="NAME", help="log this run in experiments/")
    s.add_argument("--config", help="JSON config to store with the run"); s.add_argument("--notes", default="")
    s.set_defaults(f=cmd_score)

    s = sub.add_parser("compare", help="paired bootstrap: is B really better than A?")
    s.add_argument("a", help="baseline: prediction TSV or run id (exp003)")
    s.add_argument("b", help="candidate: prediction TSV or run id")
    s.add_argument("--gt"); s.add_argument("--n-boot", type=int, default=2000)
    s.set_defaults(f=cmd_compare)

    s = sub.add_parser("sweep", help="grid-search decode thresholds on OOF probabilities")
    s.add_argument("--gt", required=True); cand_args(s)
    s.add_argument("--grid", help="JSON {param: [values]}; default gate×pair×rel")
    s.add_argument("--base", help="JSON DecodeConfig for fixed params")
    s.add_argument("--argmax", action="store_true", help="pick raw best instead of plateau centre")
    s.add_argument("--top", type=int, default=15); s.add_argument("--out"); s.add_argument("--save-config")
    s.set_defaults(f=cmd_sweep)

    s = sub.add_parser("decode", help="apply a DecodeConfig to scored candidates -> matching_results.tsv")
    cand_args(s); hdr_args(s)
    s.add_argument("--ids", required=True, help="TSV whose first column lists every S1 id (row order kept)")
    s.add_argument("--config", required=True); s.add_argument("--out", default="matching_results.tsv")
    s.set_defaults(f=cmd_decode)

    s = sub.add_parser("oracles", help="ceilings: perfect gate / perfect sets / blocking")
    s.add_argument("--gt", required=True); cand_args(s); s.add_argument("--config")
    s.set_defaults(f=cmd_oracles)

    s = sub.add_parser("blocking", help="blocking recall, cost, per-key contribution")
    s.add_argument("--gt", required=True); cand_args(s)
    s.add_argument("--n-targets", type=int, help="|S2|+|S3| for reduction ratio")
    s.add_argument("--missed", help="write missed true pairs TSV")
    s.set_defaults(f=cmd_blocking)

    s = sub.add_parser("errors", help="dump pair errors by stage (blocking/model/false merge)")
    s.add_argument("--gt", required=True); cand_args(s); s.add_argument("--config")
    s.add_argument("--out-dir", default="errors"); s.set_defaults(f=cmd_errors)

    s = sub.add_parser("calib", help="reliability table + ECE of candidate probabilities")
    s.add_argument("--gt", required=True); cand_args(s); s.add_argument("--bins", type=int, default=10)
    s.add_argument("--folds", help="folds.tsv from `split`; with --out writes cross-fitted calibrated probs")
    s.add_argument("--out"); s.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    s.set_defaults(f=cmd_calib)

    s = sub.add_parser("split", help="stratified entity-level K folds")
    s.add_argument("--gt", required=True); s.add_argument("--k", type=int, default=5)
    s.add_argument("--groups"); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="folds.tsv"); s.set_defaults(f=cmd_split)

    s = sub.add_parser("empty", help="write the all-empty probe submission")
    s.add_argument("--ids", required=True); s.add_argument("--out", default="matching_results_empty.tsv")
    s.add_argument("--gt", help="also log it (local = train singleton fraction)"); hdr_args(s)
    s.set_defaults(f=cmd_empty)

    s = sub.add_parser("lb", help="record a leaderboard score against a logged run")
    s.add_argument("run_id"); s.add_argument("score", type=float); s.add_argument("--notes", default="")
    s.set_defaults(f=cmd_lb)

    s = sub.add_parser("report", help="render experiments/results.md")
    s.add_argument("--budget", type=int, help="total submissions allowed"); s.set_defaults(f=cmd_report)

    s = sub.add_parser("validate", help="semantic pre-submission checks")
    s.add_argument("--pred", required=True); s.add_argument("--ids", required=True, help="test S1 file")
    s.add_argument("--targets", nargs="*", help="test S2 and S3 files (first column = id)")
    s.add_argument("--sample", help="official sample submission, to match headers")
    s.set_defaults(f=cmd_validate)

    s = sub.add_parser("demo", help="run everything on synthetic data")
    s.add_argument("--out-dir", default="demo_data"); s.add_argument("--n", type=int, default=5000)
    s.set_defaults(f=cmd_demo)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
