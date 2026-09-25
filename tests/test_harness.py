import math
import os

import numpy as np
import pandas as pd
import pytest

from erharness import analysis, bootstrap, metric, splits, synth, tracker
from erharness.decode import DecodeConfig, Prepared, decode
from erharness.io import read_candidates, read_match_sets, write_match_sets
from erharness.sweep import Scorer, best_config, grid_search


def ref_f05(P, T):
    """Textbook definition, independent of the closed form."""
    if not T:
        return 1.0 if not P else 0.0
    if not P:
        return 0.0
    h = len(P & T)
    if h == 0:
        return 0.0
    p, r = h / len(P), h / len(T)
    return 1.25 * p * r / (0.25 * p + r)


# ------------------------------------------------------------------ metric --
@pytest.mark.parametrize("P,T", [
    (set(), set()), ({"a"}, set()), (set(), {"a"}), ({"a"}, {"a"}), ({"a", "b"}, {"a"}),
    ({"a"}, {"a", "b"}), ({"x"}, {"a"}), ({"a", "x", "y"}, {"a", "b"}),
])
def test_closed_form_matches_definition(P, T):
    assert math.isclose(metric.f05(len(P & T), len(P), len(T)), ref_f05(P, T))


def test_random_sets_match_definition():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        T = set(rng.choice(8, rng.integers(0, 4), replace=False).tolist())
        P = set(rng.choice(8, rng.integers(0, 5), replace=False).tolist())
        assert math.isclose(metric.f05(len(P & T), len(P), len(T)), ref_f05(P, T))


def test_precision_weighting():
    # one extra wrong id hurts more than one missing id
    assert metric.f05(2, 3, 2) < metric.f05(1, 1, 2)


def test_decomposition_identity():
    gt, cands, _ = synth.make(n=3000, seed=3)
    gts = {r.s1_id: set(filter(None, r.matches.split(","))) for r in gt.itertuples()}
    cands = cands.assign(prob=cands.prob.astype(float))
    sets = decode(Prepared(cands, gts), DecodeConfig(gate=0.6, pair=0.4))
    rep = metric.evaluate(sets, gts)
    s = rep.singleton_frac
    assert math.isclose(rep.score, s * rep.spec + (1 - s) * rep.open_rate * rep.f_open, rel_tol=1e-9)
    assert math.isclose(metric.project_score(rep, s), rep.score, rel_tol=1e-9)


def test_all_empty_equals_singleton_frac():
    gt = {"a": set(), "b": {"x"}, "c": set(), "d": {"y", "z"}}
    rep = metric.evaluate({k: set() for k in gt}, gt)
    assert rep.score == 0.5 == rep.singleton_frac


def test_missing_rows_scored_empty_and_warned():
    gt = {"a": set(), "b": {"x"}}
    with pytest.warns(UserWarning):
        rep = metric.evaluate({}, gt)
    assert rep.score == 0.5
    with pytest.raises(ValueError):
        metric.evaluate({}, gt, strict=True)


# ---------------------------------------------------------------------- io --
def test_io_roundtrip(tmp_path):
    sets = {"001": {"S2_1", "S3_9"}, "002": set(), "010": {"S2_5"}}
    p = tmp_path / "sub.tsv"
    write_match_sets(sets, p, order=["010", "001", "002"])
    back = read_match_sets(p)
    assert back == sets
    assert list(pd.read_csv(p, sep="\t", dtype=str).iloc[:, 0]) == ["010", "001", "002"]  # order + leading zeros kept


def test_io_multi_column_union(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("id\ts2\ts3\nA\tx,y\tz\nB\t\t\n")
    assert read_match_sets(p) == {"A": {"x", "y", "z"}, "B": set()}


# ------------------------------------------------------------ sweep/decode --
@pytest.fixture(scope="module")
def data():
    gt, cands, groups = synth.make(n=4000, seed=7)
    gts = {r.s1_id: set(filter(None, r.matches.split(","))) for r in gt.itertuples()}
    return gts, cands.assign(prob=cands.prob.astype(float)), dict(zip(groups.s1_id, groups.region))


@pytest.mark.parametrize("cfg", [
    DecodeConfig(gate=0.7, pair=0.5),
    DecodeConfig(gate=0.5, pair=0.3, rel=0.8, max_k=2),
    DecodeConfig(gate=0.6, pair=0.2, max_per_source=1, gate_source="any"),
    DecodeConfig(mode="expected_f", ef_samples=200),
])
def test_fast_scorer_equals_slow_path(data, cfg):
    gt, cands, _ = data
    sc = Scorer(cands, gt)
    fast = sc.score(cfg)["score"]
    slow = metric.evaluate(decode(sc.prep, cfg), gt).score
    assert math.isclose(fast, slow, rel_tol=1e-12)


def test_sweep_beats_naive_and_best_config_roundtrips(data):
    gt, cands, _ = data
    sc = Scorer(cands, gt)
    res = grid_search(sc, {"gate": [0.3, 0.5, 0.7, 0.9], "pair": [0.3, 0.5, 0.7]})
    best = best_config(res)
    assert sc.score(best)["score"] >= sc.score(DecodeConfig(gate=0.3, pair=0.3))["score"]
    assert math.isclose(sc.score(best)["score"], res.iloc[0].score)


def test_oracles_are_ordered(data):
    gt, cands, _ = data
    sc = Scorer(cands, gt)
    o = analysis.oracles(sc, DecodeConfig(gate=0.6, pair=0.5)).set_index("scenario").macro_f05
    cur = o["current"]
    ceiling = o["perfect gate + sets = blocking ceiling"]
    assert o["perfect gate (your matcher)"] >= cur - 1e-12
    assert o["perfect sets (your gate)"] >= cur - 1e-12
    assert ceiling >= max(o["perfect gate (your matcher)"], o["perfect sets (your gate)"]) - 1e-12
    assert ceiling <= 1.0


def test_blocking_audit(data):
    gt, cands, _ = data
    b = analysis.blocking_audit(cands, gt)
    n_true = sum(len(v) for v in gt.values())
    assert math.isclose(b["pair_recall"], 1 - len(b["missed_true_pairs"]) / n_true)
    assert 0.9 < b["pair_recall"] < 1.0  # synth uses 95% blocking recall
    assert set(b["per_key"].key) == {"pin3_name", "phonetic_city", "tfidf_knn", "phone"}
    assert b["greedy_key_order"].cum_pair_recall.is_monotonic_increasing
    assert math.isclose(b["greedy_key_order"].cum_pair_recall.iloc[-1], b["pair_recall"])


def test_error_buckets_partition_errors(data):
    gt, cands, _ = data
    sc = Scorer(cands, gt)
    cfg = DecodeConfig(gate=0.6, pair=0.5)
    e = analysis.error_pairs(sc, cfg, gt)
    table = metric.per_entity(decode(sc.prep, cfg), gt)
    assert len(e["blocking_miss"]) + len(e["model_miss"]) == table.fn.sum()
    assert len(e["false_merge_singleton"]) + len(e["false_merge_matched"]) == table.fp.sum()


# --------------------------------------------------------------- bootstrap --
def test_bootstrap_detects_real_and_null_changes(data):
    gt, cands, _ = data
    sc = Scorer(cands, gt)
    ta = metric.per_entity(decode(sc.prep, DecodeConfig(gate=0.2, pair=0.2)), gt)
    tb = metric.per_entity(decode(sc.prep, best_config(grid_search(sc, {"gate": [0.6, 0.7, 0.8], "pair": [0.4, 0.5]}))), gt)
    assert "WINS" in bootstrap.compare(ta, tb).verdict
    assert "LOSES" in bootstrap.compare(tb, ta).verdict
    assert "NO CHANGE" in bootstrap.compare(ta, ta.copy()).verdict
    # a tiny random perturbation on a few entities must read as noise
    tc = ta.copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tc), 40, replace=False)
    tc.loc[idx, "score"] += np.r_[np.full(20, 0.3), np.full(20, -0.3)]  # zero net change
    assert "NOISE" in bootstrap.compare(ta, tc).verdict


# ------------------------------------------------------------------ splits --
def test_kfold_stratified(data):
    gt, _, groups = data
    fold = splits.kfold(gt, 5, groups)
    assert set(fold) == set(gt)
    bal = splits.fold_balance(gt, fold, groups)
    assert bal.singleton_frac.max() - bal.singleton_frac.min() < 0.01
    assert bal.n.max() - bal.n.min() <= 20


# ----------------------------------------------------------------- tracker --
def test_tracker_log_and_lb(tmp_path, data):
    gt, cands, _ = data
    root = str(tmp_path / "exp")
    sc = Scorer(cands, gt)
    for g in (0.5, 0.7, 0.9):
        t = metric.per_entity(decode(sc.prep, DecodeConfig(gate=g, pair=0.5)), gt)
        tracker.log_run(f"gate{g}", metric.report_from_table(t), t, {"gate": g}, root=root)
    ids = [r["id"] for r in tracker.load(root)]
    assert len(ids) == 3
    for rid, lb in zip(ids, (0.70, 0.72, 0.71)):
        tracker.set_lb(rid, lb, root=root)
    al = tracker.alignment(root)
    assert al["n_submitted"] == 3 and "rank_corr_local_vs_lb" in al
    md = open(os.path.join(root, "results.md")).read()
    assert ids[2] in md and "LB−local" in md
    assert len(tracker.run_table(ids[1], root)) == len(gt)
