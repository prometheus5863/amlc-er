"""Stage 3 (GPU): a multilingual cross-encoder re-scores the pairs LightGBM is unsure about.

Why: our string features cannot see that 'Black Infotech Pvt Ltd' and 'ब्लैक इंफोटेक प्राइवेट लिमिटेड'
are the same name, and the model has never seen French. A transformer pretrained on 50+ languages,
fine-tuned to read both records side by side, can.

Two halves:
  CPU pipeline (end of every Kaggle run)  -> export():  output/rerank/{train_pairs,test_pairs}.parquet
      train_pairs = OUT-OF-FOLD LightGBM probabilities (+ label) for pairs with p >= KEEP
      test_pairs  = test probabilities for pairs with p >= KEEP
  GPU notebook                            -> run(in_dir): fine-tune, measure, write matching_results.tsv

Honest measurement: train pairs are split BY SOURCE-1 BUSINESS into
  ft  (fine-tune the cross-encoder)   cal (fit the blend)   ev (score stage 1 vs blended, never touched before)
Only the uncertain band LO <= p <= HI is re-scored; outside it the LightGBM probability stands.

    python -m amlc.rerank export                    # CPU, after the pipeline
    python -m amlc.rerank run /kaggle/input/<v4-output>/output/rerank   # GPU notebook
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.environ.get("AMLC_CE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
KEEP = 0.01               # pairs below this are exported as nothing (never matches anyway)
LO, HI = float(os.environ.get("AMLC_CE_LO", "0.02")), float(os.environ.get("AMLC_CE_HI", "0.98"))
N_TRAIN = int(os.environ.get("AMLC_CE_NTRAIN", "1500000"))
MAX_LEN = int(os.environ.get("AMLC_CE_MAXLEN", "128"))
EPOCHS = float(os.environ.get("AMLC_CE_EPOCHS", "1"))
BS = int(os.environ.get("AMLC_CE_BS", "128"))
LR = float(os.environ.get("AMLC_CE_LR", "5e-5"))


def _log(msg, t0=[time.time()]):
    print(f"[rerank {time.time() - t0[0]:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- export (CPU pipeline)
def export() -> str:
    import duckdb
    from .model import W, load_gt_sets
    from .paths import output_dir
    out = output_dir() / "rerank"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    if os.path.exists(W("train_oof.parquet")):
        tr = con.execute(f"SELECT s1_id, cand_id, prob_raw::FLOAT AS p1 FROM '{W('train_oof.parquet')}' "
                         f"WHERE prob_raw >= {KEEP}").df()
        s1_all = con.execute(f"SELECT DISTINCT s1_id FROM '{W('train_oof.parquet')}'").df().s1_id
        gt = load_gt_sets(set(s1_all))
        tp = {(s, c) for s, cs in gt.items() for c in cs}
        tr["y"] = np.fromiter(((s, c) in tp for s, c in zip(tr.s1_id, tr.cand_id)), np.int8, len(tr))
        tr.to_parquet(out / "train_pairs.parquet", index=False)
        # every sampled S1 with its true set, so the GPU side can score without the ground-truth file
        pd.DataFrame({"s1_id": list(gt), "matched": [",".join(sorted(v)) for v in gt.values()]}) \
          .to_parquet(out / "train_gt.parquet", index=False)
        _log(f"train_pairs: {len(tr):,} pairs (p>={KEEP}), {tr.y.sum():,} true, {len(gt):,} S1")
    if os.path.isdir(W("test_scores")):
        con.execute(f"COPY (SELECT s1_id, cand_id, prob_raw::FLOAT AS p1 FROM '{W('test_scores')}/*.parquet' "
                    f"WHERE prob_raw >= {KEEP}) TO '{out / 'test_pairs.parquet'}' (FORMAT parquet)")
        _log(f"test_pairs: {con.execute(f'SELECT count(*) FROM {chr(39)}{out}/test_pairs.parquet{chr(39)}').fetchone()[0]:,} pairs")
    for f in ("decode.json",):
        if os.path.exists(W(f)):
            import shutil
            shutil.copy(W(f), out / f)
    return str(out)


# ----------------------------------------------------------------------------- text
def load_texts(ids) -> dict:
    """entity_id -> 'name | address' (raw text: the model reads the original scripts)."""
    import duckdb
    from .data import pq as pq_path
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"entity_id": pd.unique(np.asarray(list(ids)))}))
    frames = []
    for split in ("train", "test"):
        for s in "123":
            try:
                p = pq_path(f"{split}_source{s}")
            except Exception:
                continue
            if not os.path.exists(p):
                continue
            frames.append(con.execute(f"SELECT entity_id, business_name || ' | ' || business_address AS txt "
                                      f"FROM '{p}' SEMI JOIN ids USING (entity_id)").df())
    t = pd.concat(frames, ignore_index=True)
    return dict(zip(t.entity_id, t.txt))


# ----------------------------------------------------------------------------- model
class _Pairs:
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = list(a), list(b), y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


def _collate(tok):
    def f(batch_idx, ds):
        a = [ds.a[i] for i in batch_idx]
        b = [ds.b[i] for i in batch_idx]
        enc = tok(a, b, truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
        if ds.y is not None:
            import torch
            enc["labels"] = torch.tensor(ds.y[batch_idx], dtype=torch.float32)
        return enc
    return f


def _batches(n, bs, shuffle, seed=0):
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def fine_tune(a, b, y, save_dir=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    ds = _Pairs(a, b, np.asarray(y, np.float32))
    col = _collate(tok)
    steps = int(np.ceil(len(ds) / BS) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0, ep = 0, time.time(), 0
    while step < steps:
        for bi in _batches(len(ds), BS, True, seed=ep):
            enc = {k: v.to(dev) for k, v in col(bi, ds).items()}
            labels = enc.pop("labels")
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 500 == 0 or step == steps:
                _log(f"fine-tune step {step}/{steps} loss={loss.item():.4f} ({(time.time() - t0) / step * 1000:.0f} ms/step)")
            if step >= steps:
                break
        ep += 1
    if save_dir:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        _log(f"checkpoint saved to {save_dir}")
    return tok, model


def predict(tok, model, a, b, bs=None) -> np.ndarray:
    """Cross-encoder logits for (a[i], b[i]); sorted by length for speed, returned in input order."""
    import torch
    dev = next(model.parameters()).device
    bs = bs or BS * 4
    model.eval()
    ds = _Pairs(a, b)
    order = np.argsort([len(x) + len(y) for x, y in zip(ds.a, ds.b)])
    out = np.zeros(len(ds), np.float32)
    col = _collate(tok)
    t0 = time.time()
    with torch.no_grad():
        for k, i in enumerate(range(0, len(order), bs)):
            bi = order[i:i + bs]
            enc = {kk: v.to(dev) for kk, v in col(bi, ds).items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out[bi] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if k % 200 == 0:
                _log(f"predict {i:,}/{len(ds):,} ({(i + len(bi)) / max(time.time() - t0, 1e-9):.0f} pairs/s)")
    return out


# ----------------------------------------------------------------------------- blend + score
def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_fit(p1, ce, y):
    from sklearn.linear_model import LogisticRegression
    X = np.c_[_logit(p1), ce, _logit(p1) * ce]
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, y)


def blend_apply(m, p1, ce):
    X = np.c_[_logit(p1), ce, _logit(p1) * ce]
    return m.predict_proba(X)[:, 1]


def score(df, gt, col):
    """macro F0.5 over the S1s in gt, best (gate, pair) on a grid; same decode as the pipeline."""
    from erharness.sweep import Scorer, best_config, grid_search
    from .model import assign_targets
    c = df[["s1_id", "cand_id"]].copy()
    c["prob_raw"] = df[col].to_numpy()
    c["prob"] = assign_targets(c, "prob_raw")
    res = grid_search(Scorer(c[["s1_id", "cand_id", "prob"]], gt),
                      {"gate": [0.1, 0.2, 0.3, 0.4, 0.5], "pair": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    return float(res.iloc[0].score), best_config(res)


# ----------------------------------------------------------------------------- GPU entry point
def run(in_dir: str, out_dir: str = None) -> dict:
    from .paths import output_dir
    out_dir = out_dir or str(output_dir())
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_parquet(f"{in_dir}/train_pairs.parquet")
    g = pd.read_parquet(f"{in_dir}/train_gt.parquet")
    gt = {s: set(m.split(",")) if m else set() for s, m in zip(g.s1_id, g.matched)}
    # business-level split: ft 70% / cal 15% / ev 15%
    h = pd.util.hash_pandas_object(pd.Series(list(gt)), index=False).to_numpy() % 100
    part = dict(zip(gt, np.where(h < 70, "ft", np.where(h < 85, "cal", "ev"))))
    tr["part"] = tr.s1_id.map(part)
    band = (tr.p1 >= LO) & (tr.p1 <= HI)
    _log(f"train pairs {len(tr):,}; in band [{LO},{HI}]: {band.sum():,} ({tr.y[band].mean():.3f} true)")

    te = pd.read_parquet(f"{in_dir}/test_pairs.parquet") if os.path.exists(f"{in_dir}/test_pairs.parquet") else None
    tband = ((te.p1 >= LO) & (te.p1 <= HI)) if te is not None else None
    if te is not None:
        _log(f"test pairs {len(te):,}; in band: {tband.sum():,}")

    ft = tr[band & (tr.part == "ft")]
    if len(ft) > N_TRAIN:
        ft = ft.sample(N_TRAIN, random_state=0)
    need = set(tr.s1_id[band]) | set(tr.cand_id[band])
    if te is not None:
        need |= set(te.s1_id[tband]) | set(te.cand_id[tband])
    txt = load_texts(need)
    _log(f"texts loaded: {len(txt):,}")
    T = lambda s: [txt.get(x, "") for x in s]  # noqa: E731

    _log(f"fine-tuning {BASE} on {len(ft):,} pairs")
    tok, model = fine_tune(T(ft.s1_id), T(ft.cand_id), ft.y.to_numpy(), save_dir=f"{out_dir}/ce_model")

    held = tr[band & (tr.part != "ft")].copy()
    held["ce"] = predict(tok, model, T(held.s1_id), T(held.cand_id))
    cal = held[held.part == "cal"]
    bl = blend_fit(cal.p1, cal.ce, cal.y)
    _log(f"blend coefs {bl.coef_.round(3).tolist()} intercept {bl.intercept_.round(3).tolist()}")

    ev_ids = {s for s, p in part.items() if p == "ev"}
    ev = tr[tr.s1_id.isin(ev_ids)].copy()
    ev["p2"] = ev.p1
    m = held.part == "ev"
    ev.loc[held.index[m], "p2"] = blend_apply(bl, held.p1[m], held.ce[m])
    gev = {s: gt[s] for s in ev_ids}
    s1, c1 = score(ev, gev, "p1")
    s2, c2 = score(ev, gev, "p2")
    # the cross-encoder alone on the band (diagnostic: does it separate hard pairs?)
    from sklearn.metrics import roc_auc_score
    hv = held[held.part == "ev"]
    auc1, auc2 = roc_auc_score(hv.y, hv.p1), roc_auc_score(hv.y, hv.ce)
    res = {"ev_stage1": s1, "ev_blend": s2, "gain": s2 - s1, "band_auc_lgbm": auc1, "band_auc_ce": auc2,
           "config": c2.as_dict(), "n_ft": len(ft), "base": BASE}
    _log(f"EV (unseen businesses): stage1 {s1:.5f} -> blended {s2:.5f}  (gain {s2 - s1:+.5f}); "
         f"band AUC lgbm {auc1:.4f} vs cross-encoder {auc2:.4f}")
    json.dump(res, open(f"{out_dir}/rerank_result.json", "w"), indent=1, default=float)

    if te is not None:
        te["p2"] = te.p1
        idx = np.flatnonzero(tband.to_numpy())
        ce_t = predict(tok, model, T(te.s1_id.iloc[idx]), T(te.cand_id.iloc[idx]))
        te.iloc[idx, te.columns.get_loc("p2")] = blend_apply(bl, te.p1.iloc[idx], ce_t)
        use = "p2" if s2 > s1 else "p1"
        cfg = (c2 if use == "p2" else c1).as_dict()
        _log(f"writing test decode with {use} (gate={cfg['gate']} pair={cfg['pair']})")
        write_matching(te, use, cfg, out_dir)
    return res


def write_matching(te: pd.DataFrame, col: str, cfg: dict, out_dir: str) -> None:
    """Same rule as model.decode_sql: target -> best S1 only, then gate on the S1's max, pair threshold.
    Every test S1 gets a row (empty when nothing kept)."""
    import duckdb
    from .data import pq as pq_path
    con = duckdb.connect()
    con.register("s", te[["s1_id", "cand_id", col]].rename(columns={col: "p"}))
    con.execute("CREATE TEMP TABLE best AS SELECT cand_id, max(p) b FROM s GROUP BY 1")
    con.execute("CREATE TEMP TABLE pr AS SELECT s.s1_id, s.cand_id, CASE WHEN s.p >= best.b THEN s.p ELSE 0 END AS p "
                "FROM s JOIN best USING (cand_id)")
    con.execute("CREATE TEMP TABLE ent AS SELECT s1_id, max(p) g FROM pr GROUP BY 1")
    con.execute(f"""CREATE TEMP TABLE kept AS SELECT pr.s1_id, string_agg(pr.cand_id, ',' ORDER BY pr.cand_id) ids
                    FROM pr JOIN ent USING (s1_id) WHERE ent.g >= {cfg['gate']} AND pr.p >= {cfg['pair']} GROUP BY 1""")
    con.execute(f"CREATE TEMP TABLE s1 AS SELECT entity_id, row_number() OVER () ord FROM '{pq_path('test_source1')}'")
    con.execute(f"""COPY (SELECT s1.entity_id AS source1_entity_id, coalesce(k.ids, '') AS matched_entity_ids
                          FROM s1 LEFT JOIN kept k ON k.s1_id = s1.entity_id ORDER BY s1.ord)
                    TO '{out_dir}/matching_results.tsv' (HEADER, DELIMITER '\t', QUOTE '')""")
    n, k = con.execute("SELECT (SELECT count(*) FROM s1), (SELECT count(*) FROM kept)").fetchone()
    _log(f"wrote {out_dir}/matching_results.tsv: {n:,} rows, {k:,} non-empty")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
