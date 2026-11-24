#!/usr/bin/env python3
"""OULAD audits that complete the UCI-697 protocol.

Two modes, one shared bootstrap/metric engine (identical to the UCI-697 reanalysis):

  --mode heldout   Held-out *presentation* at an early checkpoint with a FROZEN protocol.
                   train = 2013B + 2013J, validation (threshold + post-processing calibration) = 2014B,
                   test = 2014J, run once. Students who appear in the test presentation are removed from
                   train/validation. Nothing is tuned on the test presentation.
  --mode cv        Repeated student-grouped 5-fold CV (R repeats; every student is evaluated once per repeat),
                   gender-targeted mitigations only, for a like-for-like comparison with the UCI-697 run.

Methods (binary at-risk = Fail or Withdrawn; LightGBM on MI top-12, as in the paper):
  baseline | exclude_protected | reweigh[t] | threshold_eo[t] | expgrad_eo[t]   (t = target grouping)

Bootstrap: student-level multinomial weights (B+1 replicates, row 0 = observed), every method paired with the
baseline in the same replicate, metrics averaged over repeats. Intervals are conditional on the fitted models.
Holm / Benjamini-Hochberg are computed over the cross-attribute ("side-effect") family.

Needs: OULAD csv files in $OULAD_DATA_DIR (studentInfo, studentRegistration, studentAssessment, assessments, courses).
GPU: only the bootstrap matrix products use the GPU (torch, optional; --device cuda). Model fitting is small tree
boosting and is parallelised over folds/methods with joblib (--n_jobs).

Examples (from src/):
  python oulad_audit.py --mode heldout --cutoffs 25,50 --out ../results/oulad_heldout --n_jobs 32 --device cuda
  python oulad_audit.py --mode cv --cutoffs 75 --repeats 5 --out ../results/oulad_cv_gender --n_jobs 32 --device cuda
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

warnings.filterwarnings("ignore")

CAT = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NUM = ["num_of_prev_attempts", "studied_credits", "date_registration", "module_presentation_length",
       "n_assessments_due", "n_submitted", "submission_rate", "mean_score", "weighted_mean_score",
       "mean_lateness", "n_banked"]
PROTECTED = ["gender", "disability", "imd_band"]
IMD_LOW = ["0-10%", "10-20", "20-30%"]
TARGETS = ["gender", "disability", "imd_band", "gd"]
COMPONENTS = {"gd": {"gender", "disability"}, "gi": {"gender", "imd_band"}}
METRICS = ["dir", "tpr_gap", "fpr_gap", "eo"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ----------------------------------------------------------------------------- data
def load_table(cutoff):
    from features import build_feature_table
    df, fc, tc, gc = build_feature_table(cutoff_day=cutoff)
    df = df.reset_index(drop=True)
    y = df[tc].isin(["Fail", "Withdrawn"]).astype(int).values
    return df, y


def encode(df, train_idx, cols):
    """Ordinal codes learned on the training rows only (unseen category -> NaN); numerics untouched."""
    X = np.full((len(df), len(cols)), np.nan, dtype=np.float32)
    for j, c in enumerate(cols):
        if c in CAT:
            cats = pd.Index(df[c].astype(object).where(df[c].notna(), "MISSING").iloc[train_idx].unique())
            X[:, j] = pd.Categorical(df[c].astype(object).where(df[c].notna(), "MISSING"),
                                     categories=cats).codes.astype(np.float32)
            X[X[:, j] < 0, j] = np.nan
        else:
            X[:, j] = df[c].astype(float).values
    return X


def mi_top(X, y, tr, cols, k, seed):
    Xt = X[tr].copy()
    disc = np.array([c in CAT for c in cols])
    med = np.nanmedian(Xt, axis=0)
    med = np.where(np.isnan(med), 0, med)
    Xt = np.where(np.isnan(Xt), med, Xt)
    mi = mutual_info_classif(Xt, y[tr], discrete_features=disc, random_state=seed)
    return list(np.argsort(-mi)[:k])


def fit_lgbm(X, y, seed, w=None):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                           random_state=seed, n_jobs=1, verbosity=-1)
    return m.fit(X, y, sample_weight=w)


def reweigh(y, g):
    w = np.ones(len(y))
    n = len(y)
    for gv in np.unique(g):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            if m.sum():
                w[m] = ((g == gv).mean() * (y == yv).mean()) / (m.sum() / n)
    return w


def groupings(df):
    g = {}
    for a in ["gender", "disability", "imd_band", "age_band"]:
        s = df[a].astype(object).where(df[a].notna(), "MISSING")
        g[a] = pd.factorize(s, sort=True)[0]
    g["gd"] = g["gender"] * 2 + g["disability"]
    g["gi"] = g["gender"] * 2 + df["imd_band"].isin(IMD_LOW).astype(int).values
    labels = {}
    for a in ["gender", "disability", "imd_band", "age_band"]:
        s = df[a].astype(object).where(df[a].notna(), "MISSING")
        labels[a] = list(pd.factorize(s, sort=True)[1])
    labels["gd"] = [f"{x}/{y}" for x in labels["gender"] for y in labels["disability"]] if len(labels["disability"]) == 2 \
        else [str(i) for i in range(4)]
    labels["gi"] = [f"{x}/{y}" for x in labels["gender"] for y in ("IMD>=30%", "IMD<30%")]
    return g, labels


# ----------------------------------------------------------------------------- one fit pass
def fit_methods(X, y, g, tr, thr_idx, cal_idx, te, cols, targets, a, seed, use_thr):
    """Returns {method: (pred_te, prob_te)}. Everything is fitted on `tr`; `cal_idx` is used only for the
    decision threshold (use_thr) and ThresholdOptimizer calibration; nothing touches `te`."""
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    prot_cols = {c for c in cols if c in PROTECTED}
    out = {}

    def thr_for(model, c):
        if not use_thr:
            return 0.5
        p = model.predict_proba(X[thr_idx][:, c])[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        f = [f1_score(y[thr_idx], p >= t, average="macro") for t in grid]
        return float(grid[int(np.argmax(f))])

    def rec(name, model, c):
        pr = model.predict_proba(X[te][:, c])[:, 1]
        out[name] = ((pr >= thr_for(model, c)).astype(np.int8), pr.astype(np.float32))

    ci = np.array([cols.index(c) for c in cols])
    sel = ci[mi_top(X[:, ci], y, tr, cols, a.k, seed)]
    base = fit_lgbm(X[tr][:, sel], y[tr], seed)
    rec("baseline", base, sel)

    cand = np.array([i for i, c in enumerate(cols) if c not in prot_cols])
    sel_x = cand[mi_top(X[:, cand], y, tr, [cols[i] for i in cand], a.k, seed)]
    rec("exclude_protected", fit_lgbm(X[tr][:, sel_x], y[tr], seed), sel_x)

    for t in targets:
        gt = g[t]
        mr = fit_lgbm(X[tr][:, sel], y[tr], seed, w=reweigh(y[tr], gt[tr]))
        rec(f"reweigh[{t}]", mr, sel)
        try:
            to = ThresholdOptimizer(estimator=base, constraints="equalized_odds",
                                    objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
            to.fit(X[cal_idx][:, sel], y[cal_idx], sensitive_features=gt[cal_idx])
            pt = to.predict(X[te][:, sel], sensitive_features=gt[te], random_state=seed)
            try:
                pp = to._pmf_predict(X[te][:, sel], sensitive_features=gt[te])[:, 1]
            except Exception:
                pp = pt.astype(float)
            out[f"threshold_eo[{t}]"] = (pt.astype(np.int8), pp.astype(np.float32))
        except Exception as e:
            log(f"  threshold_eo[{t}] failed: {e}")
        if a.inproc:
            try:
                eg = ExponentiatedGradient(
                    __import__("lightgbm").LGBMClassifier(n_estimators=200, learning_rate=0.08,
                                                          class_weight="balanced", random_state=seed,
                                                          n_jobs=1, verbosity=-1),
                    constraints=EqualizedOdds(difference_bound=0.02), max_iter=a.eg_iter)
                eg.fit(X[tr][:, sel], y[tr], sensitive_features=gt[tr])
                pe = eg.predict(X[te][:, sel], random_state=seed)
                try:
                    ppe = eg._pmf_predict(X[te][:, sel])[:, 1]
                except Exception:
                    ppe = pe.astype(float)
                out[f"expgrad_eo[{t}]"] = (pe.astype(np.int8), ppe.astype(np.float32))
            except Exception as e:
                log(f"  expgrad_eo[{t}] failed: {e}")
    return out


# ----------------------------------------------------------------------------- bootstrap engine
def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty(n)
    run = 1.0
    for i, idx in enumerate(o):
        run = min(run, p[idx] * n / (n - i))
        adj[idx] = run
    return adj


class Engine:
    def __init__(self, B, seed, device, S):
        self.B = B
        rng = np.random.default_rng(seed)
        self.dev = "cpu"
        self.torch = None
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    self.torch, self.dev = torch, "cuda"
            except Exception:
                pass
        log(f"bootstrap device: {self.dev}")
        if self.torch is not None:
            gen = self.torch.Generator(device="cuda").manual_seed(seed)
            probs = self.torch.full((S,), 1.0 / S, device="cuda")
            Wb = self.torch.stack([self.torch.multinomial(probs, S, replacement=True, generator=gen)
                                   .bincount(minlength=S).float() for _ in range(B)])
            self.W = self.torch.cat([self.torch.ones(1, S, device="cuda"), Wb])
        else:
            self.W = np.vstack([np.ones((1, S), np.float32),
                                rng.multinomial(S, np.full(S, 1.0 / S), size=B).astype(np.float32)])

    def mm(self, C):
        if self.torch is not None:
            return (self.W @ self.torch.as_tensor(C, dtype=self.torch.float32, device="cuda")).cpu().numpy()
        return self.W @ C.astype(np.float32)


def audit(y, sidx, preds, g, labels, targets, base, a, eng, out, tag):
    """preds: {method: (R, N) int8}. g: {grouping: (N,) int codes}. Writes long CSVs."""
    N = len(y)
    S = int(sidx.max()) + 1
    A = sp.csr_matrix((np.ones(N, np.float32), (sidx, np.arange(N))), shape=(S, N))
    gl = list(g.keys())
    masks = [(gn, v, g[gn] == v) for gn in gl for v in np.unique(g[gn])]
    masks.insert(0, ("ALL", 0, np.ones(N, bool)))
    G = len(masks)
    M = np.stack([m for _, _, m in masks], 1).astype(np.float32)           # N,G
    shared = A @ np.concatenate([M, M * y[:, None], M * (1 - y)[:, None]], 1)  # S,3G
    sh = eng.mm(shared)                                                      # B+1,3G
    n_, pos_, neg_ = sh[:, :G], sh[:, G:2 * G], sh[:, 2 * G:]
    # eligible groups (observed counts)
    elig_flag = n_[0] >= 20
    elig_tpr = elig_flag & (pos_[0] >= 5)
    elig_fpr = elig_flag & (neg_[0] >= 5)
    gidx = {gn: [i for i, (n2, _, _) in enumerate(masks) if n2 == gn] for gn in ["ALL"] + gl}
    methods = list(preds.keys())
    R = next(iter(preds.values())).shape[0]
    per = {mth: {} for mth in methods}
    for mth in methods:
        acc = None
        for r in range(R):
            p = preds[mth][r].astype(np.float32)
            Cp = A @ np.concatenate([M * p[:, None], M * (y * p)[:, None], M * ((1 - y) * p)[:, None]], 1)
            s = eng.mm(Cp)
            fl, tp, fp = s[:, :G], s[:, G:2 * G], s[:, 2 * G:]
            res = {}
            with np.errstate(invalid="ignore", divide="ignore"):
                for gn in gl:
                    ix = gidx[gn]
                    fr = np.where(elig_flag[ix], fl[:, ix] / n_[:, ix], np.nan)
                    tr_ = np.where(elig_tpr[ix], tp[:, ix] / pos_[:, ix], np.nan)
                    fr2 = np.where(elig_fpr[ix], fp[:, ix] / neg_[:, ix], np.nan)
                    d = np.nanmin(fr, 1) / np.nanmax(fr, 1)
                    tg = np.nanmax(tr_, 1) - np.nanmin(tr_, 1)
                    fg = np.nanmax(fr2, 1) - np.nanmin(fr2, 1)
                    res[gn] = dict(dir=d, tpr_gap=tg, fpr_gap=fg, eo=np.fmax(tg, fg),
                                   _tpr=tr_, _fpr=fr2, _flag=fr)
                i = gidx["ALL"][0]
                tp_, fp_, pos, neg = tp[:, i], fp[:, i], pos_[:, i], neg_[:, i]
                fn_, tn_ = pos - tp_, neg - fp_
                f1p = 2 * tp_ / (2 * tp_ + fp_ + fn_)
                f1n = 2 * tn_ / (2 * tn_ + fn_ + fp_)
                res["ALL"] = dict(macro_f1=(f1p + f1n) / 2, flag_rate=fl[:, i] / n_[:, i])
            acc = res if acc is None else {k: {kk: acc[k][kk] + res[k][kk] for kk in res[k]} for k in res}
        per[mth] = {k: {kk: vv / R for kk, vv in v.items()} for k, v in acc.items()}

    def ci(x):
        return np.nanpercentile(x[1:], 2.5, axis=0), np.nanpercentile(x[1:], 97.5, axis=0)

    def pval(d):
        d = d[1:]
        d = d[~np.isnan(d)]
        if len(d) == 0:
            return np.nan
        return max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (len(d) + 1))

    rows = []
    for mth in methods:
        tgt = mth[mth.index("[") + 1:-1] if "[" in mth else None
        for gn in gl:
            for mt in METRICS:
                v = per[mth][gn][mt]
                b = per[base][gn][mt]
                lo, hi = ci(v)
                row = dict(method=mth, target=tgt, grouping=gn, metric=mt, value=v[0], lo=lo, hi=hi, base=b[0])
                if mth != base:
                    dlt = v - b
                    dlo, dhi = ci(dlt)
                    related = (tgt == gn) or (tgt in COMPONENTS.get(gn, set())) or (gn in COMPONENTS.get(tgt, set()) if tgt else False)
                    worse = dlt[0] < 0 if mt == "dir" else dlt[0] > 0
                    row.update(delta=dlt[0], dlo=dlo, dhi=dhi, p=pval(dlt), related=bool(related) if tgt else np.nan,
                               worse=bool(worse))
                rows.append(row)
        v = per[mth]["ALL"]["macro_f1"]
        b = per[base]["ALL"]["macro_f1"]
        lo, hi = ci(v)
        row = dict(method=mth, target=tgt, grouping="ALL", metric="macro_f1", value=v[0], lo=lo, hi=hi, base=b[0])
        if mth != base:
            dlt = v - b
            dlo, dhi = ci(dlt)
            row.update(delta=dlt[0], dlo=dlo, dhi=dhi, p=pval(dlt))
        rows.append(row)
    df = pd.DataFrame(rows)
    side = df["related"].eq(False) & df["p"].notna() if "related" in df else pd.Series(False, index=df.index)
    df["p_holm_side"] = np.nan
    df["q_bh_side"] = np.nan
    if side.sum():
        df.loc[side, "p_holm_side"] = holm(df.loc[side, "p"].values)
        df.loc[side, "q_bh_side"] = bh(df.loc[side, "p"].values)
        sdf = df[side]
        log(f"[{tag}] side-effect family n={side.sum()}: unadjusted worse={(sdf.worse & (sdf.p<0.05)).sum()} "
            f"better={((~sdf.worse) & (sdf.p<0.05)).sum()}; Holm<.05 worse={(sdf.worse & (sdf.p_holm_side<0.05)).sum()} "
            f"better={((~sdf.worse) & (sdf.p_holm_side<0.05)).sum()}; BH<.05 worse={(sdf.worse & (sdf.q_bh_side<0.05)).sum()} "
            f"better={((~sdf.worse) & (sdf.q_bh_side<0.05)).sum()}")
    df.to_csv(f"{out}/{tag}_fairness.csv", index=False)

    # cell-level table (gender x disability, gender x IMD<30%): n, positives, TPR/FPR with CIs and paired changes
    cells = []
    for gn in ["gd", "gi"]:
        ix = gidx[gn]
        for j, i in enumerate(ix):
            lab = labels[gn][j] if j < len(labels[gn]) else str(j)
            if not (elig_flag[i]):
                continue
            for mth in methods:
                for mt, key, el in [("tpr", "_tpr", elig_tpr), ("fpr", "_fpr", elig_fpr)]:
                    v = per[mth][gn][key][:, j]
                    b = per[base][gn][key][:, j]
                    lo, hi = ci(v)
                    row = dict(grouping=gn, cell=lab, n=int(n_[0, i]), pos=int(pos_[0, i]), neg=int(neg_[0, i]),
                               method=mth, metric=mt, value=v[0], lo=lo, hi=hi)
                    if mth != base:
                        d = v - b
                        dlo, dhi = ci(d)
                        row.update(delta=d[0], dlo=dlo, dhi=dhi)
                    cells.append(row)
    pd.DataFrame(cells).to_csv(f"{out}/{tag}_cells.csv", index=False)
    return df


# ----------------------------------------------------------------------------- modes
def run_heldout(ck, a, out):
    df, y = load_table(ck)
    pres = df["code_presentation"].astype(str)
    te_m, va_m = pres == a.test_pres, pres == a.val_pres
    tr_m = pres.isin(a.train_pres.split(","))
    te_ids = set(df.loc[te_m, "id_student"])
    va_ids = set(df.loc[va_m, "id_student"])
    tr_m &= ~df["id_student"].isin(te_ids | va_ids)
    va_m &= ~df["id_student"].isin(te_ids)
    tr, va, te = [np.flatnonzero(m.values) for m in (tr_m, va_m, te_m)]
    log(f"[heldout C{ck}] train={len(tr)} val={len(va)} test={len(te)}  at-risk: "
        f"{y[tr].mean():.3f}/{y[va].mean():.3f}/{y[te].mean():.3f}")
    cols = [c for c in CAT + NUM]
    X = encode(df, tr, cols)
    g, labels = groupings(df)
    seeds = list(range(a.seed, a.seed + a.seeds))
    res = Parallel(n_jobs=min(a.n_jobs, len(seeds)))(
        delayed(fit_methods)(X, y, g, tr, va, va, te, cols, TARGETS, a, s, True) for s in seeds)
    # seed variability of overall performance (reported), primary = first seed
    perf = []
    for s, r_ in zip(seeds, res):
        for m_, (p_, pr_) in r_.items():
            perf.append(dict(seed=s, method=m_, macro_f1=f1_score(y[te], p_, average="macro"),
                             auroc=roc_auc_score(y[te], pr_) if len(np.unique(pr_)) > 1 else np.nan,
                             flag_rate=float(p_.mean())))
    pd.DataFrame(perf).to_csv(f"{out}/heldout_C{ck}_seed_performance.csv", index=False)
    prim = res[0]
    methods = [m_ for m_ in prim]
    preds = {m_: prim[m_][0][None, :] for m_ in methods}
    gte = {k: v[te] for k, v in g.items()}
    sidx = pd.factorize(df["id_student"].values[te])[0]
    eng = Engine(a.B, a.seed, a.device, int(sidx.max()) + 1)
    audit(y[te], sidx, preds, gte, {k: labels[k] for k in labels}, TARGETS, "baseline", a, eng, out, f"heldout_C{ck}")
    json.dump(dict(cutoff=ck, n_train=len(tr), n_val=len(va), n_test=len(te), train_pres=a.train_pres,
                   val_pres=a.val_pres, test_pres=a.test_pres, seeds=seeds, B=a.B), open(f"{out}/heldout_C{ck}_config.json", "w"), indent=2)


def cv_rep(r, X, y, g, sid, cols, a):
    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed + r)
    n = len(y)
    folds = list(skf.split(np.zeros(n), y, sid))
    def one(k, tr, te):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=a.seed + 100 * r + k)
        a_i, c_i = next(gss.split(tr, groups=sid[tr]))
        res = fit_methods(X, y, g, tr[a_i], tr[c_i] if a.cv_thr else None, tr[c_i], te, cols, a.cv_targets.split(","), a, a.seed + 100 * r + k, bool(a.cv_thr))
        return te, res
    outs = Parallel(n_jobs=a.n_jobs)(delayed(one)(k, tr, te) for k, (tr, te) in enumerate(folds))
    preds = {}
    for te, res in outs:
        for m_, (p_, _) in res.items():
            preds.setdefault(m_, np.zeros(n, np.int8))[te] = p_
    return preds


def run_cv(ck, a, out):
    df, y = load_table(ck)
    sid = df["id_student"].values
    cols = CAT + NUM
    X = encode(df, np.arange(len(df)), cols)       # categories only; no label information used
    g, labels = groupings(df)
    log(f"[cv C{ck}] n={len(df)} students={len(np.unique(sid))} at-risk={y.mean():.3f}")
    allp = []
    for r in range(a.repeats):
        log(f"  repeat {r + 1}/{a.repeats}")
        allp.append(cv_rep(r, X, y, g, sid, cols, a))
    methods = list(allp[0].keys())
    preds = {m_: np.stack([p[m_] for p in allp]) for m_ in methods}
    sidx = pd.factorize(sid)[0]
    eng = Engine(a.B, a.seed, a.device, int(sidx.max()) + 1)
    audit(y, sidx, preds, g, labels, a.cv_targets.split(","), "baseline", a, eng, out, f"cv_C{ck}")
    np.savez_compressed(f"{out}/cv_C{ck}_oof.npz", y=y, sid=sid, **{m_.replace("[", "_").replace("]", ""): v for m_, v in preds.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["heldout", "cv"], required=True)
    p.add_argument("--cutoffs", default="25,50")
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=5, help="heldout: retraining seeds for the performance range")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--inproc", type=int, default=1)
    p.add_argument("--eg_iter", type=int, default=30)
    p.add_argument("--cv_targets", default="gender")
    p.add_argument("--cv_thr", type=int, default=0, help="cv: 1 = tune the macro-F1 decision threshold on the calibration split (same procedure as the held-out run); 0 = fixed 0.5")
    p.add_argument("--train_pres", default="2013B,2013J")
    p.add_argument("--val_pres", default="2014B")
    p.add_argument("--test_pres", default="2014J")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for ck in [int(x) for x in a.cutoffs.split(",")]:
        (run_heldout if a.mode == "heldout" else run_cv)(ck, a, a.out)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
