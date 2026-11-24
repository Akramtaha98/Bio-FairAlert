#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External validation + fairness audit for Bio-FairAlert on a NEW dataset.

Dataset : UCI "Predict Students' Dropout and Academic Success" (ID 697, CC BY 4.0)
          Realinho et al. (2021), Instituto Politecnico de Portalegre, DOI 10.24432/C5MC89
          4,424 students, 36 features (demographic, socio-economic, 1st/2nd-semester
          academic records), target = Dropout / Enrolled / Graduate.

Why this dataset (vs. MOOC): it carries protected / equity attributes (gender, age,
special needs, scholarship = socio-economic proxy, international status), so the
FAIRNESS findings (not just model comparison) can be replicated on a second institution.

Checkpoints (analogue of OULAD days / MOOC days; all information strictly time-ordered):
    C0 = enrolment-time features (demographics, application, socio-economic, macro indicators)
    C1 = C0 + 1st-semester curricular units
    C2 = C1 + 2nd-semester curricular units

Stages (--stages, comma separated):
    main     8-row table: {RF, XGBoost, LightGBM, Hybrid(stack of the 3)} x {no filter, MI top-k filter}
             repeated stratified K-fold, feature selection INSIDE each training fold,
             Nadeau-Bengio corrected CIs / paired t-tests, Holm correction.
    ksweep   filter sensitivity: {MI, ANOVA-F} x k  vs no filter (LightGBM)
    fairness binary at-risk (Dropout) audit at every checkpoint, 5 protected attributes,
             mitigations: exclude-protected, reweighing, equalized-odds post-processing,
             equalized-odds in-processing (ExponentiatedGradient), single-target + joint,
             cross-attribute SIDE-EFFECT matrix, intersectional audit, student-level
             bootstrap CIs for paired deltas.
    explain  SHAP stability, SHAP-vs-permutation agreement, deletion fidelity test.

Leakage control: feature selection is fit on the training fold only; ThresholdOptimizer is
fit on an independent calibration split carved out of the training fold; every fairness
method uses the identical model-training subset.

Outputs (in --out): CSV tables, LaTeX (booktabs) tables, config.json, report.md.

Example (H200 server, many CPU cores; data are tiny so GPU is NOT needed):
    cd src && python uci697_external_validation.py --n_jobs 32
Quick smoke test (minutes):
    cd src && python uci697_external_validation.py --out ../results/uci697_smoke --quick --n_jobs 4
"""
import argparse
import io
import json
import os
import platform
import sys
import time
import urllib.request
import warnings
import zipfile
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # inherited by joblib workers

UCI_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
LABELS = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
BASE_NAMES = ["RandomForest", "XGBoost", "LightGBM"]
CKPTS = ["C0", "C1", "C2"]
PROTECTED_COLS = ["Gender", "Age at enrollment", "Scholarship holder",
                  "Educational special needs", "International", "Nationality"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_data(path, data_dir):
    if path is None:
        os.makedirs(data_dir, exist_ok=True)
        cache = os.path.join(data_dir, "data.csv")
        if not os.path.exists(cache):
            log("Downloading UCI-697 ...")
            try:
                raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    with open(cache, "wb") as f:
                        f.write(z.read("data.csv"))
            except Exception as e:  # pragma: no cover
                sys.exit(f"Download failed ({e}). Download the ZIP manually from\n  {UCI_URL}\n"
                         f"unzip data.csv and pass --data /path/to/data.csv")
        path = cache
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]          # 'Daytime/evening attendance\t'
    df = df.rename(columns={"Nacionality": "Nationality"})
    assert df.shape == (4424, 37), f"unexpected shape {df.shape}"
    return df


def build_checkpoints(feature_cols):
    c1 = [c for c in feature_cols if "Curricular units 1st sem" in c]
    c2 = [c for c in feature_cols if "Curricular units 2nd sem" in c]
    c0 = [c for c in feature_cols if c not in c1 + c2]
    return {"C0": c0, "C1": c0 + c1, "C2": c0 + c1 + c2}


def build_groups(df):
    return {
        "gender": df["Gender"].values.astype(int),                       # 0 female, 1 male
        "age_mature": (df["Age at enrollment"] >= 23).astype(int).values,  # PT 'over-23' regime
        "scholarship": df["Scholarship holder"].values.astype(int),      # socio-economic proxy
        "special_needs": df["Educational special needs"].values.astype(int),
        "international": df["International"].values.astype(int),
    }


# ----------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------
class XGBBal(XGBClassifier):
    """XGBClassifier with 'balanced' sample weights when none are given (no class_weight in XGB)."""

    def fit(self, X, y, sample_weight=None, **kw):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kw)


def make_base(name, seed, a):
    ne = a.n_estimators
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=ne, learning_rate=0.05, num_leaves=31, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                              random_state=seed, n_jobs=1, verbose=-1)
    if name == "XGBoost":
        return XGBBal(n_estimators=ne, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist",
                      device="cuda" if a.gpu else "cpu", random_state=seed, n_jobs=1, verbosity=0)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=ne, min_samples_leaf=2,
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    raise ValueError(name)


def make_model(name, seed, a):
    if name in BASE_NAMES:
        return make_base(name, seed, a)
    ests = [(n.lower(), make_base(n, seed, a)) for n in BASE_NAMES]
    if name == "Hybrid-Stack":
        return StackingClassifier(
            estimators=ests,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            stack_method="predict_proba", n_jobs=1)
    if name == "Hybrid-Vote":
        return VotingClassifier(estimators=ests, voting="soft", n_jobs=1)
    raise ValueError(name)


def select_features(Xtr, ytr, k, method, seed):
    """Indices (sorted) of selected columns; computed on TRAINING data only."""
    p = Xtr.shape[1]
    if method == "none" or k >= p:
        return np.arange(p)
    if method == "mi":
        s = mutual_info_classif(Xtr, ytr, random_state=seed, n_neighbors=3)
    elif method == "anova":
        s, _ = f_classif(Xtr, ytr)
    else:
        raise ValueError(method)
    s = np.nan_to_num(s)
    return np.sort(np.argsort(-s, kind="stable")[:k])


def mc_metrics(y, proba):
    pred = proba.argmax(1)
    yb = (y == 0).astype(int)
    pb = proba[:, 0]
    out = dict(macro_f1=f1_score(y, pred, average="macro"),
               bal_acc=balanced_accuracy_score(y, pred),
               acc=accuracy_score(y, pred),
               auroc_ovr=roc_auc_score(y, proba, multi_class="ovr", average="macro"),
               auroc_dropout=roc_auc_score(yb, pb),
               prauc_dropout=average_precision_score(yb, pb))
    return out


def nb_ci(x, k_folds, alpha=0.05):
    """Mean and Nadeau-Bengio corrected CI for scores from repeated K-fold CV."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s2 = x.var(ddof=1)
    se = np.sqrt((1.0 / n + 1.0 / (k_folds - 1)) * s2)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return m, m - t * se, m + t * se


def nb_test(d, k_folds):
    """Nadeau-Bengio corrected resampled paired t-test on per-fold differences d."""
    d = np.asarray(d, float)
    n = len(d)
    dm, s2 = d.mean(), d.var(ddof=1)
    if s2 == 0:
        return dm, np.nan, 1.0, np.nan, (dm, dm)
    se = np.sqrt((1.0 / n + 1.0 / (k_folds - 1)) * s2)
    t = dm / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    tc = stats.t.ppf(0.975, n - 1)
    dz = dm / np.sqrt(s2)
    return dm, t, p, dz, (dm - tc * se, dm + tc * se)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------------------
# Stage 1: main table (model x filter x checkpoint)
# ----------------------------------------------------------------------------------------
def main_fold(r, k, tr, te, X, y, ck_idx, a):
    seed = a.seed + 100 * r + k
    rows = []
    for ck, cols in ck_idx.items():
        Xtr, Xte = X[tr][:, cols], X[te][:, cols]
        for filt in a.filters:
            sel = select_features(Xtr, y[tr], a.k, filt, seed)
            Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]
            for name in a.models:
                t0 = time.time()
                m = make_model(name, seed, a)
                m.fit(Xtr_s, y[tr])
                p = m.predict_proba(Xte_s)
                rows.append(dict(repeat=r, fold=k, checkpoint=ck, filter=filt, model=name,
                                 n_features=len(sel), fit_s=time.time() - t0,
                                 selected="|".join(map(str, np.array(cols)[sel])) if filt != "none" else "",
                                 **mc_metrics(y[te], p)))
    return rows


def run_main(X, y, splits, ck_idx, names, a, out):
    log(f"[main] {len(splits)} folds x {len(CKPTS)} ckpts x {len(a.filters)} filters x {len(a.models)} models")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(main_fold)(r, k, tr, te, X, y, ck_idx, a) for (r, k, tr, te) in splits)
    df = pd.DataFrame([row for rs in res for row in rs])
    # map selected indices -> names (global feature index within X columns)
    df.to_csv(os.path.join(out, "main_per_fold.csv"), index=False)

    K = a.folds
    summ = []
    for (ck, filt, model), g in df.groupby(["checkpoint", "filter", "model"]):
        row = dict(checkpoint=ck, filter=filt, model=model, n=len(g), n_features=g.n_features.mean())
        for met in ["macro_f1", "bal_acc", "auroc_ovr", "auroc_dropout", "prauc_dropout"]:
            m, lo, hi = nb_ci(g[met].values, K)
            row.update({f"{met}": m, f"{met}_lo": lo, f"{met}_hi": hi, f"{met}_sd": g[met].std(ddof=1)})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, "main_summary.csv"), index=False)

    # paired tests (per-fold macro-F1), Nadeau-Bengio, Holm over the whole family
    key = ["repeat", "fold"]
    tests = []
    for ck in CKPTS:
        d = df[df.checkpoint == ck]
        piv = d.pivot_table(index=key, columns=["filter", "model"], values="macro_f1")
        # (a) filter effect: MI - none, within model
        if "mi" in a.filters and "none" in a.filters:
            for model in a.models:
                diff = piv[("mi", model)] - piv[("none", model)]
                dm, t, p, dz, ci = nb_test(diff.values, K)
                tests.append(dict(checkpoint=ck, test=f"filter effect (MI - none) | {model}",
                                  mean_diff=dm, ci_lo=ci[0], ci_hi=ci[1], t=t, p_nb=p, dz=dz))
        # (b) hybrid vs best individual, within filter
        for filt in a.filters:
            ind = [m for m in a.models if m in BASE_NAMES]
            best = max(ind, key=lambda m: piv[(filt, m)].mean())
            for hyb in [m for m in a.models if m.startswith("Hybrid")]:
                diff = piv[(filt, hyb)] - piv[(filt, best)]
                dm, t, p, dz, ci = nb_test(diff.values, K)
                tests.append(dict(checkpoint=ck, test=f"{hyb} - best individual ({best}) | filter={filt}",
                                  mean_diff=dm, ci_lo=ci[0], ci_hi=ci[1], t=t, p_nb=p, dz=dz))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = holm(tests.p_nb.values)
        tests.to_csv(os.path.join(out, "main_tests.csv"), index=False)

    # feature-selection stability (selection frequency of each feature, MI filter)
    if "mi" in a.filters:
        fr = []
        for ck, g in df[df["filter"] == "mi"].groupby("checkpoint"):
            sel = g[g.model == a.models[0]]["selected"].str.split("|")
            cnt = pd.Series([names[int(c)] for s in sel for c in s]).value_counts() / len(sel)
            fr.append(pd.DataFrame(dict(checkpoint=ck, feature=cnt.index, selection_freq=cnt.values)))
        pd.concat(fr).to_csv(os.path.join(out, "mi_selection_frequency.csv"), index=False)

    for met in ["macro_f1", "auroc_dropout", "prauc_dropout"]:
        write_tex_main(summ, met, os.path.join(out, f"table_main_{met}.tex"), a)
    return summ, tests


def write_tex_main(summ, met, path, a):
    order_f = {"none": "No filter", "mi": f"MI top-{a.k}", "anova": f"ANOVA top-{a.k}"}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{UCI-697 external validation: %s (mean and 95\%% Nadeau--Bengio CI, %d repeats $\times$ %d folds).}"
             % (met.replace("_", r"\_"), a.repeats, a.folds),
             r"\label{tab:uci697_%s}" % met,
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Model & Feature filter & C0 (enrolment) & C1 (+1st sem.) & C2 (+2nd sem.) \\", r"\midrule"]
    for model in a.models:
        for filt in a.filters:
            cells = []
            for ck in CKPTS:
                r = summ[(summ.checkpoint == ck) & (summ["filter"] == filt) & (summ.model == model)].iloc[0]
                cells.append(f"{r[met]:.3f} [{r[met + '_lo']:.3f}, {r[met + '_hi']:.3f}]")
            lines.append(f"{model} & {order_f[filt]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines))


# ----------------------------------------------------------------------------------------
# Stage 2: filter sensitivity (k sweep, MI vs ANOVA)
# ----------------------------------------------------------------------------------------
def ksweep_fold(r, k, tr, te, X, y, ck_idx, a):
    seed = a.seed + 100 * r + k
    rows = []
    for ck, cols in ck_idx.items():
        Xtr, Xte = X[tr][:, cols], X[te][:, cols]
        configs = [("none", len(cols))] + [(f, kk) for f in ("mi", "anova") for kk in (6, 12, 18, 24) if kk < len(cols)]
        for filt, kk in configs:
            sel = select_features(Xtr, y[tr], kk, filt, seed)
            m = make_base("LightGBM", seed, a).fit(Xtr[:, sel], y[tr])
            p = m.predict_proba(Xte[:, sel])
            rows.append(dict(repeat=r, fold=k, checkpoint=ck, filter=filt, k=kk, **mc_metrics(y[te], p)))
    return rows


def run_ksweep(X, y, splits, ck_idx, a, out):
    log("[ksweep] filter sensitivity")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(ksweep_fold)(r, k, tr, te, X, y, ck_idx, a) for (r, k, tr, te) in splits)
    df = pd.DataFrame([row for rs in res for row in rs])
    rows = []
    for (ck, filt, kk), g in df.groupby(["checkpoint", "filter", "k"]):
        m, lo, hi = nb_ci(g.macro_f1.values, a.folds)
        rows.append(dict(checkpoint=ck, filter=filt, k=kk, macro_f1=m, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(out, "ksweep_summary.csv"), index=False)
    df.to_csv(os.path.join(out, "ksweep_per_fold.csv"), index=False)


# ----------------------------------------------------------------------------------------
# Stage 3: fairness
# ----------------------------------------------------------------------------------------
def reweigh_weights(y, g):
    """Kamiran & Calders reweighing: w(a,y) = P(a)P(y)/P(a,y)."""
    w = np.ones(len(y))
    n = len(y)
    for gv in np.unique(g):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            if m.sum() == 0:
                continue
            w[m] = ((g == gv).mean() * (y == yv).mean()) / (m.sum() / n)
    return w


def fit_lgbm(X, y, seed, a, w=None):
    return make_base("LightGBM", seed, a).fit(X, y, sample_weight=w)


def fair_fold(r, k, tr, te, X, yb, ck_idx, col_names, groups, a):
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    seed = a.seed + 100 * r + k
    out = []
    joint = sum(groups[t] * (2 ** i) for i, t in enumerate(a.joint)) if a.joint else None
    targets = list(a.targets) + (["joint"] if joint is not None else [])
    gmap = dict(groups)
    if joint is not None:
        gmap["joint"] = joint
    prot_idx = [i for i, c in enumerate(col_names) if c in PROTECTED_COLS]

    for ck in a.fair_ckpts:
        cols = np.array(ck_idx[ck])
        tr_i, cal_i = train_test_split(tr, test_size=0.25, stratify=yb[tr], random_state=seed)

        def record(method, pred, prob):
            out.append(dict(repeat=r, checkpoint=ck, method=method, te=te, pred=pred.astype(np.int8),
                            prob=prob.astype(np.float32)))

        # -- baseline (all features incl. protected) on model-training subset
        sel = select_features(X[tr_i][:, cols], yb[tr_i], a.k, "mi", seed)
        c_sel = cols[sel]
        base = fit_lgbm(X[tr_i][:, c_sel], yb[tr_i], seed, a)
        pb = base.predict_proba(X[te][:, c_sel])[:, 1]
        record("baseline", (pb >= 0.5), pb)

        # -- exclude protected attributes (candidates only non-protected columns)
        cand = np.array([c for c in cols if c not in prot_idx])
        sel_x = select_features(X[tr_i][:, cand], yb[tr_i], a.k, "mi", seed)
        c_x = cand[sel_x]
        mx = fit_lgbm(X[tr_i][:, c_x], yb[tr_i], seed, a)
        px = mx.predict_proba(X[te][:, c_x])[:, 1]
        record("exclude_protected", (px >= 0.5), px)

        for t in targets:
            g = gmap[t]
            # -- pre-processing: reweighing
            w = reweigh_weights(yb[tr_i], g[tr_i])
            mr = fit_lgbm(X[tr_i][:, c_sel], yb[tr_i], seed, a, w=w)
            pr = mr.predict_proba(X[te][:, c_sel])[:, 1]
            record(f"reweigh[{t}]", (pr >= 0.5), pr)

            # -- post-processing: equalized odds, fit on independent calibration split
            try:
                to = ThresholdOptimizer(estimator=base, constraints="equalized_odds",
                                        objective="balanced_accuracy_score", prefit=True,
                                        predict_method="predict_proba")
                to.fit(X[cal_i][:, c_sel], yb[cal_i], sensitive_features=g[cal_i])
                pt = to.predict(X[te][:, c_sel], sensitive_features=g[te], random_state=seed)
                try:
                    ppt = to._pmf_predict(X[te][:, c_sel], sensitive_features=g[te])[:, 1]
                except Exception:
                    ppt = pt.astype(float)
                record(f"threshold_eo[{t}]", pt, ppt)
            except Exception as e:  # tiny groups etc.
                log(f"  threshold_eo[{t}] failed r={r} k={k} {ck}: {e}")

            # -- in-processing: ExponentiatedGradient with EqualizedOdds
            if a.inproc:
                try:
                    eg = ExponentiatedGradient(make_base("LightGBM", seed, a),
                                               constraints=EqualizedOdds(difference_bound=0.02),
                                               max_iter=a.eg_iter)
                    eg.fit(X[tr_i][:, c_sel], yb[tr_i], sensitive_features=g[tr_i])
                    pe = eg.predict(X[te][:, c_sel], random_state=seed)
                    try:
                        ppe = eg._pmf_predict(X[te][:, c_sel])[:, 1]
                    except Exception:
                        ppe = pe.astype(float)
                    record(f"expgrad_eo[{t}]", pe, ppe)
                except Exception as e:
                    log(f"  expgrad_eo[{t}] failed r={r} k={k} {ck}: {e}")
    return out


def fair_core(y, pred, g, vals, min_n=20, min_pos=5):
    """(DIR, TPR gap, FPR gap, EO diff) across the groups in `vals` (groups with n<min_n skipped)."""
    sel, tpr, fpr = [], [], []
    for v in vals:
        m = g == v
        if m.sum() < min_n:
            continue
        yy, pp = y[m], pred[m]
        sel.append(pp.mean())
        pos = yy == 1
        neg = ~pos
        if pos.sum() >= min_pos:
            tpr.append(pp[pos].mean())
        if neg.sum() >= min_pos:
            fpr.append(pp[neg].mean())
    if len(sel) < 2:
        return np.nan, np.nan, np.nan, np.nan
    d = min(sel) / max(sel) if max(sel) > 0 else np.nan
    tg = max(tpr) - min(tpr) if len(tpr) >= 2 else np.nan
    fg = max(fpr) - min(fpr) if len(fpr) >= 2 else np.nan
    eo = np.nan if (np.isnan(tg) and np.isnan(fg)) else np.nanmax([tg, fg])
    return d, tg, fg, eo


def core3(y, pred, g, vals, min_n=20):
    d, tg, _, eo = fair_core(y, pred, g, vals, min_n)
    return d, tg, eo


def cal_gap(y, prob, g, vals, min_n=20):
    d = [prob[g == v].mean() - y[g == v].mean() for v in vals if (g == v).sum() >= min_n]
    return max(d) - min(d) if len(d) >= 2 else np.nan


def macro_f1_bin(y, p):
    f = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return np.mean(f)


def run_fairness(X, yb, splits, ck_idx, col_names, groups, a, out):
    log(f"[fairness] {len(splits)} folds x {len(a.fair_ckpts)} ckpts; targets={a.targets} joint={a.joint}")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(fair_fold)(r, k, tr, te, X, yb, ck_idx, col_names, groups, a) for (r, k, tr, te) in splits)
    n, R = len(yb), a.repeats
    store = {}
    for rs in res:
        for d in rs:
            key = (d["checkpoint"], d["method"])
            if key not in store:
                store[key] = dict(pred=np.full((R, n), -1, np.int8), prob=np.full((R, n), np.nan, np.float32))
            store[key]["pred"][d["repeat"], d["te"]] = d["pred"]
            store[key]["prob"][d["repeat"], d["te"]] = d["prob"]
    # keep only (ck, method) that completed all repeats for all students
    store = {k: v for k, v in store.items() if (v["pred"] >= 0).all()}
    np.savez_compressed(os.path.join(out, "oof_fairness.npz"), y=yb,
                        **{f"{ck}__{m}__pred": v["pred"] for (ck, m), v in store.items()},
                        **{f"{ck}__{m}__prob": v["prob"] for (ck, m), v in store.items()})
    attrs = dict(groups)
    vals = {t: np.unique(g) for t, g in attrs.items()}

    # ---- per-repeat metrics --------------------------------------------------------
    rows = []
    for (ck, m), v in store.items():
        for r in range(R):
            pred, prob = v["pred"][r].astype(int), v["prob"][r].astype(float)
            base = dict(checkpoint=ck, method=m, repeat=r)
            rows.append(dict(base, attr="overall", acc=accuracy_score(yb, pred),
                             macro_f1=macro_f1_bin(yb, pred), auroc=roc_auc_score(yb, prob),
                             prauc=average_precision_score(yb, prob), brier=brier_score_loss(yb, prob),
                             flag_rate=pred.mean()))
            for t, g in attrs.items():
                d, tg, fg, eo = fair_core(yb, pred, g, vals[t], a.min_n)
                rows.append(dict(base, attr=t, dir=d, tpr_gap=tg, fpr_gap=fg, eo_diff=eo,
                                 cal_gap=cal_gap(yb, prob, g, vals[t], a.min_n)))
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out, "fairness_per_repeat.csv"), index=False)
    summ = per.groupby(["checkpoint", "method", "attr"]).agg(["mean", "std"]).drop(columns=["repeat"], errors="ignore")
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.dropna(axis=1, how="all").reset_index()
    summ.to_csv(os.path.join(out, "fairness_summary.csv"), index=False)

    # ---- student-level bootstrap of paired deltas vs baseline ---------------------
    log("[fairness] bootstrap deltas")
    rng = np.random.default_rng(a.seed)
    B = a.boot
    idx = [rng.integers(0, n, n) for _ in range(B)]
    methods = sorted({m for (_, m) in store})
    delta_rows = []
    for ck in a.fair_ckpts:
        if (ck, "baseline") not in store:
            continue
        M = {}
        for m in methods:
            if (ck, m) not in store:
                continue
            P = store[(ck, m)]["pred"].astype(int)
            arr = np.full((B, 1 + 3 * len(attrs)), np.nan)
            for b in range(B):
                ib = idx[b]
                yb_b = yb[ib]
                f1s = np.mean([macro_f1_bin(yb_b, P[r][ib]) for r in range(R)])
                vec = [f1s]
                for t, g in attrs.items():
                    gb = g[ib]
                    cur = np.array([core3(yb_b, P[r][ib], gb, vals[t], a.min_n) for r in range(R)], float)
                    vec += list(np.nanmean(cur, axis=0))
                arr[b] = vec
            M[m] = arr
        ref = M["baseline"]
        for m, arr in M.items():
            if m == "baseline":
                continue
            D = arr - ref
            labels = ["macro_f1"] + [f"{t}:{s}" for t in attrs for s in ("dir", "tpr_gap", "eo_diff")]
            for j, lab in enumerate(labels):
                col = D[:, j]
                col = col[~np.isnan(col)]
                if len(col) < 10:
                    continue
                if lab == "macro_f1":
                    attr, met = "overall", "macro_f1"
                else:
                    attr, met = lab.split(":")
                delta_rows.append(dict(checkpoint=ck, method=m, attr=attr, metric=met,
                                       delta_mean=col.mean(), ci_lo=np.percentile(col, 2.5),
                                       ci_hi=np.percentile(col, 97.5),
                                       excludes_zero=bool(np.percentile(col, 2.5) > 0 or np.percentile(col, 97.5) < 0)))
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(os.path.join(out, "fairness_delta_vs_baseline.csv"), index=False)

    # ---- cross-attribute side-effect matrix --------------------------------------
    se = []
    for _, row in delta.iterrows():
        if row.metric not in ("dir", "tpr_gap", "eo_diff") or "[" not in row.method:
            continue
        mtype, tgt = row.method[:-1].split("[")
        before = summ[(summ.checkpoint == row.checkpoint) & (summ.method == "baseline") & (summ.attr == row.attr)]
        after = summ[(summ.checkpoint == row.checkpoint) & (summ.method == row.method) & (summ.attr == row.attr)]
        colm = f"{row.metric}_mean"
        if len(before) and len(after):
            worse = (row.delta_mean < 0) if row.metric == "dir" else (row.delta_mean > 0)
            se.append(dict(checkpoint=row.checkpoint, mitigation=mtype, target=tgt, audited=row.attr,
                           metric=row.metric, before=before[colm].values[0], after=after[colm].values[0],
                           delta=row.delta_mean, ci_lo=row.ci_lo, ci_hi=row.ci_hi,
                           worsened=bool(worse), significant=row.excludes_zero))
    se = pd.DataFrame(se)
    se.to_csv(os.path.join(out, "fairness_side_effects.csv"), index=False)
    write_tex_side_effects(se, os.path.join(out, "table_side_effects.tex"), a)

    # ---- intersectional audit ----------------------------------------------------------
    log("[fairness] intersectional audit")
    irow = []
    names = list(attrs)
    for (ck, m), v in store.items():
        if m not in ("baseline", "exclude_protected") and not m.startswith("threshold_eo"):
            continue
        for r in range(R):
            pred = v["pred"][r].astype(int)
            singles = {t: fair_core(yb, pred, attrs[t], vals[t], a.min_n)[0] for t in names}
            for ta, tb in combinations(names, 2):
                gab = attrs[ta] * 2 + attrs[tb]
                cells = [c for c in np.unique(gab) if (gab == c).sum() >= a.min_n]
                dab = fair_core(yb, pred, gab, np.unique(gab), a.min_n)[0]
                irow.append(dict(checkpoint=ck, method=m, repeat=r, pair=f"{ta} x {tb}", dir_pair=dab,
                                 dir_a=singles[ta], dir_b=singles[tb], reliable_cells=len(cells),
                                 worse_than_both=bool(dab < min(singles[ta], singles[tb]))
                                 if not np.isnan(dab) else False))
    inter = pd.DataFrame(irow)
    inter.to_csv(os.path.join(out, "intersectional_per_repeat.csv"), index=False)
    isum = inter.groupby(["checkpoint", "method", "pair"]).agg(
        dir_pair=("dir_pair", "mean"), dir_a=("dir_a", "mean"), dir_b=("dir_b", "mean"),
        reliable_cells=("reliable_cells", "min"), repeats_worse_than_both=("worse_than_both", "sum"),
        repeats=("repeat", "nunique")).reset_index()
    isum.to_csv(os.path.join(out, "intersectional_summary.csv"), index=False)
    return summ, delta, se, isum


def write_tex_side_effects(se, path, a):
    if not len(se):
        return
    ck = a.tex_ckpt if a.tex_ckpt in set(se.checkpoint) else se.checkpoint.iloc[0]
    d = se[(se.checkpoint == ck) & (se.metric == "dir")]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Cross-attribute side effects of fairness mitigations (checkpoint %s). DIR = min/max flag-rate ratio; "
             r"$\Delta$ vs.\ baseline with 95\%% student-level bootstrap CI; $^{*}$ CI excludes 0.}" % ck,
             r"\label{tab:uci697_side_effects}", r"\begin{tabular}{lllrrl}", r"\toprule",
             r"Mitigation & Target & Audited & DIR before & DIR after & $\Delta$ [95\% CI] \\", r"\midrule"]
    for _, r in d.sort_values(["mitigation", "target", "audited"]).iterrows():
        star = "$^{*}$" if r.significant else ""
        lines.append(f"{r.mitigation.replace('_', ' ')} & {r.target} & {r.audited.replace('_', ' ')} & "
                     f"{r.before:.3f} & {r.after:.3f} & {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines))


# ----------------------------------------------------------------------------------------
# Stage 4: explainability
# ----------------------------------------------------------------------------------------
def explain_fold(r, k, tr, te, X, yb, cols, names, a):
    import shap
    seed = a.seed + 100 * r + k
    cols = np.array(cols)
    sel = select_features(X[tr][:, cols], yb[tr], a.k, "mi", seed)
    c = cols[sel]
    Xtr, Xte = X[tr][:, c], X[te][:, c]
    m = fit_lgbm(Xtr, yb[tr], seed, a)
    auc = lambda Z: roc_auc_score(yb[te], m.predict_proba(Z)[:, 1])
    sv = shap.TreeExplainer(m).shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    imp = np.abs(sv).mean(0)
    perm = permutation_importance(m, Xte, yb[te], scoring="roc_auc", n_repeats=5,
                                  random_state=seed, n_jobs=1).importances_mean
    rho = stats.spearmanr(imp, perm).correlation
    med = np.median(Xtr, 0)
    order = np.argsort(-imp)
    rng = np.random.default_rng(seed)
    full = auc(Xte)
    cs, cr = [], []
    for kk in range(1, min(6, len(c)) + 1):
        Z = Xte.copy()
        Z[:, order[:kk]] = med[order[:kk]]
        cs.append(auc(Z))
        rs = []
        for _ in range(10):
            ri = rng.choice(len(c), kk, replace=False)
            Z = Xte.copy()
            Z[:, ri] = med[ri]
            rs.append(auc(Z))
        cr.append(np.mean(rs))
    vec = np.zeros(X.shape[1])
    vec[c] = imp
    return dict(repeat=r, fold=k, vec=vec, rho_shap_perm=rho, auc_full=full, del_shap=cs, del_rand=cr)


def run_explain(X, yb, splits, ck_idx, names, a, out):
    try:
        import shap  # noqa
    except ImportError:
        log("[explain] shap not installed -> pip install shap ; skipping")
        return
    log(f"[explain] checkpoint {a.explain_ckpt}")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(explain_fold)(r, k, tr, te, X, yb, ck_idx[a.explain_ckpt], names, a) for (r, k, tr, te) in splits)
    V = np.array([d["vec"] for d in res])
    rho_mat = stats.spearmanr(V, axis=1).correlation
    iu = np.triu_indices(len(V), 1)
    top5 = [set(np.argsort(-v)[:5]) for v in V]
    jac = [len(top5[i] & top5[j]) / len(top5[i] | top5[j]) for i, j in zip(*iu)]
    rho_sp = np.array([d["rho_shap_perm"] for d in res])
    summary = dict(checkpoint=a.explain_ckpt, folds=len(V),
                   shap_rank_stability_spearman_mean=float(np.mean(rho_mat[iu])),
                   shap_rank_stability_spearman_min=float(np.min(rho_mat[iu])),
                   top5_jaccard_mean=float(np.mean(jac)),
                   shap_vs_permutation_spearman_mean=float(np.nanmean(rho_sp)),
                   shap_vs_permutation_spearman_sd=float(np.nanstd(rho_sp, ddof=1)))
    kmax = min(len(d["del_shap"]) for d in res)
    del_rows = []
    for kk in range(kmax):
        s = np.array([d["del_shap"][kk] for d in res])
        rd = np.array([d["del_rand"][kk] for d in res])
        full = np.array([d["auc_full"] for d in res])
        dm, t, p, dz, ci = nb_test(rd - s, a.folds)   # random-AUROC minus SHAP-AUROC: >0 => SHAP deletion hurts more
        del_rows.append(dict(k_removed=kk + 1, auroc_full=full.mean(), auroc_after_shap_deletion=s.mean(),
                             auroc_after_random_deletion=rd.mean(), extra_drop_vs_random=dm,
                             ci_lo=ci[0], ci_hi=ci[1], p_nb=p))
    pd.DataFrame(del_rows).to_csv(os.path.join(out, "explain_deletion.csv"), index=False)
    imp = pd.DataFrame(dict(feature=names, mean_abs_shap=V.mean(0), selection_freq=(V > 0).mean(0)))
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(os.path.join(out, "explain_top_features.csv"), index=False)
    json.dump(summary, open(os.path.join(out, "explain_summary.json"), "w"), indent=2)
    log("[explain] " + json.dumps(summary))


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="path to UCI-697 data.csv (downloaded automatically if omitted)")
    p.add_argument("--out", default="../results/uci697", help="output folder (run from src/)")
    p.add_argument("--data_dir", default="../data/uci697", help="download cache for data.csv")
    p.add_argument("--stages", default="main,ksweep,fairness,explain")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--k", type=int, default=12, help="MI top-k (paper default 12)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--models", default="RandomForest,XGBoost,LightGBM,Hybrid-Stack",
                   help="add Hybrid-Vote for a soft-voting hybrid")
    p.add_argument("--filters", default="none,mi")
    p.add_argument("--targets", default="gender,age_mature,scholarship")
    p.add_argument("--joint", default="gender,scholarship", help="joint-group target ('' to disable)")
    p.add_argument("--fair_ckpts", default="C0,C1,C2")
    p.add_argument("--explain_ckpt", default="C2")
    p.add_argument("--tex_ckpt", default="C2")
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--eg_iter", type=int, default=25)
    p.add_argument("--min_n", type=int, default=20, help="min group size for fairness metrics")
    p.add_argument("--no_inproc", action="store_true", help="skip ExponentiatedGradient (slowest)")
    p.add_argument("--gpu", action="store_true", help="XGBoost on GPU (no benefit at this data size)")
    p.add_argument("--quick", action="store_true", help="smoke test: 1x3 CV, small trees, few bootstraps")
    a = p.parse_args()
    if a.quick:
        a.repeats, a.folds, a.n_estimators, a.boot, a.eg_iter = 1, 3, 60, 40, 8
    a.stages = [s for s in a.stages.split(",") if s]
    a.models = a.models.split(",")
    a.filters = a.filters.split(",")
    a.targets = [s for s in a.targets.split(",") if s]
    a.joint = [s for s in a.joint.split(",") if s]
    a.fair_ckpts = a.fair_ckpts.split(",")
    a.inproc = not a.no_inproc
    return a


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    df = load_data(a.data, a.data_dir)
    feat = [c for c in df.columns if c != "Target"]
    y3 = df["Target"].map(LABELS).values.astype(int)
    yb = (y3 == 0).astype(int)                       # at-risk = Dropout
    X = df[feat].values.astype(float)
    groups = build_groups(df)
    cks = build_checkpoints(feat)
    ck_idx = {ck: [feat.index(c) for c in cols] for ck, cols in cks.items()}
    log(f"data {df.shape}; classes {dict(zip(*np.unique(y3, return_counts=True)))}; at-risk rate {yb.mean():.3f}")
    for t, g in groups.items():
        log(f"  group {t}: " + ", ".join(f"{v}: n={int((g == v).sum())} atrisk={yb[g == v].mean():.3f}" for v in np.unique(g)))
    log("  checkpoints: " + ", ".join(f"{k}={len(v)} feats" for k, v in cks.items()))

    splits = []
    for r in range(a.repeats):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r)
        for k, (tr, te) in enumerate(skf.split(X, y3)):
            splits.append((r, k, tr, te))

    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__)
    import sklearn, lightgbm, xgboost
    cfg.update(sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, xgboost=xgboost.__version__)
    try:
        import fairlearn
        cfg["fairlearn"] = fairlearn.__version__
    except ImportError:
        pass
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2, default=str)

    report = ["# UCI-697 external validation report", ""]
    if "main" in a.stages:
        summ, tests = run_main(X, y3, splits, ck_idx, feat, a, a.out)
        piv = summ.pivot_table(index=["model", "filter"], columns="checkpoint", values="macro_f1").round(3)
        report += ["## Macro-F1 (3-class), mean over folds", "```", piv.to_string(), "```", ""]
        if len(tests):
            report += ["## Paired tests (Nadeau-Bengio, Holm)", "```",
                       tests.round(4).to_string(index=False), "```", ""]
    if "ksweep" in a.stages:
        run_ksweep(X, y3, splits, ck_idx, a, a.out)
    if "fairness" in a.stages:
        summ, delta, se, isum = run_fairness(X, yb, splits, ck_idx, feat, groups, a, a.out)
        if len(se):
            d = se[(se.metric == "dir") & (se.checkpoint == a.tex_ckpt)]
            report += [f"## Side effects on DIR ({a.tex_ckpt})", "```", d.round(3).to_string(index=False), "```", ""]
        report += ["## Intersectional DIR (baseline)", "```",
                   isum[isum.method == "baseline"].round(3).to_string(index=False), "```", ""]
    if "explain" in a.stages:
        run_explain(X, yb, splits, ck_idx, feat, a, a.out)
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(report))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}/")


if __name__ == "__main__":
    main()
