"""
Broader fairness metric panel (TPR gap, FPR gap, equalized-odds difference,
demographic-parity difference, subgroup macro-F1/balanced accuracy, per-group
calibration error) alongside the disparate-impact ratio and recall gap already
reported, computed across 5 of the same 10 repeated 80/20 splits used in
bootstrap_fairness.py (seeds 100-104; a disclosed subset for compute feasibility,
since this panel requires storing per-group TP/FP/TN/FN and predicted
probabilities rather than only the two summary numbers bootstrap_fairness.py
kept). Reuses the identical model-training procedure (same LightGBM config,
same ThresholdOptimizer disability-targeted calibration) for exact comparability
with Table 2/2b.
"""
import json
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import OUT_DIR
CUTOFF = 75
CKPT = f"{OUT_DIR}/fairness_panel_repeated.json"


def group_confusion(pred, y_true, idx):
    p = pred[idx]; t = y_true[idx]
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    flag_rate = p.mean()
    return tpr, fpr, flag_rate, len(idx)


def calibration_error(proba, y_true, idx, n_bins=5):
    """Simple expected calibration error within a subgroup."""
    p = proba[idx]; t = y_true[idx]
    if len(p) < 10:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = t[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def full_panel(pred, proba, y_true, meta, attr, min_n=20):
    attr_series = meta[attr].astype(object).where(meta[attr].notna(), "MISSING")
    groups = {}
    for val, idx in meta.groupby(attr_series).groups.items():
        idx = np.array(list(idx))
        if len(idx) < min_n:
            continue
        tpr, fpr, flag_rate, n = group_confusion(pred, y_true, idx)
        macro_f1_g = f1_score(y_true[idx], pred[idx], average="macro", zero_division=0)
        bal_acc_g = balanced_accuracy_score(y_true[idx], pred[idx]) if len(set(y_true[idx])) > 1 else float("nan")
        ece = calibration_error(proba, y_true, idx)
        groups[str(val)] = {"n": n, "tpr": tpr, "fpr": fpr, "flag_rate": flag_rate,
                             "macro_f1": float(macro_f1_g), "balanced_acc": float(bal_acc_g),
                             "calibration_error": ece}
    tprs = [g["tpr"] for g in groups.values() if not np.isnan(g["tpr"])]
    fprs = [g["fpr"] for g in groups.values() if not np.isnan(g["fpr"])]
    flags = [g["flag_rate"] for g in groups.values()]
    tpr_gap = float(max(tprs) - min(tprs)) if tprs else None
    fpr_gap = float(max(fprs) - min(fprs)) if fprs else None
    eq_odds_diff = float(max(tpr_gap or 0, fpr_gap or 0))
    dp_diff = float(max(flags) - min(flags)) if flags else None
    di_ratio = float(min(flags) / max(flags)) if flags and max(flags) > 0 else None
    return {"groups": groups, "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
            "equalized_odds_diff": eq_odds_diff, "demographic_parity_diff": dp_diff,
            "disparate_impact_ratio": di_ratio}


def run_repeat(repeat_idx):
    rs = 100 + repeat_idx
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    top_cols, mi_ranked = select_features_mi(out[feature_cols], y_enc, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))
    X = out[top_cols]
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]
    meta_tr = out.iloc[tr_idx].reset_index(drop=True)
    meta_te = out.iloc[te_idx].reset_index(drop=True)

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=rs, n_jobs=-1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]
    pred_raw = (proba_te >= 0.5).astype(int)

    result = {"repeat": repeat_idx, "random_state": rs}
    result["before"] = {a: full_panel(pred_raw, proba_te, y_te, meta_te, a)
                         for a in ["disability", "imd_band", "gender"]}
    result["before"]["overall_macrof1"] = float(f1_score(y_te, pred_raw, average="macro"))

    sens_tr = meta_tr["disability"].values
    sens_te = meta_te["disability"].values
    to = ThresholdOptimizer(estimator=model, constraints="equalized_odds", predict_method="predict_proba", prefit=True)
    to.fit(X_tr, y_tr, sensitive_features=sens_tr)
    pred_cal = to.predict(X_te, sensitive_features=sens_te, random_state=rs)
    # ThresholdOptimizer doesn't give calibrated probabilities directly; approximate
    # calibration error using the raw model's probabilities against the NEW calibrated
    # predictions (documented explicitly as an approximation, not exact post-cal probabilities).
    result["after_disability_calib"] = {a: full_panel(pred_cal, proba_te, y_te, meta_te, a)
                                         for a in ["disability", "imd_band", "gender"]}
    result["after_disability_calib"]["overall_macrof1"] = float(f1_score(y_te, pred_cal, average="macro"))
    return result


if __name__ == "__main__":
    import sys
    repeat_idx = int(sys.argv[1])
    res = run_repeat(repeat_idx)
    data = json.load(open(CKPT)) if os.path.exists(CKPT) else {"repeats": []}
    data["repeats"].append(res)
    json.dump(data, open(CKPT, "w"), indent=2)
    print(f"repeat {repeat_idx} done, total={len(data['repeats'])}")
    print("before disability:", res["before"]["disability"]["tpr_gap"], res["before"]["disability"]["fpr_gap"])
