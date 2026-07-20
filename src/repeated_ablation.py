"""
Repeated protected-attribute inclusion/exclusion ablation at OULAD day-75,
across the same 10 student-grouped 80/20 splits (seeds 100-109) used for
Table 2's fairness-calibration analysis, addressing the reviewer's highest-
priority request that the single-split ablation result be replicated.

For each of 10 splits and 4 feature conditions (all_features, exclude_protected,
demographic_only, behavioral_only), reports: macro-F1, accuracy, balanced
accuracy, disparate-impact ratio for disability/imd_band/gender, and the
equalized-odds difference (max of TPR gap, FPR gap) for each of the three
protected attributes. Checkpoints incrementally to repeated_ablation.json.
"""
import json
import os
import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from config import OUT_DIR
CKPT = f"{OUT_DIR}/repeated_ablation.json"
CUTOFF = 75
PROTECTED = ["disability", "imd_band", "gender"]
DEMO_COLS = ["gender", "region", "highest_education", "imd_band", "age_band",
             "disability", "num_of_prev_attempts", "studied_credits", "date_registration"]
BEHAV_COLS = ["n_assessments_due", "n_submitted", "submission_rate", "mean_score",
              "weighted_mean_score", "mean_lateness", "n_banked"]


def group_confusion(pred, y_true, idx):
    p = pred[idx]; t = y_true[idx]
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return tpr, fpr


def attr_metrics(pred, y_true, meta, attr, min_n=20):
    attr_series = meta[attr].astype(object).where(meta[attr].notna(), "MISSING")
    flags, tprs, fprs = [], [], []
    for val, idx in meta.groupby(attr_series).groups.items():
        idx = np.array(list(idx))
        if len(idx) < min_n:
            continue
        flags.append(pred[idx].mean())
        tpr, fpr = group_confusion(pred, y_true, idx)
        if not np.isnan(tpr):
            tprs.append(tpr)
        if not np.isnan(fpr):
            fprs.append(fpr)
    di = float(min(flags) / max(flags)) if flags and max(flags) > 0 else None
    tpr_gap = float(max(tprs) - min(tprs)) if tprs else None
    fpr_gap = float(max(fprs) - min(fprs)) if fprs else None
    eq_odds = float(max(tpr_gap or 0, fpr_gap or 0)) if (tpr_gap is not None or fpr_gap is not None) else None
    return {"di": di, "eq_odds_diff": eq_odds}


def run_condition(name, feature_cols_override, out, target_col, y_enc, at_risk,
                   tr_idx, te_idx, rs, use_mi=True):
    if use_mi:
        cand_cat = [c for c in CAT_COLS if c in feature_cols_override]
        cand_num = [c for c in NUM_COLS if c in feature_cols_override]
        top_cols, _ = select_features_mi(out[cand_cat + cand_num], y_enc, cand_cat, cand_num,
                                          top_k=min(TOP_K, len(cand_cat) + len(cand_num)))
    else:
        top_cols = feature_cols_override

    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    X = out[top_cols]
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]
    meta_te = out.iloc[te_idx].reset_index(drop=True)

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=rs, n_jobs=1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = accuracy_score(y_te, pred)
    bal_acc = balanced_accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro")
    attr_res = {a: attr_metrics(pred, y_te, meta_te, a) for a in PROTECTED}

    return {"condition": name, "n_features": len(top_cols), "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc), "macro_f1": float(macro_f1),
            "disability_di": attr_res["disability"]["di"], "imd_band_di": attr_res["imd_band"]["di"],
            "gender_di": attr_res["gender"]["di"],
            "disability_eqodds": attr_res["disability"]["eq_odds_diff"],
            "imd_band_eqodds": attr_res["imd_band"]["eq_odds_diff"],
            "gender_eqodds": attr_res["gender"]["eq_odds_diff"]}


def run_split(split_idx):
    rs = 100 + split_idx
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    tr_idx, te_idx = next(gss.split(out[feature_cols], at_risk, out[group_col]))

    conditions = []
    conditions.append(run_condition("all_features", feature_cols, out, target_col, y_enc, at_risk,
                                     tr_idx, te_idx, rs, use_mi=True))
    non_protected = [c for c in feature_cols if c not in PROTECTED]
    conditions.append(run_condition("exclude_protected", non_protected, out, target_col, y_enc, at_risk,
                                     tr_idx, te_idx, rs, use_mi=True))
    conditions.append(run_condition("demographic_only", DEMO_COLS, out, target_col, y_enc, at_risk,
                                     tr_idx, te_idx, rs, use_mi=False))
    conditions.append(run_condition("behavioral_only", BEHAV_COLS, out, target_col, y_enc, at_risk,
                                     tr_idx, te_idx, rs, use_mi=False))
    return {"split": split_idx, "random_state": rs, "conditions": conditions}


if __name__ == "__main__":
    split_idx = int(sys.argv[1])
    res = run_split(split_idx)
    data = json.load(open(CKPT)) if os.path.exists(CKPT) else {"splits": []}
    data["splits"].append(res)
    json.dump(data, open(CKPT, "w"), indent=2)
    print(f"split {split_idx} (seed {res['random_state']}) done, total={len(data['splits'])}")
    for c in res["conditions"]:
        print(" ", c["condition"], "macroF1=", round(c["macro_f1"], 4),
              "DI(imd)=", round(c["imd_band_di"], 3) if c["imd_band_di"] else None)
