"""
Protected-attribute inclusion/exclusion ablation at OULAD day-75, on the SAME
train/test split (random_state=42) used throughout the main fairness analysis,
so results are directly comparable to Table 2's "before calibration" row.

Conditions:
 1. all_features        - top-12 MI-selected features (existing baseline; imd_band
                           happens to be MI-selected, disability/gender are not)
 2. exclude_protected    - MI selection re-run with disability/imd_band/gender
                           removed from the candidate pool entirely before ranking
 3. demographic_only     - only demographic/administrative features (no behavioral
                           engagement signals)
 4. behavioral_only      - only behavioral/assessment-engagement features (no
                           demographic attributes at all, including imd_band)

Fairness is audited on the held-out test set in every condition using the metadata
table (disability/imd_band/gender), even in conditions where these are excluded
from the model's input features -- this is what "excluded from prediction but
retained for auditing" means operationally.
"""
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from config import OUT_DIR
CUTOFF = 75
RANDOM_STATE = 42
PROTECTED = ["disability", "imd_band", "gender"]
DEMO_COLS = ["gender", "region", "highest_education", "imd_band", "age_band",
             "disability", "num_of_prev_attempts", "studied_credits", "date_registration"]
BEHAV_COLS = ["n_assessments_due", "n_submitted", "submission_rate", "mean_score",
              "weighted_mean_score", "mean_lateness", "n_banked"]


def fairness_report(pred, y_true, meta, attr, min_n=20):
    attr_series = meta[attr].astype(object).where(meta[attr].notna(), "MISSING")
    groups = {}
    for val, idx in meta.groupby(attr_series).groups.items():
        idx = list(idx)
        if len(idx) < min_n:
            continue
        p = pred[idx]; t = y_true[idx]
        rec = f1_score(t, p, average="binary", zero_division=0) if t.sum() > 0 else float("nan")
        groups[str(val)] = {"n": len(idx), "flag_rate": float(p.mean())}
    flags = [g["flag_rate"] for g in groups.values()]
    di = float(min(flags) / max(flags)) if flags and max(flags) > 0 else None
    return di


def run_condition(name, feature_cols_override, out, feature_cols, target_col, group_col,
                   y_enc, at_risk, tr_idx, te_idx, use_mi=True):
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
                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro")
    di = {a: fairness_report(pred, y_te, meta_te, a) for a in PROTECTED}

    return {"condition": name, "features_used": top_cols, "n_features": len(top_cols),
            "accuracy": float(acc), "macro_f1": float(macro_f1), "disparate_impact": di}


def main():
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(out[feature_cols], at_risk, out[group_col]))

    results = []
    # 1. all features (replicate original MI top-12, includes imd_band)
    results.append(run_condition("all_features", feature_cols, out, feature_cols, target_col,
                                  group_col, y_enc, at_risk, tr_idx, te_idx, use_mi=True))
    # 2. protected attributes excluded from candidate pool entirely
    non_protected = [c for c in feature_cols if c not in PROTECTED]
    results.append(run_condition("exclude_protected", non_protected, out, feature_cols, target_col,
                                  group_col, y_enc, at_risk, tr_idx, te_idx, use_mi=True))
    # 3. demographic-only features (no MI reduction needed, already <= 12)
    results.append(run_condition("demographic_only", DEMO_COLS, out, feature_cols, target_col,
                                  group_col, y_enc, at_risk, tr_idx, te_idx, use_mi=False))
    # 4. behavioral-only features (no demographic attributes at all, including imd_band)
    results.append(run_condition("behavioral_only", BEHAV_COLS, out, feature_cols, target_col,
                                  group_col, y_enc, at_risk, tr_idx, te_idx, use_mi=False))

    with open(f"{OUT_DIR}/ablation_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(r["condition"], "n_feat=", r["n_features"], "acc=", round(r["accuracy"], 4),
              "macroF1=", round(r["macro_f1"], 4), "DI=", {k: (round(v, 3) if v else v) for k, v in r["disparate_impact"].items()})


if __name__ == "__main__":
    main()
