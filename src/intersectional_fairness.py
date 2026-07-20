"""
Intersectional fairness analysis at OULAD day-75, on the same train/test split
(random_state=42) and same uncalibrated LightGBM model as fairness_calibration.py,
so the intersectional disparate-impact ratios are directly comparable to Table 2's
single-attribute "before calibration" row.

Reports subgroup n explicitly and excludes/flags any intersectional cell with
n < 20 in the test set as too small to interpret, per reviewer guidance.
"""
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from config import OUT_DIR
CUTOFF = 75
RANDOM_STATE = 42
MIN_N = 20


def intersectional_report(pred, meta, attr1, attr2, min_n=MIN_N):
    s1 = meta[attr1].astype(object).where(meta[attr1].notna(), "MISSING")
    s2 = meta[attr2].astype(object).where(meta[attr2].notna(), "MISSING")
    combo = s1.astype(str) + " x " + s2.astype(str)
    groups = {}
    for val, idx in meta.groupby(combo).groups.items():
        idx = list(idx)
        n = len(idx)
        flag_rate = float(pred[idx].mean())
        groups[val] = {"n": n, "flag_rate": flag_rate, "reliable": n >= min_n}
    reliable_flags = [g["flag_rate"] for g in groups.values() if g["reliable"]]
    di = float(min(reliable_flags) / max(reliable_flags)) if reliable_flags and max(reliable_flags) > 0 else None
    n_reliable = sum(1 for g in groups.values() if g["reliable"])
    n_total = len(groups)
    return {"groups": groups, "disparate_impact_ratio_reliable_cells": di,
            "n_reliable_cells": n_reliable, "n_total_cells": n_total}


def main():
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    top_cols, _ = select_features_mi(out[feature_cols], y_enc, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))
    X = out[top_cols]
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr = at_risk[tr_idx]
    meta_te = out.iloc[te_idx].reset_index(drop=True)

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    pairs = [("disability", "gender"), ("imd_band", "gender"), ("disability", "imd_band")]
    results = {}
    for a1, a2 in pairs:
        results[f"{a1}_x_{a2}"] = intersectional_report(pred, meta_te, a1, a2)

    with open(f"{OUT_DIR}/intersectional_fairness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    for key, r in results.items():
        print(f"\n{key}: DI(reliable cells only)={r['disparate_impact_ratio_reliable_cells']}, "
              f"reliable={r['n_reliable_cells']}/{r['n_total_cells']} cells")
        for g, v in sorted(r["groups"].items(), key=lambda kv: -kv[1]["n"]):
            flag = "" if v["reliable"] else "  [n<20, excluded from DI]"
            print(f"  {g}: n={v['n']}, flag_rate={v['flag_rate']:.3f}{flag}")


if __name__ == "__main__":
    main()
