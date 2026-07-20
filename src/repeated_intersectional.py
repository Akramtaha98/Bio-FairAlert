"""
Repeated intersectional fairness analysis at OULAD day-75, across 5 of the
same repeated 80/20 splits used for the broader fairness panel (seeds
100-104), addressing the reviewer's request to test whether the single-split
intersectional disparate-impact ratios (disability x gender, imd_band x
gender, disability x imd_band) are stable across resampling.

For each split and each attribute-pair intersection, computes the
disparate-impact ratio restricted to cells with >= 20 test students, and
records how many splits keep that ratio below the four-fifths (0.80)
threshold plus the minimum reliable-cell count observed.
"""
import json
import os
import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K

from config import OUT_DIR
CKPT = f"{OUT_DIR}/repeated_intersectional.json"
CUTOFF = 75
MIN_N = 20
PAIRS = [("disability", "gender"), ("imd_band", "gender"), ("disability", "imd_band")]


def intersectional_report(pred, meta, attr1, attr2, min_n=MIN_N):
    s1 = meta[attr1].astype(object).where(meta[attr1].notna(), "MISSING")
    s2 = meta[attr2].astype(object).where(meta[attr2].notna(), "MISSING")
    combo = s1.astype(str) + " x " + s2.astype(str)
    n_reliable = 0
    n_total = 0
    reliable_flags = []
    min_cell_n = None
    for val, idx in meta.groupby(combo).groups.items():
        idx = list(idx)
        n = len(idx)
        n_total += 1
        if min_cell_n is None or n < min_cell_n:
            min_cell_n = n
        if n >= min_n:
            n_reliable += 1
            reliable_flags.append(float(pred[idx].mean()))
    di = float(min(reliable_flags) / max(reliable_flags)) if reliable_flags and max(reliable_flags) > 0 else None
    return {"di": di, "n_reliable": n_reliable, "n_total": n_total, "min_cell_n": min_cell_n}


def run_split(split_idx):
    rs = 100 + split_idx
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    top_cols, _ = select_features_mi(out[feature_cols], y_enc, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))
    X = out[top_cols]
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr = at_risk[tr_idx]
    meta_te = out.iloc[te_idx].reset_index(drop=True)

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=rs, n_jobs=1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    result = {"split": split_idx, "random_state": rs, "pairs": {}}
    for a1, a2 in PAIRS:
        result["pairs"][f"{a1}_x_{a2}"] = intersectional_report(pred, meta_te, a1, a2)
    return result


if __name__ == "__main__":
    split_idx = int(sys.argv[1])
    res = run_split(split_idx)
    data = json.load(open(CKPT)) if os.path.exists(CKPT) else {"splits": []}
    data["splits"].append(res)
    json.dump(data, open(CKPT, "w"), indent=2)
    print(f"split {split_idx} (seed {res['random_state']}) done, total={len(data['splits'])}")
    for k, v in res["pairs"].items():
        print(" ", k, "DI=", round(v["di"], 3) if v["di"] else None, "reliable=", f"{v['n_reliable']}/{v['n_total']}",
              "min_cell_n=", v["min_cell_n"])
