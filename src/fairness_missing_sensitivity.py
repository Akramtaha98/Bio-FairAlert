import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, RANDOM_STATE, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import OUT_DIR
CUTOFF = 75

out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
le = LabelEncoder()
y_enc = le.fit_transform(out[target_col])
top_cols, mi_ranked = select_features_mi(out[feature_cols], y_enc, CAT_COLS, NUM_COLS, TOP_K)
cat_sel = [c for c in top_cols if c in CAT_COLS]
num_sel = [c for c in top_cols if c in NUM_COLS]

at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))

X = out[top_cols]
X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]
meta_te = out.iloc[te_idx].reset_index(drop=True)

prep = make_preprocessor(cat_sel, num_sel)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
model = Pipeline([("prep", prep), ("clf", clf)])
model.fit(X_tr, y_tr)

proba_te = model.predict_proba(X_te)[:, 1]
pred_te_raw = (proba_te >= 0.5).astype(int)


def fairness_report(pred, y_true, meta, attr, min_n=20, exclude_missing=False):
    groups = {}
    attr_series = meta[attr].astype(object).where(meta[attr].notna(), "MISSING")
    for val, idx in meta.groupby(attr_series).groups.items():
        if exclude_missing and val == "MISSING":
            continue
        idx = list(idx)
        if len(idx) < min_n:
            continue
        p = pred[idx]
        t = y_true[idx]
        rec = recall_score(t, p, zero_division=0) if t.sum() > 0 else float("nan")
        flag_rate = p.mean()
        groups[str(val)] = {"n": len(idx), "recall": float(rec), "flag_rate": float(flag_rate)}
    recalls = [g["recall"] for g in groups.values() if not np.isnan(g["recall"])]
    flags = [g["flag_rate"] for g in groups.values()]
    gap = float(max(recalls) - min(recalls)) if recalls else None
    di = float(min(flags) / max(flags)) if flags and max(flags) > 0 else None
    return {"groups": groups, "max_min_recall_gap": gap, "disparate_impact_ratio": di,
            "n_groups_used": len(groups), "n_students_used": sum(g["n"] for g in groups.values())}

n_missing = (meta_te["imd_band"].isna()).sum()

results = {
    "cutoff": CUTOFF,
    "n_test_total": len(X_te),
    "n_missing_imd_band": int(n_missing),
    "imd_band_including_missing": fairness_report(pred_te_raw, y_te, meta_te, "imd_band", exclude_missing=False),
    "imd_band_excluding_missing": fairness_report(pred_te_raw, y_te, meta_te, "imd_band", exclude_missing=True),
}

with open(f"{OUT_DIR}/fairness_missing_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps({
    "n_test_total": results["n_test_total"],
    "n_missing_imd_band": results["n_missing_imd_band"],
    "DI_including_missing": results["imd_band_including_missing"]["disparate_impact_ratio"],
    "gap_including_missing": results["imd_band_including_missing"]["max_min_recall_gap"],
    "DI_excluding_missing": results["imd_band_excluding_missing"]["disparate_impact_ratio"],
    "gap_excluding_missing": results["imd_band_excluding_missing"]["max_min_recall_gap"],
}, indent=2))
