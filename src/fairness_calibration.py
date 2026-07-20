import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
from fairlearn.postprocessing import ThresholdOptimizer

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

# binary "at-risk" target: Fail or Withdrawn = 1
at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))

X = out[top_cols]
X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]
meta_tr = out.iloc[tr_idx].reset_index(drop=True)
meta_te = out.iloc[te_idx].reset_index(drop=True)

prep = make_preprocessor(cat_sel, num_sel)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
model = Pipeline([("prep", prep), ("clf", clf)])
model.fit(X_tr, y_tr)

proba_te = model.predict_proba(X_te)[:, 1]
pred_te_raw = (proba_te >= 0.5).astype(int)

def fairness_report(pred, y_true, meta, attr, min_n=20):
    groups = {}
    attr_series = meta[attr].astype(object).where(meta[attr].notna(), "MISSING")
    for val, idx in meta.groupby(attr_series).groups.items():
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
    return {"groups": groups, "max_min_recall_gap": gap, "disparate_impact_ratio": di}

results = {"cutoff": CUTOFF, "n_train": len(X_tr), "n_test": len(X_te), "selected_features": top_cols}

results["overall_before"] = {
    "accuracy": float(accuracy_score(y_te, pred_te_raw)),
    "macro_f1": float(f1_score(y_te, pred_te_raw, average="macro")),
    "auc": float(roc_auc_score(y_te, proba_te)),
}
results["fairness_before"] = {
    "disability": fairness_report(pred_te_raw, y_te, meta_te, "disability"),
    "imd_band": fairness_report(pred_te_raw, y_te, meta_te, "imd_band"),
    "gender": fairness_report(pred_te_raw, y_te, meta_te, "gender"),
}

# --- Equalized-odds post-processing calibration on 'disability' ---
sensitive_tr = meta_tr["disability"].values
sensitive_te = meta_te["disability"].values

to = ThresholdOptimizer(estimator=model, constraints="equalized_odds", predict_method="predict_proba", prefit=True)
to.fit(X_tr, y_tr, sensitive_features=sensitive_tr)
pred_te_calibrated = to.predict(X_te, sensitive_features=sensitive_te, random_state=RANDOM_STATE)

results["overall_after_disability_calibration"] = {
    "accuracy": float(accuracy_score(y_te, pred_te_calibrated)),
    "macro_f1": float(f1_score(y_te, pred_te_calibrated, average="macro")),
}
results["fairness_after_disability_calibration"] = {
    "disability": fairness_report(pred_te_calibrated, y_te, meta_te, "disability"),
    "imd_band": fairness_report(pred_te_calibrated, y_te, meta_te, "imd_band"),
    "gender": fairness_report(pred_te_calibrated, y_te, meta_te, "gender"),
}

with open(f"{OUT_DIR}/fairness_calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps({
    "overall_before": results["overall_before"],
    "disability_before": results["fairness_before"]["disability"],
    "overall_after": results["overall_after_disability_calibration"],
    "disability_after": results["fairness_after_disability_calibration"]["disability"],
}, indent=2))
