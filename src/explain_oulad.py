import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, RANDOM_STATE, TOP_K

from config import OUT_DIR
CUTOFF = 75

out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values
top_cols, mi_ranked = select_features_mi(out[feature_cols], at_risk, CAT_COLS, NUM_COLS, TOP_K)
cat_sel = [c for c in top_cols if c in CAT_COLS]
num_sel = [c for c in top_cols if c in NUM_COLS]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))
X = out[top_cols]
X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]

prep = make_preprocessor(cat_sel, num_sel)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
model = Pipeline([("prep", prep), ("clf", clf)])
model.fit(X_tr, y_tr)

# ---- Layer 1: SHAP (TreeExplainer on the LightGBM step, using transformed features) ----
X_te_trans = model.named_steps["prep"].transform(X_te)
feat_names = model.named_steps["prep"].get_feature_names_out()
explainer = shap.TreeExplainer(model.named_steps["clf"])
shap_values = explainer.shap_values(X_te_trans)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:15]
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh([feat_names[i] for i in order][::-1], mean_abs[order][::-1], color="#4C72B0")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Bio-FairAlert (OULAD): Global feature importance\nat-risk flag, LightGBM, day-75 checkpoint", fontsize=11)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_shap_global.png", dpi=150)
plt.close(fig)

shap_importance = [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]

# ---- Layer 2: LIME on a few individual at-risk students ----
lime_explainer = LimeTabularExplainer(
    training_data=model.named_steps["prep"].transform(X_tr),
    feature_names=list(feat_names),
    class_names=["not_at_risk", "at_risk"],
    mode="classification",
    discretize_continuous=True,
)

proba_te = model.predict_proba(X_te)[:, 1]
pred_te = (proba_te >= 0.5).astype(int)
high_risk_idx = np.where((y_te == 1) & (proba_te > 0.7))[0][:3]

lime_reports = []
for idx in high_risk_idx:
    exp = lime_explainer.explain_instance(
        X_te_trans[idx], model.named_steps["clf"].predict_proba, num_features=8
    )
    lime_reports.append({
        "test_row": int(idx), "predicted_proba_at_risk": float(proba_te[idx]),
        "top_factors": [{"factor": f, "weight": float(w)} for f, w in exp.as_list()],
    })

# ---- Layer 3: rule-based natural-language reports ----
active_mask = (y_te == 0)
thresholds = {
    "low_submission_rate": float(np.nanpercentile(X_te.loc[active_mask, "submission_rate"], 25)),
    "low_mean_score": float(np.nanpercentile(X_te.loc[active_mask, "mean_score"].dropna(), 25)),
    "high_lateness": float(np.nanpercentile(X_te.loc[active_mask, "mean_lateness"].dropna(), 75)),
}

def rule_report(row_idx):
    row = X_te.iloc[row_idx]
    triggered = []
    if pd.notna(row["submission_rate"]) and row["submission_rate"] < thresholds["low_submission_rate"]:
        triggered.append(f"Low assignment submission rate ({row['submission_rate']:.2f} < {thresholds['low_submission_rate']:.2f})")
    if pd.notna(row["mean_score"]) and row["mean_score"] < thresholds["low_mean_score"]:
        triggered.append(f"Low average assessment score ({row['mean_score']:.1f} < {thresholds['low_mean_score']:.1f})")
    if pd.notna(row["mean_lateness"]) and row["mean_lateness"] > thresholds["high_lateness"]:
        triggered.append(f"Frequent late submissions (avg lateness {row['mean_lateness']:.1f} days > {thresholds['high_lateness']:.1f})")
    risk_tier = "HIGH" if len(triggered) >= 2 else ("MEDIUM" if len(triggered) == 1 else "LOW")
    return {"row": int(row_idx), "risk_tier": risk_tier, "triggered_rules": triggered,
            "predicted_proba_at_risk": float(proba_te[row_idx])}

rule_reports = [rule_report(int(i)) for i in high_risk_idx]

results = {
    "cutoff": CUTOFF,
    "shap_global_importance": shap_importance,
    "lime_reports": lime_reports,
    "rule_thresholds": thresholds,
    "rule_reports": rule_reports,
}
with open(f"{OUT_DIR}/explainability_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
