import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from mooc_features import build_feature_table
from mooc_mi_stacking import select_top_k, make_preprocessor, RANDOM_STATE, TOP_K

from config import OUT_DIR
CUTOFF = 21

agg, feature_cols, target_col = build_feature_table(cutoff_day=CUTOFF)
X = agg[feature_cols]
y = agg[target_col].values
top_cols, ranked = select_top_k(X, y, TOP_K)

X_tr, X_te, y_tr, y_te = train_test_split(
    X[top_cols], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

prep = make_preprocessor()
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
model = Pipeline([("prep", prep), ("clf", clf)])
model.fit(X_tr, y_tr)

X_te_trans = model.named_steps["prep"].transform(X_te)
explainer = shap.TreeExplainer(model.named_steps["clf"])
shap_values = explainer.shap_values(X_te_trans)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:15]
feat_names = top_cols

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh([feat_names[i] for i in order][::-1], mean_abs[order][::-1], color="#4C72B0")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Bio-FairAlert (MOOC): Global feature importance\nat-risk label, LightGBM, day-21 checkpoint", fontsize=11)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_shap_global_mooc.png", dpi=150)
plt.close(fig)

shap_importance = [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
results = {"cutoff": CUTOFF, "selected_features": top_cols, "shap_global_importance": shap_importance}
with open(f"{OUT_DIR}/explainability_results_mooc.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
