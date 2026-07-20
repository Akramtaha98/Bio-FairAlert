"""
Explanation stability and fidelity tests for the OULAD day-75 at-risk LightGBM
model (identical training procedure to explain_oulad.py / fairness_calibration.py,
random_state=42), addressing the reviewer concern that SHAP/LIME/rule-based
agreement demonstrates convergence but not usability or faithfulness.

Tests:
 1. Perturbation stability: small Gaussian noise added to transformed numeric
    test features; Spearman correlation between original and perturbed global
    SHAP importance rankings, averaged over 3 noise seeds.
 2. Deletion test: replacing the top-5 SHAP features (by original-column
    aggregated importance) with their training-set median/mode, one group at a
    time, and measuring the resulting drop in macro-F1, contrasted against
    deleting the bottom-5 features as a control.
 3. SHAP vs. permutation importance: rank correlation between SHAP's global
    importance (aggregated to original-column level across one-hot dummies)
    and sklearn's permutation_importance on the same test set.
 4. Fold/split consistency: top-5 SHAP features recomputed on two additional
    independent 80/20 splits (random_state=100, 200) of the same day-75 cohort,
    compared against the main split's (random_state=42) top-5 for overlap.
"""
import json
import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K

from config import OUT_DIR
CUTOFF = 75
RANDOM_STATE = 42


def fit_model(random_state):
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values
    top_cols, _ = select_features_mi(out[feature_cols], at_risk, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    X = out[top_cols]
    tr_idx, te_idx = next(gss.split(X, at_risk, out[group_col]))
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)
    return model, X_tr, X_te, y_tr, y_te, top_cols, cat_sel, num_sel


def original_col_of(transformed_name, cat_sel, num_sel):
    """Map a ColumnTransformer output feature name back to its source original column."""
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__"):]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[len("cat__"):]
        for c in cat_sel:
            if rest.startswith(c + "_"):
                return c
        return rest
    return transformed_name


def aggregated_shap_importance(sv, feat_names, cat_sel, num_sel):
    mean_abs = np.abs(sv).mean(axis=0)
    agg = {}
    for name, val in zip(feat_names, mean_abs):
        orig = original_col_of(name, cat_sel, num_sel)
        agg[orig] = agg.get(orig, 0.0) + float(val)
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return agg, ranked


def main():
    model, X_tr, X_te, y_tr, y_te, top_cols, cat_sel, num_sel = fit_model(RANDOM_STATE)
    X_te_trans = model.named_steps["prep"].transform(X_te)
    feat_names = list(model.named_steps["prep"].get_feature_names_out())
    explainer = shap.TreeExplainer(model.named_steps["clf"])

    sv_orig = explainer.shap_values(X_te_trans)
    sv_orig = sv_orig[1] if isinstance(sv_orig, list) else sv_orig
    agg_orig, ranked_orig = aggregated_shap_importance(sv_orig, feat_names, cat_sel, num_sel)
    top5 = [c for c, _ in ranked_orig[:5]]
    bottom5 = [c for c, _ in ranked_orig[-5:]]
    print("Top-5 (main split, seed 42):", top5)
    print("Bottom-5:", bottom5)

    results = {"cutoff": CUTOFF, "top5_features_main_split": top5, "bottom5_features_main_split": bottom5}

    # ---- Test 1: perturbation stability ----
    rng_seeds = [1, 2, 3]
    corrs = []
    X_te_trans_dense = np.asarray(X_te_trans.todense()) if hasattr(X_te_trans, "todense") else np.asarray(X_te_trans)
    std = X_te_trans_dense.std(axis=0)
    std[std == 0] = 1.0
    for seed in rng_seeds:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 0.05 * std, size=X_te_trans_dense.shape)
        X_perturbed = X_te_trans_dense + noise
        sv_pert = explainer.shap_values(X_perturbed)
        sv_pert = sv_pert[1] if isinstance(sv_pert, list) else sv_pert
        agg_pert, ranked_pert = aggregated_shap_importance(sv_pert, feat_names, cat_sel, num_sel)
        common_cols = list(agg_orig.keys())
        rho, p = spearmanr([agg_orig[c] for c in common_cols], [agg_pert[c] for c in common_cols])
        corrs.append({"seed": seed, "spearman_rho": float(rho), "p_value": float(p)})
    results["perturbation_stability"] = corrs
    results["perturbation_stability_mean_rho"] = float(np.mean([c["spearman_rho"] for c in corrs]))
    print("Perturbation stability mean rho:", results["perturbation_stability_mean_rho"])

    # ---- Test 2: deletion test (top-5 vs bottom-5 vs random control) ----
    baseline_pred = model.predict(X_te)
    baseline_f1 = f1_score(y_te, baseline_pred, average="macro")

    def delete_and_eval(cols_to_delete):
        X_mod = X_te.copy()
        for c in cols_to_delete:
            if c in num_sel:
                X_mod[c] = X_tr[c].median()
            elif c in cat_sel:
                X_mod[c] = X_tr[c].mode().iloc[0]
        pred_mod = model.predict(X_mod)
        return float(f1_score(y_te, pred_mod, average="macro"))

    f1_delete_top5 = delete_and_eval(top5)
    f1_delete_bottom5 = delete_and_eval(bottom5)
    rng4 = np.random.RandomState(4)
    random5 = list(rng4.choice(top_cols, size=min(5, len(top_cols)), replace=False))
    f1_delete_random5 = delete_and_eval(random5)

    results["deletion_test"] = {
        "baseline_macroF1": float(baseline_f1),
        "delete_top5_macroF1": f1_delete_top5, "delete_top5_drop": float(baseline_f1 - f1_delete_top5),
        "delete_bottom5_macroF1": f1_delete_bottom5, "delete_bottom5_drop": float(baseline_f1 - f1_delete_bottom5),
        "delete_random5_macroF1": f1_delete_random5, "delete_random5_drop": float(baseline_f1 - f1_delete_random5),
        "random5_features": random5,
    }
    print("Deletion test:", results["deletion_test"])

    # ---- Test 3: SHAP vs permutation importance ----
    perm = permutation_importance(model, X_te, y_te, scoring="f1_macro", n_repeats=10,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    perm_importance = dict(zip(top_cols, perm.importances_mean))
    common_cols = [c for c in top_cols if c in agg_orig]
    shap_vals_common = [agg_orig[c] for c in common_cols]
    perm_vals_common = [perm_importance[c] for c in common_cols]
    rho_sp, p_sp = spearmanr(shap_vals_common, perm_vals_common)
    results["shap_vs_permutation_importance"] = {
        "spearman_rho": float(rho_sp), "p_value": float(p_sp),
        "shap_ranking": sorted(common_cols, key=lambda c: -agg_orig[c]),
        "permutation_ranking": sorted(common_cols, key=lambda c: -perm_importance[c]),
    }
    print("SHAP vs permutation importance rho:", rho_sp, "p=", p_sp)

    # ---- Test 4: split/fold consistency ----
    overlaps = []
    for seed in [100, 200]:
        model_s, X_tr_s, X_te_s, y_tr_s, y_te_s, top_cols_s, cat_sel_s, num_sel_s = fit_model(seed)
        X_te_trans_s = model_s.named_steps["prep"].transform(X_te_s)
        feat_names_s = list(model_s.named_steps["prep"].get_feature_names_out())
        explainer_s = shap.TreeExplainer(model_s.named_steps["clf"])
        sv_s = explainer_s.shap_values(X_te_trans_s)
        sv_s = sv_s[1] if isinstance(sv_s, list) else sv_s
        agg_s, ranked_s = aggregated_shap_importance(sv_s, feat_names_s, cat_sel_s, num_sel_s)
        top5_s = [c for c, _ in ranked_s[:5]]
        overlap = len(set(top5) & set(top5_s))
        overlaps.append({"seed": seed, "top5": top5_s, "overlap_with_main_split": overlap})
        print(f"seed {seed} top5:", top5_s, "overlap with main:", overlap)
    results["split_consistency"] = overlaps

    with open(f"{OUT_DIR}/explanation_stability_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
