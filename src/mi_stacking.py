import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb

from features import build_feature_table

RANDOM_STATE = 42
from config import OUT_DIR
CUTOFFS = [25, 50, 75, 100]
TOP_K = 12

CAT_COLS = ["code_module", "code_presentation", "gender", "region", "highest_education",
            "imd_band", "age_band", "disability"]
NUM_COLS = ["num_of_prev_attempts", "studied_credits", "date_registration",
            "module_presentation_length", "n_assessments_due", "n_submitted",
            "submission_rate", "mean_score", "weighted_mean_score", "mean_lateness", "n_banked"]


def select_features_mi(X, y_enc, cat_cols, num_cols, top_k=TOP_K):
    """Encode categoricals ordinally for MI scoring, compute MI per column, keep top_k columns overall."""
    enc_df = X.copy()
    discrete_mask = []
    for c in cat_cols:
        enc_df[c] = enc_df[c].astype("category").cat.codes
        discrete_mask.append(True)
    for c in num_cols:
        enc_df[c] = enc_df[c].fillna(enc_df[c].median())
        discrete_mask.append(False)
    cols = cat_cols + num_cols
    enc_df = enc_df[cols]
    mi = mutual_info_classif(enc_df, y_enc, discrete_features=discrete_mask, random_state=RANDOM_STATE)
    ranked = sorted(zip(cols, mi), key=lambda t: t[1], reverse=True)
    top_cols = [c for c, _ in ranked[:top_k]]
    return top_cols, ranked


def make_preprocessor(cat_cols, num_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)


def make_stacking(cat_cols, num_cols):
    prep = make_preprocessor(cat_cols, num_cols)
    base_estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8,
                                   colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]
    # NOTE: final_estimator intentionally does NOT use class_weight="balanced" -- combining
    # balanced base learners with a further-balanced meta-learner was empirically found to
    # over-correct and severely miscalibrate predictions (verified during pipeline development).
    # passthrough=True gives the meta-learner access to the original features in addition to
    # base-learner probabilities, which empirically improved macro-F1 substantially over
    # passthrough=False in this dataset.
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=5, n_jobs=1, passthrough=True,
    )
    return Pipeline([("prep", prep), ("clf", stack)])


def make_best_individual(cat_cols, num_cols):
    prep = make_preprocessor(cat_cols, num_cols)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    return Pipeline([("prep", prep), ("clf", clf)])


N_SPLITS = 3


def prep_window(cutoff_day):
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=cutoff_day)
    X = out[feature_cols]
    y = out[target_col]
    groups = out[group_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    top_cols, mi_ranked = select_features_mi(X, y_enc, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    X_sel = X[top_cols]
    return X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, len(out)


def run_fold(cutoff_day, fold_idx):
    X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, n_cohort = prep_window(cutoff_day)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(X_sel, y_enc, groups))
    tr_idx, va_idx = splits[fold_idx]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    stack_model = make_stacking(cat_sel, num_sel)
    stack_model.fit(X_tr, y_tr)
    pred = stack_model.predict(X_va)
    proba = stack_model.predict_proba(X_va)
    stack_f1 = f1_score(y_va, pred, average="macro")
    try:
        stack_auc = roc_auc_score(y_va, proba, multi_class="ovr", average="macro")
    except Exception:
        stack_auc = float("nan")

    best_model = make_best_individual(cat_sel, num_sel)
    best_model.fit(X_tr, y_tr)
    pred_b = best_model.predict(X_va)
    proba_b = best_model.predict_proba(X_va)
    best_f1 = f1_score(y_va, pred_b, average="macro")
    try:
        best_auc = roc_auc_score(y_va, proba_b, multi_class="ovr", average="macro")
    except Exception:
        best_auc = float("nan")

    result = {
        "cutoff_day": cutoff_day, "fold": fold_idx, "n_cohort": n_cohort,
        "selected_features": top_cols,
        "top_features_mi": [{"feature": c, "mi": float(m)} for c, m in mi_ranked[:TOP_K]],
        "stacking": {"macroF1": float(stack_f1), "macroAUC": float(stack_auc)},
        "best_individual_lightgbm": {"macroF1": float(best_f1), "macroAUC": float(best_auc)},
    }
    return result


if __name__ == "__main__":
    import sys
    import os
    cutoff = int(sys.argv[1])
    fold_idx = int(sys.argv[2])
    res = run_fold(cutoff, fold_idx)
    print(json.dumps(res, indent=2))
    ckpt = f"{OUT_DIR}/mi_stacking_partial.json"
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            data = json.load(f)
    else:
        data = {"folds": []}
    data["folds"].append(res)
    with open(ckpt, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved checkpoint: cutoff={cutoff} fold={fold_idx}, total saved={len(data['folds'])}")
