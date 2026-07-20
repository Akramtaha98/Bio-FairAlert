import json
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from mi_stacking import prep_window, make_preprocessor, RANDOM_STATE, N_SPLITS, OUT_DIR

CUTOFF = 100


def make_stacking_altmeta(cat_cols, num_cols):
    prep = make_preprocessor(cat_cols, num_cols)
    base_estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8,
                                   colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]
    # Alternative meta-learner: a shallow, low-capacity LightGBM classifier (few leaves, low depth)
    # in place of the plain Logistic Regression meta-learner, to test whether the OULAD
    # stacking-underperformance finding (Section 4.1) is attributable to this specific,
    # comparatively weak meta-learner choice rather than a deeper property of stacking on OULAD.
    meta = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, max_depth=3, learning_rate=0.1,
                               random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    stack = StackingClassifier(estimators=base_estimators, final_estimator=meta,
                                cv=5, n_jobs=1, passthrough=True)
    return Pipeline([("prep", prep), ("clf", stack)])


def run_fold(fold_idx):
    X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, n_cohort = prep_window(CUTOFF)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(X_sel, y_enc, groups))
    tr_idx, va_idx = splits[fold_idx]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    model = make_stacking_altmeta(cat_sel, num_sel)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    proba = model.predict_proba(X_va)
    f1 = f1_score(y_va, pred, average="macro")
    try:
        auc = roc_auc_score(y_va, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"cutoff_day": CUTOFF, "fold": fold_idx, "altmeta_stacking": {"macroF1": float(f1), "macroAUC": float(auc)}}


if __name__ == "__main__":
    fold_idx = int(sys.argv[1])
    res = run_fold(fold_idx)
    print(json.dumps(res, indent=2))
    ckpt = f"{OUT_DIR}/alt_meta_test_partial.json"
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            data = json.load(f)
    else:
        data = {"folds": []}
    data["folds"].append(res)
    with open(ckpt, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved checkpoint: fold={fold_idx}, total saved={len(data['folds'])}")
