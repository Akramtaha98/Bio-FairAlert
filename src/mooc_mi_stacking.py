import json
import os
import sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from mooc_features import build_feature_table

RANDOM_STATE = 42
from config import OUT_DIR
TOP_K = 12
N_SPLITS = 3


def select_top_k(X, y, top_k=TOP_K):
    Xf = X.fillna(X.median())
    mi = mutual_info_classif(Xf, y, discrete_features=False, random_state=RANDOM_STATE)
    ranked = sorted(zip(X.columns, mi), key=lambda t: t[1], reverse=True)
    top_cols = [c for c, _ in ranked[:top_k]]
    return top_cols, ranked


def make_preprocessor():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def make_stacking():
    base_estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8,
                                   colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]
    stack = StackingClassifier(estimators=base_estimators,
                                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                cv=5, n_jobs=1, passthrough=True)
    return Pipeline([("prep", make_preprocessor()), ("clf", stack)])


def make_individual(name):
    clfs = {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "xgb": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8,
                                  colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "lgbm": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    }
    return Pipeline([("prep", make_preprocessor()), ("clf", clfs[name])])


def prep_window(cutoff_day):
    agg, feature_cols, target_col = build_feature_table(cutoff_day)
    X = agg[feature_cols]
    y = agg[target_col].values
    top_cols, ranked = select_top_k(X, y, TOP_K)
    return X[top_cols], y, top_cols, ranked, len(agg)


def run_fold(cutoff_day, fold_idx, individual_name="lgbm"):
    X_sel, y, top_cols, ranked, n_users = prep_window(cutoff_day)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_sel, y))
    tr_idx, va_idx = splits[fold_idx]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    stack_model = make_stacking()
    stack_model.fit(X_tr, y_tr)
    pred_s = stack_model.predict(X_va)
    proba_s = stack_model.predict_proba(X_va)[:, 1]
    stack_f1 = f1_score(y_va, pred_s, average="macro")
    stack_auc = roc_auc_score(y_va, proba_s)

    ind_model = make_individual(individual_name)
    ind_model.fit(X_tr, y_tr)
    pred_i = ind_model.predict(X_va)
    proba_i = ind_model.predict_proba(X_va)[:, 1]
    ind_f1 = f1_score(y_va, pred_i, average="macro")
    ind_auc = roc_auc_score(y_va, proba_i)

    return {
        "cutoff_day": cutoff_day, "fold": fold_idx, "n_users": n_users,
        "selected_features": top_cols,
        "top_features_mi": [{"feature": c, "mi": float(m)} for c, m in ranked[:TOP_K]],
        "stacking": {"macroF1": float(stack_f1), "auc": float(stack_auc)},
        f"best_individual_{individual_name}": {"macroF1": float(ind_f1), "auc": float(ind_auc)},
    }


if __name__ == "__main__":
    cutoff = int(sys.argv[1])
    fold_idx = int(sys.argv[2])
    res = run_fold(cutoff, fold_idx)
    print(json.dumps(res, indent=2))
    ckpt = f"{OUT_DIR}/mooc_mi_stacking_partial.json"
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            data = json.load(f)
    else:
        data = {"folds": []}
    data["folds"].append(res)
    with open(ckpt, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved checkpoint: cutoff={cutoff} fold={fold_idx}, total saved={len(data['folds'])}")
