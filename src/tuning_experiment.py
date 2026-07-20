"""
Fair (equal-budget) hyperparameter tuning at representative checkpoints, addressing
the reviewer concern that untuned default configurations may drive the reported
stacking-vs-individual gaps. Scope, disclosed explicitly in the manuscript: rather
than tuning at all 9 checkpoints (compute-infeasible in this study's budget), tuning
is done at three representative checkpoints -- OULAD day 100 (the widest untuned
gap) and MOOC day 7 (the untuned "crossover" point) and day 21 (the endpoint) --
using RandomizedSearchCV (n_iter=6, cv=3, training-fold-only, no test leakage) with
an equal search budget applied to all four base learners. Tuning is performed on the
SAME train/test split already used for the main-analysis result (repeat_seed=42,
fold=0), so the tuned-vs-untuned comparison is apples-to-apples on identical data.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from config import OUT_DIR
CKPT = f"{OUT_DIR}/tuning_experiment_results.json"
RANDOM_STATE = 42
N_ITER = 6
CV = 3

PARAM_DISTS = {
    "lr": {"clf__C": uniform(0.01, 10)},
    "rf": {"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(3, 20),
           "clf__min_samples_leaf": randint(1, 10)},
    "xgb": {"clf__n_estimators": randint(80, 400), "clf__max_depth": randint(3, 10),
            "clf__learning_rate": uniform(0.02, 0.25), "clf__subsample": uniform(0.6, 0.4)},
    "lgbm": {"clf__n_estimators": randint(80, 400), "clf__num_leaves": randint(15, 80),
             "clf__learning_rate": uniform(0.02, 0.25)},
}


def build_estimator(name):
    # n_jobs=1 on the estimator itself (rather than -1) to avoid nested-parallelism
    # deadlocks/hangs with RandomizedSearchCV's own n_jobs=-1 across candidates.
    if name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)
    if name == "xgb":
        return xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=1)
    if name == "lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)


def tune_one(name, prep, X_tr, y_tr, cv_splitter):
    pipe = Pipeline([("prep", prep), ("clf", build_estimator(name))])
    search = RandomizedSearchCV(pipe, PARAM_DISTS[name], n_iter=N_ITER, cv=cv_splitter,
                                 scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_, search.best_score_


def run_oulad(cutoff):
    import mi_stacking as ms
    X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, n = ms.prep_window(cutoff)
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, va_idx = list(sgkf.split(X_sel, y_enc, groups))[0]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]
    groups_tr = groups.iloc[tr_idx]
    prep = ms.make_preprocessor(cat_sel, num_sel)
    inner_cv = list(StratifiedGroupKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE).split(X_tr, y_tr, groups_tr))

    tuned = {}
    t0 = time.time()
    for name in ["lr", "rf", "xgb", "lgbm"]:
        best_est, best_params, best_cv_score = tune_one(name, prep, X_tr, y_tr, inner_cv)
        pred = best_est.predict(X_va)
        f1 = f1_score(y_va, pred, average="macro")
        tuned[name] = {"best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                                        for k, v in best_params.items()},
                       "inner_cv_f1": float(best_cv_score), "test_macroF1": float(f1)}
        print(f"  {name} tuned test macroF1={f1:.4f} (elapsed {time.time()-t0:.0f}s)")

    # tuned stacking: rebuild base estimators with tuned hyperparams, tuned LR meta (kept simple, untuned meta per Sec 3.6 rationale)
    def apply_params(est, params):
        clean = {k.replace("clf__", ""): v for k, v in params.items()}
        return est.set_params(**clean)

    base_estimators = [
        ("lr", apply_params(build_estimator("lr"), tuned["lr"]["best_params"])),
        ("rf", apply_params(build_estimator("rf"), tuned["rf"]["best_params"])),
        ("xgb", apply_params(build_estimator("xgb"), tuned["xgb"]["best_params"])),
        ("lgbm", apply_params(build_estimator("lgbm"), tuned["lgbm"]["best_params"])),
    ]
    stack = StackingClassifier(estimators=base_estimators,
                                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                cv=5, n_jobs=1, passthrough=True)
    stack_pipe = Pipeline([("prep", ms.make_preprocessor(cat_sel, num_sel)), ("clf", stack)])
    stack_pipe.fit(X_tr, y_tr)
    stack_pred = stack_pipe.predict(X_va)
    stack_f1_tuned = f1_score(y_va, stack_pred, average="macro")
    tuned["stacking_tuned_bases"] = {"test_macroF1": float(stack_f1_tuned)}
    print(f"  stacking (tuned bases) test macroF1={stack_f1_tuned:.4f} (elapsed {time.time()-t0:.0f}s)")

    return {"dataset": "oulad", "cutoff": cutoff, "tuned": tuned,
            "untuned_lightgbm_macroF1": None, "untuned_stacking_macroF1": None}


def run_mooc(cutoff):
    import mooc_mi_stacking as ms
    X_sel, y, top_cols, ranked, n = ms.prep_window(cutoff)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, va_idx = list(skf.split(X_sel, y))[0]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    prep = ms.make_preprocessor()
    inner_cv = CV  # plain int fine, no groups needed for MOOC

    tuned = {}
    t0 = time.time()
    for name in ["lr", "rf", "xgb", "lgbm"]:
        best_est, best_params, best_cv_score = tune_one(name, prep, X_tr, y_tr, inner_cv)
        pred = best_est.predict(X_va)
        f1 = f1_score(y_va, pred, average="macro")
        tuned[name] = {"best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                                        for k, v in best_params.items()},
                       "inner_cv_f1": float(best_cv_score), "test_macroF1": float(f1)}
        print(f"  {name} tuned test macroF1={f1:.4f} (elapsed {time.time()-t0:.0f}s)")

    def apply_params(est, params):
        clean = {k.replace("clf__", ""): v for k, v in params.items()}
        return est.set_params(**clean)

    base_estimators = [
        ("lr", apply_params(build_estimator("lr"), tuned["lr"]["best_params"])),
        ("rf", apply_params(build_estimator("rf"), tuned["rf"]["best_params"])),
        ("xgb", apply_params(build_estimator("xgb"), tuned["xgb"]["best_params"])),
        ("lgbm", apply_params(build_estimator("lgbm"), tuned["lgbm"]["best_params"])),
    ]
    stack = StackingClassifier(estimators=base_estimators,
                                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                cv=5, n_jobs=1, passthrough=True)
    stack_pipe = Pipeline([("prep", ms.make_preprocessor()), ("clf", stack)])
    stack_pipe.fit(X_tr, y_tr)
    stack_pred = stack_pipe.predict(X_va)
    stack_f1_tuned = f1_score(y_va, stack_pred, average="macro")
    tuned["stacking_tuned_bases"] = {"test_macroF1": float(stack_f1_tuned)}
    print(f"  stacking (tuned bases) test macroF1={stack_f1_tuned:.4f} (elapsed {time.time()-t0:.0f}s)")

    return {"dataset": "mooc", "cutoff": cutoff, "tuned": tuned,
            "untuned_lightgbm_macroF1": None, "untuned_stacking_macroF1": None}


if __name__ == "__main__":
    which = sys.argv[1]
    cutoff = int(sys.argv[2])
    if which == "oulad":
        res = run_oulad(cutoff)
    else:
        res = run_mooc(cutoff)

    if os.path.exists(CKPT):
        with open(CKPT) as f:
            all_res = json.load(f)
    else:
        all_res = []
    all_res.append(res)
    with open(CKPT, "w") as f:
        json.dump(all_res, f, indent=2)
    print("saved", which, cutoff)
