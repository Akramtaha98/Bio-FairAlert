import json
import time
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier

from config import OUT_DIR


def run_oulad(cutoff):
    import mi_stacking as ms
    X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, n = ms.prep_window(cutoff)
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, va_idx = list(sgkf.split(X_sel, y_enc, groups))[0]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    dummy_f1 = f1_score(y_va, dummy.predict(X_va), average="macro")

    prep = ms.make_preprocessor(cat_sel, num_sel)
    ebm_pipe = Pipeline([("prep", prep),
                          ("clf", ExplainableBoostingClassifier(random_state=42))])
    t0 = time.time()
    ebm_pipe.fit(X_tr, y_tr)
    ebm_f1 = f1_score(y_va, ebm_pipe.predict(X_va), average="macro")
    print("ebm fit+predict", time.time() - t0)

    return {"dataset": "oulad", "cutoff": cutoff,
            "majority_class_macroF1": float(dummy_f1), "ebm_macroF1": float(ebm_f1)}


def run_mooc(cutoff):
    import mooc_mi_stacking as ms
    X_sel, y, top_cols, ranked, n = ms.prep_window(cutoff)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, va_idx = list(skf.split(X_sel, y))[0]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    dummy_f1 = f1_score(y_va, dummy.predict(X_va), average="macro")

    prep = ms.make_preprocessor()
    ebm_pipe = Pipeline([("prep", prep),
                          ("clf", ExplainableBoostingClassifier(random_state=42))])
    t0 = time.time()
    ebm_pipe.fit(X_tr, y_tr)
    ebm_f1 = f1_score(y_va, ebm_pipe.predict(X_va), average="macro")
    print("ebm fit+predict", time.time() - t0)

    return {"dataset": "mooc", "cutoff": cutoff,
            "majority_class_macroF1": float(dummy_f1), "ebm_macroF1": float(ebm_f1)}


if __name__ == "__main__":
    import sys
    which, cutoff = sys.argv[1], int(sys.argv[2])
    res = run_oulad(cutoff) if which == "oulad" else run_mooc(cutoff)
    print(json.dumps(res))
    import os
    ck = f"{OUT_DIR}/baseline_experiment_results.json"
    all_res = json.load(open(ck)) if os.path.exists(ck) else []
    all_res.append(res)
    json.dump(all_res, open(ck, "w"), indent=2)
