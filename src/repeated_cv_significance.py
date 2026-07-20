"""
Repeated cross-validation for the stacking-vs-best-individual-model comparison,
extending the original 3-fold (single random_state=42) design in mi_stacking.py /
mooc_mi_stacking.py to 3 independent repeats (random_state = 42, 142, 242), so that
the reported gaps (Table 1 / Table 3) can be tested for statistical significance
rather than reported only as a 3-fold mean +/- SD from a single split assignment.

repeat_seed=42 reuses the ALREADY-EXECUTED, already-reported fold results from
mi_stacking_partial.json / mooc_mi_stacking_partial.json (loaded directly, not
re-run) since re-running them would waste compute reproducing numbers already in
the manuscript. repeat_seed=142 and 242 are NEW executions.

Checkpoints to a single shared JSON file, one (dataset, cutoff, repeat_seed, fold)
result per call, so this can be run across many bounded bash calls.
"""
import json
import os
import sys
import time

from config import OUT_DIR
CKPT = f"{OUT_DIR}/repeated_cv_significance.json"

OULAD_CUTOFFS = [25, 50, 75, 100]
MOOC_CUTOFFS = [1, 3, 7, 14, 21]
REPEAT_SEEDS = [42, 142, 242]
N_SPLITS = 3


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {"oulad": [], "mooc": []}


def save_ckpt(data):
    with open(CKPT, "w") as f:
        json.dump(data, f, indent=2)


def seed_existing_from_originals():
    """One-time import of the repeat_seed=42 results already computed in the
    original (non-repeated) mi_stacking / mooc_mi_stacking checkpoint files."""
    data = load_ckpt()
    have_oulad = {(r["cutoff_day"], r["repeat_seed"], r["fold"]) for r in data["oulad"]}
    have_mooc = {(r["cutoff_day"], r["repeat_seed"], r["fold"]) for r in data["mooc"]}

    with open(f"{OUT_DIR}/mi_stacking_partial.json") as f:
        orig_oulad = json.load(f)["folds"]
    for f_ in orig_oulad:
        key = (f_["cutoff_day"], 42, f_["fold"])
        if key not in have_oulad:
            data["oulad"].append({
                "cutoff_day": f_["cutoff_day"], "repeat_seed": 42, "fold": f_["fold"],
                "stacking_macroF1": f_["stacking"]["macroF1"],
                "individual_macroF1": f_["best_individual_lightgbm"]["macroF1"],
                "source": "original_mi_stacking_partial.json",
            })

    with open(f"{OUT_DIR}/mooc_mi_stacking_partial.json") as f:
        orig_mooc = json.load(f)["folds"]
    for f_ in orig_mooc:
        key = (f_["cutoff_day"], 42, f_["fold"])
        if key not in have_mooc:
            data["mooc"].append({
                "cutoff_day": f_["cutoff_day"], "repeat_seed": 42, "fold": f_["fold"],
                "stacking_macroF1": f_["stacking"]["macroF1"],
                "individual_macroF1": f_["best_individual_lgbm"]["macroF1"],
                "source": "original_mooc_mi_stacking_partial.json",
            })
    save_ckpt(data)
    print(f"Seeded: oulad={len(data['oulad'])} mooc={len(data['mooc'])}")


def run_oulad_fold(cutoff, repeat_seed, fold_idx):
    import mi_stacking as ms
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score

    X_sel, y_enc, groups, cat_sel, num_sel, top_cols, mi_ranked, n = ms.prep_window(cutoff)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat_seed)
    splits = list(sgkf.split(X_sel, y_enc, groups))
    tr_idx, va_idx = splits[fold_idx]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    stack_model = ms.make_stacking(cat_sel, num_sel)
    stack_model.fit(X_tr, y_tr)
    stack_f1 = f1_score(y_va, stack_model.predict(X_va), average="macro")

    best_model = ms.make_best_individual(cat_sel, num_sel)
    best_model.fit(X_tr, y_tr)
    best_f1 = f1_score(y_va, best_model.predict(X_va), average="macro")

    return {"cutoff_day": cutoff, "repeat_seed": repeat_seed, "fold": fold_idx,
            "stacking_macroF1": float(stack_f1), "individual_macroF1": float(best_f1),
            "source": "repeated_cv_significance.py"}


def run_mooc_fold(cutoff, repeat_seed, fold_idx):
    import mooc_mi_stacking as ms
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    X_sel, y, top_cols, ranked, n = ms.prep_window(cutoff)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat_seed)
    splits = list(skf.split(X_sel, y))
    tr_idx, va_idx = splits[fold_idx]
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    stack_model = ms.make_stacking()
    stack_model.fit(X_tr, y_tr)
    stack_f1 = f1_score(y_va, stack_model.predict(X_va), average="macro")

    ind_model = ms.make_individual("lgbm")
    ind_model.fit(X_tr, y_tr)
    ind_f1 = f1_score(y_va, ind_model.predict(X_va), average="macro")

    return {"cutoff_day": cutoff, "repeat_seed": repeat_seed, "fold": fold_idx,
            "stacking_macroF1": float(stack_f1), "individual_macroF1": float(ind_f1),
            "source": "repeated_cv_significance.py"}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "seed":
        seed_existing_from_originals()
    elif cmd == "oulad":
        cutoff, repeat_seed, fold_idx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        t0 = time.time()
        res = run_oulad_fold(cutoff, repeat_seed, fold_idx)
        data = load_ckpt()
        data["oulad"].append(res)
        save_ckpt(data)
        print(res, "elapsed", time.time() - t0)
    elif cmd == "mooc":
        cutoff, repeat_seed, fold_idx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        t0 = time.time()
        res = run_mooc_fold(cutoff, repeat_seed, fold_idx)
        data = load_ckpt()
        data["mooc"].append(res)
        save_ckpt(data)
        print(res, "elapsed", time.time() - t0)
