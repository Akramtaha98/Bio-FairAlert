import json
import sys
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
from fairlearn.postprocessing import ThresholdOptimizer

from features import build_feature_table
from mi_stacking import select_features_mi, make_preprocessor, CAT_COLS, NUM_COLS, TOP_K
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import OUT_DIR
CUTOFF = 75
CKPT = f"{OUT_DIR}/bootstrap_fairness_partial.json"


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
    return {"max_min_recall_gap": gap, "disparate_impact_ratio": di}


def run_repeat(repeat_idx):
    rs = 100 + repeat_idx  # distinct split seed per repeat, independent of the main-analysis seed (42)
    out, feature_cols, target_col, group_col = build_feature_table(cutoff_day=CUTOFF)
    le = LabelEncoder()
    y_enc = le.fit_transform(out[target_col])
    top_cols, mi_ranked = select_features_mi(out[feature_cols], y_enc, CAT_COLS, NUM_COLS, TOP_K)
    cat_sel = [c for c in top_cols if c in CAT_COLS]
    num_sel = [c for c in top_cols if c in NUM_COLS]
    at_risk = out[target_col].isin(["Fail", "Withdrawn"]).astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    tr_idx, te_idx = next(gss.split(out[top_cols], at_risk, out[group_col]))

    X = out[top_cols]
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = at_risk[tr_idx], at_risk[te_idx]
    meta_tr = out.iloc[tr_idx].reset_index(drop=True)
    meta_te = out.iloc[te_idx].reset_index(drop=True)

    prep = make_preprocessor(cat_sel, num_sel)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, class_weight="balanced",
                              random_state=rs, n_jobs=-1, verbosity=-1)
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_tr, y_tr)

    proba_te = model.predict_proba(X_te)[:, 1]
    pred_raw = (proba_te >= 0.5).astype(int)

    result = {"repeat": repeat_idx, "random_state": rs, "n_test": len(X_te)}
    result["before"] = {
        "overall_acc": float(accuracy_score(y_te, pred_raw)),
        "overall_macrof1": float(f1_score(y_te, pred_raw, average="macro")),
        "disability": fairness_report(pred_raw, y_te, meta_te, "disability"),
        "imd_band": fairness_report(pred_raw, y_te, meta_te, "imd_band"),
        "gender": fairness_report(pred_raw, y_te, meta_te, "gender"),
    }

    # disability-targeted calibration
    sens_tr_dis = meta_tr["disability"].values
    sens_te_dis = meta_te["disability"].values
    to_dis = ThresholdOptimizer(estimator=model, constraints="equalized_odds", predict_method="predict_proba", prefit=True)
    to_dis.fit(X_tr, y_tr, sensitive_features=sens_tr_dis)
    pred_dis = to_dis.predict(X_te, sensitive_features=sens_te_dis, random_state=rs)
    result["after_disability_calib"] = {
        "overall_acc": float(accuracy_score(y_te, pred_dis)),
        "overall_macrof1": float(f1_score(y_te, pred_dis, average="macro")),
        "disability": fairness_report(pred_dis, y_te, meta_te, "disability"),
        "imd_band": fairness_report(pred_dis, y_te, meta_te, "imd_band"),
        "gender": fairness_report(pred_dis, y_te, meta_te, "gender"),
    }

    # imd_band-targeted calibration
    sens_tr_imd = meta_tr["imd_band"].astype(object).where(meta_tr["imd_band"].notna(), "MISSING").values
    sens_te_imd = meta_te["imd_band"].astype(object).where(meta_te["imd_band"].notna(), "MISSING").values
    to_imd = ThresholdOptimizer(estimator=model, constraints="equalized_odds", predict_method="predict_proba", prefit=True)
    to_imd.fit(X_tr, y_tr, sensitive_features=sens_tr_imd)
    pred_imd = to_imd.predict(X_te, sensitive_features=sens_te_imd, random_state=rs)
    result["after_imdband_calib"] = {
        "overall_acc": float(accuracy_score(y_te, pred_imd)),
        "overall_macrof1": float(f1_score(y_te, pred_imd, average="macro")),
        "disability": fairness_report(pred_imd, y_te, meta_te, "disability"),
        "imd_band": fairness_report(pred_imd, y_te, meta_te, "imd_band"),
        "gender": fairness_report(pred_imd, y_te, meta_te, "gender"),
    }
    return result


if __name__ == "__main__":
    repeat_idx = int(sys.argv[1])
    res = run_repeat(repeat_idx)
    print(json.dumps(res, indent=2))
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            data = json.load(f)
    else:
        data = {"repeats": []}
    data["repeats"].append(res)
    with open(CKPT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved checkpoint: repeat={repeat_idx}, total saved={len(data['repeats'])}")
