import json
import numpy as np
from scipy import stats

from config import OUT_DIR

with open(f"{OUT_DIR}/repeated_cv_significance.json") as f:
    data = json.load(f)


def nadeau_bengio_test(diffs, n_test, n_train):
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003) for repeated k-fold CV,
    which corrects the naive paired t-test's underestimated variance caused by overlapping
    train/test sets across folds/repeats (the same violation Dietterich, 1998 identifies)."""
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    mean_d = diffs.mean()
    var_d = diffs.var(ddof=1)
    correction = (1.0 / k) + (n_test / n_train)
    se_corrected = np.sqrt(correction * var_d) if var_d > 0 else 1e-12
    t_stat = mean_d / se_corrected
    df = k - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    # 95% CI
    tcrit = stats.t.ppf(0.975, df)
    ci_lo = mean_d - tcrit * se_corrected
    ci_hi = mean_d + tcrit * se_corrected
    cohens_dz = mean_d / np.sqrt(var_d) if var_d > 0 else float("nan")
    return {
        "k": k, "mean_diff": float(mean_d), "sd_diff": float(np.sqrt(var_d)),
        "t_stat": float(t_stat), "df": df, "p_value": float(p_val),
        "ci95": [float(ci_lo), float(ci_hi)], "cohens_dz": float(cohens_dz),
    }


def holm_bonferroni(pvals_dict):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted[k] = min(running_max, 1.0)
    return adjusted


results = {"oulad": {}, "mooc": {}}
raw_pvals = {}

# OULAD: 3-fold CV, test fraction = 1/3, train fraction = 2/3
n_test_ratio_oulad = (1 / 3) / (2 / 3)
for cutoff in [25, 50, 75, 100]:
    recs = [r for r in data["oulad"] if r["cutoff_day"] == cutoff]
    diffs = [r["stacking_macroF1"] - r["individual_macroF1"] for r in recs]
    res = nadeau_bengio_test(diffs, n_test=1, n_train=2)
    res["n_folds_used"] = len(recs)
    res["stacking_mean"] = float(np.mean([r["stacking_macroF1"] for r in recs]))
    res["stacking_sd"] = float(np.std([r["stacking_macroF1"] for r in recs], ddof=1))
    res["individual_mean"] = float(np.mean([r["individual_macroF1"] for r in recs]))
    res["individual_sd"] = float(np.std([r["individual_macroF1"] for r in recs], ddof=1))
    # repeat-level Wilcoxon (n=3 pairs, supplementary/low-power check)
    by_repeat = {}
    for r in recs:
        by_repeat.setdefault(r["repeat_seed"], []).append(r)
    repeat_means_stack = [np.mean([x["stacking_macroF1"] for x in v]) for v in by_repeat.values()]
    repeat_means_ind = [np.mean([x["individual_macroF1"] for x in v]) for v in by_repeat.values()]
    try:
        wstat, wp = stats.wilcoxon(repeat_means_stack, repeat_means_ind)
    except Exception:
        wstat, wp = float("nan"), float("nan")
    res["wilcoxon_repeat_level_n3"] = {"statistic": float(wstat) if wstat == wstat else None,
                                        "p_value": float(wp) if wp == wp else None}
    results["oulad"][cutoff] = res
    raw_pvals[f"oulad_{cutoff}"] = res["p_value"]

# MOOC: 3-fold CV, same ratio
for cutoff in [1, 3, 7, 14, 21]:
    recs = [r for r in data["mooc"] if r["cutoff_day"] == cutoff]
    diffs = [r["stacking_macroF1"] - r["individual_macroF1"] for r in recs]
    res = nadeau_bengio_test(diffs, n_test=1, n_train=2)
    res["n_folds_used"] = len(recs)
    res["stacking_mean"] = float(np.mean([r["stacking_macroF1"] for r in recs]))
    res["stacking_sd"] = float(np.std([r["stacking_macroF1"] for r in recs], ddof=1))
    res["individual_mean"] = float(np.mean([r["individual_macroF1"] for r in recs]))
    res["individual_sd"] = float(np.std([r["individual_macroF1"] for r in recs], ddof=1))
    by_repeat = {}
    for r in recs:
        by_repeat.setdefault(r["repeat_seed"], []).append(r)
    repeat_means_stack = [np.mean([x["stacking_macroF1"] for x in v]) for v in by_repeat.values()]
    repeat_means_ind = [np.mean([x["individual_macroF1"] for x in v]) for v in by_repeat.values()]
    try:
        wstat, wp = stats.wilcoxon(repeat_means_stack, repeat_means_ind)
    except Exception:
        wstat, wp = float("nan"), float("nan")
    res["wilcoxon_repeat_level_n3"] = {"statistic": float(wstat) if wstat == wstat else None,
                                        "p_value": float(wp) if wp == wp else None}
    results["mooc"][cutoff] = res
    raw_pvals[f"mooc_{cutoff}"] = res["p_value"]

adjusted = holm_bonferroni(raw_pvals)
for key, adj_p in adjusted.items():
    ds, cutoff = key.split("_")
    results[ds][int(cutoff)]["p_value_holm"] = adj_p

with open(f"{OUT_DIR}/repeated_cv_significance_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== OULAD ===")
for cutoff, r in results["oulad"].items():
    print(f"day {cutoff}: stack={r['stacking_mean']:.3f}+/-{r['stacking_sd']:.3f} "
          f"lgbm={r['individual_mean']:.3f}+/-{r['individual_sd']:.3f} "
          f"diff={r['mean_diff']:.4f} t={r['t_stat']:.2f} p={r['p_value']:.4f} "
          f"p_holm={r['p_value_holm']:.4f} dz={r['cohens_dz']:.2f} CI={r['ci95']}")
print("=== MOOC ===")
for cutoff, r in results["mooc"].items():
    print(f"day {cutoff}: stack={r['stacking_mean']:.3f}+/-{r['stacking_sd']:.3f} "
          f"lgbm={r['individual_mean']:.3f}+/-{r['individual_sd']:.3f} "
          f"diff={r['mean_diff']:.4f} t={r['t_stat']:.2f} p={r['p_value']:.4f} "
          f"p_holm={r['p_value_holm']:.4f} dz={r['cohens_dz']:.2f} CI={r['ci95']}")
