import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR

with open(f"{OUT_DIR}/mi_stacking_summary.json") as f:
    summary = json.load(f)
with open(f"{OUT_DIR}/fairness_calibration_results.json") as f:
    fc = json.load(f)

# --- Fig A: Stacking vs LightGBM across windows ---
cutoffs = sorted(int(c) for c in summary.keys())
stack_means = [summary[str(c)]["stacking_macroF1_mean"] for c in cutoffs]
stack_stds = [summary[str(c)]["stacking_macroF1_std"] for c in cutoffs]
lgbm_means = [summary[str(c)]["lightgbm_macroF1_mean"] for c in cutoffs]
lgbm_stds = [summary[str(c)]["lightgbm_macroF1_std"] for c in cutoffs]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(cutoffs))
w = 0.35
ax.bar(x - w/2, stack_means, w, yerr=stack_stds, capsize=4, label="Stacking (MI-selected, passthrough)", color="#DD8452")
ax.bar(x + w/2, lgbm_means, w, yerr=lgbm_stds, capsize=4, label="Best individual (LightGBM, MI-selected)", color="#4C72B0")
ax.set_xticks(x); ax.set_xticklabels([f"Day {c}" for c in cutoffs])
ax.set_ylabel("Macro-F1 (3-fold, student-grouped CV)")
ax.set_title("Bio-FairAlert (OULAD): Stacking vs. best individual model\nacross early-warning checkpoints")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_stacking_vs_individual.png", dpi=150)
plt.close(fig)

# --- Fig B: Fairness before/after calibration ---
attrs = ["disability", "imd_band", "gender"]
before_di = [fc["fairness_before"][a]["disparate_impact_ratio"] for a in attrs]
after_di = [fc["fairness_after_disability_calibration"][a]["disparate_impact_ratio"] for a in attrs]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(attrs))
ax.bar(x - w/2, before_di, w, label="Before calibration", color="#C44E52")
ax.bar(x + w/2, after_di, w, label="After equalized-odds calibration\n(fit on disability)", color="#55A868")
ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="Four-fifths threshold (0.80)")
ax.set_xticks(x); ax.set_xticklabels(["Disability", "IMD band", "Gender"])
ax.set_ylabel("Disparate-impact ratio (at-risk flagging)")
ax.set_ylim(0, 1.05)
ax.set_title("Bio-FairAlert (OULAD): Disparate-impact ratio before/after\nequalized-odds calibration on disability status")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_fairness_before_after.png", dpi=150)
plt.close(fig)

print("Saved fig_stacking_vs_individual.png and fig_fairness_before_after.png")
