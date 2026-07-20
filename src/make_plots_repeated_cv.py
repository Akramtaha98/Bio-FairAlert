import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR
from config import MEDIA_DIR as MEDIA

with open(f"{OUT_DIR}/repeated_cv_significance_summary.json") as f:
    res = json.load(f)

# ---- OULAD ----
cutoffs_o = [25, 50, 75, 100]
stack_m = [res["oulad"][str(c)]["stacking_mean"] for c in cutoffs_o]
stack_s = [res["oulad"][str(c)]["stacking_sd"] for c in cutoffs_o]
ind_m = [res["oulad"][str(c)]["individual_mean"] for c in cutoffs_o]
ind_s = [res["oulad"][str(c)]["individual_sd"] for c in cutoffs_o]
p_holm_o = [res["oulad"][str(c)]["p_value_holm"] for c in cutoffs_o]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(cutoffs_o))
ax.errorbar(x, stack_m, yerr=stack_s, marker="o", capsize=4, label="Stacking ensemble", color="#4C72B0")
ax.errorbar(x, ind_m, yerr=ind_s, marker="s", capsize=4, label="Best individual (LightGBM)", color="#DD8452")
for i, p in enumerate(p_holm_o):
    marker = "*" if p < 0.05 else "n.s."
    ax.annotate(marker, (x[i], max(stack_m[i], ind_m[i]) + max(stack_s[i], ind_s[i]) + 0.012),
                ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"day {c}" for c in cutoffs_o])
ax.set_ylabel("Macro-F1 (mean ± SD, 3 repeats × 3 folds)")
ax.set_title("Bio-FairAlert (OULAD): Stacking vs. best individual model\nacross early-warning checkpoints (repeated CV)")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
fig.savefig(f"{MEDIA}/fig_stacking_vs_individual.png", dpi=150)
plt.close(fig)
print("Saved OULAD repeated-CV figure")

# ---- MOOC ----
cutoffs_m = [1, 3, 7, 14, 21]
stack_m2 = [res["mooc"][str(c)]["stacking_mean"] for c in cutoffs_m]
stack_s2 = [res["mooc"][str(c)]["stacking_sd"] for c in cutoffs_m]
ind_m2 = [res["mooc"][str(c)]["individual_mean"] for c in cutoffs_m]
ind_s2 = [res["mooc"][str(c)]["individual_sd"] for c in cutoffs_m]
p_holm_m = [res["mooc"][str(c)]["p_value_holm"] for c in cutoffs_m]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(cutoffs_m))
ax.errorbar(x, stack_m2, yerr=stack_s2, marker="o", capsize=4, label="Stacking ensemble", color="#4C72B0")
ax.errorbar(x, ind_m2, yerr=ind_s2, marker="s", capsize=4, label="Best individual (LightGBM)", color="#DD8452")
for i, p in enumerate(p_holm_m):
    marker = "*" if p < 0.05 else "n.s."
    ax.annotate(marker, (x[i], max(stack_m2[i], ind_m2[i]) + max(stack_s2[i], ind_s2[i]) + 0.012),
                ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"day {c}" for c in cutoffs_m])
ax.set_ylabel("Macro-F1 (mean ± SD, 3 repeats × 3 folds)")
ax.set_title("Bio-FairAlert (MOOC): Stacking vs. best individual model\nacross early-warning checkpoints (repeated CV)")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
fig.savefig(f"{MEDIA}/fig_stacking_vs_individual_mooc.png", dpi=150)
plt.close(fig)
print("Saved MOOC repeated-CV figure")

print("p_holm oulad", dict(zip(cutoffs_o, p_holm_o)))
print("p_holm mooc", dict(zip(cutoffs_m, p_holm_m)))
