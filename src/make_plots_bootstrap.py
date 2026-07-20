import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR

with open(f"{OUT_DIR}/bootstrap_fairness_summary.json") as f:
    bs = json.load(f)

attrs = ["disability", "imd_band", "gender"]
labels = ["Disability", "IMD band", "Gender"]

before_di = [bs["before"][a]["DI_mean"] for a in attrs]
before_sd = [bs["before"][a]["DI_std"] for a in attrs]
after_dis_di = [bs["after_disability_calib"][a]["DI_mean"] for a in attrs]
after_dis_sd = [bs["after_disability_calib"][a]["DI_std"] for a in attrs]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(attrs))
w = 0.35
ax.bar(x - w/2, before_di, w, yerr=before_sd, capsize=4, label="Before calibration", color="#C44E52")
ax.bar(x + w/2, after_dis_di, w, yerr=after_dis_sd, capsize=4, label="After equalized-odds calibration\n(fit on disability, 10 repeated splits)", color="#55A868")
ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="Four-fifths threshold (0.80)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Disparate-impact ratio (at-risk flagging)")
ax.set_ylim(0, 1.1)
ax.set_title("Bio-FairAlert (OULAD): Disparate-impact ratio before/after\nequalized-odds calibration on disability status\n(mean ± SD across 10 repeated 80/20 splits)")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_fairness_before_after.png", dpi=150)
plt.close(fig)

print("Saved fig_fairness_before_after.png (bootstrap version, disability-targeted)")
print(json.dumps({"before_di": before_di, "before_sd": before_sd,
                   "after_dis_di": after_dis_di, "after_dis_sd": after_dis_sd}, indent=2))
