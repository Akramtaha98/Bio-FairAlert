"""
Professional redesign of the Bio-FairAlert architecture and intervention-
workflow figures: phase-grouped cards, subtle drop shadows, a restrained
academic (Elsevier-style) navy/slate palette, bold title + gray subtitle text
inside each card, and a colored phase-group bar on the left margin instead of
plain uniform boxes. Replaces the earlier flat-box version.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

from config import MEDIA_DIR as OUT_DIR

NAVY = "#1F3864"
SLATE = "#44546A"
LIGHT_BG = "#F4F6FA"
ACCENT = "#2E74B5"
SHADOW = "#B7C3D6"

PHASE_PALETTE = [
    ("#D9E3F2", "#1F3864"),  # data/feature
    ("#FCEBD2", "#8A5A00"),  # prediction
    ("#F3D6D6", "#7A1F1F"),  # fairness/explain
    ("#D7ECD9", "#1E5C2C"),  # deployment/outcome
]


def rounded_card(ax, x, y, w, h, title, subtitle, face, edge_text, shadow=True):
    if shadow:
        shadow_box = FancyBboxPatch((x + 0.045, y - 0.045), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.10",
                                     linewidth=0, facecolor=SHADOW, alpha=0.55, zorder=1)
        ax.add_patch(shadow_box)
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.10",
                          linewidth=1.3, edgecolor=edge_text, facecolor=face, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=11.3, fontweight="bold", color="#1B1B1B", zorder=3)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
            fontsize=8.6, fontweight="normal", style="italic", color=SLATE, zorder=3,
            wrap=True)


def phase_bar(ax, x, y0, y1, color, label):
    ax.add_patch(plt.Rectangle((x, y0), 0.09, y1 - y0, facecolor=color, edgecolor="none", zorder=2))
    ax.text(x - 0.14, (y0 + y1) / 2, label, ha="center", va="center", rotation=90,
            fontsize=9.5, fontweight="bold", color=color, zorder=3)


def flow_diagram(steps, phase_of_step, title, out_path, figsize, phase_labels, box_w=5.2, box_h=0.95, gap=0.5):
    n = len(steps)
    total_h = n * box_h + (n - 1) * gap
    fig_h = total_h + 1.6
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor("white")
    left_margin = 0.9
    ax.set_xlim(0, left_margin + box_w + 0.6)
    ax.set_ylim(0, total_h + 1.3)
    ax.axis("off")

    y = total_h + 0.15
    centers = []
    phase_spans = {}
    for i, (title_txt, sub_txt) in enumerate(steps):
        y_box = y - box_h
        x = left_margin
        phase_idx = phase_of_step[i]
        face, edge = PHASE_PALETTE[phase_idx]
        rounded_card(ax, x, y_box, box_w, box_h, title_txt, sub_txt, face, edge)
        centers.append((x + box_w / 2, y_box))
        phase_spans.setdefault(phase_idx, [y_box + box_h, y_box])
        phase_spans[phase_idx][1] = y_box
        y = y_box - gap

    for i in range(n - 1):
        x_c, y_bottom_of_box_i = centers[i]
        y_top_of_next = centers[i + 1][1] + box_h
        arrow = FancyArrowPatch((x_c, y_bottom_of_box_i), (x_c, y_top_of_next),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.5, color=SLATE, zorder=2)
        ax.add_patch(arrow)

    for idx, (top, bottom) in phase_spans.items():
        phase_bar(ax, left_margin - 0.28, bottom, top, PHASE_PALETTE[idx][1], phase_labels[idx])

    ax.text(left_margin + box_w / 2, total_h + 0.85, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY)
    ax.add_patch(plt.Rectangle((left_margin, total_h + 0.55), box_w, 0.03, facecolor=ACCENT, zorder=3))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---- Figure 1: system architecture ----
arch_steps = [
    ("Student Data", "Demographics, VLE logs, assessment / interaction traces"),
    ("Feature Engineering", "Checkpointed aggregation + mutual-information selection"),
    ("Bio-FairAlert Model", "Stacking ensemble or best individual base learner"),
    ("Prediction", "At-risk probability score at each checkpoint"),
    ("Fairness Module", "Equalized-odds calibration on one selected attribute;\nre-audit of all protected attributes"),
    ("Explainability", "SHAP + LIME + rule-based reports"),
    ("Instructor / Advisor Dashboard", "Risk flag + plain-language explanation"),
    ("Educational Intervention", "Outreach, advising, support referral"),
]
arch_phase = [0, 0, 1, 1, 2, 2, 3, 3]
arch_phase_labels = ["DATA & FEATURES", "PREDICTION", "FAIRNESS & EXPLAINABILITY", "DEPLOYMENT"]
flow_diagram(arch_steps, arch_phase, "Bio-FairAlert System Architecture",
             f"{OUT_DIR}/fig_system_architecture.png", figsize=(6.6, 11.2), phase_labels=arch_phase_labels,
             box_h=1.02, gap=0.42)

# ---- Figure 2: intervention workflow ----
workflow_steps = [
    ("Student Activity", "Ongoing coursework, engagement, submissions"),
    ("Risk Prediction", "Bio-FairAlert flags at-risk status at a checkpoint"),
    ("Advisor Notification", "Risk flag routed to instructor/advisor dashboard"),
    ("Explanation", "SHAP / LIME / rule-based reason for the flag"),
    ("Instructor Action", "Targeted outreach, resource referral, check-in"),
    ("Student Support", "Advising, tutoring, accommodation as needed"),
    ("Intended Outcome: Re-engagement / Retention", "Potential outcome, not measured here;\nrequires prospective evaluation"),
]
workflow_phase = [0, 0, 1, 1, 2, 2, 3]
workflow_phase_labels = ["MONITORING", "NOTIFICATION", "ACTION & SUPPORT", "OUTCOME"]
flow_diagram(workflow_steps, workflow_phase, "Bio-FairAlert Intervention Workflow in Practice",
             f"{OUT_DIR}/fig_intervention_workflow.png", figsize=(6.6, 10.0), phase_labels=workflow_phase_labels,
             box_h=1.05, gap=0.42)
