import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import MEDIA_DIR as OUT_DIR

def flow_diagram(steps, colors, title, out_path, figsize=(6.2, 10.5), box_w=4.6, box_h=0.72, gap=0.42):
    n = len(steps)
    total_h = n * box_h + (n - 1) * gap
    fig_h = total_h + 1.4
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.set_xlim(0, box_w + 1.0)
    ax.set_ylim(0, total_h + 1.2)
    ax.axis("off")

    y = total_h + 0.2
    centers = []
    for i, (label) in enumerate(steps):
        y_box = y - box_h
        x = 0.5
        color = colors[i]
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                              boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.4, edgecolor="#2E3440", facecolor=color)
        ax.add_patch(box)
        ax.text(x + box_w / 2, y_box + box_h / 2, label, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#1B1B1B", wrap=True)
        centers.append((x + box_w / 2, y_box))
        y = y_box - gap

    # arrows between consecutive boxes
    for i in range(n - 1):
        x_c, y_top_of_next_box = centers[i]
        y_start = y_top_of_next_box  # bottom of box i
        y_end = centers[i + 1][1] + box_h  # top of box i+1
        arrow = FancyArrowPatch((x_c, y_start), (x_c, y_end),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.6, color="#2E3440")
        ax.add_patch(arrow)

    ax.text((box_w + 1.0) / 2, total_h + 0.9, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1B1B1B")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---- Figure: system architecture ----
arch_steps = [
    "Student Data\n(demographics, VLE logs, assessments / interaction traces)",
    "Feature Engineering\n(checkpointed aggregation + mutual-information selection)",
    "Bio-FairAlert Model\n(stacking ensemble / best base learner)",
    "Prediction\n(at-risk probability score)",
    "Fairness Module\n(equalized-odds calibration across protected attributes)",
    "Explainability\n(SHAP + LIME + rule-based reports)",
    "Instructor / Advisor Dashboard\n(risk flag + plain-language explanation)",
    "Educational Intervention\n(outreach, advising, support referral)",
]
arch_colors = [
    "#D6E4F0", "#D6E4F0", "#FCE8B4", "#FCE8B4",
    "#F6C6C6", "#C9E4CA", "#B7D8E8", "#CDEAC0",
]
flow_diagram(arch_steps, arch_colors,
             "Bio-FairAlert System Architecture",
             f"{OUT_DIR}/fig_system_architecture.png",
             figsize=(6.4, 11.0), box_h=0.95, gap=0.38)

# ---- Figure: intervention workflow ----
workflow_steps = [
    "Student Activity\n(ongoing coursework, engagement, submissions)",
    "Risk Prediction\n(Bio-FairAlert flags at-risk status at a checkpoint)",
    "Advisor Notification\n(risk flag routed to instructor/advisor dashboard)",
    "Explanation\n(SHAP / LIME / rule-based reason for the flag)",
    "Instructor Action\n(targeted outreach, resource referral, check-in)",
    "Student Support\n(advising, tutoring, accommodation as needed)",
    "Intended Outcome: Re-engagement / Retention\n(potential outcome, not measured here; requires prospective evaluation)",
]
workflow_colors = [
    "#D6E4F0", "#FCE8B4", "#B7D8E8", "#C9E4CA",
    "#F6C6C6", "#CDEAC0", "#D9C6EC",
]
flow_diagram(workflow_steps, workflow_colors,
             "Bio-FairAlert Intervention Workflow in Practice",
             f"{OUT_DIR}/fig_intervention_workflow.png",
             figsize=(6.4, 9.8), box_h=0.95, gap=0.38)
