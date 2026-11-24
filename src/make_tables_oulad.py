#!/usr/bin/env python3
"""LaTeX tables for the OULAD held-out and common gender-targeted analyses (reads results/oulad_* and results/uci697)."""
import argparse, os
import numpy as np, pandas as pd

LAB = {"gender": "Gender", "disability": "Disability", "imd_band": "IMD band", "age_band": "Age band",
       "gd": "Gender$\\times$disab.", "gi": "Gender$\\times$IMD$<$30\\%"}
MLAB = {"reweigh": "Reweighing", "expgrad_eo": "Exp.\\ gradient", "threshold_eo": "Threshold opt."}


def f(x, d=3, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(v, lo, hi, d=3, sign=True, bold=False):
    s = f"{f(v, d, sign)} [{f(lo, d, sign)}, {f(hi, d, sign)}]"
    return "\\textbf{" + s + "}" if bold else s


def tab_heldout_base(res, out):
    rows = []
    for ck, nm in [("C25", "Day 25"), ("C50", "Day 50")]:
        d = pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_fairness.csv")
        sp = pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_seed_performance.csv")
        b = sp[sp.method == "baseline"]
        cfg = pd.read_json(f"{res}/oulad_heldout/heldout_{ck}_config.json", typ="series")
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{nm}: train {cfg.n_train:,}, val. {cfg.n_val:,}, test {cfg.n_test:,}}}}} \\\\")
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\quad macro-F1 {b.macro_f1.mean():.3f}$\\pm${b.macro_f1.std():.3f}, AUROC {b.auroc.mean():.3f}$\\pm${b.auroc.std():.3f} (5 seeds)}} \\\\")
        for g in ["gender", "disability", "imd_band", "age_band", "gd", "gi"]:
            x = d[(d.method == "baseline") & (d.grouping == g)].set_index("metric")
            r = [LAB[g]] + [f"{x.loc[m, 'value']:.3f}" for m in ["dir", "tpr_gap", "fpr_gap", "eo"]]
            rows.append(" & ".join(r) + " \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{OULAD held-out presentation (train 2013B+2013J; validation 2014B; test 2014J; frozen protocol, "
           "single run). Baseline fairness of the validation-tuned LightGBM on the primary seed. DIR = disparate-impact ratio; gaps are max minus min across groups.}\n"
           "\\label{tab:oulad_heldout}\n\\begin{tabular}{lcccc}\n\\toprule\nGrouping & DIR & TPR gap & FPR gap & EO diff. \\\\\n\\midrule\n"
           + "\n".join(rows).replace("\\multicolumn{6}{l}", "\\multicolumn{5}{l}") + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{out}/tab_oulad_heldout.tex", "w").write(tex)


def tab_heldout_crit(res, out):
    rows = []
    for ck, nm in [("C25", "Day 25"), ("C50", "Day 50")]:
        d = pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_fairness.csv")
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{nm}}}}} \\\\")
        for t in ["gender", "disability", "imd_band", "gd"]:
            for fam in ["reweigh", "expgrad_eo", "threshold_eo"]:
                m = f"{fam}[{t}]"
                x = d[(d.method == m) & (d.grouping == t)].set_index("metric")
                a = d[(d.method == m) & (d.grouping == "ALL")].iloc[0]
                rows.append(" & ".join([f"{MLAB[fam]}, {LAB[t]}",
                    ci(a.delta, a.dlo, a.dhi, bold=(a.dlo > 0 or a.dhi < 0)),
                    ci(x.loc["dir", "delta"], x.loc["dir", "dlo"], x.loc["dir", "dhi"], bold=(x.loc["dir", "dlo"] > 0 or x.loc["dir", "dhi"] < 0)),
                    ci(x.loc["eo", "delta"], x.loc["eo", "dlo"], x.loc["eo", "dhi"], bold=(x.loc["eo", "dlo"] > 0 or x.loc["eo", "dhi"] < 0)),
                    f"{x.loc['eo', 'base']:.3f}"]) + " \\\\")
            rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\tiny\n\\caption{OULAD held-out presentation: change in the targeted grouping against the baseline "
           "(paired difference [95\\% student-bootstrap interval]; bold = interval excludes 0). Lower is better for EO, higher for DIR. "
           "Reweighing and the baseline use a decision threshold tuned on the validation presentation; threshold optimisation and exponentiated gradient "
           "use their own decision rules, so flag-rate changes are not purely due to the mitigation.}\n\\label{tab:oulad_heldout_crit}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\nMethod, target & $\\Delta$ macro-F1 & $\\Delta$ DIR & $\\Delta$ EO diff. & Baseline EO \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_oulad_heldout_crit.tex", "w").write(tex)


def tab_cells(res, out):
    rows = []
    srcs = [("Held-out, day 25", f"{res}/oulad_heldout/heldout_C25_cells.csv"),
            ("Held-out, day 50", f"{res}/oulad_heldout/heldout_C50_cells.csv"),
            ("Repeated CV, day 75", f"{res}/oulad_cv_gender/cv_C75_cells.csv")]
    for nm, p in srcs:
        c = pd.read_csv(p)
        c = c[(c.method == "baseline") & (c.grouping == "gi")]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{nm}}}}} \\\\")
        for cell in c.cell.unique():
            x = c[c.cell == cell].set_index("metric")
            lab = cell.replace("F/", "Women, ").replace("M/", "Men, ").replace("IMD<30%", "IMD $<$ 30\\%").replace("IMD>=30%", "IMD $\\geq$ 30\\%")
            rows.append(" & ".join([lab, f"{int(x.loc['tpr', 'n']):,}", f"{int(x.loc['tpr', 'pos']):,}",
                f"{x.loc['tpr', 'value']:.3f} [{x.loc['tpr', 'lo']:.3f}, {x.loc['tpr', 'hi']:.3f}]",
                f"{x.loc['fpr', 'value']:.3f} [{x.loc['fpr', 'lo']:.3f}, {x.loc['fpr', 'hi']:.3f}]"]) + " \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{OULAD gender$\\times$deprivation cells (baseline model; 95\\% student-bootstrap intervals). "
           "$n$ = students (rows), Pos = students who failed or withdrew. The cell detected least is women in less deprived areas; women in the most deprived areas "
           "have the highest false-alert rate.}\n\\label{tab:oulad_cells}\n\\begin{tabular}{lrrcc}\n\\toprule\nCell & $n$ & Pos & TPR & FPR \\\\\n\\midrule\n"
           + "\n".join(rows).replace("\\multicolumn{6}{l}", "\\multicolumn{5}{l}") + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{out}/tab_oulad_cells.tex", "w").write(tex)


def tab_common(res, uci, out):
    u = pd.read_csv(f"{uci}/reanalysis_criteria.csv")
    d = pd.read_csv(f"{res}/oulad_cv_gender/cv_C75_fairness.csv")
    rows = []
    def bold(lo, hi, s):
        return "\\textbf{" + s + "}" if (lo > 0 or hi < 0) else s
    rows.append("\\multicolumn{6}{l}{\\emph{OULAD, repeated student-grouped CV, day 75}} \\\\")
    for fam, nm in [("reweigh", "Reweighing"), ("expgrad_eo", "Exp.\\ gradient"), ("threshold_eo", "Threshold opt.")]:
        m = f"{fam}[gender]"
        x = d[(d.method == m) & (d.grouping == "gender")].set_index("metric")
        a = d[(d.method == m) & (d.grouping == "ALL")].iloc[0]
        cells = [ci(a.delta, a.dlo, a.dhi, bold=(a.dlo > 0 or a.dhi < 0))]
        for mt in ["dir", "tpr_gap", "fpr_gap", "eo"]:
            r = x.loc[mt]; cells.append(ci(r.delta, r.dlo, r.dhi, bold=(r.dlo > 0 or r.dhi < 0)))
        rows.append(" & ".join([nm] + cells) + " \\\\")
    b = d[(d.method == "baseline") & (d.grouping == "gender")].set_index("metric")
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\quad baseline: DIR {b.loc['dir','value']:.3f}, TPR gap {b.loc['tpr_gap','value']:.3f}, FPR gap {b.loc['fpr_gap','value']:.3f}, EO {b.loc['eo','value']:.3f}}} \\\\")
    rows.append("\\addlinespace")
    for ck, nm in [("C0", "UCI-697, $5\\times5$ CV, C0 (enrolment)"), ("C2", "UCI-697, $5\\times5$ CV, C2")]:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{nm}}}}} \\\\")
        for fam, lab in [("reweigh", "Reweighing"), ("expgrad_eo", "Exp.\\ gradient"), ("threshold_eo", "Threshold opt.")]:
            r = u[(u.checkpoint == ck) & (u.family == fam) & (u.target == "gender")].iloc[0]
            cells = [ci(r.f1, r.f1_lo, r.f1_hi, bold=(r.f1_lo > 0 or r.f1_hi < 0))]
            for mt in ["dir", "tpr_gap", "fpr_gap", "eo_diff"]:
                cells.append(ci(r[mt], r[mt + "_lo"], r[mt + "_hi"], bold=(r[mt + "_lo"] > 0 or r[mt + "_hi"] < 0)))
            rows.append(" & ".join([lab] + cells) + " \\\\")
        r = u[(u.checkpoint == ck) & (u.family == "reweigh") & (u.target == "gender")].iloc[0]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\quad baseline: DIR {r.dir_base:.3f}, TPR gap {r.tpr_gap_base:.3f}, FPR gap {r.fpr_gap_base:.3f}, EO {r.eo_diff_base:.3f}}} \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\tiny\n\\caption{Like-for-like comparison of gender-targeted mitigation on OULAD and UCI-697 (same metric definitions, "
           "same repeated-CV protocol with student-level bootstrap, paired change [95\\% interval], bold = interval excludes 0). Gender is the only attribute "
           "audited identically in both datasets.}\n\\label{tab:common_gender}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\n"
           "Method & $\\Delta$ macro-F1 & $\\Delta$ DIR & $\\Delta$ TPR gap & $\\Delta$ FPR gap & $\\Delta$ EO diff. \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_common_gender.tex", "w").write(tex)


def tab_sameday(res, out):
    """Random-split CV (threshold tuned on a calibration split) vs held-out presentation, same day, same procedure."""
    cols = []
    for ck in ["C25", "C50"]:
        cols.append(pd.read_csv(f"{res}/oulad_cv_sameday/cv_{ck}_fairness.csv"))
        cols.append(pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_fairness.csv"))
    def base(d, g, m):
        return d[(d.method == "baseline") & (d.grouping == g) & (d.metric == m)].iloc[0]
    def tgt(d, fam):
        r = d[(d.method == f"{fam}[gender]") & (d.grouping == "gender") & (d.metric == "eo")].iloc[0]
        return f"{r.delta:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}]"
    rows = []
    r = [base(d, "ALL", "macro_f1") for d in cols]
    rows.append("Macro-F1 & " + " & ".join(f"{x.value:.3f} [{x.lo:.3f}, {x.hi:.3f}]" for x in r) + " \\\\")
    dif = []
    for i in (0, 2):
        a, b = r[i], r[i + 1]
        se = np.hypot((a.hi - a.lo) / 3.92, (b.hi - b.lo) / 3.92)
        dif.append(f"\\multicolumn{{2}}{{c}}{{{a.value - b.value:+.3f} [{a.value - b.value - 1.96 * se:+.3f}, {a.value - b.value + 1.96 * se:+.3f}]}}")
    rows.append("CV minus held-out (approx.) & " + " & ".join(dif) + " \\\\")
    rows.append("\\addlinespace")
    for g, m, lab in [("gender", "dir", "Gender DIR"), ("gender", "eo", "Gender EO diff."), ("imd_band", "dir", "IMD DIR"), ("imd_band", "eo", "IMD EO diff."), ("gi", "eo", "Gender$\\times$IMD$<$30\\% EO diff.")]:
        rows.append(f"{lab} & " + " & ".join(f"{base(d, g, m).value:.3f}" for d in cols) + " \\\\")
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Change in gender EO difference under gender-targeted mitigation}} \\\\")
    for fam, lab in [("reweigh", "Reweighing"), ("expgrad_eo", "Exp.\\ gradient"), ("threshold_eo", "Threshold opt.")]:
        rows.append(f"{lab} & " + " & ".join(tgt(d, fam) for d in cols) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\\tiny\n\\caption{Random-split cross-validation against the held-out presentation at the same prediction day, with the same binary outcome, features, "
           "MI top-12 LightGBM, and a macro-F1-maximising decision threshold tuned on a disjoint calibration or validation split. CV: $5\\times5$ student-grouped folds over all presentations; "
           "held-out: train 2013B+2013J, tune 2014B, test 2014J (primary seed). Intervals are conditional student-bootstrap intervals; the difference interval treats the two estimates as independent and is approximate. "
           "The designs also differ in training-set size and cohort mix, so the difference is not attributable to presentation shift alone.}\n\\label{tab:sameday}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Day 25} & \\multicolumn{2}{c}{Day 50} \\\\\n"
           " & Random CV & Held-out & Random CV & Held-out \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_oulad_sameday.tex", "w").write(tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results")
    ap.add_argument("--uci", default="../results/uci697")
    ap.add_argument("--out", default="../../writing/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab_heldout_base(a.res, a.out); tab_heldout_crit(a.res, a.out); tab_cells(a.res, a.out); tab_common(a.res, a.uci, a.out)
    if os.path.isdir(f"{a.res}/oulad_cv_sameday"):
        tab_sameday(a.res, a.out)
    print("wrote oulad tables")
