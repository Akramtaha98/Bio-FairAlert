#!/usr/bin/env python3
"""Round-4 tables: IMD group counts/rates, five-seed ranges, fixed-capacity, no-scores and known-IMD checks."""
import argparse, os
import numpy as np, pandas as pd

SEEDS = range(42, 47)
def sg(x): return f"{x:+.3f}"

def tab_imd_groups(res, out):
    rows = []
    g = {ck: pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_groups.csv") for ck in ("C25", "C50")}
    g = {k: v[v.grouping == "imd_band"].set_index("group") for k, v in g.items()}
    for grp in g["C25"].index:
        a, b = g["C25"].loc[grp], g["C50"].loc[grp]
        rows.append(f"{grp.replace('%', chr(92) + '%')} & {int(a.n):,} & {a.flag:.3f} [{a.flag_lo:.3f}, {a.flag_hi:.3f}] & {int(b.n):,} & {b.flag:.3f} [{b.flag_lo:.3f}, {b.flag_hi:.3f}] \\\\")
    dirs = []
    for ck in ("C25", "C50"):
        t = g[ck]; mx, mn = t.flag.idxmax(), t.flag.idxmin()
        k = t.drop("MISSING"); kmn = k.flag.idxmin()
        dirs.append((t.flag.min() / t.flag.max(), mn, mx.replace('%', '\\%'), k.flag.min() / k.flag.max(), kmn.replace('%', '\\%')))
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{OULAD held-out test presentation (2014J): IMD categories, group sizes and baseline flag rates [95\\% bootstrap interval], primary seed. "
           "Bands are IMD deciles (0--10\\% = most deprived); MISSING = IMD not recorded, kept as its own category. The disparate-impact ratio is the minimum over the maximum group flag rate: "
           f"day 25, {dirs[0][0]:.3f} (lowest: {dirs[0][1]}; highest: {dirs[0][2]}); day 50, {dirs[1][0]:.3f} (lowest: {dirs[1][1]}; highest: {dirs[1][2]}). "
           f"Excluding the MISSING category the ratio on the same model is {dirs[0][3]:.3f} and {dirs[1][3]:.3f} (lowest known band: {dirs[0][4]} and {dirs[1][4]}).}}\n"
           "\\label{tab:oulad_imd}\n\\begin{tabular}{lrcrc}\n\\toprule\n & \\multicolumn{2}{c}{Day 25} & \\multicolumn{2}{c}{Day 50} \\\\\nIMD band & $n$ & Flag rate & $n$ & Flag rate \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{out}/tab_oulad_imd.tex", "w").write(tex)
    return dirs

def seed_deltas(res, ck, m, g, me):
    v = []
    for s in SEEDS:
        d = pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}_seed{s}_fairness.csv")
        r = d[(d.method == m) & (d.grouping == g) & (d.metric == me)].iloc[0]
        v.append((r.delta, r.dlo, r.dhi))
    return np.array(v)

def tab_seeds(res, out):
    rows = []
    spec = [("Gender-targeted, gender EO", "gender", "eo", ["reweigh", "expgrad_eo", "threshold_eo"]),
            ("IMD-targeted, IMD EO", "imd_band", "eo", ["reweigh", "expgrad_eo", "threshold_eo"])]
    nm = {"reweigh": "Reweighing", "expgrad_eo": "Exp.\\ gradient", "threshold_eo": "Threshold opt."}
    for lab, tg, me, fams in spec:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for fam in fams:
            cells = []
            for ck in ("C25", "C50"):
                v = seed_deltas(res, ck, f"{fam}[{tg}]", tg, me)
                npos = int((v[:, 0] > 0).sum()); nsig = int(((v[:, 1] > 0) | (v[:, 2] < 0)).sum())
                cells += [f"{v[:, 0].min():+.3f} to {v[:, 0].max():+.3f}", f"{npos}/5; {nsig}/5"]
            rows.append(f"{nm[fam]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Holm-adverse side effects of the primary seed}} \\\\")
    for lab, ck, m, g, me in [("Exp.\\ gradient (disability) $\\to$ age-band FPR gap", "C25", "expgrad_eo[disability]", "age_band", "fpr_gap"),
                              ("Reweighing (gender$\\times$disab.) $\\to$ gender$\\times$IMD FPR gap", "C50", "reweigh[gd]", "gi", "fpr_gap")]:
        cells = ["", ""] if ck == "C50" else []
        v = seed_deltas(res, ck, m, g, me)
        txt = [f"{v[:, 0].min():+.3f} to {v[:, 0].max():+.3f}", f"{int((v[:, 0] > 0).sum())}/5; {int(((v[:, 1] > 0) | (v[:, 2] < 0)).sum())}/5"]
        rows.append(f"{lab} & " + (" & ".join(txt + ["", ""]) if ck == "C25" else " & ".join(["", ""] + txt)) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{OULAD held-out presentation: variation of selected contrasts over five retraining seeds (42--46) with the same data split. "
           "For each, the range of the paired change against the baseline (EO difference; lower is better), the number of seeds with an increase, and the number of seeds whose own 95\\% bootstrap interval excludes zero. "
           "Seed ranges show training variability only; they are not confidence intervals.}\n\\label{tab:oulad_seeds}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Day 25} & \\multicolumn{2}{c}{Day 50} \\\\\n"
           "Contrast & Range of change & Increase; excl.\\ 0 & Range of change & Increase; excl.\\ 0 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_oulad_seeds.tex", "w").write(tex)

def tab_checks(res, out):
    rows = []
    runs = [("Main run", "oulad_heldout"), ("Without score features", "oulad_heldout_noscores"), ("Known IMD only", "oulad_heldout_knownimd")]
    for ck, d in (("C25", "Day 25"), ("C50", "Day 50")):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{d}}}}} \\\\")
        for lab, r in runs:
            fa = pd.read_csv(f"{res}/{r}/heldout_{ck}_fairness.csv"); sp = pd.read_csv(f"{res}/{r}/heldout_{ck}_seed_performance.csv")
            b = sp[sp.method == "baseline"]
            x = lambda g, m: fa[(fa.method == "baseline") & (fa.grouping == g) & (fa.metric == m)].iloc[0].value
            tg = fa[(fa.method == "expgrad_eo[gender]") & (fa.grouping == "gender") & (fa.metric == "eo")].iloc[0]
            rows.append(" & ".join([lab, f"{b.macro_f1.mean():.3f}", f"{b.auroc.mean():.3f}", f"{x('gender', 'eo'):.3f}", f"{x('imd_band', 'dir'):.3f}",
                                    f"{x('imd_band', 'eo'):.3f}", f"{tg.delta:+.3f}"]) + " \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{OULAD held-out robustness checks. Without score features: assessment-score features removed from the model. "
           "Known IMD only: students with missing IMD removed from training and test. Macro-F1 and AUROC are means over five seeds (three for the no-scores run); fairness values are the primary seed. "
           "Last column: change in gender EO difference under gender-targeted exponentiated gradient.}\n\\label{tab:oulad_checks}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\nRun & Macro-F1 & AUROC & Gender EO & IMD DIR & IMD EO & $\\Delta$ gender EO (EG) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_oulad_checks.tex", "w").write(tex)

def tab_fixedcap(res, out):
    rows = []
    nm = {"reweigh": "Reweighing", "expgrad_eo": "Exp.\\ gradient", "threshold_eo": "Threshold opt."}
    for tg, tl in (("gender", "Gender"), ("imd_band", "IMD band")):
        for fam in nm:
            c = []
            for ck in ("C25", "C50"):
                for sfx in ("", "_fixedcap"):
                    fa = pd.read_csv(f"{res}/oulad_heldout/heldout_{ck}{sfx}_fairness.csv")
                    r = fa[(fa.method == f"{fam}[{tg}]") & (fa.grouping == tg) & (fa.metric == "eo")].iloc[0]
                    c.append(f"{r.delta:+.3f} [{r.dlo:+.3f}, {r.dhi:+.3f}]")
            rows.append(f"{nm[fam]}, {tl} & " + " & ".join(c) + " \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\tiny\n\\caption{OULAD held-out presentation: change in the targeted EO difference [95\\% bootstrap interval] under the method's own decision rule and when every method is "
           "restricted to flag as many students as the baseline (each method flags its top-$k$ students by its own score, $k$ = number flagged by the baseline).}\n\\label{tab:oulad_fixedcap}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Day 25} & \\multicolumn{2}{c}{Day 50} \\\\\n"
           " & Own rule & Fixed capacity & Own rule & Fixed capacity \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{out}/tab_oulad_fixedcap.tex", "w").write(tex)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="../results"); ap.add_argument("--out", default="../../writing/tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = tab_imd_groups(a.res, a.out); tab_seeds(a.res, a.out); tab_checks(a.res, a.out); tab_fixedcap(a.res, a.out)
    print(d)
