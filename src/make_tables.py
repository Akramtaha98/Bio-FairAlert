#!/usr/bin/env python3
"""Build the LaTeX tables of the UCI-697 external validation from results/uci697/*.csv.

Usage:  python make_tables.py --res ../results/uci697 --data ../data/uci697/data.csv --out ../../writing/tables
Every number in the paper's UCI-697 tables is produced here, none is typed by hand.
"""
import argparse
import os

import numpy as np
import pandas as pd

LBL = {"gender": "Gender", "age_mature": "Age $\\geq$ 23", "scholarship": "Scholarship",
       "special_needs": "Special needs", "international": "International", "joint": "Gender$\\times$Schol."}
MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "Hybrid-Stack"]


def f3(x):
    return f"{x:.3f}"


def sgn(x):
    return f"{x:+.3f}"


def write(path, s):
    with open(path, "w") as fh:
        fh.write(s)
    print("wrote", path)


def tab_main(res, out):
    d = pd.read_csv(f"{res}/main_summary.csv")
    rows = []
    for m in MODEL_ORDER:
        for flt, fl in [("none", "None"), ("mi", "MI top-12")]:
            cells = []
            for ck in ["C0", "C1", "C2"]:
                r = d[(d.model == m) & (d["filter"] == flt) & (d.checkpoint == ck)].iloc[0]
                cells.append(f"{r.macro_f1:.3f} [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}]")
            rows.append(f"{m} & {fl} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    s = ("\\begin{table}[t]\n\\centering\n\\small\n"
         "\\caption{UCI-697, three-class task (Dropout / Enrolled / Graduate): macro-F1 with 95\\% Nadeau--Bengio "
         "intervals (5$\\times$5 repeated stratified CV). C0 = enrolment data only; C1 = + first-semester; "
         "C2 = + second-semester records.}\n\\label{tab:uci_main}\n"
         "\\begin{tabular}{llccc}\n\\toprule\nModel & Filter & C0 & C1 & C2 \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"{out}/tab_uci_main.tex", s)


def tab_tests(res, out):
    t = pd.read_csv(f"{res}/main_tests.csv")
    t["kind"] = t.test.str.split("|").str[0].str.strip()
    t["sub"] = t.test.str.split("|").str[-1].str.strip()
    order = []
    for m in MODEL_ORDER:
        order.append((f"filter effect (MI - none) | {m}", f"MI filter vs.\\ none, {m}"))
    order.append(("Hybrid-Stack - best individual (XGBoost) | filter=none", "Hybrid vs.\\ best single (XGBoost), no filter"))
    order.append(("Hybrid-Stack - best individual (XGBoost) | filter=mi", "Hybrid vs.\\ best single (XGBoost), MI filter"))
    rows = []
    for key, lab in order:
        cells = []
        for ck in ["C0", "C1", "C2"]:
            r = t[(t.test == key) & (t.checkpoint == ck)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            star = "$^{*}$" if r.p_holm < 0.05 else ""
            cells.append(f"{sgn(r.mean_diff)} [{sgn(r.ci_lo)}, {sgn(r.ci_hi)}]; {r.p_holm:.3f}{star}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{table}[t]\n\\centering\n\\footnotesize\n"
         "\\caption{Paired macro-F1 differences (mean [95\\% Nadeau--Bengio CI]; Holm-adjusted $p$). "
         "$^{*}$ Holm $p<0.05$. Negative values mean the first-named condition is worse.}\n\\label{tab:uci_tests}\n"
         "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccc}\n\\toprule\nComparison & C0 & C1 & C2 \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    write(f"{out}/tab_uci_tests.tex", s)


def tab_ksweep(res, out):
    k = pd.read_csv(f"{res}/ksweep_summary.csv")
    rows = []
    for flt, fl in [("mi", "Mutual information"), ("anova", "ANOVA $F$")]:
        for kk in sorted(k[k["filter"] == flt].k.unique()):
            cells = []
            for ck in ["C0", "C1", "C2"]:
                r = k[(k["filter"] == flt) & (k.k == kk) & (k.checkpoint == ck)]
                cells.append("--" if r.empty else f"{r.iloc[0].macro_f1:.3f}")
            rows.append(f"{fl} & {kk} & " + " & ".join(cells) + " \\\\")
    for ck_full, n in [("C0", 24), ("C1", 30), ("C2", 36)]:
        pass
    nf = k[k["filter"] == "none"]
    cells = []
    for ck in ["C0", "C1", "C2"]:
        r = nf[nf.checkpoint == ck]
        cells.append("--" if r.empty else f"{r.iloc[0].macro_f1:.3f}")
    rows.append("No filter (all features) & all & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{table}[t]\n\\centering\n\\small\n"
         "\\caption{Feature-filter sensitivity (LightGBM, three-class): macro-F1 against the number of retained features $k$ "
         "(selection refit inside every training fold).}\n\\label{tab:uci_ksweep}\n"
         "\\begin{tabular}{lcccc}\n\\toprule\nFilter & $k$ & C0 & C1 & C2 \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"{out}/tab_uci_ksweep.tex", s)


def tab_fair(res, out, ck="C2"):
    f = pd.read_csv(f"{res}/fairness_summary.csv")
    f = f[f.checkpoint == ck]
    attrs = ["gender", "age_mature", "scholarship"]

    def row(method, label, target=None):
        ov = f[(f.method == method) & (f.attr == "overall")].iloc[0]
        cells = [f"{ov.macro_f1_mean:.3f}"]
        for a in attrs:
            r = f[(f.method == method) & (f.attr == a)].iloc[0]
            c = f"{r.dir_mean:.3f}"
            cells.append(f"\\textbf{{{c}}}" if target and (target == a or (target == 'joint' and a in ('gender', 'scholarship'))) else c)
        for a in attrs:
            r = f[(f.method == method) & (f.attr == a)].iloc[0]
            c = f"{r.eo_diff_mean:.3f}"
            cells.append(f"\\textbf{{{c}}}" if target and (target == a or (target == 'joint' and a in ('gender', 'scholarship'))) else c)
        return f"{label} & " + " & ".join(cells) + " \\\\"

    rows = [row("baseline", "Baseline (all features)"), row("exclude_protected", "Exclude protected attributes"), "\\midrule"]
    fam = [("reweigh", "Reweighing"), ("expgrad_eo", "Exp.\\ gradient (EO)"), ("threshold_eo", "Threshold opt.\\ (EO)")]
    for mk, ml in fam:
        for t in ["gender", "age_mature", "scholarship", "joint"]:
            rows.append(row(f"{mk}[{t}]", f"{ml}, target: {LBL[t]}", t))
        rows.append("\\addlinespace")
    s = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n"
         f"\\caption{{UCI-697 ({ck}): binary Dropout flag. Binary macro-F1, disparate-impact ratio (DIR, higher is fairer) "
         "and equalized-odds difference (EO, lower is fairer) for three protected attributes, mean of 25 out-of-fold evaluations. "
         "Bold = attribute the mitigation was fitted for (``Gender$\\times$Schol.'' targets both).}\n"
         "\\label{tab:uci_fair}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccccc}\n\\toprule\n"
         " & & \\multicolumn{3}{c}{DIR} & \\multicolumn{3}{c}{EO difference} \\\\\n"
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
         "Method & Macro-F1 & Gender & Age $\\geq$ 23 & Scholarship & Gender & Age $\\geq$ 23 & Scholarship \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    write(f"{out}/tab_uci_fair.tex", s)


def tab_side(res, out):
    s = pd.read_csv(f"{res}/fairness_side_effects.csv")
    comp = {"gender": ["gender"], "age_mature": ["age_mature"], "scholarship": ["scholarship"],
            "joint": ["gender", "scholarship"]}
    s = s[[a not in comp[t] for a, t in zip(s.audited, s.target)]].copy()  # cross-attribute only
    s["grp"] = np.where(s.audited.isin(["special_needs", "international"]), "Small groups", "Large groups")
    s["sig_worse"] = s.worsened & s.significant
    s["sig_better"] = (~s.worsened) & s.significant
    names = {"reweigh": "Reweighing", "expgrad_eo": "Exp.\\ gradient (EO)", "threshold_eo": "Threshold opt.\\ (EO)"}
    rows = []
    for m in ["reweigh", "expgrad_eo", "threshold_eo"]:
        for g in ["Large groups", "Small groups"]:
            x = s[(s.mitigation == m) & (s.grp == g)]
            rows.append(f"{names[m]} & {g} & {len(x)} & {int(x.worsened.sum())} & {int(x.sig_worse.sum())} & {int(x.sig_better.sum())} \\\\")
    tot = s
    rows.append("\\midrule")
    rows.append(f"All & -- & {len(tot)} & {int(tot.worsened.sum())} & {int(tot.sig_worse.sum())} & {int(tot.sig_better.sum())} \\\\")
    body = ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\caption{Cross-attribute side effects of single-attribute mitigations, UCI-697 (all checkpoints, DIR / TPR-gap / EO-difference). "
            "Each count is one (checkpoint, mitigation, target, audited attribute, metric) comparison against the baseline model; "
            "``sig.'' = 95\\% student-level bootstrap interval excludes zero (not corrected for multiplicity). "
            "Large groups: gender, age, scholarship; small groups: special needs ($n=51$), international ($n=110$).}\n"
            "\\label{tab:uci_side}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
            "Mitigation & Audited & Comparisons & Point est.\\ worse & Sig.\\ worse & Sig.\\ better \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"{out}/tab_uci_side.tex", body)
    # print the significant worsenings for the text
    w = s[s.sig_worse][["checkpoint", "mitigation", "target", "audited", "metric", "delta", "ci_lo", "ci_hi"]]
    print(w.round(3).to_string(index=False))
    print("totals", len(s), int(s.worsened.sum()), int(s.sig_worse.sum()), int(s.sig_better.sum()))


def tab_inter(res, out, ck="C2"):
    i = pd.read_csv(f"{res}/intersectional_summary.csv")
    b = i[(i.checkpoint == ck) & (i.method == "baseline")].sort_values("dir_pair")
    rows = []
    for _, r in b.iterrows():
        a, c = [x.strip() for x in r.pair.split(" x ")]
        rows.append(f"{LBL[a]} $\\times$ {LBL[c]} & {f3(r.dir_pair)} & {f3(r.dir_a)} & {f3(r.dir_b)} & {int(r.reliable_cells)} & {int(r.repeats_worse_than_both)}/{int(r.repeats)} \\\\")
    s = ("\\begin{table}[t]\n\\centering\n\\small\n"
         f"\\caption{{UCI-697 ({ck}), baseline model: intersectional DIR for every attribute pair next to the two single-attribute DIRs. "
         "Cells with fewer than 20 test students are dropped; ``Cells'' = cells retained.}\n\\label{tab:uci_inter}\n"
         "\\begin{tabular}{lccccc}\n\\toprule\nPair & DIR (pair) & DIR (first) & DIR (second) & Cells & Pair $<$ both \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"{out}/tab_uci_inter.tex", s)


def tab_cells(res, data, out, ck="C2"):
    z = np.load(f"{res}/oof_fairness.npz", allow_pickle=True)
    df = pd.read_csv(data, sep=None, engine="python")
    y = (df["Target"] == "Dropout").astype(int).values
    g = df["Gender"].values.astype(int)
    sch = df["Scholarship holder"].values.astype(int)
    labels = {(0, 0): "Woman, no scholarship", (0, 1): "Woman, scholarship", (1, 0): "Man, no scholarship", (1, 1): "Man, scholarship"}
    meths = [("baseline", "Baseline"), ("threshold_eo[joint]", "Threshold opt.\\ (EO), joint")]
    rows = []
    for (gv, sv), lab in labels.items():
        m = (g == gv) & (sch == sv)
        cells = [str(int(m.sum())), f"{y[m].mean():.3f}"]
        for key, _ in meths:
            P = z[f"{ck}__{key}__pred"].astype(float)
            P = np.where(np.isnan(P), np.nan, P)
            flag = np.mean([P[r][m].mean() for r in range(P.shape[0])])
            tpr = np.mean([P[r][m & (y == 1)].mean() for r in range(P.shape[0])])
            fpr = np.mean([P[r][m & (y == 0)].mean() for r in range(P.shape[0])])
            cells += [f"{flag:.3f}", f"{tpr:.3f}", f"{fpr:.3f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{table}[t]\n\\centering\n\\footnotesize\n"
         f"\\caption{{UCI-697 ({ck}): the four gender$\\times$scholarship cells. Observed dropout prevalence, and flag rate, TPR and FPR of the baseline model and after "
         "equalized-odds threshold optimisation fitted on the joint attribute (mean over 5 repeats of out-of-fold predictions).}\n"
         "\\label{tab:uci_cells}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrcccccccc}\n\\toprule\n"
         " & & & \\multicolumn{3}{c}{Baseline} & \\multicolumn{3}{c}{Threshold opt.\\ (joint)} \\\\\n"
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
         "Cell & $n$ & Dropout rate & Flag & TPR & FPR & Flag & TPR & FPR \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    write(f"{out}/tab_uci_cells.tex", s)


def tab_explain(res, out):
    d = pd.read_csv(f"{res}/explain_deletion.csv")
    rows = [f"{int(r.k_removed)} & {r.auroc_full:.3f} & {r.auroc_after_shap_deletion:.3f} & {r.auroc_after_random_deletion:.3f} & "
            f"{sgn(-r.extra_drop_vs_random)} [{sgn(-r.ci_hi)}, {sgn(-r.ci_lo)}] \\\\" for _, r in d.iterrows()]
    s = ("\\begin{table}[t]\n\\centering\n\\small\n"
         "\\caption{UCI-697 (C2): deletion-fidelity test. AUROC (Dropout) after replacing the top-$k$ SHAP features by their training median, "
         "against replacing $k$ random features. Last column: AUROC after SHAP deletion minus AUROC after random deletion (95\\% Nadeau--Bengio interval).}\n"
         "\\label{tab:uci_explain}\n\\begin{tabular}{ccccc}\n\\toprule\n$k$ & Full model & SHAP deletion & Random deletion & Difference \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"{out}/tab_uci_explain.tex", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results/uci697")
    ap.add_argument("--data", default="../data/uci697/data.csv")
    ap.add_argument("--out", default="../../writing/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab_main(a.res, a.out)
    tab_tests(a.res, a.out)
    tab_ksweep(a.res, a.out)
    tab_fair(a.res, a.out)
    tab_side(a.res, a.out)
    tab_inter(a.res, a.out)
    tab_cells(a.res, a.data, a.out)
    tab_explain(a.res, a.out)
