#!/usr/bin/env python3
"""Gender contrasts WITHIN scholarship strata (UCI-697), their interaction, and the effect of scholarship within gender,
with student-level bootstrap intervals (all repeats of a student resampled together).

Cells: women = Gender 0, men = Gender 1. Contrast = TPR(men) - TPR(women) within a scholarship stratum.
Interaction = contrast among scholarship holders minus contrast among non-holders.
The two extreme cells highlighted in the paper (women with scholarship, men without) were identified AFTER inspecting
the four cell rates, so every comparison among the four cells is reported here, not only the extreme pair.
Usage: python within_scholarship_uci697.py --full ../results/uci697 --notiming ../results/uci697_notiming --data ../data/uci697/data.csv --out ../results/uci697
"""
import argparse
import numpy as np
import pandas as pd
from gap_ci_uci697 import weights, tpr, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="../results/uci697")
    ap.add_argument("--notiming", default="../results/uci697_notiming")
    ap.add_argument("--data", default="../data/uci697/data.csv")
    ap.add_argument("--out", default="../results/uci697")
    ap.add_argument("--tex", default="../../writing/tables/tab_uci_within.tex")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    df = pd.read_csv(a.data, sep=None, engine="python")
    y = (df["Target"] == "Dropout").astype(int).values
    gen, sch = df["Gender"].values.astype(int), df["Scholarship holder"].values.astype(int)
    N = len(y)
    W = weights(N, a.B, a.seed)
    zf, zn = np.load(f"{a.full}/oof_fairness.npz"), np.load(f"{a.notiming}/oof_fairness.npz")
    cells = {"WS": (gen == 0) & (sch == 1), "MS": (gen == 1) & (sch == 1), "WN": (gen == 0) & (sch == 0), "MN": (gen == 1) & (sch == 0)}
    print({k: (int(m.sum()), int((m & (y == 1)).sum())) for k, m in cells.items()})
    runs = [("full feature set", zf, [0, 1, 2, 3, 4], ["C0", "C1", "C2"]), ("without Debtor/Tuition", zn, [0, 1, 2], ["C0", "C1", "C2"])]
    rows = []
    for rn, z, reps, cks in runs:
        for ck in cks:
            P = z[f"{ck}__baseline__pred"][reps].astype(np.float32)
            t = {k: tpr(P, y, m, W) for k, m in cells.items()}
            quant = {
                "TPR women, scholarship": t["WS"], "TPR men, scholarship": t["MS"],
                "TPR women, no scholarship": t["WN"], "TPR men, no scholarship": t["MN"],
                "gender contrast (men-women), scholarship holders": t["MS"] - t["WS"],
                "gender contrast (men-women), no scholarship": t["MN"] - t["WN"],
                "interaction (contrast holders - contrast non-holders)": (t["MS"] - t["WS"]) - (t["MN"] - t["WN"]),
                "scholarship effect (no-sch - sch), women": t["WN"] - t["WS"],
                "scholarship effect (no-sch - sch), men": t["MN"] - t["MS"],
                "TPR scholarship holders (marginal)": tpr(P, y, sch == 1, W),
                "TPR non-holders (marginal)": tpr(P, y, sch == 0, W),
                "marginal scholarship TPR gap (non-holders - holders)": tpr(P, y, sch == 0, W) - tpr(P, y, sch == 1, W),
                "TPR women (marginal)": tpr(P, y, gen == 0, W), "TPR men (marginal)": tpr(P, y, gen == 1, W),
                "marginal gender TPR gap (men - women)": tpr(P, y, gen == 1, W) - tpr(P, y, gen == 0, W),
            }
            for q, d in quant.items():
                v, lo, hi = ci(d)
                rows.append(dict(run=rn, checkpoint=ck, quantity=q, value=v, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.out}/within_scholarship_ci.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    # table: full feature set and restricted, C0 and C2
    order = ["TPR women, scholarship", "TPR men, scholarship", "TPR women, no scholarship", "TPR men, no scholarship",
             "gender contrast (men-women), scholarship holders", "gender contrast (men-women), no scholarship",
             "interaction (contrast holders - contrast non-holders)", "scholarship effect (no-sch - sch), women", "scholarship effect (no-sch - sch), men"]
    lab = {"TPR women, scholarship": "TPR, women with scholarship", "TPR men, scholarship": "TPR, men with scholarship",
           "TPR women, no scholarship": "TPR, women without scholarship", "TPR men, no scholarship": "TPR, men without scholarship",
           "gender contrast (men-women), scholarship holders": "Gender contrast among holders (men $-$ women)",
           "gender contrast (men-women), no scholarship": "Gender contrast among non-holders",
           "interaction (contrast holders - contrast non-holders)": "Interaction (holders $-$ non-holders)",
           "scholarship effect (no-sch - sch), women": "Scholarship effect, women (none $-$ holders)",
           "scholarship effect (no-sch - sch), men": "Scholarship effect, men (none $-$ holders)"}
    cols = [("full feature set", "C0"), ("without Debtor/Tuition", "C0"), ("full feature set", "C2")]
    rws = []
    for q in order:
        cs = []
        for rn, ck in cols:
            r = out[(out.run == rn) & (out.checkpoint == ck) & (out.quantity == q)].iloc[0]
            sg = "+" if ("contrast" in q or "effect" in q or "interaction" in q) else ""
            cs.append(f"{r.value:{sg}.3f} [{r.lo:{sg}.3f}, {r.hi:{sg}.3f}]")
        rws.append(f"{lab[q]} & " + " & ".join(cs) + " \\\\")
        if q in ("TPR men, no scholarship", "interaction (contrast holders - contrast non-holders)"):
            rws.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Detection rates (TPR) in the four gender$\\times$scholarship cells and the contrasts between them, with 95\\% student-bootstrap intervals "
           "(full feature set: $5\\times5$ CV; restricted C0: $3\\times5$ CV without \\emph{Debtor} and \\emph{Tuition fees up to date}). The cells at the extremes were identified after inspecting the four rates, "
           "so all contrasts are shown. Scholarship status, not gender, accounts for most of the difference at C0.}\n\\label{tab:uci_within}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccc}\n\\toprule\n & C0, full & C0, restricted & C2, full \\\\\n\\midrule\n" + "\n".join(rws) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(a.tex, "w").write(tex)


if __name__ == "__main__":
    main()
