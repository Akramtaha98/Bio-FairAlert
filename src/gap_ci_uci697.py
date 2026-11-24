#!/usr/bin/env python3
"""Baseline detection gaps between gender x scholarship cells, with student-level bootstrap intervals, and a paired
comparison of the full feature set against the no-timing-fields rerun on IDENTICAL folds.

Both runs use seed 2026 and StratifiedKFold(random_state=seed + repeat), so repeats 0-2 of the full run share their
folds with the 3-repeat no-timing run; the comparison therefore uses repeats 0-2 of the full run only.
Students are resampled with multinomial weights once per replicate; all repeats of a student stay together, and metrics
are averaged over repeats inside each replicate (same engine as reanalysis_uci697.py).
Usage: python gap_ci_uci697.py --full ../results/uci697 --notiming ../results/uci697_notiming --data ../data/uci697/data.csv --out ../results/uci697
"""
import argparse
import numpy as np
import pandas as pd

CK = ["C0", "C1", "C2"]


def weights(N, B, seed):
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones((1, N), np.float32), rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float32)])


def tpr(P, y, m, W):
    mf = (m & (y == 1)).astype(np.float32)
    return ((W @ (P * mf).T) / (W @ mf)[:, None]).mean(1)


def macro_f1(P, y, W):
    yf = y.astype(np.float32)
    tp, fp = W @ (P * yf).T, W @ (P * (1 - yf)).T
    fn, tn = W @ ((1 - P) * yf).T, W @ ((1 - P) * (1 - yf)).T
    return (0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))).mean(1)


def ci(d):
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    return d[0], lo, hi


def write_tex(out, path):
    runs = [("full feature set, $5\\times5$ CV", "full feature set, 5 repeats", ["TPR women+scholarship", "TPR men, no scholarship", "gap (men-no-sch minus women+sch)", "macro-F1"]),
            ("full feature set, repeats 1--3 (same folds as below)", "full feature set, repeats 0-2 (identical folds)", ["TPR women+scholarship", "TPR men, no scholarship", "gap (men-no-sch minus women+sch)", "macro-F1"]),
            ("without Debtor and Tuition fees up to date, $3\\times5$ CV", "without Debtor/Tuition-up-to-date, 3 repeats", ["TPR women+scholarship", "TPR men, no scholarship", "gap (men-no-sch minus women+sch)", "macro-F1"]),
            ("paired change, no fields minus full (same folds)", "paired: no-timing minus full (identical folds)", ["change in TPR women+scholarship", "change in gap", "change in macro-F1"])]
    short = {"TPR women+scholarship": "TPR, women with scholarship", "TPR men, no scholarship": "TPR, men without scholarship",
             "gap (men-no-sch minus women+sch)": "Gap (men without minus women with)", "macro-F1": "Macro-F1",
             "change in TPR women+scholarship": "Change in TPR, women with scholarship", "change in gap": "Change in gap", "change in macro-F1": "Change in macro-F1"}
    rows = []
    for title, run, qs in runs:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for q in qs:
            cells = []
            for ck in CK:
                r = out[(out.run == run) & (out.checkpoint == ck) & (out.quantity == q)].iloc[0]
                sg = "+" if q.startswith("change") else ""
                cells.append(f"{r.value:{sg}.3f} [{r.lo:{sg}.3f}, {r.hi:{sg}.3f}]")
            rows.append(f"\\quad {short[q]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Baseline detection rates of the two extreme gender$\\times$scholarship cells and their difference, "
           "with 95\\% student-level bootstrap intervals (all repeats of a student resampled together), and the effect of removing the two fields of undocumented recording time. "
           "Macro-F1 is for the binary Dropout task. The full-feature and no-field runs share seed and fold assignment for repeats 1--3, so the last block is a paired comparison on identical folds.}\n"
           "\\label{tab:uci_gapci}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccc}\n\\toprule\n & C0 & C1 & C2 \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(path, "w").write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="../results/uci697")
    ap.add_argument("--notiming", default="../results/uci697_notiming")
    ap.add_argument("--data", default="../data/uci697/data.csv")
    ap.add_argument("--out", default="../results/uci697")
    ap.add_argument("--tex", default="../../writing/tables/tab_uci_gapci.tex")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    df = pd.read_csv(a.data, sep=None, engine="python")
    y = (df["Target"] == "Dropout").astype(int).values
    gen, sch = df["Gender"].values.astype(int), df["Scholarship holder"].values.astype(int)
    N = len(y)
    W = weights(N, a.B, a.seed)
    zf, zn = np.load(f"{a.full}/oof_fairness.npz"), np.load(f"{a.notiming}/oof_fairness.npz")
    # women = Gender 0 (checked against the reported 0.279 / 0.709 values below)
    cells = {"women_sch": (gen == 0) & (sch == 1), "men_nosch": (gen == 1) & (sch == 0)}
    rows = []

    def gap_rows(tag, z, reps, ck):
        P = z[f"{ck}__baseline__pred"][reps].astype(np.float32)
        t = {k: tpr(P, y, m, W) for k, m in cells.items()}
        for nm, d in [("TPR women+scholarship", t["women_sch"]), ("TPR men, no scholarship", t["men_nosch"]),
                      ("gap (men-no-sch minus women+sch)", t["men_nosch"] - t["women_sch"])]:
            v, lo, hi = ci(d)
            rows.append(dict(run=tag, checkpoint=ck, quantity=nm, value=v, lo=lo, hi=hi, repeats=len(reps)))
        v, lo, hi = ci(macro_f1(P, y, W))
        rows.append(dict(run=tag, checkpoint=ck, quantity="macro-F1", value=v, lo=lo, hi=hi, repeats=len(reps)))
        return P, t

    store = {}
    for ck in CK:
        store[("full5", ck)] = gap_rows("full feature set, 5 repeats", zf, [0, 1, 2, 3, 4], ck)
        store[("full3", ck)] = gap_rows("full feature set, repeats 0-2 (identical folds)", zf, [0, 1, 2], ck)
        store[("nt3", ck)] = gap_rows("without Debtor/Tuition-up-to-date, 3 repeats", zn, [0, 1, 2], ck)
        # paired effect of removing the two fields on identical folds
        Pf, tf = store[("full3", ck)]
        Pn, tn = store[("nt3", ck)]
        for nm, d in [("change in TPR women+scholarship", tn["women_sch"] - tf["women_sch"]),
                      ("change in TPR men, no scholarship", tn["men_nosch"] - tf["men_nosch"]),
                      ("change in gap", (tn["men_nosch"] - tn["women_sch"]) - (tf["men_nosch"] - tf["women_sch"])),
                      ("change in macro-F1", macro_f1(Pn, y, W) - macro_f1(Pf, y, W))]:
            v, lo, hi = ci(d)
            rows.append(dict(run="paired: no-timing minus full (identical folds)", checkpoint=ck, quantity=nm, value=v, lo=lo, hi=hi, repeats=3))
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.out}/gap_ci.csv", index=False)
    write_tex(out, a.tex)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
