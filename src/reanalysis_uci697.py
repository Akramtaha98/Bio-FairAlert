#!/usr/bin/env python3
"""Criterion-specific, multiplicity-aware and cell-level reanalysis of the UCI-697 fairness run.

Uses only the saved out-of-fold predictions (results/uci697/oof_fairness.npz), so it needs no model refitting.

Bootstrap: students are resampled WITH replacement once per replicate (multinomial weights over the 4,424 students).
Every method's predictions for a student are kept together (and the same replicate is applied to the baseline), so every
reported difference is a PAIRED difference.  Each metric is computed separately for each of the 5 repeats on the
resampled students and then averaged over repeats.  The intervals therefore describe sampling uncertainty of the
evaluation set *conditional on the fitted models*; variability due to retraining is reflected only through the 5 repeats
being averaged, not through the interval.

Usage: python reanalysis_uci697.py --res ../results/uci697 --data ../data/uci697/data.csv --out ../../writing/tables --B 2000
"""
import argparse
import os

import numpy as np
import pandas as pd

CKPTS = ["C0", "C1", "C2"]
ATTRS = ["gender", "age_mature", "scholarship", "special_needs", "international"]
LBL = {"gender": "Gender", "age_mature": "Age $\\geq$ 23", "scholarship": "Scholarship",
       "special_needs": "Special needs", "international": "International", "joint": "Gender$\\times$Schol."}
FAMS = [("reweigh", "Reweighing"), ("expgrad_eo", "Exp.\\ gradient"), ("threshold_eo", "Threshold opt.")]


def load(res, data):
    z = np.load(f"{res}/oof_fairness.npz", allow_pickle=True)
    df = pd.read_csv(data, sep=None, engine="python")
    y = (df["Target"] == "Dropout").astype(int).values
    g = {"gender": df["Gender"].values.astype(int),
         "age_mature": (df["Age at enrollment"] >= 23).astype(int).values,
         "scholarship": df["Scholarship holder"].values.astype(int),
         "special_needs": df["Educational special needs"].values.astype(int),
         "international": df["International"].values.astype(int)}
    masks = {a: [(g[a] == v) for v in (0, 1)] for a in ATTRS}
    cells = {(gv, sv): (g["gender"] == gv) & (g["scholarship"] == sv) for gv in (0, 1) for sv in (0, 1)}
    masks["joint"] = list(cells.values())
    return z, y, masks, cells


def weights(N, B, seed):
    rng = np.random.default_rng(seed)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float32)
    return np.vstack([np.ones((1, N), np.float32), W])  # row 0 = observed data


def group_stats(P, y, mlist, W):
    """flag, tpr, fpr per group -> arrays (G, B+1, R)."""
    out = {"flag": [], "tpr": [], "fpr": []}
    for m in mlist:
        mf = m.astype(np.float32)
        pos, neg = mf * y, mf * (1 - y)
        n, npos, nneg = W @ mf, W @ pos, W @ neg
        with np.errstate(invalid="ignore", divide="ignore"):
            out["flag"].append((W @ (P * mf).T) / n[:, None])
            out["tpr"].append((W @ (P * pos).T) / npos[:, None])
            out["fpr"].append((W @ (P * neg).T) / nneg[:, None])
    return {k: np.stack(v) for k, v in out.items()}


def metrics(P, y, mlist, W):
    gs = group_stats(P, y, mlist, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        dir_ = np.nanmin(gs["flag"], 0) / np.nanmax(gs["flag"], 0)
    tg = np.nanmax(gs["tpr"], 0) - np.nanmin(gs["tpr"], 0)
    fg = np.nanmax(gs["fpr"], 0) - np.nanmin(gs["fpr"], 0)
    return {"dir": dir_.mean(1), "tpr_gap": tg.mean(1), "fpr_gap": fg.mean(1), "eo_diff": np.maximum(tg, fg).mean(1)}, gs


def macro_f1(P, y, W):
    yf = y.astype(np.float32)
    tp = W @ (P * yf).T
    fp = W @ (P * (1 - yf)).T
    fn = W @ ((1 - P) * yf).T
    tn = W @ ((1 - P) * (1 - yf)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))
    return f1.mean(1)


def ci_p(d):
    """d[0] observed paired difference, d[1:] bootstrap replicates."""
    b = d[1:]
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2 * min((np.sum(b <= 0) + 1) / (len(b) + 1), (np.sum(b >= 0) + 1) / (len(b) + 1))
    return d[0], lo, hi, min(p, 1.0)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    run = 1.0
    for i in range(n - 1, -1, -1):
        idx = o[i]
        run = min(run, p[idx] * n / (i + 1))
        adj[idx] = run
    return adj


def fmt(d, lo, hi, nd=3):
    return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results/uci697")
    ap.add_argument("--data", default="../data/uci697/data.csv")
    ap.add_argument("--out", default="../../writing/tables")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--suffix", default="", help="appended to output file names (e.g. _notiming)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z, y, masks, cells = load(a.res, a.data)
    N = len(y)
    W = weights(N, a.B, a.seed)
    global FAMS
    FAMS = [(f, l) for f, l in FAMS if f"C0__{f}[gender]__pred" in z.files]
    sfx = a.suffix

    def P(ck, m):
        return z[f"{ck}__{m}__pred"].astype(np.float32)

    # ------------------------------------------------------------------ 1. criterion-specific, targeted attribute
    crit, side = [], []
    for ck in CKPTS:
        base = P(ck, "baseline")
        bm = {s: metrics(base, y, masks[s], W)[0] for s in masks}
        bf1 = macro_f1(base, y, W)
        for fam, _ in FAMS:
            for t in ["gender", "age_mature", "scholarship", "joint"]:
                Pm = P(ck, f"{fam}[{t}]")
                mm = {s: metrics(Pm, y, masks[s], W)[0] for s in masks}
                f1 = macro_f1(Pm, y, W)
                row = dict(checkpoint=ck, family=fam, target=t)
                d, lo, hi, p = ci_p(f1 - bf1)
                row.update(f1=d, f1_lo=lo, f1_hi=hi, f1_p=p, base_f1=bf1[0])
                for met in ["dir", "tpr_gap", "fpr_gap", "eo_diff"]:
                    d, lo, hi, p = ci_p(mm[t][met] - bm[t][met])
                    row.update({f"{met}": d, f"{met}_lo": lo, f"{met}_hi": hi, f"{met}_p": p, f"{met}_base": bm[t][met][0]})
                crit.append(row)
                comps = ["gender", "scholarship"] if t == "joint" else [t]
                for aud in ATTRS:
                    if aud in comps:
                        continue
                    for met in ["dir", "tpr_gap", "eo_diff"]:
                        d, lo, hi, p = ci_p(mm[aud][met] - bm[aud][met])
                        harm = -d if met == "dir" else d
                        side.append(dict(checkpoint=ck, family=fam, target=t, audited=aud, metric=met, delta=d, lo=lo, hi=hi,
                                         p=p, worse=bool(harm > 0), small_group=aud in ("special_needs", "international")))
    crit = pd.DataFrame(crit)
    side = pd.DataFrame(side)
    side["p_holm"] = holm(side.p.values)
    side["q_bh"] = bh(side.p.values)
    side["sig_raw"] = (side.lo > 0) | (side.hi < 0)
    crit.to_csv(f"{a.res}/reanalysis_criteria{sfx}.csv", index=False)
    side.to_csv(f"{a.res}/reanalysis_side_effects{sfx}.csv", index=False)

    # ---- tables: criterion-specific
    def tab_crit(cks, label, caption):
        rows = []
        for ck in cks:
            rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{Checkpoint {ck}}}}} \\\\")
            x = crit[crit.checkpoint == ck]
            for fam, fl in FAMS:
                for t in ["gender", "age_mature", "scholarship", "joint"]:
                    r = x[(x.family == fam) & (x.target == t)].iloc[0]
                    def c(m, nd=3):
                        s = fmt(r[m], r[f"{m}_lo"], r[f"{m}_hi"], nd)
                        return f"\\textbf{{{s}}}" if r[f"{m}_p"] < 0.05 else s
                    s1 = fmt(r.f1, r.f1_lo, r.f1_hi)
                    s1 = f"\\textbf{{{s1}}}" if r.f1_p < 0.05 else s1
                    rows.append(f"{fl}, {LBL[t]} & {s1} & {c('dir')} & {c('tpr_gap')} & {c('fpr_gap')} & {c('eo_diff')} & {r.eo_diff_base:.3f} \\\\")
                rows.append("\\addlinespace")
        s = ("\\begin{table}[t]\n\\centering\n\\tiny\n"
             f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
             "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\n"
             "Method, target & $\\Delta$ macro-F1 & $\\Delta$ DIR & $\\Delta$ TPR gap & $\\Delta$ FPR gap & $\\Delta$ EO diff. & Baseline EO \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
        return s
    cap = ("Targeted-attribute changes against the baseline for every criterion (paired difference [95\\% student-bootstrap interval]; bold = interval excludes 0). "
           "For ``Gender$\\times$Schol.'' the metrics are computed over the four gender$\\times$scholarship cells. Lower is better for gap measures, higher for DIR. "
           "Baseline EO = baseline equalized-odds difference of the same grouping.")
    with open(f"{a.out}/tab_uci_crit{sfx}.tex", "w") as fh:
        fh.write(tab_crit(["C0", "C2"], "tab:uci_crit" + sfx, cap))

    # ------------------------------------------------------------------ 2. multiplicity summary
    tot = len(side)
    rows = []
    for lab, mask in [("Unadjusted 95\\% interval excludes 0", side.sig_raw),
                      ("Holm-adjusted $p<0.05$ (405 comparisons)", side.p_holm < 0.05),
                      ("Benjamini--Hochberg $q<0.05$", side.q_bh < 0.05)]:
        rows.append(f"{lab} & {int((mask & side.worse).sum())} & {int((mask & ~side.worse).sum())} \\\\")
    exp = tot * 0.025
    s = ("\\begin{table}[t]\n\\centering\n\\small\n"
         f"\\caption{{Cross-attribute side effects: {tot} paired comparisons (3 checkpoints $\\times$ 3 mitigation families $\\times$ targets $\\times$ non-targeted attributes $\\times$ DIR / TPR gap / EO diff.). "
         f"Under a global null with correct coverage one expects about {exp:.1f} significant results in each direction at the unadjusted 5\\% level. "
         "Bootstrap $p$-values are two-sided; the comparisons are exploratory and not independent.}\n\\label{tab:uci_multi}\n"
         "\\begin{tabular}{lcc}\n\\toprule\nCriterion & Significantly worse & Significantly better \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{a.out}/tab_uci_multi{sfx}.tex", "w").write(s)
    sw = side[side.sig_raw & side.worse].copy()
    sw["mag"] = sw.delta.abs()
    print("== significant worsenings (raw):", len(sw))
    print(sw[["checkpoint", "family", "target", "audited", "metric", "delta", "lo", "hi", "p", "p_holm", "q_bh"]].round(3).to_string(index=False))
    for lab, mask in [("raw", side.sig_raw), ("holm", side.p_holm < .05), ("bh", side.q_bh < .05)]:
        print(lab, "worse", int((mask & side.worse).sum()), "better", int((mask & ~side.worse).sum()))
    print("worse point estimates:", int(side.worse.sum()), "of", tot, "; large groups:", int((side.worse & ~side.small_group).sum()), "of", int((~side.small_group).sum()))
    print("median |delta| of sig worse:", sw.mag.median().round(3), " max:", sw.mag.max().round(3))

    # ------------------------------------------------------------------ 3. cells
    crow = []
    cell_lab = {(0, 0): "Woman, no scholarship", (0, 1): "Woman, scholarship", (1, 0): "Man, no scholarship", (1, 1): "Man, scholarship"}
    ml = list(cells.values())
    for ck in CKPTS:
        base = P(ck, "baseline")
        gb = group_stats(base, y, ml, W)
        for fam, _ in FAMS:
            Pm = P(ck, f"{fam}[joint]")
            gm = group_stats(Pm, y, ml, W)
            fo = (W @ Pm.T).mean(1) / N  # overall flag rate
            fb = (W @ base.T).mean(1) / N
            for i, (key, m) in enumerate(cells.items()):
                n, pos = int(m.sum()), int(y[m].sum())
                prev = pos / n
                r = dict(checkpoint=ck, family=fam, cell=cell_lab[key], n=n, dropouts=pos, nondropouts=n - pos)
                for met in ["tpr", "fpr", "flag"]:
                    bb = gb[met][i].mean(1)
                    aa = gm[met][i].mean(1)
                    d, lo, hi, p = ci_p(aa - bb)
                    r.update({f"{met}_base": bb[0], f"{met}_after": aa[0], f"d{met}": d, f"d{met}_lo": lo, f"d{met}_hi": hi, f"d{met}_p": p})
                r["extra_detect_per1000"] = r["dtpr"] * prev * 1000
                r["extra_false_alerts_per1000"] = r["dfpr"] * (1 - prev) * 1000
                crow.append(r)
            d, lo, hi, p = ci_p(fo - fb)
            crow.append(dict(checkpoint=ck, family=fam, cell="ALL", n=N, dropouts=int(y.sum()), nondropouts=int(N - y.sum()),
                             flag_base=fb[0], flag_after=fo[0], dflag=d, dflag_lo=lo, dflag_hi=hi, dflag_p=p,
                             extra_contacted_per1000=d * 1000))
    cr = pd.DataFrame(crow)
    cr.to_csv(f"{a.res}/reanalysis_cells{sfx}.csv", index=False)

    def tab_cells(cks):
        rows = []
        for ck in cks:
            rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{Checkpoint {ck}; threshold optimisation fitted on gender$\\times$scholarship}}}} \\\\")
            x = cr[(cr.checkpoint == ck) & (cr.family == "threshold_eo")]
            for lab in cell_lab.values():
                r = x[x.cell == lab].iloc[0]
                rows.append(f"{lab} & {int(r.n)} & {int(r.dropouts)}/{int(r.nondropouts)} & {r.tpr_base:.3f} & {r.fpr_base:.3f} & "
                            f"{fmt(r.dtpr, r.dtpr_lo, r.dtpr_hi)} & {fmt(r.dfpr, r.dfpr_lo, r.dfpr_hi)} & {r.extra_detect_per1000:+.0f} & {r.extra_false_alerts_per1000:+.0f} \\\\")
            r = x[x.cell == "ALL"].iloc[0]
            rows.append(f"All students & {int(r.n)} & {int(r.dropouts)}/{int(r.nondropouts)} & -- & -- & -- & -- & \\multicolumn{{2}}{{c}}{{{r.extra_contacted_per1000:+.0f} contacted [{r.dflag_lo*1000:+.0f}, {r.dflag_hi*1000:+.0f}]}} \\\\")
            rows.append("\\addlinespace")
        s = ("\\begin{table}[t]\n\\centering\n\\tiny\n"
             "\\caption{Gender$\\times$scholarship cells. $n$ = students; D/N = dropouts / non-dropouts. Baseline TPR and FPR, then the paired change after threshold optimisation fitted on the joint attribute "
             "[95\\% student-bootstrap interval]. The last two columns convert the changes to students per 1,000 students in the cell (extra dropouts detected; extra false alerts). "
             "Cell $n$ is the pooled out-of-fold count per repeat (every student is evaluated once per repeat), so the 20-student rule applies to pooled counts, not to individual folds.}\n"
             "\\label{tab:uci_cells}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrcccccc}\n\\toprule\n"
             "Cell & $n$ & D/N & Base TPR & Base FPR & $\\Delta$ TPR & $\\Delta$ FPR & Extra detected /1000 & Extra false alerts /1000 \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
        return s
    open(f"{a.out}/tab_uci_cells{sfx}.tex", "w").write(tab_cells(["C0", "C2"]))
    print("== cells C0/C2 threshold joint")
    print(cr[(cr.family == "threshold_eo") & cr.checkpoint.isin(["C0", "C2"])][["checkpoint", "cell", "n", "dropouts", "tpr_base", "tpr_after", "fpr_base", "fpr_after", "extra_detect_per1000", "extra_false_alerts_per1000", "extra_contacted_per1000"]].round(3).to_string(index=False))
    print("== criteria C0 threshold & reweigh")
    pd.set_option("display.width", 250)
    print(crit[crit.checkpoint.isin(["C0", "C2"]) & (crit.family != "expgrad_eo")][["checkpoint", "family", "target", "f1", "dir", "tpr_gap", "fpr_gap", "eo_diff", "eo_diff_base", "eo_diff_lo", "eo_diff_hi"]].round(3).to_string(index=False))
    print("== four-cell baseline + intersect per-ckpt baseline: tpr/fpr/flag by cell")
    for ck in CKPTS:
        print(ck, cr[(cr.checkpoint == ck) & (cr.family == "threshold_eo") & (cr.cell != "ALL")][["cell", "n", "dropouts", "tpr_base", "fpr_base", "flag_base"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
