#!/usr/bin/env python3
"""Compare regenerated result JSONs (results_regen/) with the committed ones (results/): max abs difference of every numeric leaf."""
import json, os, sys, math
a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("../results", "../results_regen")
def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f"{p}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x):
        yield p, float(x)
for f in sorted(os.listdir(b)):
    if not f.endswith(".json") or not os.path.exists(f"{a}/{f}"): continue
    A, B = dict(leaves(json.load(open(f"{a}/{f}")))), dict(leaves(json.load(open(f"{b}/{f}"))))
    keys = A.keys() & B.keys()
    if not keys: print(f"{f}: no comparable numbers"); continue
    d = sorted(((abs(A[k] - B[k]), k) for k in keys), reverse=True)
    print(f"{f}: {len(keys)} numbers, max |diff| = {d[0][0]:.4g} at {d[0][1]}; >0.005: {sum(x > 0.005 for x, _ in d)}; missing={len(A.keys() ^ B.keys())}")
