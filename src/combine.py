#!/usr/bin/env python3
"""
Post-processing of submission CSVs.

Bond-dimension extrapolation (Richardson-style), using two MPS runs:
    python combine.py extrap low_bond.csv high_bond.csv K out.csv
        value = v_high - K * (v_low - v_high)

Correlation between two CSVs (e.g. approximate vs exact):
    python combine.py corr a.csv b.csv
"""
import csv, sys

import numpy as np


def load(path):
    return {r[0]: float(r[1]) for r in csv.reader(open(path)) if r and r[0] != "circuit_id"}


def save(d, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit_id", "value"])
        for k in sorted(d, key=lambda k: int(k.rsplit("_", 1)[1])):
            w.writerow([k, d[k]])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "extrap":
        lo, hi, k = load(sys.argv[2]), load(sys.argv[3]), float(sys.argv[4])
        save({c: hi[c] - k * (lo[c] - hi[c]) for c in hi}, sys.argv[5])
    elif mode == "corr":
        a, b = load(sys.argv[2]), load(sys.argv[3])
        keys = sorted(set(a) & set(b))
        r = np.corrcoef([a[k] for k in keys], [b[k] for k in keys])[0, 1]
        print(f"{len(keys)} IDs  corr={r:.4f}  corr^2={r * r:.4f}")
