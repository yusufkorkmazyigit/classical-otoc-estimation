#!/usr/bin/env python3
"""
Calibrate the Pauli-path truncation threshold against exact reference values.

    python pauli_path_threshold.py <folder_with_qasm> <exact.csv>
"""
import csv, glob, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pauli_path import expect_z, parse_qasm  # noqa: E402

folder, ref_csv = sys.argv[1], sys.argv[2]
ref = {r[0]: float(r[1]) for r in csv.reader(open(ref_csv)) if r and r[0] != "circuit_id"}
files = sorted(glob.glob(os.path.join(folder, "*.qasm")), key=lambda p: int(p.rsplit("_", 1)[1][:-5]))
for thr in [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]:
    t0, xs, ys = time.time(), [], []
    for f in files:
        cid = os.path.basename(f)[:-5]
        n, gates = parse_qasm(f)
        xs.append(ref[cid])
        ys.append(expect_z(n, gates, int(cid.rsplit("_", 1)[1]), thr, 2_000_000)[0])
    r = np.corrcoef(xs, ys)[0, 1] if np.std(ys) > 0 else 0.0
    zeros = sum(abs(y) < 1e-12 for y in ys)
    print(f"threshold {thr:<7}: corr^2 = {r * r:.4f}   values truncated to 0: {zeros}/{len(ys)}   ({time.time() - t0:.0f}s)")
