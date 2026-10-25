#!/usr/bin/env python3
"""
Heisenberg-picture Pauli-path simulator for <0...0| C^dagger Z_q C |0...0>.

The observable Z_q is propagated backwards through the circuit (gate by gate,
last gate first). It is stored as a sum of Pauli strings:
    O = sum_k c_k P_k,   P_k encoded as two bitmasks (x, z) over the qubits.

* CZ is Clifford: it maps each Pauli string to exactly one Pauli string (plus a sign),
  so it never increases the number of terms.
* A single-qubit gate (u3) maps X/Y/Z to a linear combination of X/Y/Z, so a
  non-Clifford rotation can split one term into up to three ("branching").
* Terms whose |coefficient| falls below a threshold are dropped (truncation).
  threshold = 0 gives the exact result.

At the end, <0|P|0> = 1 if P contains only I/Z (x-mask == 0), else 0.

Supported gates: u3, cz, x, z  (the gate set of the challenge circuits).

Usage:
    python pauli_path.py <folder_with_qasm> <out.csv> [threshold] [max_terms]
"""
import csv
import glob
import os
import re
import sys
import time

import numpy as np

PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]], complex)
PZ = np.array([[1, 0], [0, -1]], complex)
PAULIS = [PX, PY, PZ]               # order: X, Y, Z
XZ_BITS = [(1, 0), (1, 1), (0, 1)]  # (x, z) bits of X, Y, Z


def u3(theta, phi, lam):
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
    ])


def heisenberg_matrix(U):
    """3x3 real matrix M with  U^dagger sigma_a U = sum_b M[b, a] sigma_b."""
    M = np.zeros((3, 3))
    for a in range(3):
        O = U.conj().T @ PAULIS[a] @ U
        for b in range(3):
            M[b, a] = np.real(np.trace(PAULIS[b] @ O)) / 2
    M[np.abs(M) < 1e-12] = 0
    return M


def _angle(s):
    return float(eval(s.strip(), {"__builtins__": {}}, {"pi": np.pi}))


def parse_qasm(path):
    """Minimal OpenQASM 2.0 parser for the u3/cz/x/z gate set."""
    gates, n = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("qreg"):
            n = int(re.search(r"\[(\d+)\]", line).group(1))
        elif line.startswith("u3("):
            args, q = re.match(r"u3\((.*)\)\s*q\[(\d+)\];", line).groups()
            gates.append(("1q", int(q), heisenberg_matrix(u3(*[_angle(a) for a in args.split(",")]))))
        elif line.startswith(("x ", "z ")):
            q = int(re.search(r"\[(\d+)\]", line).group(1))
            gates.append(("1q", q, heisenberg_matrix(PX if line[0] == "x" else PZ)))
        elif line.startswith("cz "):
            a, b = map(int, re.findall(r"\[(\d+)\]", line))
            gates.append(("cz", a, b))
        elif line and not line.startswith(("OPENQASM", "include", "//", "barrier")):
            raise ValueError("unsupported line: " + line)
    return n, gates


def _merge(X, Z, C, threshold):
    """Combine identical Pauli strings and drop small coefficients."""
    key = X | (Z << np.uint64(32))
    uniq, inv = np.unique(key, return_inverse=True)
    coeff = np.zeros(len(uniq))
    np.add.at(coeff, inv, C)
    keep = np.abs(coeff) > max(threshold, 1e-14)
    uniq = uniq[keep]
    return uniq & np.uint64(0xFFFFFFFF), uniq >> np.uint64(32), coeff[keep]


def expect_z(n, gates, q, threshold=0.0, max_terms=None):
    """Return (<Z_q>, peak number of Pauli terms). Supports up to 32 qubits per mask half."""
    one = np.uint64(1)
    X = np.array([0], np.uint64)
    Z = np.array([1 << q], np.uint64)
    C = np.array([1.0])
    peak = 1
    for g in reversed(gates):
        if g[0] == "cz":
            a, b = np.uint64(g[1]), np.uint64(g[2])
            xa, xb = (X >> a) & one, (X >> b) & one
            za, zb = (Z >> a) & one, (Z >> b) & one
            C = np.where((xa & xb & (za ^ zb)).astype(bool), -C, C)
            Z = Z ^ (xb << a) ^ (xa << b)
            continue

        qq, M = np.uint64(g[1]), g[2]
        xq, zq = (X >> qq) & one, (Z >> qq) & one
        # index of the Pauli on qubit q: -1 = I, 0 = X, 1 = Y, 2 = Z
        idx = np.where(xq == 0, np.where(zq == 0, -1, 2), np.where(zq == 0, 0, 1))
        act = idx >= 0
        if not act.any():
            continue
        bx, bz = X & ~(one << qq), Z & ~(one << qq)
        nX, nZ, nC = [X[~act]], [Z[~act]], [C[~act]]
        for b in range(3):
            coef = M[b, idx[act]] * C[act]
            m = coef != 0
            if m.any():
                xb_, zb_ = XZ_BITS[b]
                nX.append(bx[act][m] | (np.uint64(xb_) << qq))
                nZ.append(bz[act][m] | (np.uint64(zb_) << qq))
                nC.append(coef[m])
        X, Z, C = np.concatenate(nX), np.concatenate(nZ), np.concatenate(nC)
        if (M != 0).sum(axis=0).max() > 1:          # gate branched: merge + truncate
            X, Z, C = _merge(X, Z, C, threshold)
            if max_terms and len(C) > max_terms:     # memory guard
                keep = np.argsort(-np.abs(C))[:max_terms]
                X, Z, C = X[keep], Z[keep], C[keep]
        peak = max(peak, len(C))
    return float(C[X == 0].sum()), peak


def main():
    folder, out = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    max_terms = int(sys.argv[4]) if len(sys.argv) > 4 else None
    files = sorted(glob.glob(os.path.join(folder, "*.qasm")),
                   key=lambda p: int(p.rsplit("_", 1)[1][:-5]))
    rows = []
    for f in files:
        cid = os.path.basename(f)[:-5]
        q = int(cid.rsplit("_", 1)[1])
        t0 = time.time()
        n, gates = parse_qasm(f)
        val, peak = expect_z(n, gates, q, threshold, max_terms)
        rows.append((cid, val))
        print(f"{cid}: {val:+.6f}  (peak {peak} terms, {time.time() - t0:.1f}s)", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["circuit_id", "value"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
