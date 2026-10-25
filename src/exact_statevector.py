#!/usr/bin/env python3
"""Exact <Z_q> for every QASM file in a folder (statevector; fine up to ~28 qubits locally).

Usage: python exact_statevector.py <folder_with_qasm> <out.csv>
"""
import csv, glob, os, sys
from qiskit import qasm2
from qiskit.quantum_info import SparsePauliOp, Statevector


def z_string(n, q):
    # Qiskit Pauli strings are little-endian: qubit 0 is the rightmost character
    return "I" * (n - 1 - q) + "Z" + "I" * q


def main():
    folder, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(folder, "*.qasm")),
                   key=lambda p: int(p.rsplit("_", 1)[1][:-5]))
    rows = []
    for f in files:
        cid = os.path.basename(f)[:-5]
        q = int(cid.rsplit("_", 1)[1])
        qc = qasm2.load(f)
        val = Statevector(qc).expectation_value(SparsePauliOp(z_string(qc.num_qubits, q))).real
        rows.append((cid, val))
        print(f"{cid}: {val:+.6f}")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["circuit_id", "value"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
