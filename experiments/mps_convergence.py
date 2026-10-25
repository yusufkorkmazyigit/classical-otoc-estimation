#!/usr/bin/env python3
"""
How well does truncated MPS reproduce exact <Z_q> on surrogate circuits?
Reports corr^2 (the challenge score fraction) per bond dimension, plus the
bond-extrapolated estimate  v = v_hi - 0.5 * (v_lo - v_hi).

    python mps_convergence.py 1d 22 12 11   16 32 64 128     # n, layers, butterfly, bonds...
    python mps_convergence.py 2d 4x5 12 0   8 16 32 64       # grid, layers, butterfly, bonds...
"""
import sys, time

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from surrogates import brickwork_1d, brickwork_2d, echo


def values(U, b, method, bond=None):
    n = U.num_qubits
    opts = dict(method=method, max_memory_mb=10**6)
    if bond:
        opts.update(matrix_product_state_max_bond_dimension=bond,
                    matrix_product_state_truncation_threshold=1e-12)
    sim = AerSimulator(**opts)
    out = []
    for q in range(n):
        qc = echo(U, b, q)
        qc.save_expectation_value(SparsePauliOp("I" * (n - 1 - q) + "Z" + "I" * q), range(n))
        out.append(float(np.real(sim.run(transpile(qc, sim)).result().data()["expectation_value"])))
    return np.array(out)


if __name__ == "__main__":
    kind, size, layers, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    bonds = [int(x) for x in sys.argv[5:]]
    if kind == "1d":
        U = brickwork_1d(int(size), layers)
    else:
        r, c = map(int, size.split("x"))
        U = brickwork_2d(r, c, layers)
    exact = values(U, b, "statevector")
    print("exact:", np.round(exact, 3))
    prev = None
    for bond in bonds:
        t0 = time.time()
        v = values(U, b, "matrix_product_state", bond)
        line = f"bond {bond:4d}: corr^2 = {np.corrcoef(exact, v)[0, 1] ** 2:.4f}  ({time.time() - t0:.0f}s)"
        if prev is not None:
            e = v - 0.5 * (prev - v)
            line += f" | extrapolated corr^2 = {np.corrcoef(exact, e)[0, 1] ** 2:.4f}"
        print(line, flush=True)
        prev = v
