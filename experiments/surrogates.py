"""
Surrogate echo circuits with the same structure as the challenge circuits,
small enough to simulate exactly, used to calibrate approximate methods.

Echo circuit (as in the challenge):  U, X_b, U^dagger, Z_q, U, X_b, U^dagger
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary


def brickwork_1d(n, layers, seed=7):
    """1D chain of Haar-random two-qubit gates (G1/G2-like)."""
    rng = np.random.default_rng(seed)
    U = QuantumCircuit(n)
    for layer in range(layers):
        for i in range(layer % 2, n - 1, 2):
            U.unitary(random_unitary(4, seed=int(rng.integers(1 << 30))), [i, i + 1])
    return transpile(U, basis_gates=["u3", "cz"], optimization_level=1)


def brickwork_2d(rows, cols, layers, seed=3):
    """2D grid of Haar-random two-qubit gates, 4 alternating edge patterns (G4-like)."""
    rng = np.random.default_rng(seed)
    patterns = []
    for dr, dc, off in [(0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1)]:
        p = []
        for r in range(rows):
            for c in range(cols):
                if dc and c % 2 == off and c + 1 < cols:
                    p.append((r * cols + c, r * cols + c + 1))
                if dr and r % 2 == off and r + 1 < rows:
                    p.append((r * cols + c, (r + 1) * cols + c))
        patterns.append(p)
    U = QuantumCircuit(rows * cols)
    for layer in range(layers):
        for a, b in patterns[layer % 4]:
            U.unitary(random_unitary(4, seed=int(rng.integers(1 << 30))), [a, b])
    return transpile(U, basis_gates=["u3", "cz"], optimization_level=1)


def echo(U, butterfly, q):
    n = U.num_qubits
    qc = QuantumCircuit(n)
    qc.compose(U, inplace=True); qc.x(butterfly); qc.compose(U.inverse(), inplace=True)
    qc.z(q)
    qc.compose(U, inplace=True); qc.x(butterfly); qc.compose(U.inverse(), inplace=True)
    return qc
