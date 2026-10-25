# Classical OTOC Estimation — Duke Quantum Hackathon 2026

My entry to the **BlueQubit × DuQuantum (Duke University) Quantum Hackathon**, 24–25 October 2026 (24 hours, online).

The challenge was built around Google's *Quantum Echoes* experiment ([Nature, 2025](https://doi.org/10.1038/s41586-025-09526-6)). Google measured out-of-time-order correlators (OTOCs) on its Willow chip and reported a verifiable quantum advantage. Participants took the classical side: estimate those OTOCs with classical simulators, for circuits designed to be hard to simulate.

## The task

Each problem is an echo circuit `U · X_b · U† · Z_q · U · X_b · U†` on *n* qubits, where *b* is the "butterfly" qubit. The value to estimate for each qubit *q* is ⟨Z_q⟩ after running the circuit from |0…0⟩.

The score for each problem is `corr² × points`, where `corr` is the Pearson correlation with the exact values. Scale and offset don't matter, but sign does.

| Problem | Qubits | Layout | Gates in U | Points |
|---|---|---|---|---|
| S1 | 4 | 1D chain | Haar-random | 100 |
| S2 | 16 | 1D chain | Haar-random | 100 |
| S3 | 30 | 5×6 grid | Clifford + some Rz | 100 |
| G1 | 30 | 1D chain, 32 layers | Haar-random | 300 |
| G2 | 36 | 1D chain, butterfly in middle | Haar-random | 600 |
| G3 | 36 | 6×6 grid | Clifford + some Rz | 800 |
| G4 | 38 | 6×6 grid + 2 | Haar-random | 1000 |

The S circuits came with open QASM. The G circuits were hidden and could only be run on BlueQubit's cloud by ID, using a $30 hackathon credit.

## Approach

The main idea: **calibrate before spending.** Each simulation method has a knob (bond dimension, truncation threshold). For every hidden circuit, I first built or used a circuit with the *same structure* whose exact answer I could compute. I measured which method and setting would score well on it, and only then spent credit on the graded circuit.

### 1. Exact statevector — S1, S2, G1
Up to ~30 qubits the full state fits in memory, so the result is exact. G1 (30 qubits) ran on BlueQubit's GPU at about $0.20 per qubit.

### 2. Own Pauli-path simulator — S3 ([`src/pauli_path.py`](src/pauli_path.py))
This method works in the Heisenberg picture. It propagates the observable Z_q backwards through the circuit as a sum of Pauli strings, each stored as two bitmasks.
- **Clifford gates** (CZ and most single-qubit gates in S3/G3) map one Pauli string to exactly one string. They never grow the sum.
- **Non-Clifford rotations** split a term into up to three.
- **Small coefficients are dropped.**

Because the S3 and G3 circuits are mostly Clifford, the number of terms stays manageable. The simulator matches exact statevector results to machine precision, and calibrating the threshold on S3 gave:

| Truncation threshold | 0.1 | 0.03 | 0.01 | 0.003 | 0.001 | 0.0003 |
|---|---|---|---|---|---|---|
| corr² vs exact (S3) | 0.21 | 0.68 | 0.91 | 0.992 | 0.9994 | 1.0000 |

### 3. MPS with bond-dimension extrapolation — G2
Matrix product states handle 1D circuits well. On a G2-like surrogate (22-qubit chain), corr² rose with bond dimension χ:

| χ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| corr² (surrogate) | 0.76 | 0.93 | 0.991 | 0.9996 |

Truncated MPS systematically **overestimates** ⟨Z⟩ in the scrambling front, because truncation removes entanglement. So I ran G2 at χ = 32 and χ = 64 and extrapolated, `v = v₆₄ − k·(v₃₂ − v₆₄)`. This took G2 from **426 → 519 / 600** with no extra simulation cost.

### 4. What did not work — and why
- **MPS on 2D grids (S3/G3/G4).** A 2D grid flattened into a 1D MPS needs far more entanglement. On S3 itself, MPS scored corr² ≈ 0.19 at χ = 16 and ≈ 0 at χ = 64. On a fully scrambled 2D Haar surrogate (G4-like), every bond dimension up to 64 gave corr² ≈ 0.
- **G3.** On BlueQubit, Pauli-path for circuits over 1000 gates requires an account balance of at least $25. After G1/G2 the hackathon credit was below that, so G3 could not be run. The method calibrated on S3 (threshold 0.001 → corr² 0.999) was ready but went unused.
- **G4.** A 38-qubit Haar-random 2D circuit, scrambled from the corner. This is exactly the regime where Google claims quantum advantage. MPS (χ = 16) and simple distance heuristics all scored ≈ 0, consistent with the surrogate experiments.

## Results

| Problem | Method | Score |
|---|---|---|
| S1 | Exact statevector | 100 / 100 |
| S2 | Exact statevector | 100 / 100 |
| S3 | Own Pauli-path simulator | _TBD_ / 100 |
| G1 | Exact statevector (BlueQubit GPU) | 300 / 300 |
| G2 | MPS χ = 32/64 + extrapolation | 519 / 600 |
| G3 | — (blocked by credit minimum) | 0 / 800 |
| G4 | MPS χ = 16, heuristics | ≈ 0 / 1000 |

## Repository layout

```
src/
  pauli_path.py          Heisenberg-picture Pauli-path simulator (u3/cz/x/z)
  exact_statevector.py   exact <Z_q> for small circuits (Qiskit)
  run_by_id.py           run hidden circuits on BlueQubit by ID -> submission CSV
  combine.py             bond-dimension extrapolation, CSV correlation
experiments/
  surrogates.py          1D / 2D Haar echo circuits with the challenge's structure
  mps_convergence.py     MPS accuracy vs bond dimension on surrogates
  pauli_path_threshold.py  threshold calibration against exact values
```

The challenge circuits and answers are not included.

```bash
pip install -r requirements.txt
python src/pauli_path.py path/to/qasm_folder out.csv 0.001
python experiments/mps_convergence.py 1d 22 12 11  16 32 64
python experiments/mps_convergence.py 2d 4x5 12 0  8 16 32 64
```

## What I learned

- Which classical method works is set by the **structure** of the circuit, not its size. A 1D chain is easy for MPS; a mostly-Clifford circuit is easy for Pauli propagation; a scrambled 2D Haar circuit defeats both.
- Validating an approximate method on cheap surrogates with known answers before spending compute is the same discipline as using a validation set in ML.
- Systematic errors can be corrected cheaply. Bond-dimension extrapolation recovered a large part of G2's gap with no extra simulation.

---
Built during the hackathon with AI assistance (Claude), which the organizers explicitly encouraged. MIT License.
