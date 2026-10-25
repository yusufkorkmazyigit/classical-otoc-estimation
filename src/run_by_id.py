#!/usr/bin/env python3
"""
Run the hidden challenge circuits on BlueQubit by ID and write a submission CSV.

    python run_by_id.py G1 --device gpu --batch 1
    python run_by_id.py G2 --device mps.gpu --bond 64 --batch 1
    python run_by_id.py G3 --device pauli-path --thr 0.001
    python run_by_id.py G2 --device mps.gpu --bond 32 --qubits 0-9   # subset first

* Jobs are submitted asynchronously in batches, then awaited.
* Results are saved after every batch; re-running the same command resumes.
* If the platform's parallel-job limit is hit, submission is retried.
Requires BLUEQUBIT_API_TOKEN (preset inside BlueQubit Hybrid notebooks).
"""
import argparse, csv, os, sys, time

import numpy as np


def parse_qubits(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [q for q in out if q < n]


def load_csv(path):
    if not os.path.exists(path):
        return {}
    return {r[0]: float(r[1]) for r in csv.reader(open(path)) if r and r[0] != "circuit_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")                         # S1..S3, G1..G4
    ap.add_argument("--device", default="cpu")         # cpu gpu mps.cpu mps.gpu pauli-path
    ap.add_argument("--bond", type=int)                # mps_bond_dimension
    ap.add_argument("--thr", type=float)               # pauli_path_truncation_threshold
    ap.add_argument("--qubits", default="")            # e.g. "0-9" or "0,5,12"
    ap.add_argument("--batch", type=int, default=20)   # jobs in flight at once
    a = ap.parse_args()

    import bluequbit
    bq = bluequbit.init(execution_mode="cloud")        # circuit IDs only run in the cloud

    n = bq.get_circuit_info(f"Duke_{a.circuit}_0").num_qubits
    opts, tag = {}, a.device
    if a.bond:
        opts["mps_bond_dimension"] = a.bond
        tag += f"_b{a.bond}"
    if a.thr is not None:
        opts["pauli_path_truncation_threshold"] = a.thr
        tag += f"_t{a.thr}"
    out = f"{a.circuit}_{tag}.csv"
    done = load_csv(out)
    todo = [q for q in parse_qubits(a.qubits, n) if f"Duke_{a.circuit}_{q}" not in done]
    print(f"{a.circuit}: {n} qubits | {a.device} {opts} | {len(todo)} to run -> {out}")

    def save():
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit_id", "value"])
            for k in sorted(done, key=lambda s: int(s.rsplit("_", 1)[1])):
                w.writerow([k, done[k]])

    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        jobs = {}
        for q in todo[i:i + a.batch]:
            cid = f"Duke_{a.circuit}_{q}"
            z = "I" * (n - 1 - q) + "Z" + "I" * q          # qubit 0 = rightmost character
            kw = dict(device=a.device, pauli_sum=[(z, 1.0)], asynchronous=True,
                      job_name=f"{a.circuit}_{q}_{tag}")
            if opts:
                kw["options"] = opts
            for attempt in range(60):
                try:
                    jobs[cid] = bq.run(cid, **kw)
                    break
                except Exception as e:                     # parallel-job limit: wait, retry
                    if "PARALLEL" in str(e) and attempt < 59:
                        time.sleep(10)
                        continue
                    print(f"  {cid}: submit failed: {e}")
                    break
        for cid, job in jobs.items():
            try:
                done[cid] = float(bq.wait(job.job_id).expectation_value)
                print(f"  {cid}: {done[cid]:+.6f}   ({time.time() - t0:.0f}s)")
            except Exception as e:
                print(f"  {cid}: ERROR: {e}")
        save()

    print(f"done: {out} ({len(done)}/{n} IDs)")
    if len(done) == n and np.std(list(done.values())) < 1e-12:
        print("warning: all values equal -> Pearson correlation undefined (scores 0)")


if __name__ == "__main__":
    main()
