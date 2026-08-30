"""
Stage 12b - Extended-n scaling sweep, with SA as the competitive classical
baseline instead of (only) brute force.
=============================================================================
Stage 08's sweep stops at n=12-14 because that's the range brute force is
still both exact AND fast enough to use as ground truth for every seed.
This sweep pushes further, to sizes where brute force is either skipped
entirely or run once (not per-seed) as an expensive reference point, and
where the SCIENTIFICALLY MEANINGFUL comparison for "is there an edge" is
QAOA vs. SA - not QAOA vs. an exact solver that stops being available.

Two honestly-reported caveats baked into the output:
  - Above `exact_cutoff_n`, `bf_val`/`bf_feasible` are None - there is no
    known optimum to normalize against, so approximation ratios there are
    computed relative to SA's result instead (labelled accordingly), not
    silently reused from a smaller n.
  - QAOA's own simulation cost is exponential in n (statevector size), so
    this sweep is deliberately capped at a size chosen to keep total run
    time reasonable - it is not evidence that QAOA "stops scaling" beyond
    the cap, only that a laptop-simulator study cannot see further, exactly
    the honest limitation already documented in stage 08/09's write-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from markowitz import brute_force_exact, objective
from qiskit_qubo import build_qubo
from exact_qubo_solver import solve_exact_qubo
from qaoa_solver import solve_qaoa
from simulated_annealing import simulated_annealing
from scaling_benchmark import make_synthetic_universe


@dataclass
class ExtendedSizeResult:
    n_assets: int
    k: int
    bf_val: Optional[float]      # None above exact_cutoff_n
    bf_time_s: Optional[float]
    sa_val: float
    sa_time_s: float
    qaoa_val: float
    qaoa_feasible: bool
    qaoa_converged: bool
    qaoa_time_s: float
    ratio_qaoa_vs_bf: Optional[float]     # QAOA / true optimum, where known
    ratio_qaoa_vs_sa: float               # QAOA / SA - always available, the honest headline number
    ratio_sa_vs_bf: Optional[float]       # SA / true optimum, where known


def run_extended_scaling(
    sizes: List[int], exact_cutoff_n: int = 20, q: float = 0.5,
    reps: int = 2, seed: int = 42, sa_iters: int = 6000,
) -> List[ExtendedSizeResult]:
    rows: List[ExtendedSizeResult] = []
    for n in sizes:
        k = n // 2
        mu, sigma = make_synthetic_universe(n, seed=seed)

        bf_val = bf_time = None
        if n <= exact_cutoff_n:
            bf = brute_force_exact(mu, sigma, k, q)
            bf_val, bf_time = bf.value, bf.elapsed_s

        sa = simulated_annealing(mu, sigma, k, q, n_iters=sa_iters, seed=seed)

        qp = build_qubo(mu, sigma, k, q)
        qaoa_run = solve_qaoa(qp, reps=reps, seed=seed)
        qaoa_feasible = bool(np.isclose(qaoa_run.x.sum(), k))
        qaoa_val = objective(mu, sigma, qaoa_run.x, q)

        ratio_bf = (qaoa_val / bf_val) if (bf_val and qaoa_feasible and bf_val != 0) else None
        ratio_sa_bf = (sa.value / bf_val) if (bf_val and bf_val != 0) else None
        ratio_sa = (qaoa_val / sa.value) if (qaoa_feasible and sa.value != 0) else float("nan")

        rows.append(ExtendedSizeResult(
            n_assets=n, k=k, bf_val=bf_val, bf_time_s=bf_time,
            sa_val=sa.value, sa_time_s=sa.elapsed_s,
            qaoa_val=qaoa_val, qaoa_feasible=qaoa_feasible, qaoa_converged=qaoa_run.converged,
            qaoa_time_s=qaoa_run.elapsed_s,
            ratio_qaoa_vs_bf=ratio_bf, ratio_qaoa_vs_sa=ratio_sa, ratio_sa_vs_bf=ratio_sa_bf,
        ))
        r = rows[-1]
        bf_str = f"{bf_val:.4f}" if bf_val is not None else "n/a (past exact cutoff)"
        conv_note = "" if r.qaoa_converged else " [QAOA DID NOT CONVERGE - near-uniform output]"
        print(f"n={n:2d} k={k:2d} | brute-force={bf_str} | SA={sa.value:.4f} ({sa.elapsed_s:.2f}s) | "
              f"QAOA={qaoa_val:.4f} feas={qaoa_feasible} ({qaoa_run.elapsed_s:.2f}s){conv_note} | "
              f"QAOA/SA={ratio_sa:.4f}")
    return rows


def results_to_dicts(rows: List[ExtendedSizeResult]) -> List[dict]:
    return [asdict(r) for r in rows]
