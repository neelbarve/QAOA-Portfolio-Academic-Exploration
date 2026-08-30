"""
Stage 14b - Warm-start QAOA vs. the same sizes stage 12 already tested.
=============================================================================
Runs solve_warm_start_qaoa at the EXACT (n, seed, q, reps) combinations
stage 12's extended_scaling sweep used, so the two are directly comparable:
same synthetic instance, same QUBO, same optimizer/shots/maxiter, only the
mixer and initial state differ. brute-force/SA/standard-QAOA numbers are not
recomputed here (already in results/extended_scaling.json, and standard
QAOA alone costs minutes per size) - scripts/run_warm_start_comparison.py
merges this sweep's output with that existing file into one comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from markowitz import objective
from warm_start_qaoa import solve_warm_start_qaoa
from scaling_benchmark import make_synthetic_universe


@dataclass
class WarmStartSizeResult:
    n_assets: int
    k: int
    warm_start_available: bool
    warm_val: float
    warm_feasible: bool
    warm_converged: bool
    warm_time_s: float


def run_warm_start_scaling(
    sizes: List[int], q: float = 0.5, reps: int = 2, seed: int = 42,
) -> List[WarmStartSizeResult]:
    rows: List[WarmStartSizeResult] = []
    for n in sizes:
        k = n // 2
        mu, sigma = make_synthetic_universe(n, seed=seed)

        res = solve_warm_start_qaoa(mu, sigma, k, q, reps=reps, seed=seed)
        x = res.qaoa.x
        feasible = bool(np.isclose(x.sum(), k))
        val = objective(mu, sigma, x, q)

        rows.append(WarmStartSizeResult(
            n_assets=n, k=k, warm_start_available=res.warm_start.available,
            warm_val=val, warm_feasible=feasible, warm_converged=res.qaoa.converged,
            warm_time_s=res.qaoa.elapsed_s,
        ))
        r = rows[-1]
        conv_note = "" if r.warm_converged else " [DID NOT CONVERGE]"
        print(f"n={n:2d} k={k:2d} | warm-start QAOA={val:.4f} feas={feasible} "
              f"({r.warm_time_s:.2f}s){conv_note}")
    return rows


def results_to_dicts(rows: List[WarmStartSizeResult]) -> List[dict]:
    return [asdict(r) for r in rows]
