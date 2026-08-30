"""
Stage 07 - Single-instance comparison: brute force vs. relaxation vs.
exact-QUBO vs. QAOA.
==========================================================================
Runs all four solvers (stages 04 and 06) on ONE (mu, sigma, k, q) instance
and packages a single comparable result. Stage 08 calls this in a loop
across problem sizes and random seeds to build the scaling study; this
module is deliberately kept single-instance so it can also be called
directly by the dashboard for an interactive "run it now" demo without
paying for a full multi-seed sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from markowitz import brute_force_exact, continuous_relaxation_rounded, objective, ClassicalResult
from qiskit_qubo import build_qubo
from exact_qubo_solver import solve_exact_qubo
from qaoa_solver import solve_qaoa, QAOARunResult


@dataclass
class InstanceComparison:
    n: int
    k: int
    q: float
    brute_force: ClassicalResult
    relaxation: Optional[ClassicalResult]
    exact_qubo_true_val: float
    exact_qubo_feasible: bool
    qaoa_true_val: float
    qaoa_feasible: bool
    qaoa: QAOARunResult
    approx_ratio_qaoa: float
    approx_ratio_exact_qubo: float
    approx_ratio_relaxation: Optional[float]


def run_comparison(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    reps: int = 2, seed: int = 42,
) -> InstanceComparison:
    bf = brute_force_exact(mu, sigma, k, q)

    relaxed = continuous_relaxation_rounded(mu, sigma, k, q)

    qp = build_qubo(mu, sigma, k, q)

    exact_qubo_result = solve_exact_qubo(qp)
    x_exact = np.array(exact_qubo_result.x)
    exact_true_val = objective(mu, sigma, x_exact, q)
    exact_feasible = bool(np.isclose(x_exact.sum(), k))

    qaoa_run = solve_qaoa(qp, reps=reps, seed=seed)
    qaoa_feasible = bool(np.isclose(qaoa_run.x.sum(), k))
    qaoa_true_val = objective(mu, sigma, qaoa_run.x, q)

    def ratio(val: float, feasible: bool) -> float:
        return val / bf.value if feasible and bf.value != 0 else float("nan")

    return InstanceComparison(
        n=len(mu), k=k, q=q,
        brute_force=bf,
        relaxation=relaxed,
        exact_qubo_true_val=exact_true_val, exact_qubo_feasible=exact_feasible,
        qaoa_true_val=qaoa_true_val, qaoa_feasible=qaoa_feasible, qaoa=qaoa_run,
        approx_ratio_qaoa=ratio(qaoa_true_val, qaoa_feasible),
        approx_ratio_exact_qubo=ratio(exact_true_val, exact_feasible),
        approx_ratio_relaxation=ratio(relaxed.value, True) if relaxed else None,
    )
