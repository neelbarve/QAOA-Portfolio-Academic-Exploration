"""
Stage 04 - Classical cardinality-constrained Markowitz problem.
===================================================================
    maximize    mu^T x - q * x^T Sigma x
    subject to  sum(x) = k        (pick exactly k of n assets)
                x in {0,1}^n

`mu` = expected return per asset, `Sigma` = covariance matrix, `q` =
risk-aversion coefficient, `k` = cardinality (budget) constraint. Dropping
the cardinality constraint turns this into a QP solvable in polynomial
time; forcing exactly k assets makes it a combinatorial search over C(n,k)
subsets - the NP-hard piece this whole project exists to test QAOA against.

Three solvers live here, each answering a different question:
  - brute_force_exact: the true combinatorial optimum. Ground truth for
    everything else, cost O(C(n,k)).
  - continuous_relaxation_rounded: drop x in {0,1}, solve the box-relaxed QP
    (0<=x<=1, sum x=k) via CVXPY in polynomial time, then round to the
    top-k assets by relaxed weight. This is the realistic "what would a
    classical practitioner reach for first" baseline - fast but not exact
    (rounding is not guaranteed optimal once the covariance term is
    involved, which is exactly why the cardinality constraint is hard).
  - A real MIQP solve (branch-and-bound on the exact boolean problem) is
    intentionally NOT attempted with a free/open solver here: HiGHS (the
    open solver installed with this project via cvxpy) only reliably
    handles MIP with linear objectives, not the exact boolean quadratic
    program. A commercial MIQP solver (Gurobi/CPLEX/XPRESS) is the
    honest bar to beat at real scale, per the benchmarking discussion in
    stage 08/09 - not something this project claims to demonstrate,
    stated explicitly rather than silently substituting brute force for it.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ClassicalResult:
    method: str
    value: float
    selection: Tuple[int, ...]     # indices of selected assets
    feasible: bool
    elapsed_s: float


def objective(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, q: float) -> float:
    return float(mu @ x - q * (x @ sigma @ x))


def brute_force_exact(mu: np.ndarray, sigma: np.ndarray, k: int, q: float) -> ClassicalResult:
    n = len(mu)
    t0 = time.time()
    best_val, best_set = -np.inf, None
    for subset in itertools.combinations(range(n), k):
        x = np.zeros(n)
        x[list(subset)] = 1
        val = objective(mu, sigma, x, q)
        if val > best_val:
            best_val, best_set = val, subset
    elapsed = time.time() - t0
    return ClassicalResult("brute_force_exact", best_val, best_set, True, elapsed)


def continuous_relaxation_rounded(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float
) -> Optional[ClassicalResult]:
    """CVXPY box relaxation (0<=x<=1, sum x=k) of the same objective, then
    round to the top-k assets by relaxed weight. Returns None (rather than
    raising) if no compatible solver is available in this environment, so
    callers can report "unavailable" instead of crashing a benchmark run."""
    import cvxpy as cp

    n = len(mu)
    x = cp.Variable(n)
    quad = cp.quad_form(x, cp.psd_wrap(sigma))
    prob = cp.Problem(
        cp.Maximize(mu @ x - q * quad),
        [cp.sum(x) == k, x >= 0, x <= 1],
    )
    t0 = time.time()
    try:
        prob.solve()
    except Exception:
        return None
    if x.value is None:
        return None

    top_k = tuple(sorted(np.argsort(-x.value)[:k].tolist()))
    x_rounded = np.zeros(n)
    x_rounded[list(top_k)] = 1
    elapsed = time.time() - t0
    val = objective(mu, sigma, x_rounded, q)
    return ClassicalResult("continuous_relaxation_rounded", val, top_k, True, elapsed)
