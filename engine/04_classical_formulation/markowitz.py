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


def brute_force_exact_vectorized(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float, batch_size: int = 200_000,
) -> ClassicalResult:
    """Same exact search as brute_force_exact - identical answer, guaranteed
    (verified against it in tests/sanity_checks.py) - but batches C(n,k)
    subsets through vectorized numpy calls instead of evaluating the
    objective one Python-level function call at a time.

    brute_force_exact's inner loop pays full Python + array-allocation
    overhead per subset (np.zeros(n), a list->array conversion, two
    small matmuls) C(n,k) times. Here, each batch of subsets becomes ONE
    (batch_size, n) boolean matrix built via a single fancy-index
    assignment, and the whole batch's linear and quadratic objective terms
    are computed in two vectorized calls (a matmul and an einsum) that BLAS
    handles as compiled, multi-threaded C code - the same reason "brute
    force is dominated by simulator cost, not asymptotic complexity" holds
    for classical loops too: a huge constant-factor gap between an
    interpreted per-item loop and a batched array operation, not a change
    in the underlying O(C(n,k)) exponential complexity itself. This does
    not change WHERE the combinatorial wall is - it moves it a few sizes
    further out for the same wall-clock budget."""
    n = len(mu)
    t0 = time.time()
    best_val, best_set = -np.inf, None
    combos_iter = itertools.combinations(range(n), k)
    while True:
        batch = list(itertools.islice(combos_iter, batch_size))
        if not batch:
            break
        b = len(batch)
        rows = np.repeat(np.arange(b), k)
        cols = np.asarray(batch, dtype=np.int64).reshape(-1)
        x = np.zeros((b, n))
        x[rows, cols] = 1.0

        linear = x @ mu
        quad = np.einsum("bi,ij,bj->b", x, sigma, x)
        vals = linear - q * quad

        local_best = int(np.argmax(vals))
        if vals[local_best] > best_val:
            best_val, best_set = float(vals[local_best]), batch[local_best]

    elapsed = time.time() - t0
    return ClassicalResult("brute_force_exact_vectorized", best_val, best_set, True, elapsed)


def _solve_box_relaxation(mu: np.ndarray, sigma: np.ndarray, k: int, q: float) -> Optional[np.ndarray]:
    """Shared CVXPY solve for the box relaxation (0<=x<=1, sum x=k) of the
    same objective. Returns the raw fractional solution, or None if no
    compatible solver is available in this environment."""
    import cvxpy as cp

    n = len(mu)
    x = cp.Variable(n)
    quad = cp.quad_form(x, cp.psd_wrap(sigma))
    prob = cp.Problem(
        cp.Maximize(mu @ x - q * quad),
        [cp.sum(x) == k, x >= 0, x <= 1],
    )
    try:
        prob.solve()
    except Exception:
        return None
    return x.value


def continuous_relaxation_rounded(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float
) -> Optional[ClassicalResult]:
    """CVXPY box relaxation, rounded to the top-k assets by relaxed weight.
    This is the realistic "what would a classical practitioner reach for
    first" baseline - fast but not exact (rounding is not guaranteed optimal
    once the covariance term is involved, which is exactly why the
    cardinality constraint is hard). Returns None (rather than raising) if
    no compatible solver is available, so callers can report "unavailable"
    instead of crashing a benchmark run."""
    n = len(mu)
    t0 = time.time()
    xval = _solve_box_relaxation(mu, sigma, k, q)
    if xval is None:
        return None

    top_k = tuple(sorted(np.argsort(-xval)[:k].tolist()))
    x_rounded = np.zeros(n)
    x_rounded[list(top_k)] = 1
    elapsed = time.time() - t0
    val = objective(mu, sigma, x_rounded, q)
    return ClassicalResult("continuous_relaxation_rounded", val, top_k, True, elapsed)


def continuous_relaxation_fractional(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float
) -> Optional[np.ndarray]:
    """The RAW fractional box-relaxation solution (not rounded to a
    selection) - used by stage 14's warm-start QAOA as the per-qubit bias
    signal (Egger, Marecek & Woerner 2021's warm-starting construction).
    Returns None if the relaxation can't be solved."""
    return _solve_box_relaxation(mu, sigma, k, q)
