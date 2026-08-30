"""
Stage 11 - A harder, more realistic constrained variant: per-sector caps.
==========================================================================
Motivation (per the literature review synthesized on the dashboard's "Is
There An Edge" page): the vanilla single-cardinality-constraint problem
used everywhere else in this project (sum(x) = k) is the case where the
literature is most unanimous that QAOA shows no edge - Brandhofer et al.,
Aggarwal et al., and Yalovetzky et al. all test something in this family
and none find an advantage. Uotila et al.'s more speculative argument is
that a HARDER constraint structure - one a continuous relaxation handles
poorly and that adds genuine ruggedness to the penalized QUBO landscape -
is a more plausible place to look, even though their own QAOA runs on such
a formulation were the weakest method in their benchmark. This stage adds
one such harder structure: a SECOND constraint family (a maximum number of
assets per "sector") on top of the existing budget constraint, which is
exactly the kind of discrete, non-relaxation-friendly rule real portfolios
carry (concentration limits) and that Gemini_search_why_qaoa.txt's own
"discrete constraint" argument for QAOA is about.

Two independent penalty terms (budget equality + per-sector inequalities)
instead of one make the unconstrained QUBO landscape more rugged - directly
matching what Uotila et al. (Fig. 5c) show happens when constraint
complexity grows. Whether QAOA's relative standing (vs. brute force / SA)
gets worse, better, or unchanged as this ruggedness increases is exactly
the open, honestly-reported question this stage generates data for.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, OptimizationResult

from markowitz import objective, ClassicalResult


def assign_sectors(n_assets: int, n_sectors: int, seed: int) -> np.ndarray:
    """Deterministic, roughly-balanced synthetic sector labels (0..n_sectors-1)
    for the synthetic scaling-study universes, which have no real ticker/GICS
    metadata to draw sectors from. Real equity/crypto runs (stage 01) could
    swap this for actual sector membership without touching anything below."""
    rng = np.random.default_rng(seed + 7919)  # offset so sectors don't correlate with mu/Sigma draw
    labels = np.array([i % n_sectors for i in range(n_assets)])
    rng.shuffle(labels)
    return labels


def build_sector_constrained_qubo(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    sectors: np.ndarray, sector_cap: int,
) -> QuadraticProgram:
    """maximize  mu^T x - q * x^T Sigma x
       s.t.      sum(x) = k                              (original budget constraint)
                 sum(x_i for i in sector s) <= sector_cap  for every sector s

    Built directly with qiskit-optimization's QuadraticProgram (rather than
    qiskit-finance's PortfolioOptimization, which only expresses the single
    budget equality) - MinimumEigenOptimizer's converter chain still handles
    the equality-to-penalty AND inequality-to-penalty conversion automatically
    from here, so the "use the library, don't hand-roll the converter" rule
    from stage 05 still holds."""
    n = len(mu)
    qp = QuadraticProgram(name="sector_constrained_portfolio")
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(mu[i]) for i in range(n)}
    quadratic = {
        (f"x{i}", f"x{j}"): float(-q * sigma[i, j])
        for i in range(n) for j in range(n)
    }
    qp.maximize(linear=linear, quadratic=quadratic)

    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(n)}, sense="==", rhs=k, name="budget",
    )
    for s in sorted(set(sectors.tolist())):
        members = [f"x{i}" for i in range(n) if sectors[i] == s]
        qp.linear_constraint(
            linear={m: 1 for m in members}, sense="<=", rhs=sector_cap, name=f"sector_{s}_cap",
        )
    return qp


def brute_force_sector_constrained(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    sectors: np.ndarray, sector_cap: int,
) -> ClassicalResult:
    """Same exhaustive search as markowitz.brute_force_exact, with an added
    per-sector feasibility filter. Still exact - the extra constraint makes
    the FEASIBLE region smaller (a strict subset of the budget-only case),
    it doesn't change the O(C(n,k)) search cost, since every size-k subset
    is still enumerated and simply skipped if any sector cap is violated."""
    n = len(mu)
    t0 = time.time()
    best_val, best_set = -np.inf, None
    for subset in itertools.combinations(range(n), k):
        counts = np.bincount(sectors[list(subset)], minlength=sectors.max() + 1)
        if np.any(counts > sector_cap):
            continue
        x = np.zeros(n)
        x[list(subset)] = 1
        val = objective(mu, sigma, x, q)
        if val > best_val:
            best_val, best_set = val, subset
    elapsed = time.time() - t0
    if best_set is None:
        return ClassicalResult("brute_force_sector_constrained", float("nan"), (), False, elapsed)
    return ClassicalResult("brute_force_sector_constrained", best_val, best_set, True, elapsed)


def is_sector_feasible(x: np.ndarray, k: int, sectors: np.ndarray, sector_cap: int) -> bool:
    if not np.isclose(x.sum(), k):
        return False
    counts = np.bincount(sectors[x.astype(bool)], minlength=sectors.max() + 1)
    return bool(np.all(counts <= sector_cap))


@dataclass
class SectorHardnessResult:
    n_assets: int
    k: int
    n_sectors: int
    sector_cap: int
    bf_val: float
    bf_feasible: bool
    bf_time_s: float
    exact_qubo_val: float
    exact_qubo_feasible: bool
    qaoa_val: float
    qaoa_feasible: bool
    qaoa_converged: bool
    qaoa_time_s: float
    approx_ratio_qaoa: float
    approx_ratio_exact_qubo: float
