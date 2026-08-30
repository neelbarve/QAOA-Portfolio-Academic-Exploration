"""
Stage 08 - Scaling benchmark: solution quality + wall-clock vs. problem
size, across multiple random seeds per size.
==========================================================================
qaoa_v1's original benchmark() ran ONE seed per problem size - fine for a
first pass, but not enough to say anything statistically about how the
QAOA/classical gap changes with n (stage 09 needs a distribution per size,
not a point estimate, to fit a scaling exponent with a confidence interval
or run a significance test). This stage is that multi-seed generalization:
for each n in `sizes`, run `n_seeds` independent random instances and keep
every InstanceComparison, so stage 09 can compute means, bootstrap CIs, and
regression fits directly from the raw per-seed values.

Synthetic instances (make_synthetic_universe) are used for the pure scaling
study, exactly as in qaoa_v1 - real crypto/equity data (stage 01) is used
for the single "run it on my actual universe" pipeline path (pipeline.py),
not for the scaling sweep, since the sweep needs many different n values
and real market universes of every size aren't available on demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from comparison import run_comparison, InstanceComparison
from markowitz import brute_force_exact


def make_synthetic_universe(n_assets: int, seed: int):
    """Same construction as qaoa_v1/portfolio_qaoa.py's make_universe:
    random mu, and a random covariance built as A@A.T/n (guarantees PSD)
    then rescaled to a sane volatility range."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(loc=0.08, scale=0.05, size=n_assets)
    A = rng.normal(size=(n_assets, n_assets))
    sigma = A @ A.T / n_assets
    sigma = sigma / np.max(np.abs(sigma)) * 0.05
    return mu, sigma


@dataclass
class SizeSeedResult:
    n_assets: int
    seed: int
    k: int
    bf_val: float
    bf_time_s: float
    exact_qubo_val: float
    exact_qubo_feasible: bool
    qaoa_val: float
    qaoa_feasible: bool
    qaoa_time_s: float
    approx_ratio_qaoa: float
    approx_ratio_exact_qubo: float


def run_scaling_benchmark(
    sizes: List[int], n_seeds: int = 10, q: float = 0.5, reps: int = 2, base_seed: int = 42,
) -> List[SizeSeedResult]:
    rows: List[SizeSeedResult] = []
    for n in sizes:
        k = n // 2
        for trial in range(n_seeds):
            seed = base_seed * 1000 + trial  # deterministic, distinct per (n, trial)
            mu, sigma = make_synthetic_universe(n, seed=seed)
            cmp: InstanceComparison = run_comparison(mu, sigma, k, q, reps=reps, seed=seed)
            rows.append(SizeSeedResult(
                n_assets=n, seed=seed, k=k,
                bf_val=cmp.brute_force.value, bf_time_s=cmp.brute_force.elapsed_s,
                exact_qubo_val=cmp.exact_qubo_true_val, exact_qubo_feasible=cmp.exact_qubo_feasible,
                qaoa_val=cmp.qaoa_true_val, qaoa_feasible=cmp.qaoa_feasible,
                qaoa_time_s=cmp.qaoa.elapsed_s,
                approx_ratio_qaoa=cmp.approx_ratio_qaoa,
                approx_ratio_exact_qubo=cmp.approx_ratio_exact_qubo,
            ))
            r = rows[-1]
            print(f"n={n:2d} seed_trial={trial:2d} k={k:2d} | brute-force={r.bf_val:.4f} | "
                  f"exact-QUBO={r.exact_qubo_val:.4f} feas={r.exact_qubo_feasible} | "
                  f"QAOA={r.qaoa_val:.4f} feas={r.qaoa_feasible} ({r.qaoa_time_s:.2f}s)")
    return rows


def classical_scaling_only(sizes: List[int], q: float = 0.5, seed: int = 42):
    """Brute-force-only runtime scaling, pushed past where statevector
    simulation of QAOA becomes the practical bottleneck on a laptop - the
    crossover itself is part of the honest scaling story (today it's
    simulator cost; on real quantum hardware it would be qubit count and
    noise instead). Unchanged from qaoa_v1/portfolio_qaoa.py."""
    rows = []
    for n in sizes:
        k = n // 2
        mu, sigma = make_synthetic_universe(n, seed=seed)
        bf = brute_force_exact(mu, sigma, k, q)
        rows.append(dict(n_assets=n, k=k, bf_val=bf.value, bf_time_s=bf.elapsed_s))
        print(f"n={n:2d} k={k:2d} | brute-force={bf.value:.4f} ({bf.elapsed_s:.3f} s, "
              f"C(n,k)={math.comb(n, k):,} subsets)")
    return rows


def results_to_dicts(rows: List[SizeSeedResult]) -> List[dict]:
    return [asdict(r) for r in rows]
