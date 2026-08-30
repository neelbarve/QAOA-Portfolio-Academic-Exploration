"""
Stage 11b - Hardness sweep: fixed n, tightening sector caps.
==================================================================
Holds problem size fixed and turns up ONLY constraint hardness (more
sectors, tighter per-sector caps -> a smaller feasible region and a more
rugged two-penalty-term QUBO landscape), to isolate the effect Uotila et
al. attribute to constraint complexity from the effect of qubit count
alone (already covered by stage 08's n-sweep). n_sectors=1 with a cap of k
reduces to the original unconstrained-by-sector problem (sanity check: the
sector-constrained results at n_sectors=1 should match the stage 08/07
budget-only results for the same (n, k, seed)).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

from sector_constrained import (
    assign_sectors, build_sector_constrained_qubo, brute_force_sector_constrained,
    is_sector_feasible, SectorHardnessResult,
)
from exact_qubo_solver import solve_exact_qubo
from qaoa_solver import solve_qaoa
from markowitz import objective
from scaling_benchmark import make_synthetic_universe


def run_hardness_sweep(
    n_assets: int, n_sectors_list: List[int], q: float = 0.5,
    reps: int = 2, seed: int = 42,
) -> List[SectorHardnessResult]:
    k = n_assets // 2
    mu, sigma = make_synthetic_universe(n_assets, seed=seed)
    rows: List[SectorHardnessResult] = []

    for n_sectors in n_sectors_list:
        sectors = assign_sectors(n_assets, n_sectors, seed=seed)
        # Tightest cap that still leaves the problem feasible for a
        # roughly-balanced sector split: ceil(k / n_sectors), so hardness
        # increases monotonically with n_sectors without ever making the
        # instance infeasible outright.
        sector_cap = max(1, -(-k // n_sectors))

        bf = brute_force_sector_constrained(mu, sigma, k, q, sectors, sector_cap)

        qp = build_sector_constrained_qubo(mu, sigma, k, q, sectors, sector_cap)
        exact_result = solve_exact_qubo(qp)
        x_exact = exact_result.x
        exact_val = objective(mu, sigma, x_exact, q)
        exact_feasible = is_sector_feasible(x_exact, k, sectors, sector_cap)

        qaoa_run = solve_qaoa(qp, reps=reps, seed=seed)
        qaoa_feasible = is_sector_feasible(qaoa_run.x, k, sectors, sector_cap)
        qaoa_val = objective(mu, sigma, qaoa_run.x, q)

        def ratio(val, feasible):
            return val / bf.value if feasible and bf.feasible and bf.value != 0 else float("nan")

        rows.append(SectorHardnessResult(
            n_assets=n_assets, k=k, n_sectors=n_sectors, sector_cap=sector_cap,
            bf_val=bf.value, bf_feasible=bf.feasible, bf_time_s=bf.elapsed_s,
            exact_qubo_val=exact_val, exact_qubo_feasible=exact_feasible,
            qaoa_val=qaoa_val, qaoa_feasible=qaoa_feasible, qaoa_converged=qaoa_run.converged,
            qaoa_time_s=qaoa_run.elapsed_s,
            approx_ratio_qaoa=ratio(qaoa_val, qaoa_feasible),
            approx_ratio_exact_qubo=ratio(exact_val, exact_feasible),
        ))
        r = rows[-1]
        conv_note = "" if r.qaoa_converged else " [QAOA DID NOT CONVERGE - near-uniform output]"
        print(f"n_sectors={n_sectors:2d} cap={sector_cap} | brute-force={r.bf_val:.4f} feas={r.bf_feasible} | "
              f"exact-QUBO={r.exact_qubo_val:.4f} feas={r.exact_qubo_feasible} | "
              f"QAOA={r.qaoa_val:.4f} feas={r.qaoa_feasible} ({r.qaoa_time_s:.2f}s){conv_note}")
    return rows


def results_to_dicts(rows: List[SectorHardnessResult]) -> List[dict]:
    return [asdict(r) for r in rows]
