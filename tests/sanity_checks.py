r"""
Sanity checks for manual debugging and knowledge transfer.
==============================================================
Not a full pytest-style unit-test suite - a small set of targeted checks
that catch the specific failure modes this project already ran into once
(silent PSD violations, an infeasible QAOA answer being accepted, the
by-hand Ising derivation silently disagreeing with the qiskit-finance
QUBO). Run directly:

    python tests/sanity_checks.py

Each check prints PASS/FAIL and a one-line reason; exits non-zero if
anything fails, so it's also usable as a lightweight CI gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

import numpy as np  # noqa: E402

from scaling_benchmark import make_synthetic_universe  # noqa: E402
from markowitz import brute_force_exact, brute_force_exact_vectorized, objective  # noqa: E402
from qiskit_qubo import build_qubo  # noqa: E402
from exact_qubo_solver import solve_exact_qubo  # noqa: E402
from manual_ising import to_ising_hamiltonian, find_minimal_penalty  # noqa: E402
from cleaning import preprocess_price_panel, MISSING_DROP_THRESHOLD  # noqa: E402
from base_adapter import AssetUniverseAdapter  # noqa: E402
from sector_constrained import assign_sectors, brute_force_sector_constrained  # noqa: E402
from simulated_annealing import simulated_annealing  # noqa: E402
from warm_start_qaoa import compute_warm_start_angles, EPSILON  # noqa: E402

_failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def check_covariance_is_psd():
    mu, sigma = make_synthetic_universe(10, seed=1)
    eigvals = np.linalg.eigvalsh(sigma)
    check("synthetic covariance is PSD", bool(np.all(eigvals >= -1e-8)), f"min eigenvalue={eigvals.min():.2e}")


def check_align_and_clean_drops_sparse_symbols():
    import pandas as pd
    dates = pd.bdate_range("2023-01-01", periods=100)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "GOOD": 100 * np.cumprod(1 + rng.normal(0, 0.01, 100)),
        "SPARSE": np.concatenate([np.full(75, np.nan), 100 * np.cumprod(1 + rng.normal(0, 0.01, 25))]),
    }, index=dates)

    class _Dummy(AssetUniverseAdapter):
        def default_universe(self, n):
            return []
        def fetch_price_history(self, symbols, start, end):
            return prices

    cleaned = _Dummy()._align_and_clean(prices)
    check("adapter drops a symbol missing >10% of history", "SPARSE" not in cleaned.columns)
    check("adapter keeps a fully-populated symbol", "GOOD" in cleaned.columns)


def check_preprocessing_drops_at_35_percent_missing():
    import pandas as pd
    dates = pd.bdate_range("2023-01-01", periods=100)
    prices = pd.DataFrame({
        "OK": np.linspace(100, 110, 100),
        "MOSTLY_MISSING": [np.nan] * 40 + list(np.linspace(50, 55, 60)),
    }, index=dates)
    cleaned, report = preprocess_price_panel(prices)
    check(
        f"preprocessing drops a column at/above {MISSING_DROP_THRESHOLD:.0%} missing",
        "MOSTLY_MISSING" in report.columns_dropped_missing,
    )


def check_exact_qubo_matches_brute_force_for_small_n():
    mu, sigma = make_synthetic_universe(8, seed=2)
    k, q = 4, 0.5
    bf = brute_force_exact(mu, sigma, k, q)
    qp = build_qubo(mu, sigma, k, q)
    exact = solve_exact_qubo(qp)
    x = np.array(exact.x)
    val = objective(mu, sigma, x, q)
    feasible = bool(np.isclose(x.sum(), k))
    check("exact-QUBO diagonalization recovers the true optimum", feasible and np.isclose(val, bf.value, atol=1e-6),
          f"exact-QUBO={val:.6f} vs brute-force={bf.value:.6f}, feasible={feasible}")


def check_manual_ising_agrees_with_qiskit_qubo():
    """Cross-check stage 05a (qiskit-finance auto QUBO) against stage 05b
    (the by-hand Ising derivation): both should identify the same ground
    state as optimal, since they encode the identical constrained problem."""
    mu, sigma = make_synthetic_universe(7, seed=3)
    k, q = 7 // 2, 0.5
    bf = brute_force_exact(mu, sigma, k, q)

    penalty, _ = find_minimal_penalty(mu, sigma, k, q)
    H = to_ising_hamiltonian(mu, sigma, k, q, penalty=penalty)

    import itertools
    best_energy, best_z = None, None
    for bits in itertools.product([-1, 1], repeat=H.n_qubits):
        z = np.array(bits)
        e = H.energy(z)
        if best_energy is None or e < best_energy:
            best_energy, best_z = e, z
    x = ((1 - best_z) // 2)
    feasible = bool(x.sum() == k)
    val = objective(mu, sigma, x, q)
    check(
        "by-hand Ising ground state matches brute-force optimum",
        feasible and np.isclose(val, bf.value, atol=1e-6),
        f"manual-Ising-derived x -> objective={val:.6f} vs brute-force={bf.value:.6f}, feasible={feasible}",
    )


def check_budget_is_never_hardcoded_downstream():
    """Regression guard: the whole point of RunConfig.budget/n_assets is
    that nothing downstream assumes a fixed k or n. This simply checks the
    solver core handles an odd, non-default (n, k) pair without special-casing."""
    mu, sigma = make_synthetic_universe(9, seed=4)
    k, q = 2, 0.3  # deliberately not n//2
    bf = brute_force_exact(mu, sigma, k, q)
    check("solver core handles an arbitrary (n, k) pair", bf.selection is not None and len(bf.selection) == k)


def check_sector_constraint_is_actually_enforced():
    """A tight per-sector cap should force a DIFFERENT (worse-or-equal, never
    better) selection than the unconstrained brute-force optimum - if it
    didn't, the constraint wouldn't be doing anything. Also checks the
    returned selection genuinely respects every sector's cap."""
    mu, sigma = make_synthetic_universe(10, seed=5)
    k, q = 5, 0.5
    unconstrained = brute_force_exact(mu, sigma, k, q)

    sectors = assign_sectors(10, n_sectors=5, seed=5)   # cap=1 forces one-per-sector
    constrained = brute_force_sector_constrained(mu, sigma, k, q, sectors, sector_cap=1)

    counts = np.bincount(sectors[list(constrained.selection)], minlength=sectors.max() + 1)
    respects_cap = bool(np.all(counts <= 1))
    no_better_than_unconstrained = constrained.value <= unconstrained.value + 1e-9
    check(
        "sector-constrained brute force respects the cap and never beats the unconstrained optimum",
        constrained.feasible and respects_cap and no_better_than_unconstrained,
        f"constrained={constrained.value:.6f} (sector counts {counts.tolist()}) "
        f"vs unconstrained={unconstrained.value:.6f}",
    )


def check_simulated_annealing_reaches_known_optimum():
    """On a small instance where brute force is exact, SA (given a generous
    iteration budget relative to the tiny search space) should reliably find
    the same optimum - if it consistently didn't, the neighbor-move/cooling
    logic would be broken, not just imprecise."""
    mu, sigma = make_synthetic_universe(8, seed=6)
    k, q = 4, 0.5
    bf = brute_force_exact(mu, sigma, k, q)
    sa = simulated_annealing(mu, sigma, k, q, n_iters=3000, seed=6)
    check(
        "simulated annealing reaches the known optimum on a small instance",
        np.isclose(sa.value, bf.value, rtol=1e-6),
        f"SA={sa.value:.6f} vs brute-force={bf.value:.6f}",
    )


def check_vectorized_brute_force_matches_original():
    """stage 04's batched-numpy brute force is claimed to be an exact,
    faster reimplementation of the original per-subset Python loop, not an
    approximation - this is the check that claim actually rests on."""
    mu, sigma = make_synthetic_universe(11, seed=7)
    k, q = 5, 0.4
    r1 = brute_force_exact(mu, sigma, k, q)
    r2 = brute_force_exact_vectorized(mu, sigma, k, q)
    check(
        "vectorized brute force matches the original exactly",
        np.isclose(r1.value, r2.value, atol=1e-9) and set(r1.selection) == set(r2.selection),
        f"original={r1.value:.6f} {r1.selection} vs vectorized={r2.value:.6f} {r2.selection}",
    )


def check_warm_start_angles_reduce_to_standard_mixer_at_c_half():
    """Egger et al.'s construction is built to strictly generalize the
    standard X-mixer: at c_i=0.5 (relaxation has no opinion on asset i),
    theta_i should come out to exactly pi/2 - the angle at which the
    warm-start mixer's rotated basis coincides with the standard one."""
    mu, sigma = make_synthetic_universe(6, seed=8)
    # Force a relaxation-neutral case is impractical to construct directly;
    # instead check the theta(c) formula itself at its defined boundary,
    # since compute_warm_start_angles clips into [EPSILON, 1-EPSILON] and
    # applies theta = 2*arcsin(sqrt(c)) - c=0.5 must map to theta=pi/2.
    c_half = np.array([0.5])
    theta = 2 * np.arcsin(np.sqrt(np.clip(c_half, EPSILON, 1 - EPSILON)))
    check(
        "warm-start theta(c=0.5) equals pi/2 (standard-mixer-equivalent point)",
        np.isclose(theta[0], np.pi / 2, atol=1e-9),
        f"theta={theta[0]:.6f} vs pi/2={np.pi/2:.6f}",
    )

    warm = compute_warm_start_angles(mu, sigma, k=3, q=0.5)
    check(
        "warm-start angles are computed for every asset and stay in [0, pi]",
        warm.available and warm.thetas is not None and len(warm.thetas) == 6
        and bool(np.all((warm.thetas >= 0) & (warm.thetas <= np.pi))),
        f"available={warm.available}, thetas={warm.thetas}",
    )


def run_all() -> tuple[list[str], bool]:
    """Run every check, returning the list of failure names (empty = all
    passed) - importable by the dashboard's sanity_runner.py so both the
    CLI and the live dashboard button run the exact same checks."""
    _failures.clear()
    check_covariance_is_psd()
    check_align_and_clean_drops_sparse_symbols()
    check_preprocessing_drops_at_35_percent_missing()
    check_exact_qubo_matches_brute_force_for_small_n()
    check_manual_ising_agrees_with_qiskit_qubo()
    check_budget_is_never_hardcoded_downstream()
    check_sector_constraint_is_actually_enforced()
    check_simulated_annealing_reaches_known_optimum()
    check_vectorized_brute_force_matches_original()
    check_warm_start_angles_reduce_to_standard_mixer_at_c_half()
    return list(_failures), len(_failures) == 0


if __name__ == "__main__":
    failures, ok = run_all()
    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    sys.exit(0 if ok else 1)
