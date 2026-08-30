r"""
Portfolio dashboard orchestrator - the Part 2 analogue of engine/pipeline.py.
=====================================================================
Ties the whole "select sector -> select n stocks -> select holding period
-> run" flow together for ONE request, reusing the academic engine's
stages 01-07 AS A LIBRARY (not copied - imported directly, exactly per
the instructions: "use the same resources... keep processing/result files
separate") and adding the Part 2 layers on top: sector/holding-period
selection (01_universe), shrinkage covariance + weight construction
(02_weighting), historical VaR (03_risk_metrics), and an optional
transaction-cost adjustment (04_constraints).

QAOA is only attempted up to QAOA_MAX_N assets. This is not a UI
convenience cutoff - it is a direct, evidence-based consequence of this
project's OWN academic-page findings (README section 9): standard QAOA
stops converging at all above n=14-16 on a noiseless simulator, taking
minutes to fail rather than seconds. Running it anyway above that size
would either hang the dashboard for a long time to produce a known-useless
result, or silently mislead a user into thinking a spinner meant progress.
Above the cutoff, classical relaxation + simulated annealing (both from
the academic engine, both proven reliable at these sizes) stand in for
the "quantum" comparison column, with that substitution stated explicitly
in the UI rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_sources import UniverseSelection, fetch_universe, HOLDING_PERIODS
from shrinkage_covariance import shrinkage_covariance
from mean_variance_weights import mean_variance_weights, equal_weights, WeightResult
from hierarchical_risk_parity import hierarchical_risk_parity
from value_at_risk import historical_var, portfolio_returns, VaRResult, VAR_CONFIDENCE_LEVELS
from transaction_cost import transaction_cost_adjusted_mu

from markowitz import brute_force_exact, continuous_relaxation_rounded, objective
from qiskit_qubo import build_qubo
from qaoa_solver import solve_qaoa
from simulated_annealing import simulated_annealing

QAOA_MAX_N = 16
BRUTE_FORCE_MAX_N = 20  # beyond this, exact search itself is impractically slow for a dashboard click


@dataclass
class SelectionResult:
    method: str                # "qaoa" | "brute_force" | "simulated_annealing" | "relaxation"
    selected_symbols: List[str]
    selected_indices: np.ndarray
    value: float
    feasible: bool
    converged: Optional[bool]  # None where not applicable (classical exact/relaxation methods)
    elapsed_s: float


@dataclass
class WeightedPortfolio:
    selection: SelectionResult
    weights: WeightResult
    weight_method: str
    per_asset_var: Dict[str, VaRResult]     # symbol -> VaR (at the selected weight, standalone asset returns)
    portfolio_var: VaRResult


@dataclass
class PortfolioRunResult:
    symbols: List[str]
    n_assets: int
    k: int
    mu: np.ndarray
    sigma: np.ndarray
    sigma_shrinkage_intensity: Optional[float]
    log_returns: pd.DataFrame
    quantum_portfolio: Optional[WeightedPortfolio]   # None if n_assets > QAOA_MAX_N
    classical_portfolio: WeightedPortfolio
    quantum_skipped_reason: Optional[str]


def _select_qaoa(mu, sigma, k, q, reps, seed) -> SelectionResult:
    qp = build_qubo(mu, sigma, k, q)
    run = solve_qaoa(qp, reps=reps, seed=seed)
    feasible = bool(np.isclose(run.x.sum(), k))
    val = objective(mu, sigma, run.x, q)
    return SelectionResult("qaoa", [], np.flatnonzero(run.x), val, feasible, run.converged, run.elapsed_s)


def _select_classical(mu, sigma, k, q) -> SelectionResult:
    n = len(mu)
    if n <= BRUTE_FORCE_MAX_N:
        bf = brute_force_exact(mu, sigma, k, q)
        return SelectionResult("brute_force", [], np.array(bf.selection), bf.value, bf.feasible, None, bf.elapsed_s)
    sa = simulated_annealing(mu, sigma, k, q, n_iters=6000, seed=42)
    return SelectionResult("simulated_annealing", [], np.array(sa.selection), sa.value, sa.feasible, None, sa.elapsed_s)


def _weight_and_assess(
    selection: SelectionResult, mu: np.ndarray, sigma: np.ndarray, symbols: List[str],
    log_returns: pd.DataFrame, weight_method: str, holding_period: str,
) -> WeightedPortfolio:
    idx = selection.selected_indices
    selection.selected_symbols = [symbols[i] for i in idx]
    mu_sub, sigma_sub = mu[idx], sigma[np.ix_(idx, idx)]

    if weight_method == "mean_variance":
        weights = mean_variance_weights(mu_sub, sigma_sub, q=0.5)
    elif weight_method == "hrp":
        weights = hierarchical_risk_parity(mu_sub, sigma_sub)
    else:
        weights = equal_weights(mu_sub, sigma_sub)

    holding_days = HOLDING_PERIODS[holding_period]
    selected_returns = log_returns.iloc[:, idx]
    per_asset_var = {
        sym: historical_var(selected_returns.iloc[:, j], holding_days)
        for j, sym in enumerate(selection.selected_symbols)
    }
    port_ret = portfolio_returns(selected_returns, weights.weights)
    portfolio_var = historical_var(port_ret, holding_days)

    return WeightedPortfolio(
        selection=selection, weights=weights, weight_method=weight_method,
        per_asset_var=per_asset_var, portfolio_var=portfolio_var,
    )


def run_portfolio_pipeline(
    selection_cfg: UniverseSelection, weight_method: str = "mean_variance",
    reps: int = 2, seed: int = 42, use_shrinkage: bool = True,
    transaction_cost_coefficient: Optional[float] = None,
) -> PortfolioRunResult:
    """Live-fetch path: resolves symbols from the sector/n_assets/engine
    selection, fetches real prices (engine stage 01, cached to
    data_cache/), then hands off to run_portfolio_pipeline_from_returns -
    the SAME downstream logic the file-upload path (dashboard's
    "Upload your own data" mode) uses, so a live fetch and an uploaded
    file are indistinguishable to everything past this point."""
    prices, log_returns, symbols = fetch_universe(selection_cfg)
    periods_per_year = 252 if selection_cfg.engine == "equity" else 365
    return run_portfolio_pipeline_from_returns(
        log_returns, symbols, selection_cfg.q, selection_cfg.holding_period, periods_per_year,
        weight_method=weight_method, reps=reps, seed=seed, use_shrinkage=use_shrinkage,
        transaction_cost_coefficient=transaction_cost_coefficient,
    )


def run_portfolio_pipeline_from_returns(
    log_returns: pd.DataFrame, symbols: List[str], q: float, holding_period: str,
    periods_per_year: int, weight_method: str = "mean_variance",
    reps: int = 2, seed: int = 42, use_shrinkage: bool = True,
    transaction_cost_coefficient: Optional[float] = None,
) -> PortfolioRunResult:
    """Everything downstream of "having a clean log-return panel": mu/Sigma
    (with optional shrinkage), QAOA-vs-classical selection, weighting, and
    VaR. Used directly by the uploaded-file path (which has no
    UniverseSelection - a file has no engine/sector/n_assets to resolve),
    and internally by run_portfolio_pipeline() above for the live-fetch
    path."""
    n = len(symbols)
    k = max(2, min(n - 1, n // 2))

    mu = log_returns.mean().values * periods_per_year
    if use_shrinkage:
        sigma, shrink_intensity = shrinkage_covariance(log_returns, periods_per_year)
    else:
        sigma, shrink_intensity = log_returns.cov().values * periods_per_year, None

    if transaction_cost_coefficient is not None:
        mu = transaction_cost_adjusted_mu(mu, sigma, transaction_cost_coefficient)

    quantum_wp, skip_reason = None, None
    if n <= QAOA_MAX_N:
        qaoa_sel = _select_qaoa(mu, sigma, k, q, reps, seed)
        quantum_wp = _weight_and_assess(
            qaoa_sel, mu, sigma, symbols, log_returns, weight_method, holding_period,
        )
    else:
        skip_reason = (
            f"QAOA is skipped above n={QAOA_MAX_N} assets: this project's own academic-page "
            f"benchmarking (README section 9) found standard QAOA stops converging at all past "
            f"n=14-16 on a noiseless simulator, taking minutes to fail rather than seconds. "
            f"At n={n}, simulated annealing (a realistic, reliable classical heuristic) stands "
            f"in for the quantum column instead of a QAOA run known in advance not to converge."
        )

    classical_sel = _select_classical(mu, sigma, k, q)
    classical_wp = _weight_and_assess(
        classical_sel, mu, sigma, symbols, log_returns, weight_method, holding_period,
    )

    return PortfolioRunResult(
        symbols=symbols, n_assets=n, k=k, mu=mu, sigma=sigma,
        sigma_shrinkage_intensity=shrink_intensity, log_returns=log_returns,
        quantum_portfolio=quantum_wp, classical_portfolio=classical_wp,
        quantum_skipped_reason=skip_reason,
    )
