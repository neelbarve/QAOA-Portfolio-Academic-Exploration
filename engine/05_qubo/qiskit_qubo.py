"""
Stage 05a - QUBO construction via qiskit-finance (primary path).
=====================================================================
qiskit_finance.applications.optimization.PortfolioOptimization builds the
docplex model

    min  q * x^T Sigma x  -  mu^T x
    s.t. sum(x) = k

then MinimumEigenOptimizer's internal converter chain (IntegerToBinary ->
LinearEqualityToPenalty) turns the equality constraint into a quadratic
penalty term lambda*(sum(x) - k)^2 added to the objective, with lambda
auto-scaled to the objective's coefficient magnitudes.

Don't build this translation by hand for the "does it run" path - the
library exists precisely so effort goes into the formulation and analysis,
not reimplementing a converter chain qiskit-optimization already ships. The
BY-HAND derivation of the same math (useful for the academic explainer
page, and for cross-checking this library's output) lives in
05_qubo/manual_ising.py.
"""

from __future__ import annotations

import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization import QuadraticProgram


def build_qubo(mu: np.ndarray, sigma: np.ndarray, k: int, q: float) -> QuadraticProgram:
    portfolio = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=q, budget=k
    )
    return portfolio.to_quadratic_program()
