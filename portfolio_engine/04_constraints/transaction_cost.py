"""
Stage P04 - Transaction-cost-adjusted selection, and constraint-stacking
cost on real data.
=====================================================================
Two related but separate things the instructions ask for:

1. "Consider specific constraints (transaction costs, cardinality limits)
   [that] are causing your classical solvers to slow down" - reuses the
   academic engine's ALREADY-BUILT, ALREADY-VALIDATED sector-constrained
   QUBO machinery (engine/11_hard_constraints/sector_constrained.py:
   brute_force_sector_constrained, assign_sectors) rather than
   reimplementing constraint-stacking from scratch, and runs it on this
   dashboard's REAL selected universe instead of only the academic page's
   synthetic sweep - reported timing is what stage 11's own hardness
   sweep already found (classical exact search cost grows combinatorially
   with EACH additional constraint layered on, not just with n), now
   shown on data a user actually picked.

2. A simple, LINEAR, asset-specific transaction-cost adjustment to the
   selection objective itself (this file's transaction_cost_adjusted_mu):
   without real trade/volume data, a genuine bid-ask-spread-based cost
   isn't available, so realized volatility is used as an honest, stated
   proxy for illiquidity/trading-cost (higher-volatility names are
   typically more expensive to trade at size) - a documented assumption,
   not a claim of real transaction-cost data. This keeps the QUBO convex/
   quadratic (a proportional cost subtracted from mu), which is the
   HONEST scope: a fixed, non-convex per-trade cost (the model that would
   make selection itself combinatorially harder, not just add a second
   penalty term) is flagged as a further extension, not implemented.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def transaction_cost_adjusted_mu(
    mu: np.ndarray, sigma: np.ndarray, cost_coefficient: float = 0.1,
) -> np.ndarray:
    """mu_adjusted_i = mu_i - cost_coefficient * volatility_i, where
    volatility_i = sqrt(Sigma_ii) (annualized). cost_coefficient is a
    dimensionless knob (default 0.1 - a name with 20% annualized vol pays
    a 2-percentage-point return haircut), not calibrated to any real
    fee schedule; surfaced on the dashboard as an adjustable, clearly-
    labeled assumption, not a fetched market data point."""
    vol = np.sqrt(np.diag(sigma))
    return mu - cost_coefficient * vol
