"""
Stage P02a - Shrinkage covariance estimation.
=====================================================================
gemini_search_techniques.txt (this project's own research note for Part 2)
names this as the first fix to Modern Portfolio Theory's core practical
flaw, in almost these words: "small data errors cause MVO to produce
extreme, unstable asset weights... shrinkage estimators adjust covariance
matrices mathematically to reduce estimation errors." The academic
engine's Sigma (engine/01_data/base_adapter.py's get_mu_sigma) is the raw
sample covariance of log returns - exact, but noisy with the amount of
history a dashboard realistically has (a year or two of daily data for a
20-50 asset universe is a small-sample regime for an NxN covariance
matrix). Ledoit-Wolf shrinkage (scikit-learn, already a project
dependency via cvxpy/qiskit-optimization - no new install) pulls the
sample covariance toward a structured target (a scaled identity matrix)
by an amount the data itself determines, which is the standard, most-cited
fix for exactly this problem (Ledoit & Wolf, 2004).

Kept as a SEPARATE, optional step (not baked into base_adapter.py) so the
academic page's results (built and reported before this stage existed)
stay unchanged - the portfolio dashboard opts into shrinkage explicitly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrinkage_covariance(log_returns: pd.DataFrame, periods_per_year: int) -> Tuple[np.ndarray, float]:
    """Returns (annualized_shrunk_sigma, shrinkage_intensity). shrinkage_intensity
    in [0, 1] is how much weight Ledoit-Wolf put on the structured target -
    0 means the sample covariance was already well-conditioned enough that
    shrinkage barely mattered, close to 1 means the raw sample estimate was
    unreliable and the shrunk target dominates. Surfaced to the dashboard
    so a user can see how much this step actually changed anything, rather
    than a shrunk number appearing with no indication of its own
    confidence."""
    lw = LedoitWolf().fit(log_returns.values)
    sigma_annual = lw.covariance_ * periods_per_year
    return sigma_annual, float(lw.shrinkage_)
