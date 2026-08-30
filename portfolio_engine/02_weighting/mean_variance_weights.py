"""
Stage P02b - Continuous weights among a FIXED, already-selected subset.
=====================================================================
The academic engine (stage 04-07) answers WHICH k of n assets to hold -
every academic-page result is equal-weighted among the selection, by
design (README section 6, assumption 2: "the optimizer chooses which k
assets to hold... it does not solve for continuous portfolio weights").
The portfolio dashboard needs actual weights (instructions: "table of
selected stocks and corresponding portfolio weights"), so this stage adds
exactly that ONE extra step, applied AFTER QAOA/classical selection, never
instead of it: given the k selected assets' own (mu, Sigma) sub-block,
solve the classical (long-only, fully-invested) continuous mean-variance
QP for their weights. This is deliberately a separate, later step from
cardinality selection, not a different formulation of the same problem -
the combinatorial "which k" decision and the continuous "how much of
each" decision are solved by different tools because they are different
kinds of problems (the first is what makes this whole project's QAOA
question interesting; the second is a standard convex QP with a unique
global optimum, nothing for QAOA to contribute here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class WeightResult:
    method: str
    weights: np.ndarray        # aligned with the selected symbol order, sums to 1
    expected_return: float     # mu . weights
    expected_variance: float   # weights . Sigma . weights


def mean_variance_weights(mu: np.ndarray, sigma: np.ndarray, q: float) -> WeightResult:
    """Long-only, fully-invested continuous mean-variance QP over the
    SELECTED subset only:
        maximize    mu^T w - q * w^T Sigma w
        subject to  sum(w) = 1, w >= 0
    Falls back to equal weights (matching the academic page's own
    convention) if the solver is unavailable, rather than raising -
    consistent with continuous_relaxation_rounded's fallback behavior in
    the academic engine's markowitz.py."""
    import cvxpy as cp

    n = len(mu)
    w = cp.Variable(n)
    quad = cp.quad_form(w, cp.psd_wrap(sigma))
    prob = cp.Problem(cp.Maximize(mu @ w - q * quad), [cp.sum(w) == 1, w >= 0])
    try:
        prob.solve()
        wval = w.value
    except Exception:
        wval = None

    if wval is None:
        wval = np.full(n, 1.0 / n)
        method = "equal_weight_fallback"
    else:
        wval = np.clip(wval, 0, None)
        wval = wval / wval.sum()
        method = "mean_variance_qp"

    return WeightResult(
        method=method, weights=wval,
        expected_return=float(mu @ wval),
        expected_variance=float(wval @ sigma @ wval),
    )


def equal_weights(mu: np.ndarray, sigma: np.ndarray) -> WeightResult:
    """The academic page's convention, offered here too as an explicit,
    named alternative rather than only ever being an error fallback - a
    fair baseline to compare mean-variance/HRP weights against."""
    n = len(mu)
    wval = np.full(n, 1.0 / n)
    return WeightResult(
        method="equal_weight", weights=wval,
        expected_return=float(mu @ wval), expected_variance=float(wval @ sigma @ wval),
    )
