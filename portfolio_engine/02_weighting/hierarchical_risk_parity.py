"""
Stage P02c - Hierarchical Risk Parity (HRP) weighting.
=====================================================================
Lopez de Prado (2016), "Building Diversified Portfolios that Outperform
Out-of-Sample" - named explicitly in gemini_search_techniques.txt as a
core modern technique that fixes MVO's error-maximization problem by
avoiding covariance matrix INVERSION entirely (mean_variance_weights.py's
QP implicitly inverts/solves against Sigma; with a noisy, small-sample
covariance matrix that inversion is exactly where estimation error gets
amplified into extreme weights). HRP sidesteps this with three steps that
never invert the full matrix:

  1. TREE CLUSTERING: convert the correlation matrix into a distance
     matrix (d_ij = sqrt(0.5*(1-corr_ij)), a proper metric unlike raw
     correlation) and hierarchically cluster assets by it.
  2. QUASI-DIAGONALIZATION: reorder assets by the dendrogram's leaf order,
     so similar (highly correlated) assets sit next to each other - this
     alone doesn't allocate anything, it just makes the next step's
     recursive splits fall along genuinely similar/dissimilar boundaries.
  3. RECURSIVE BISECTION: walk DOWN the tree, at each split allocating
     weight between the two child clusters INVERSELY to each child's own
     (inverse-variance-weighted) cluster variance - a diversified cluster
     with lower risk gets more weight than a riskier one, at every level
     of the hierarchy, never requiring Sigma^-1.

Implemented from scratch (not via an external portfolio-optimization
library) so the mechanism is fully inspectable, consistent with this
project's "by-hand derivation alongside the library path" approach
already used for the QUBO/Ising formulation in the academic engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from mean_variance_weights import WeightResult


def _correlation_distance(sigma: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - corr))


def _cluster_variance(sigma: np.ndarray, items: List[int]) -> float:
    """Inverse-variance weights WITHIN this cluster (the one place HRP
    still uses a variance, but only the diagonal - no inversion of the
    full covariance block), then that sub-portfolio's own variance."""
    sub = sigma[np.ix_(items, items)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def _recursive_bisection(sigma: np.ndarray, sorted_items: List[int]) -> np.ndarray:
    n = sigma.shape[0]
    weights = np.ones(n)
    clusters = [sorted_items]

    while clusters:
        clusters = [
            c[start:stop]
            for c in clusters if len(c) > 1
            for start, stop in [(0, len(c) // 2), (len(c) // 2, len(c))]
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(sigma, left)
            var_right = _cluster_variance(sigma, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    return weights


def hierarchical_risk_parity(mu: np.ndarray, sigma: np.ndarray) -> WeightResult:
    n = len(mu)
    if n < 2:
        w = np.ones(n)
        return WeightResult("hierarchical_risk_parity", w, float(mu @ w), float(w @ sigma @ w))

    dist = _correlation_distance(sigma)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = leaves_list(link).tolist()

    weights = _recursive_bisection(sigma, order)
    weights = weights / weights.sum()

    return WeightResult(
        method="hierarchical_risk_parity", weights=weights,
        expected_return=float(mu @ weights), expected_variance=float(weights @ sigma @ weights),
    )
