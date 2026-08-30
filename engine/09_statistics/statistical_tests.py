r"""
Stage 09 - Statistical analysis of the scaling benchmark (stage 08).
=========================================================================
Turns a list of per-(n, seed) results into the PhD-level-demonstration
findings the project instructions call for: confidence intervals, a
regression-fit scaling exponent, and a significance test - not just point
estimates. Methodology follows Yalovetzky et al. (2026) "Quantum-Informed
Portfolio Selection" (the most statistically rigorous of the five reference
papers): percentile bootstrap CIs, OLS regression of log(time) vs. problem
size to extract a scaling exponent with its standard error and R^2, and a
paired significance test between two methods' approximation ratios.

Nothing here needs anything beyond numpy/scipy - no extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class BootstrapCI:
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int


def bootstrap_mean_ci(
    values: Sequence[float], confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 0,
) -> BootstrapCI:
    """Nonparametric percentile bootstrap on the sample mean - no
    distributional assumption on `values`, which matters here since
    approximation ratios and wall-clock times are neither normal nor
    symmetric in general."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    rng = np.random.default_rng(seed)
    if len(arr) == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), confidence, n_resamples)
    resample_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)
    ])
    alpha = 1 - confidence
    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(arr.mean()), float(lo), float(hi), confidence, n_resamples)


@dataclass
class ScalingFit:
    """log2(y) = intercept + slope * n  =>  y ~ 2^(intercept) * 2^(slope*n),
    i.e. an exponential-in-n fit (matched to the C(n,k) combinatorial
    scaling the classical brute-force solver actually has). slope is the
    scaling exponent quoted in the write-up; slope_se its standard error."""
    slope: float
    slope_se: float
    intercept: float
    r_squared: float
    p_value: float
    n_points: int


def fit_scaling_exponent(n_values: Sequence[int], y_values: Sequence[float]) -> ScalingFit:
    n = np.asarray(n_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    n, y = n[mask], y[mask]
    log2_y = np.log2(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(n, log2_y)
    return ScalingFit(
        slope=float(slope), slope_se=float(std_err), intercept=float(intercept),
        r_squared=float(r_value ** 2), p_value=float(p_value), n_points=int(len(n)),
    )


@dataclass
class PairedTestResult:
    test: str
    statistic: float
    p_value: float
    n_pairs: int
    mean_difference: float


def paired_comparison(method_a: Sequence[float], method_b: Sequence[float]) -> PairedTestResult:
    """Wilcoxon signed-rank test on paired (method_a - method_b) values -
    used instead of a paired t-test because approximation-ratio differences
    are typically not normally distributed and can be bounded/skewed near
    the ratio's ceiling of 1.0. Falls back to a paired t-test if the
    Wilcoxon test degenerates (e.g. all differences identical or too few
    non-zero differences), reporting which test actually ran."""
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    if len(diff) < 2 or np.all(diff == diff[0]):
        return PairedTestResult("insufficient variation", float("nan"), float("nan"), len(diff), float(diff.mean()) if len(diff) else float("nan"))
    try:
        stat, p = stats.wilcoxon(a, b)
        return PairedTestResult("wilcoxon_signed_rank", float(stat), float(p), len(diff), float(diff.mean()))
    except ValueError:
        stat, p = stats.ttest_rel(a, b)
        return PairedTestResult("paired_t_test (wilcoxon degenerate)", float(stat), float(p), len(diff), float(diff.mean()))


def two_sample_slope_difference_test(fit_a: ScalingFit, fit_b: ScalingFit) -> Tuple[float, float, float]:
    """Two-sample t-test on the difference between two independently-fit
    scaling exponents (e.g. QAOA time-scaling slope vs. brute-force
    time-scaling slope), following the same test used by Yalovetzky et al.
    to reject "equal scaling" between two methods. Returns
    (delta_slope, t_statistic, p_value)."""
    delta = fit_a.slope - fit_b.slope
    se = np.sqrt(fit_a.slope_se ** 2 + fit_b.slope_se ** 2)
    if se == 0:
        return float(delta), float("nan"), float("nan")
    t_stat = delta / se
    dof = max(fit_a.n_points + fit_b.n_points - 4, 1)  # 2 params fit per side
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))
    return float(delta), float(t_stat), float(p_value)
