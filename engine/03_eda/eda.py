"""
Stage 03 - Exploratory data analysis.
========================================
Kept brief on purpose (per instructions: "do EDA in brief and keep it for
later fine-tuning, sanity check etc") - this is not a modeling stage, it's
the diagnostic layer the dashboard's "sanity check" tab and Fama-French page
both draw from. Everything here is descriptive statistics on log returns,
computed once per engine run and cached alongside the benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class EDAReport:
    symbols: List[str]
    summary_stats: pd.DataFrame            # count/mean/std/min/quartiles/max per asset
    skewness: Dict[str, float]
    excess_kurtosis: Dict[str, float]
    correlation_matrix: pd.DataFrame
    return_variance_s2_ret: float          # var({mu_i}) - Brandhofer et al. 2023 hardness proxy
    correlation_variance_s2_cor: float     # var(pairwise correlations) - same paper
    predicted_hardness: str                # "easier" | "harder" (relative label, not absolute)


def run_eda(log_returns: pd.DataFrame, mu: np.ndarray) -> EDAReport:
    """log_returns: cleaned per-period log-return panel (output of stage
    01's get_mu_sigma, before annualization) - one column per kept symbol.
    mu: the annualized expected-return vector already computed for these
    same symbols, used for the hardness proxy below."""
    symbols = list(log_returns.columns)

    summary_stats = log_returns.describe().T
    skewness = log_returns.skew().to_dict()
    excess_kurtosis = log_returns.kurtosis().to_dict()  # pandas kurtosis() is already excess (Fisher)
    corr = log_returns.corr()

    # Brandhofer et al. (2023) "Benchmarking the performance of portfolio
    # optimization with QAOA" found that instance hardness for QAOA
    # correlates with the variance of the return distribution across assets
    # and the variance of the pairwise-correlation distribution: broadly
    # spread returns/correlations give a more distinct cost landscape and
    # tend to be EASIER for the optimizer; a tightly clustered, nearly-flat
    # landscape tends to be HARDER (more near-degenerate optima to confuse
    # the classical outer loop). This is a cheap pre-screening diagnostic,
    # not a proof - reported as a label, not a hard guarantee.
    s2_ret = float(np.var(mu, ddof=1)) if len(mu) > 1 else 0.0
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    s2_cor = float(np.var(off_diag, ddof=1)) if len(off_diag) > 1 else 0.0

    # Relative label only meaningful when compared across instances of the
    # same size; the dashboard shows the raw numbers alongside this label
    # rather than treating it as absolute.
    hardness = "harder (low spread in returns/correlations)" if (s2_ret < 1e-3 or s2_cor < 1e-3) \
        else "easier (broadly spread returns/correlations)"

    return EDAReport(
        symbols=symbols,
        summary_stats=summary_stats,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        correlation_matrix=corr,
        return_variance_s2_ret=s2_ret,
        correlation_variance_s2_cor=s2_cor,
        predicted_hardness=hardness,
    )
