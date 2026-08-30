"""
Stage P03 - Historical (empirical) Value at Risk, at three confidence levels.
=====================================================================
"Result: table of selected stocks and corresponding portfolio weights,
along with expected returns for 3 different VaR" (instructions) - three
VaR columns per row (asset or portfolio), computed from REAL historical
return data, scaled to the user's chosen holding period.

Historical/empirical VaR, not parametric (Gaussian) VaR: gemini_search_
techniques.txt flags exactly this as MVO's second core weakness -
"[it] assumes asset returns follow a normal bell curve, which ignores
severe market crashes." Reading the loss quantile directly off the
asset's own realized return history sidesteps that assumption entirely,
at the cost of needing enough history to have a meaningfully-sized tail -
noted explicitly below rather than silently producing an unreliable
number from too little data.

Method: roll the DAILY log-return series into overlapping H-day windows
(H = the holding period in trading days), sum log returns within each
window (a standard log-return approximation to the H-day compounded
return), then take the empirical loss quantile of that distribution at
each confidence level. Overlapping windows are used deliberately - a
non-overlapping split of even two years of daily data leaves only a
handful of independent 1-year windows, nowhere near enough for a stable
99% quantile; overlapping windows trade window independence for sample
size, a standard, explicitly-acknowledged trade-off in practice (not
introducing autocorrelation that wasn't already there - it reuses real
observed transitions, not synthetic ones).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]


@dataclass
class VaRResult:
    holding_period_days: int
    n_windows: int
    var_by_level: Dict[float, float]     # confidence -> VaR (positive = a loss, as a fraction)
    reliable: bool                        # False if n_windows is too small to trust the tail


def historical_var(
    returns: pd.Series, holding_period_days: int, levels: List[float] = VAR_CONFIDENCE_LEVELS,
) -> VaRResult:
    """returns: a single daily LOG return series (one asset, or a
    portfolio's weighted combination - see portfolio_returns() below)."""
    if holding_period_days <= 1:
        window_returns = returns.dropna()
    else:
        window_returns = returns.rolling(holding_period_days).sum().dropna()

    n_windows = len(window_returns)
    # A 99% quantile needs enough tail observations to mean anything - 100
    # overlapping windows gives roughly one genuinely independent worst-case
    # observation at the 99% level, already thin; below that, flag it
    # rather than report a number that looks precise but isn't.
    reliable = n_windows >= 100

    var_by_level = {}
    for level in levels:
        loss_quantile = window_returns.quantile(1.0 - level)
        var_by_level[level] = float(max(-loss_quantile, 0.0))

    return VaRResult(
        holding_period_days=holding_period_days, n_windows=n_windows,
        var_by_level=var_by_level, reliable=reliable,
    )


def portfolio_returns(log_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted sum of the SELECTED assets' daily log returns. Treating a
    weighted sum of log returns as the portfolio's own log return is a
    standard, small approximation (exact for simple/arithmetic returns,
    not log returns) - accurate to first order for the daily magnitudes
    involved here, and consistent with how mu/Sigma themselves are
    already computed from log returns throughout this project."""
    return pd.Series(log_returns.values @ weights, index=log_returns.index)
