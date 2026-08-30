"""
Stage 01a - Shared adapter interface + shared math.
=====================================================
The solver core (stages 04-07) only ever consumes (mu, sigma, k, q). Every
asset-class-specific concern (where prices come from, how a "return" is
defined, how the calendar is annualized) lives behind one interface here, so
swapping "crypto" for "equity" is a one-line config change, not a rewrite.

To add a new asset class: subclass AssetUniverseAdapter and implement
default_universe() and fetch_price_history(). Return-calculation, calendar
alignment/cleaning and annualization are handled once, in the base class.

This is a direct, commented extension of qaoa_v1_update1/data_adapters.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame, return_freq: str = "D"
) -> pd.DataFrame:
    """Shared by get_mu_sigma() below and by stage 03 (EDA), so both work
    off the exact same return series rather than two subtly different
    recomputations of "the returns"."""
    resampled = prices.resample(return_freq).last() if return_freq != "D" else prices
    log_returns = np.log(resampled / resampled.shift(1)).dropna(how="all")
    return log_returns.dropna(axis=1, how="any")


@dataclass
class RunConfig:
    """Everything one pipeline run needs to specify, independent of engine.

    engine        "crypto" | "equity" - picks the adapter (stage 01b/01c).
    symbols       explicit universe; if None the adapter picks a default.
    n_assets      only used when symbols is None.
    start / end   lookback window for price history.
    return_freq   "D" (daily) / "W" (weekly) / "M" (monthly) - the "holding
                  period" knob called out in the project instructions as a
                  user-selectable parameter, not something the algorithm
                  itself decides.
    risk_factor   q, the risk-aversion coefficient in the Markowitz objective
                  mu^T x - q * x^T Sigma x.
    budget        k, the cardinality constraint (pick exactly k of n assets).
                  Defaults to n_assets // 2 if not given. Deliberately never
                  hardcoded elsewhere in the pipeline - this field is the only
                  place k is set, so the whole engine scales to any n, k.
    """

    engine: str
    symbols: Optional[List[str]] = None
    n_assets: int = 10
    start: date = date(2023, 1, 1)
    end: date = date(2024, 1, 1)
    return_freq: str = "D"
    risk_factor: float = 0.5
    budget: Optional[int] = None


class AssetUniverseAdapter(ABC):
    """Common base for every asset-class adapter.

    trading_days_per_year: periods per year at DAILY frequency, used to
    annualize mu/sigma. Crypto trades 24/7/365; equities trade on an
    exchange calendar (~252 sessions/year). Set by each subclass.
    """

    trading_days_per_year: int = 252

    @abstractmethod
    def default_universe(self, n: int) -> List[str]:
        """Pick n reasonable default symbols for this asset class. Only
        called when RunConfig.symbols is None."""

    @abstractmethod
    def fetch_price_history(
        self, symbols: List[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by date, one column per symbol, values
        = a single clean price series (adjusted close for equities, close
        for crypto). This is the ONLY place network/IO happens - keeping it
        isolated here is what makes the rest of the pipeline testable
        without a network connection (see tests/sanity_checks.py)."""

    # ------------------------------------------------------------------
    # Shared for every engine: turn a price panel into (mu, sigma).
    # This is stage 01's contribution to preprocessing; the *generic*
    # cleaning/imputation checklist (drop >=35% missing columns, EDA, etc.)
    # lives in stage 02/03 and is applied on top of this.
    # ------------------------------------------------------------------
    def get_mu_sigma(
        self, prices: pd.DataFrame, return_freq: str = "D"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        prices = self._align_and_clean(prices)
        log_returns = compute_log_returns(prices, return_freq)

        periods_per_year = {"D": self.trading_days_per_year, "W": 52, "M": 12}[return_freq]
        mu = log_returns.mean().values * periods_per_year
        sigma = log_returns.cov().values * periods_per_year
        symbols = list(log_returns.columns)
        return mu, sigma, symbols

    def _align_and_clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop symbols missing too much history, forward-fill small gaps
        (an exchange holiday on one market but not another, a brief API
        hiccup). Subclasses may override for asset-class-specific cleaning
        (e.g. winsorizing crypto's fatter tails)."""
        prices = prices.copy()
        min_coverage = 0.9
        keep = prices.columns[prices.notna().mean() >= min_coverage]
        prices = prices[keep].ffill().dropna(how="any")
        return prices
