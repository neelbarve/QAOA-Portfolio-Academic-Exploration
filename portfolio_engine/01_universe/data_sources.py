"""
Stage P01a - Universe resolution for the portfolio dashboard.
=====================================================================
Wraps the academic engine's RunConfig / fetch_raw_prices (engine stage 01)
with the SELECTION FLOW the instructions specify: pick an engine (crypto or
equity) -> optionally filter to one sector -> pick n stocks (2-50) ->
pick a holding period. This module does not re-implement data fetching -
it reuses engine's adapters directly (same Tiingo -> free-fallback chain,
same data_cache/), it only adds the sector/holding-period layer on top,
per the instructions: "use the same resources... keep processing/result
files separate."

Holding period is a DIFFERENT knob from RunConfig.return_freq. return_freq
controls what interval returns are computed over for mu/Sigma (kept at
daily here, for statistical power - see below); holding period is how
long the user actually intends to hold the portfolio, which matters for
risk metrics (03_risk_metrics/value_at_risk.py scales VaR to this horizon)
and is surfaced on the dashboard exactly as named in the instructions
("3 months, 1 month, daily, 6 month, 1 year").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from base_adapter import RunConfig, compute_log_returns
from universe_builder import fetch_raw_prices, ENGINES
from equity_adapter import EquityAdapter, SECTOR_UNIVERSES as EQUITY_SECTORS
from crypto_adapter import CryptoAdapter, SECTOR_UNIVERSES as CRYPTO_SECTORS
from cleaning import preprocess_price_panel

SECTOR_CHOICES: Dict[str, List[str]] = {
    "equity": sorted(EQUITY_SECTORS),
    "crypto": sorted(CRYPTO_SECTORS),
}

# name -> approximate trading days, used both to pick a sensible fetch
# lookback and, downstream, to scale VaR to the chosen horizon
# (square-root-of-time / linear scaling - see value_at_risk.py). Trading
# days, not calendar days, consistent with the rest of this project's
# annualization conventions (252/year for equities, but kept simple and
# uniform across engines here since this is a HORIZON label, not a
# per-engine annualization factor).
HOLDING_PERIODS: Dict[str, int] = {
    "Daily": 1,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
}

# However short the holding period, mu/Sigma estimation needs enough
# history to be stable (this is exactly the small-sample covariance
# problem gemini_search_techniques.txt flags as MVO's core weakness,
# addressed here by (a) a floor on lookback length and (b) shrinkage
# covariance in 02_weighting/shrinkage_covariance.py, not by shortening
# the estimation window to match the holding period).
MIN_LOOKBACK_TRADING_DAYS = 252 * 2  # 2 years


@dataclass
class UniverseSelection:
    engine: str                     # "crypto" | "equity"
    sector: Optional[str]           # None = no sector filter
    n_assets: int
    holding_period: str             # a HOLDING_PERIODS key
    q: float = 0.5


def resolve_symbols(selection: UniverseSelection) -> List[str]:
    """Sector filter -> n symbols, or the engine's own diversified default
    if no sector was chosen - the exact "select sector -> select n stocks"
    flow from the instructions."""
    adapter = ENGINES[selection.engine]()
    if selection.sector is not None:
        return adapter.sector_universe(selection.sector, selection.n_assets)
    return adapter.default_universe(selection.n_assets)


def fetch_universe(
    selection: UniverseSelection, end: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Returns (cleaned_prices, log_returns, symbols). Runs stage 02's
    generic preprocessing checklist (drop >=35% missing, impute) on top of
    stage 01's own cleaning, exactly like the academic pipeline does, so
    both dashboard pages apply the identical preprocessing contract to
    real data - not two subtly different cleaning paths for the same kind
    of input."""
    end = end or date.today()
    start = end - timedelta(days=MIN_LOOKBACK_TRADING_DAYS * 7 // 5)  # trading days -> calendar days, roughly

    symbols = resolve_symbols(selection)
    config = RunConfig(engine=selection.engine, symbols=symbols, start=start, end=end,
                        return_freq="D", risk_factor=selection.q)
    raw_prices, adapter, _symbols = fetch_raw_prices(config)
    cleaned_prices, _report = preprocess_price_panel(raw_prices)

    log_returns = compute_log_returns(cleaned_prices, return_freq="D")
    return cleaned_prices, log_returns, list(log_returns.columns)
