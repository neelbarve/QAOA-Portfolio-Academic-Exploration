"""
Stage 01d - Equity engine.
============================
Primary source: Tiingo (if TIINGO_API_KEY set) - clean EOD daily data, deep
history, per data_requirements.txt.
Fallback 1: yfinance (free, no key; Yahoo throttles under heavy use but the
call volumes here - a handful of tickers, once - are fine).
Fallback 2: Stooq (free, no key), if yfinance itself raises.

Fundamentals (EPS, P/E, ...) and full OHLCV are deliberately NOT fetched
here: the solver core only ever needs (mu, sigma) derived from one price
series per asset. A fundamentals pre-filter is a separate, optional module,
not part of the QAOA/QUBO core (see data_requirements.txt).

trading_days_per_year = 252: exchange trading-day calendar, not 365.
"""

from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd

from base_adapter import AssetUniverseAdapter
from tiingo_source import tiingo_available, fetch_equity_prices_tiingo


class EquityAdapter(AssetUniverseAdapter):
    trading_days_per_year = 252

    # Small static fallback universe (sector-diversified large caps) so
    # default_universe() works with zero configuration. Override via
    # RunConfig(symbols=[...]) for anything else, e.g. a specific index's
    # constituents or a sector-specific set.
    DEFAULT_UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "DIS", "BAC",
    ]

    def default_universe(self, n: int) -> List[str]:
        return self.DEFAULT_UNIVERSE[:n]

    def fetch_price_history(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        if tiingo_available():
            try:
                df = fetch_equity_prices_tiingo(symbols, start, end)
                if not df.empty:
                    return df
            except Exception:
                pass  # fall through to yfinance below

        try:
            return self._fetch_from_yfinance(symbols, start, end)
        except Exception:
            return self._fetch_from_stooq(symbols, start, end)

    def _fetch_from_yfinance(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
        # auto_adjust=True folds splits/dividends into "Close" directly
        prices = data["Close"] if len(symbols) > 1 else data[["Close"]].rename(
            columns={"Close": symbols[0]}
        )
        return prices

    def _fetch_from_stooq(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        """Stooq CSV endpoint, one request per symbol - used only as a last
        resort when yfinance itself raises (e.g. Yahoo rate-limit)."""
        import io
        import requests

        series = {}
        for sym in symbols:
            resp = requests.get(
                "https://stooq.com/q/d/l/",
                params={"s": sym.lower(), "i": "d"},
                timeout=15,
            )
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
            df = df.set_index("Date").loc[str(start):str(end)]
            if not df.empty:
                series[sym] = df["Close"]
        return pd.DataFrame(series).sort_index()
