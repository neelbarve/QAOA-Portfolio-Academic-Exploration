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
from typing import Dict, List

import pandas as pd

from base_adapter import AssetUniverseAdapter
from tiingo_source import tiingo_available, fetch_equity_prices_tiingo


# ---------------------------------------------------------------------
# Sector universes (Part 2 / portfolio dashboard requirement): a one-word
# switch instead of pasting a new ticker list every time a user wants a
# sector-scoped universe instead of the fully diversified default. Static
# lists of liquid, well-known large caps (real tickers, fetchable through
# the same Tiingo -> yfinance -> Stooq chain as everything else in this
# adapter) - not an attempt at an exhaustive or auto-updating index
# membership feed, which is out of scope here the same way real-time index
# reconstitution is out of scope for DEFAULT_UNIVERSE above.
# ---------------------------------------------------------------------
SECTOR_UNIVERSES: Dict[str, List[str]] = {
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB", "KMI",
        "VLO", "HES", "BKR", "HAL", "DVN", "FANG", "TRGP", "OKE", "CTRA", "EQT",
    ],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ORCL", "CRM", "ADBE", "CSCO",
        "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU", "PANW",
    ],
    "healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "CVS", "CI", "ELV", "ISRG", "VRTX", "REGN", "ZTS", "BSX",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "MS", "GS", "C", "SCHW", "BLK", "AXP", "USB",
        "PNC", "TFC", "COF", "MET", "AIG", "PGR", "TRV", "ALL", "AON", "MMC",
    ],
    "consumer": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO",
        "PEP", "COST", "WMT", "DIS", "BKNG", "CMG", "MAR", "YUM", "EL", "KMB",
    ],
    "industrials": [
        "BA", "CAT", "HON", "UPS", "GE", "LMT", "RTX", "DE", "UNP", "MMM",
        "GD", "NOC", "FDX", "EMR", "ETN", "ITW", "PH", "CSX", "NSC", "WM",
    ],
    # A broad cross-sector mix, deliberately kept to real, liquid large caps
    # rather than padded to hit exactly 50 - if a run asks for more than
    # this list has, the adapter truncates and the dashboard reports the
    # ACTUAL count used rather than silently proceeding as if the request
    # were fully honored (see portfolio_page.py's universe-size caption).
    "diversified": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "DIS", "BAC", "XOM", "JNJ", "KO", "CAT", "LLY",
        "AVGO", "ORCL", "CRM", "ADBE", "MRK", "ABBV", "WFC", "COST", "PEP", "TMO",
        "MCD", "CSCO", "ABT", "GE", "LMT", "AMD", "INTC", "NKE", "TXN", "UNP",
    ],
}


class EquityAdapter(AssetUniverseAdapter):
    trading_days_per_year = 252

    # Sector-diversified large-cap fallback so default_universe() works
    # with zero configuration. The FULL diversified list (40 tickers), not
    # a further-truncated slice of it - this adapter is used both by the
    # academic page (n <= 14, unaffected either way) and the portfolio
    # dashboard (n up to 50), and silently capping the default universe
    # below what a user actually requested there is exactly the kind of
    # bug that produces a portfolio smaller than asked for with no
    # indication why (found and fixed while testing the dashboard's n=20
    # path - see portfolio_page.py's universe-size caption, which now
    # reports the ACTUAL count used against what was requested).
    DEFAULT_UNIVERSE = SECTOR_UNIVERSES["diversified"]

    def default_universe(self, n: int) -> List[str]:
        return self.DEFAULT_UNIVERSE[:n]

    def sector_universe(self, sector: str, n: int) -> List[str]:
        """Same contract as default_universe(), scoped to one sector -
        used directly by the portfolio dashboard's sector dropdown."""
        if sector not in SECTOR_UNIVERSES:
            raise ValueError(f"Unknown sector '{sector}'. Choices: {sorted(SECTOR_UNIVERSES)}")
        return SECTOR_UNIVERSES[sector][:n]

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
