"""
Stage 01b - Optional Tiingo price source.
============================================
Per data_requirements.txt: Tiingo is the more reliable option for equities
(clean EOD daily data, deep history) and also has a crypto endpoint
aggregating 150+ exchanges. Both CryptoAdapter and EquityAdapter try Tiingo
FIRST if a key is available, and fall back to their free/keyless source
(CoinGecko, yfinance) otherwise - so the pipeline runs out of the box with
no key, and gets better data automatically once one is configured.

Per the project's background instructions, the key is read from an
environment variable so it never has to live in code:

    setx TIINGO_API_KEY "your-key-here"      (Windows, permanent)
    $env:TIINGO_API_KEY = "your-key-here"    (Windows, current session)

If TIINGO_API_KEY is not set, tiingo_available() returns False and callers
silently use their free fallback - no crash, no prompt.
"""

from __future__ import annotations

import os
from datetime import date
from typing import List

import pandas as pd
import requests

TIINGO_BASE_URL = "https://api.tiingo.com"


def tiingo_available() -> bool:
    return bool(os.environ.get("TIINGO_API_KEY"))


def _headers() -> dict:
    key = os.environ.get("TIINGO_API_KEY", "")
    return {"Content-Type": "application/json", "Authorization": f"Token {key}"}


def fetch_equity_prices_tiingo(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    """Adjusted close, one column per symbol. Raises if the request fails -
    callers are expected to catch and fall back to yfinance/Stooq."""
    series = {}
    for sym in symbols:
        resp = requests.get(
            f"{TIINGO_BASE_URL}/tiingo/daily/{sym}/prices",
            params={"startDate": start.isoformat(), "endDate": end.isoformat()},
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json()
        s = pd.Series(
            {pd.Timestamp(r["date"]).normalize(): r["adjClose"] for r in rows}
        )
        series[sym] = s
    return pd.DataFrame(series).sort_index()


def fetch_crypto_prices_tiingo(symbols: List[str], start: date, end: date) -> pd.DataFrame:
    """Tiingo crypto tickers are like 'btcusd'; symbols passed in should
    already be in that form (CryptoAdapter maps CoinGecko-style ids to
    Tiingo tickers before calling this)."""
    series = {}
    for sym in symbols:
        resp = requests.get(
            f"{TIINGO_BASE_URL}/tiingo/crypto/prices",
            params={
                "tickers": sym,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "resampleFreq": "1day",
            },
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            continue
        rows = payload[0]["priceData"]
        s = pd.Series(
            {pd.Timestamp(r["date"]).normalize(): r["close"] for r in rows}
        )
        series[sym] = s
    return pd.DataFrame(series).sort_index()
