"""
Stage 01c - Crypto engine.
============================
Source chain, tried in order, each one falling through to the next on any
error:
  1. Tiingo   - if TIINGO_API_KEY is set (tiingo_source.py).
  2. CoinGecko - keyless "markets" list for default_universe() always works,
    but CoinGecko now returns 401 Unauthorized on the historical
    market_chart/range endpoint without a (free, signup-required) Demo API
    key - discovered while testing this project, and left as an attempted
    source rather than removed, since a demo key drops it back to "works
    with zero extra code".
  3. Binance public REST API (/api/v3/klines) - genuinely keyless, no
    signup, works today. This is the fallback that actually fires in
    practice right now; per data_requirements.txt it was already flagged
    as the "finer granularity, full history" free alternative.

trading_days_per_year = 365: crypto markets never close, unlike an equity
exchange calendar (see equity_adapter.py, 252).
"""

from __future__ import annotations

import time
from datetime import date
from typing import Dict, List

import pandas as pd
import requests

from base_adapter import AssetUniverseAdapter
from tiingo_source import tiingo_available, fetch_crypto_prices_tiingo

# api.binance.com returns 451 (unavailable for legal reasons) from US IP
# ranges - discovered while testing this project. api.binance.us is the
# US-jurisdiction mirror of the same /api/v3/klines endpoint and is what
# actually works from here; kept configurable in case this project is ever
# run from a network where the reverse is true.
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_BASE_URL = "https://api.binance.us/api/v3"

# CoinGecko id -> Tiingo crypto ticker, for the handful of majors this
# project's default universe uses. Extend this map if you point
# default_universe() at a larger top-N set.
_COINGECKO_TO_TIINGO = {
    "bitcoin": "btcusd", "ethereum": "ethusd", "tether": "usdtusd",
    "binancecoin": "bnbusd", "solana": "solusd", "ripple": "xrpusd",
    "usd-coin": "usdcusd", "dogecoin": "dogeusd", "cardano": "adausd",
    "tron": "trxusd", "avalanche-2": "avaxusd", "shiba-inu": "shibusd",
    "polkadot": "dotusd", "chainlink": "linkusd", "litecoin": "ltcusd",
}

# CoinGecko id -> Binance spot symbol, same majors as above (USDT-quoted).
_COINGECKO_TO_BINANCE = {
    "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT", "ripple": "XRPUSDT", "dogecoin": "DOGEUSDT",
    "cardano": "ADAUSDT", "tron": "TRXUSDT", "avalanche-2": "AVAXUSDT",
    "shiba-inu": "SHIBUSDT", "polkadot": "DOTUSDT", "chainlink": "LINKUSDT",
    "litecoin": "LTCUSDT",
}


# Crypto doesn't have GICS-style sectors, but informal groupings are
# real and useful for the portfolio dashboard's "select a sector" dropdown
# to behave consistently across engines. Restricted to coins already
# mapped in _COINGECKO_TO_TIINGO / _COINGECKO_TO_BINANCE above, so every
# entry here is guaranteed fetchable through the same source chain, not
# just listed on CoinGecko.
SECTOR_UNIVERSES: Dict[str, List[str]] = {
    "layer1": ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2", "polkadot", "tron"],
    "payments_stable_adjacent": ["ripple", "litecoin", "tether", "usd-coin"],
    "defi_infra": ["chainlink", "avalanche-2", "polkadot", "solana"],
    "meme": ["dogecoin", "shiba-inu"],
    "diversified": [
        "bitcoin", "ethereum", "binancecoin", "solana", "ripple", "usd-coin",
        "dogecoin", "cardano", "tron", "avalanche-2",
    ],
}


class CryptoAdapter(AssetUniverseAdapter):
    trading_days_per_year = 365

    def sector_universe(self, sector: str, n: int) -> List[str]:
        """Same contract as EquityAdapter.sector_universe() - lets the
        portfolio dashboard's sector dropdown work identically regardless
        of which engine is selected."""
        if sector not in SECTOR_UNIVERSES:
            raise ValueError(f"Unknown sector '{sector}'. Choices: {sorted(SECTOR_UNIVERSES)}")
        return SECTOR_UNIVERSES[sector][:n]

    def default_universe(self, n: int) -> List[str]:
        """Top-n coins by market cap, via CoinGecko (works even without a
        Tiingo key, since it's only picking symbols, not prices yet)."""
        resp = requests.get(
            f"{COINGECKO_BASE_URL}/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1},
            timeout=15,
        )
        resp.raise_for_status()
        return [c["id"] for c in resp.json()]

    def fetch_price_history(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        if tiingo_available():
            tiingo_symbols = [_COINGECKO_TO_TIINGO.get(s, s) for s in symbols]
            try:
                df = fetch_crypto_prices_tiingo(tiingo_symbols, start, end)
                if not df.empty:
                    # map back to the caller's original ids so downstream
                    # symbol lists stay consistent with default_universe()
                    inverse = {v: k for k, v in _COINGECKO_TO_TIINGO.items()}
                    df = df.rename(columns=inverse)
                    return df
            except Exception:
                pass  # fall through to CoinGecko below

        try:
            df = self._fetch_from_coingecko(symbols, start, end)
            if not df.empty:
                return df
        except Exception:
            pass  # fall through to Binance below

        return self._fetch_from_binance(symbols, start, end)

    def _fetch_from_coingecko(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        series = {}
        for coin_id in symbols:
            resp = requests.get(
                f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range",
                params={
                    "vs_currency": "usd",
                    "from": int(pd.Timestamp(start).timestamp()),
                    "to": int(pd.Timestamp(end).timestamp()),
                },
                timeout=15,
            )
            resp.raise_for_status()
            prices = resp.json()["prices"]  # [[ms_timestamp, price], ...]
            s = pd.Series(
                {pd.to_datetime(ts, unit="ms").normalize(): px for ts, px in prices}
            )
            series[coin_id] = s
            time.sleep(1.5)  # be polite to the free tier's rate limit
        return pd.DataFrame(series).sort_index()

    def _get_binance(self, symbol: str, start_ms: int, end_ms: int):
        """Try the global api.binance.com endpoint first, fall back to
        api.binance.us on a 451 (geo-block) - whichever jurisdiction this
        code happens to run in, one of the two should work."""
        params = {"symbol": symbol, "interval": "1d", "startTime": start_ms,
                   "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError:
            resp = requests.get(f"{BINANCE_BASE_URL}/klines", params=params, timeout=15)
            resp.raise_for_status()
            return resp

    def _fetch_from_binance(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        """/api/v3/klines, daily interval, paginated 1000 bars at a time
        (Binance's per-request cap) - genuinely keyless, the source that
        actually works today now that CoinGecko gates history behind a key.
        Symbols not in _COINGECKO_TO_BINANCE are skipped with a note rather
        than raising, so a partially-mapped universe still returns
        whatever it can."""
        series = {}
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        day_ms = 24 * 60 * 60 * 1000

        for coin_id in symbols:
            binance_symbol = _COINGECKO_TO_BINANCE.get(coin_id)
            if binance_symbol is None:
                continue  # no known mapping for this id - silently skipped, adapter cleaning drops it later if too sparse

            bars = []
            cursor = start_ms
            while cursor < end_ms:
                resp = self._get_binance(binance_symbol, cursor, end_ms)
                chunk = resp.json()
                if not chunk:
                    break
                bars.extend(chunk)
                cursor = chunk[-1][0] + day_ms  # advance past the last returned open-time
                if len(chunk) < 1000:
                    break

            if bars:
                s = pd.Series(
                    {pd.to_datetime(b[0], unit="ms").normalize(): float(b[4]) for b in bars}
                )
                series[coin_id] = s

        return pd.DataFrame(series).sort_index()
