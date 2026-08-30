"""
Stage 01e - Ties a RunConfig to the right adapter and caches the result.
===========================================================================
build_universe() is the one function the rest of the pipeline calls. It:
  1. picks CryptoAdapter or EquityAdapter by RunConfig.engine,
  2. resolves the symbol universe (explicit or adapter default),
  3. fetches raw prices - or reads them back from data_cache/ if a network
     fetch isn't possible, so a demo/offline run always has something to
     show (per the project instructions: "store data so that data will be
     available even when fetches aren't possible"),
  4. hands the cleaned (mu, sigma) back for the classical/QUBO/QAOA stages,
     which never see a raw price or know which engine produced them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from base_adapter import RunConfig, AssetUniverseAdapter
from crypto_adapter import CryptoAdapter
from equity_adapter import EquityAdapter

ENGINES = {
    "crypto": CryptoAdapter,
    "equity": EquityAdapter,
}

# qaoa_academic_engine/data_cache/<engine>_<start>_<end>.csv
DATA_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _cache_path(config: RunConfig, symbols: List[str]) -> Path:
    tag = "-".join(sorted(symbols))[:120]  # keep filenames filesystem-safe
    return DATA_CACHE_DIR / f"{config.engine}_{config.start}_{config.end}_{tag}.csv"


def fetch_raw_prices(config: RunConfig) -> Tuple[pd.DataFrame, AssetUniverseAdapter, List[str]]:
    """Lower-level half of build_universe(): resolves the adapter + symbol
    universe and returns the RAW (uncleaned) price panel, so a caller (the
    pipeline orchestrator) can run stage 02's generic preprocessing on it
    before stage 01's own get_mu_sigma() annualization math runs. Falls
    back to data_cache/ when a live fetch isn't possible."""
    adapter: AssetUniverseAdapter = ENGINES[config.engine]()
    symbols = config.symbols or adapter.default_universe(config.n_assets)

    cache_path = _cache_path(config, symbols)
    prices = None
    try:
        prices = adapter.fetch_price_history(symbols, config.start, config.end)
        if prices is None or prices.empty:
            raise ValueError("empty price panel returned")
        DATA_CACHE_DIR.mkdir(exist_ok=True)
        prices.to_csv(cache_path)
    except Exception as fetch_error:
        if cache_path.exists():
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            raise RuntimeError(
                f"Could not fetch live prices for engine={config.engine} "
                f"({fetch_error}) and no cached copy exists at {cache_path}. "
                f"Run once with network access to populate data_cache/, or "
                f"pass symbols already covered by an existing cache file."
            ) from fetch_error
    return prices, adapter, symbols


def build_universe(config: RunConfig) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Convenience path used when you just want (mu, sigma) with stage 01's
    own cleaning only (no stage 02/03 report needed) - e.g. quick scripts,
    tests. The full dashboard pipeline (pipeline.py) calls
    fetch_raw_prices() directly instead, so it can run stage 02/03 in
    between."""
    prices, adapter, _symbols = fetch_raw_prices(config)
    mu, sigma, kept_symbols = adapter.get_mu_sigma(prices, config.return_freq)
    budget = config.budget or len(kept_symbols) // 2
    return mu, sigma, kept_symbols, budget
