"""
Stage P01b - Generic extraction from a user-uploaded data file.
=====================================================================
"Generalization of extraction, identification from data file uploaded by
user" (instructions). A user's CSV/Excel export can look like almost
anything: dates as the index or a named column in any common format,
one price column per ticker (wide) or a long/tidy format with
symbol/date/price columns, different column-name conventions ("Close",
"Adj Close", "price", "close_price"...). This module makes a best-effort,
transparent guess at the shape and hands back the SAME (prices, adapter)
contract engine's own adapters produce, so everything downstream (stage
02 preprocessing, mu/Sigma, QAOA/classical solve) is identical whether the
data came from a live fetch or a user's file - no special-cased "uploaded
data" branch anywhere else in the pipeline.

Deliberately conservative: raises a clear, specific error rather than
silently guessing wrong when the shape is genuinely ambiguous, since a
wrong guess here (e.g. treating a returns column as a price column) would
silently corrupt every result downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

_DATE_COLUMN_CANDIDATES = ["date", "Date", "DATE", "timestamp", "Timestamp", "time"]
_PRICE_COLUMN_CANDIDATES = ["close", "Close", "CLOSE", "adj close", "Adj Close", "price", "Price"]
_SYMBOL_COLUMN_CANDIDATES = ["symbol", "Symbol", "ticker", "Ticker", "SYMBOL", "TICKER"]


@dataclass
class UploadParseReport:
    detected_format: str      # "wide" | "long"
    date_column: Optional[str]
    symbol_column: Optional[str]
    price_column: Optional[str]
    n_symbols: int
    n_rows: int
    dropped_columns: List[str]


def parse_uploaded_price_file(raw: pd.DataFrame) -> tuple[pd.DataFrame, UploadParseReport]:
    """raw: whatever pandas.read_csv/read_excel returned, untouched.
    Returns (prices, report) where prices is a date-indexed DataFrame, one
    column per symbol, matching AssetUniverseAdapter.fetch_price_history's
    own contract exactly - the file_upload "adapter" plugs into the same
    downstream pipeline as CryptoAdapter/EquityAdapter."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in _DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col is None:
        # fall back to: any column that parses as a date for >90% of rows
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            "Could not identify a date column in the uploaded file. Expected one of "
            f"{_DATE_COLUMN_CANDIDATES}, or a column where >90% of values parse as dates."
        )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    symbol_col = next((c for c in _SYMBOL_COLUMN_CANDIDATES if c in df.columns), None)
    price_col = next((c for c in _PRICE_COLUMN_CANDIDATES if c in df.columns), None)

    if symbol_col is not None and price_col is not None:
        # long/tidy format: one row per (date, symbol), a price column, an
        # optional symbol column - pivot to wide.
        wide = df.pivot_table(index=df.index, columns=symbol_col, values=price_col, aggfunc="last")
        dropped = [c for c in df.columns if c not in (symbol_col, price_col)]
        report = UploadParseReport(
            detected_format="long", date_column=date_col, symbol_column=symbol_col,
            price_column=price_col, n_symbols=wide.shape[1], n_rows=wide.shape[0],
            dropped_columns=dropped,
        )
        return wide, report

    # wide format: every remaining numeric column is treated as one
    # symbol's price series. Non-numeric columns (e.g. a stray text note
    # column) are dropped, not silently coerced.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dropped = [c for c in df.columns if c not in numeric_cols]
    if not numeric_cols:
        raise ValueError(
            "No numeric price columns found after identifying the date column "
            f"('{date_col}'). If this is a long/tidy file, it needs a recognizable "
            f"symbol column ({_SYMBOL_COLUMN_CANDIDATES}) and price column "
            f"({_PRICE_COLUMN_CANDIDATES})."
        )
    wide = df[numeric_cols]
    report = UploadParseReport(
        detected_format="wide", date_column=date_col, symbol_column=None, price_column=None,
        n_symbols=wide.shape[1], n_rows=wide.shape[0], dropped_columns=dropped,
    )
    return wide, report
