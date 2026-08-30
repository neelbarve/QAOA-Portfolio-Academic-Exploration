"""
Stage 02 - Generic data pre-processing pipeline.
===================================================
Applied to the raw price panel returned by stage 01 (universe_builder),
BEFORE stage 01's own get_mu_sigma() return/annualization math runs. This
stage is deliberately generic (works on any wide numeric time-series panel,
crypto or equity) and follows the project's preprocessing checklist:

  1. Identify and delete irrelevant columns (if any).
  2. Report data distributions + dtypes for relevant columns.
  3. Check missing values; drop columns with >=35% missing, and record
     exactly what was dropped and why (surfaced in the dashboard + README).
  4. Fill missing values: mode for categorical, median for skewed numeric
     distributions, mean otherwise. A daily price series is numeric and
     typically NOT symmetric (crypto especially), so median imputation is
     used for price gaps; this is stated explicitly in the report below
     rather than silently choosing one.
  5. Encode categorical variables - N/A for this dataset (price panels are
     purely numeric, one column per asset); reported explicitly rather than
     skipped silently, per the instruction to "mention it if any is missing".

Returns a PreprocessingReport dataclass consumed directly by the Streamlit
academic page's "sanity check" tab and by the README data section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

MISSING_DROP_THRESHOLD = 0.35  # per instructions: drop columns >=35% missing


@dataclass
class PreprocessingReport:
    n_rows_in: int
    n_cols_in: int
    irrelevant_columns_dropped: List[str] = field(default_factory=list)
    missing_pct_by_column: Dict[str, float] = field(default_factory=dict)
    columns_dropped_missing: List[str] = field(default_factory=list)
    columns_imputed: Dict[str, str] = field(default_factory=dict)  # col -> method used
    categorical_columns_encoded: List[str] = field(default_factory=list)
    dtypes_by_column: Dict[str, str] = field(default_factory=dict)
    n_rows_out: int = 0
    n_cols_out: int = 0
    notes: List[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = [
            f"- Input panel: {self.n_rows_in} rows x {self.n_cols_in} columns.",
            f"- Irrelevant columns dropped: "
            f"{', '.join(self.irrelevant_columns_dropped) or 'none found'}.",
            f"- Columns dropped for >= {MISSING_DROP_THRESHOLD:.0%} missing data: "
            f"{', '.join(self.columns_dropped_missing) or 'none'}.",
            f"- Columns imputed: "
            f"{', '.join(f'{c} ({m})' for c, m in self.columns_imputed.items()) or 'none needed'}.",
            f"- Categorical columns encoded: "
            f"{', '.join(self.categorical_columns_encoded) or 'N/A (price panel is purely numeric)'}.",
            f"- Output panel: {self.n_rows_out} rows x {self.n_cols_out} columns.",
        ]
        lines.extend(f"- Note: {n}" for n in self.notes)
        return "\n".join(lines)


def _is_skewed(series: pd.Series, threshold: float = 0.5) -> bool:
    """Simple, transparent skew check (Pearson's moment coefficient of
    skewness via pandas .skew()) - used only to decide median vs mean
    imputation, not for any modeling claim."""
    s = series.dropna()
    if len(s) < 3:
        return False
    return abs(s.skew()) > threshold


def preprocess_price_panel(
    prices: pd.DataFrame, irrelevant_columns: List[str] | None = None
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Run the full checklist against a raw wide price panel (index = date,
    columns = asset symbols). Returns the cleaned panel plus a report object
    documenting every decision made, so nothing is silently dropped."""
    report = PreprocessingReport(n_rows_in=len(prices), n_cols_in=prices.shape[1])
    df = prices.copy()

    # 1. Irrelevant columns (e.g. an accidental index/ID column that isn't a
    # price series - none expected from stage 01's adapters, but checked
    # explicitly so the step exists and is auditable rather than assumed).
    irrelevant_columns = irrelevant_columns or []
    present_irrelevant = [c for c in irrelevant_columns if c in df.columns]
    if present_irrelevant:
        df = df.drop(columns=present_irrelevant)
    report.irrelevant_columns_dropped = present_irrelevant

    # 2. Distributions / dtypes (recorded, not printed - the dashboard
    # renders histograms straight from the returned cleaned frame).
    report.dtypes_by_column = {c: str(t) for c, t in df.dtypes.items()}

    # 3. Missing-value audit + >=35% drop rule.
    missing_pct = df.isna().mean()
    report.missing_pct_by_column = missing_pct.round(4).to_dict()
    drop_cols = missing_pct[missing_pct >= MISSING_DROP_THRESHOLD].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    report.columns_dropped_missing = drop_cols

    # 4. Impute what's left: median for skewed numeric columns, mean for
    # roughly-symmetric numeric columns. (Mode/categorical branch is defined
    # for completeness but never triggers on a numeric price panel - see
    # note 5 below.)
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if _is_skewed(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    report.columns_imputed[col] = "median (skewed numeric)"
                else:
                    df[col] = df[col].fillna(df[col].mean())
                    report.columns_imputed[col] = "mean (approx. symmetric numeric)"
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
                report.columns_imputed[col] = "mode (categorical)"

    # 5. Categorical encoding - explicitly N/A here; recorded so the README
    # and dashboard say so rather than the step silently disappearing.
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        report.categorical_columns_encoded = categorical_cols
    else:
        report.notes.append(
            "No categorical columns in a price panel - one-hot encoding step is N/A."
        )

    df = df.dropna(how="any")  # any residual full-row gaps after imputation
    report.n_rows_out, report.n_cols_out = df.shape
    return df, report
