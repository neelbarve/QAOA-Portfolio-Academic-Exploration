r"""
Stage 10 - Fama-French factor model -> QAOA Hamiltonian.
=============================================================
The dashboard page this module backs is explicitly requested by the project
instructions: "explore the mathematical formulation of how a Fama-French
matrix converts into a QAOA Hamiltonian". This module runs that conversion
for real, end to end, on actual asset return data - it is not a prose-only
explainer.

------------------------------------------------------------------
The math, in order:

1. Fama-French three-factor model (Fama & French, 1993). For each asset i,
   regress its excess return on three factors via OLS:

       R_i(t) - RF(t) = alpha_i + b_i1*MktRF(t) + b_i2*SMB(t) + b_i3*HML(t) + eps_i(t)

   MktRF = market excess return, SMB = "Small Minus Big" (size factor),
   HML = "High Minus Low" (value factor). Factor returns are Ken French's
   published daily series (Dartmouth Tuck Data Library, a standard,
   public, citation-grade academic source - see fetch_fama_french_factors
   below); a clearly-labeled synthetic fallback is used if that fetch
   fails, exactly like the price-data caching in stage 01.

2. Factor covariance/idiosyncratic-risk decomposition. Let B be the (n x 3)
   matrix of factor loadings [b_i1, b_i2, b_i3], F the (3x3) sample
   covariance of the three factor return series, and D = diag(var(eps_i))
   the idiosyncratic (stock-specific) variances. The factor model implies

       Sigma_FF = B @ F @ B^T + D

   This Sigma_FF is a genuinely different covariance estimate than stage
   01's plain sample covariance of raw returns - it explains covariance
   through shared exposure to three priced risk factors rather than raw
   historical co-movement, and is the standard practitioner alternative
   precisely because it's better-conditioned on small samples (n assets,
   3 factors, versus estimating n(n+1)/2 raw covariance entries directly).

   Expected returns under the same model:

       mu_FF_i = alpha_i + B_i @ mean(factor returns)

3. QUBO -> Ising conversion. (mu_FF, Sigma_FF) are just another (mu, Sigma)
   pair from the solver core's point of view - stage 05's
   to_ising_hamiltonian(mu_FF, Sigma_FF, k, q) runs completely unmodified
   and produces the same H(z) = sum h_i z_i + sum J_ij z_i z_j + const
   structure used everywhere else in this project. That reuse IS the
   point: the Fama-French model only ever changes what (mu, Sigma) means,
   never how the combinatorial optimization or the QAOA circuit works.
------------------------------------------------------------------
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

FF_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def fetch_fama_french_factors(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    """Daily Mkt-RF, SMB, HML, RF (%, as published) from the Ken French
    Data Library, indexed by date. Returns (factors_df, source_label) so
    callers/UI can show whether real or synthetic factors were used - the
    synthetic fallback is deliberately visible, never silent."""
    try:
        resp = requests.get(FF_FACTORS_URL, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            raw = zf.read(csv_name).decode("latin-1")

        # The published file has a text header, a daily-data block keyed by
        # YYYYMMDD row labels, then a blank line and an ANNUAL block below -
        # keep only rows whose first field is an 8-digit date.
        lines = raw.splitlines()
        data_lines = [ln for ln in lines if len(ln) > 8 and ln[:8].strip().isdigit()]
        buf = io.StringIO("\n".join(data_lines))
        df = pd.read_csv(
            buf, header=None,
            names=["date", "MktRF", "SMB", "HML", "RF"],
        )
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.set_index("date").astype(float) / 100.0  # published in percent
        df = df.loc[(df.index >= start) & (df.index <= end)]
        if df.empty:
            raise ValueError("no Fama-French rows in requested date range")
        return df, "Ken French Data Library (live fetch)"
    except Exception:
        return _synthetic_factors(start, end), "SYNTHETIC placeholder (live fetch failed)"


def _synthetic_factors(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Clearly-labeled stand-in with realistic factor-return statistics
    (annualized vol/mean roughly matching historical Mkt-RF/SMB/HML), used
    only so the pipeline still runs end to end without network access -
    never mixed silently with real data."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(7)
    mkt = rng.normal(0.0003, 0.010, len(dates))
    smb = rng.normal(0.0000, 0.006, len(dates))
    hml = rng.normal(0.0000, 0.006, len(dates))
    rf = np.full(len(dates), 0.00007)
    return pd.DataFrame({"MktRF": mkt, "SMB": smb, "HML": hml, "RF": rf}, index=dates)


@dataclass
class FamaFrenchModel:
    symbols: List[str]
    alpha: np.ndarray            # (n,)
    beta: np.ndarray             # (n, 3) loadings on [MktRF, SMB, HML]
    factor_cov: np.ndarray       # (3, 3), annualized
    idiosyncratic_var: np.ndarray  # (n,), annualized
    mu_ff: np.ndarray            # (n,), annualized
    sigma_ff: np.ndarray         # (n, n), annualized
    r_squared: np.ndarray        # (n,), OLS fit quality per asset
    factor_source: str
    trading_days_per_year: int


def fit_fama_french(
    asset_log_returns: pd.DataFrame, factors: pd.DataFrame, trading_days_per_year: int = 252,
) -> FamaFrenchModel:
    """asset_log_returns: cleaned per-period log returns, one column per
    symbol (stage 01/02 output). factors: output of
    fetch_fama_french_factors, aligned by date."""
    aligned = asset_log_returns.join(factors, how="inner").dropna()
    symbols = list(asset_log_returns.columns)
    X = sm.add_constant(aligned[["MktRF", "SMB", "HML"]].values)

    n = len(symbols)
    alpha = np.zeros(n)
    beta = np.zeros((n, 3))
    r2 = np.zeros(n)
    residuals = np.zeros((len(aligned), n))

    for i, sym in enumerate(symbols):
        y = (aligned[sym] - aligned["RF"]).values  # excess return
        model = sm.OLS(y, X).fit()
        alpha[i] = model.params[0]
        beta[i, :] = model.params[1:4]
        r2[i] = model.rsquared
        residuals[:, i] = model.resid

    factor_cov = np.cov(aligned[["MktRF", "SMB", "HML"]].values, rowvar=False) * trading_days_per_year
    idio_var = residuals.var(axis=0, ddof=1) * trading_days_per_year

    sigma_ff = beta @ factor_cov @ beta.T + np.diag(idio_var)
    mu_ff = (alpha + beta @ aligned[["MktRF", "SMB", "HML"]].mean().values) * trading_days_per_year

    return FamaFrenchModel(
        symbols=symbols, alpha=alpha * trading_days_per_year, beta=beta,
        factor_cov=factor_cov, idiosyncratic_var=idio_var,
        mu_ff=mu_ff, sigma_ff=sigma_ff, r_squared=r2,
        factor_source="", trading_days_per_year=trading_days_per_year,
    )


def build_hamiltonian_from_fama_french(
    model: FamaFrenchModel, k: int, q: float,
):
    """The stage-05 -> stage-10 handoff described in the module docstring:
    feed (mu_FF, Sigma_FF) straight into the same manual Ising derivation
    used for the plain sample-covariance path, unmodified."""
    from manual_ising import to_ising_hamiltonian  # stage 05b, added to sys.path by engine bootstrap
    return to_ising_hamiltonian(model.mu_ff, model.sigma_ff, k, q)
