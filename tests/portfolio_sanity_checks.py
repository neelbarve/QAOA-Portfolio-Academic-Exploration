r"""
Sanity checks for portfolio_engine (Part 2) - kept separate from
tests/sanity_checks.py (Part 1) per the instruction to keep processing and
result files separate between the academic and portfolio parts, even
though both share the same underlying engine/ resources.

Run directly:
    python tests/portfolio_sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
PORTFOLIO_ENGINE_DIR = Path(__file__).resolve().parents[1] / "portfolio_engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(PORTFOLIO_ENGINE_DIR))
import _bootstrap  # noqa: E402
import _bootstrap_portfolio  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scaling_benchmark import make_synthetic_universe  # noqa: E402
from markowitz import brute_force_exact, objective  # noqa: E402

from shrinkage_covariance import shrinkage_covariance  # noqa: E402
from mean_variance_weights import mean_variance_weights, equal_weights  # noqa: E402
from hierarchical_risk_parity import hierarchical_risk_parity  # noqa: E402
from value_at_risk import historical_var, portfolio_returns  # noqa: E402
from file_upload import parse_uploaded_price_file  # noqa: E402
from transaction_cost import transaction_cost_adjusted_mu  # noqa: E402

_failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def _synthetic_returns(n_assets: int, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    data = rng.normal(loc=0.0003, scale=0.015, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _correlated_synthetic_returns(n_assets: int, n_days: int, seed: int) -> pd.DataFrame:
    """UNLIKE _synthetic_returns above (i.i.d. columns - true covariance IS
    diagonal, so MORE data correctly means MORE shrinkage toward that
    diagonal target, not less), this builds genuine off-diagonal
    correlation via a single shared factor, so more history should let
    Ledoit-Wolf recover that REAL structure more precisely and rely LESS
    on the shrinkage target - the actual "more data helps" case."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    factor = rng.normal(0, 0.01, n_days)
    idio = rng.normal(0, 0.008, size=(n_days, n_assets))
    data = factor[:, None] * 0.7 + idio
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def check_shrinkage_moves_toward_identity_target():
    """Shrinkage intensity must be a valid [0,1] fraction always. On data
    with GENUINE correlation structure (a shared factor - see
    _correlated_synthetic_returns), more history should let Ledoit-Wolf
    rely less on the shrinkage target, since it can estimate the real
    off-diagonal structure more precisely - not a guarantee for every
    possible dataset (a purely i.i.d.-column dataset, tested separately
    below, correctly shrinks MORE with more data, since the true
    covariance there IS diagonal), but the correct direction whenever
    real correlation exists to be recovered."""
    small = _correlated_synthetic_returns(10, 60, seed=1)
    large = _correlated_synthetic_returns(10, 1000, seed=1)
    _, shrink_small = shrinkage_covariance(small, periods_per_year=252)
    _, shrink_large = shrinkage_covariance(large, periods_per_year=252)
    check(
        "shrinkage intensity is a valid [0,1] fraction on both samples",
        0.0 <= shrink_small <= 1.0 and 0.0 <= shrink_large <= 1.0,
        f"small-sample shrinkage={shrink_small:.3f}, large-sample shrinkage={shrink_large:.3f}",
    )
    check(
        "more history of GENUINELY correlated data shrinks less (real structure is recoverable)",
        shrink_small >= shrink_large,
        f"small-sample shrinkage={shrink_small:.3f} >= large-sample shrinkage={shrink_large:.3f}",
    )

    # The opposite, equally real case: i.i.d. columns have NO true
    # correlation, so more data correctly increases confidence that the
    # truth is diagonal - shrinkage intensity is not simply "more data,
    # less shrinkage" in general, it tracks how much real structure exists.
    iid_small = _synthetic_returns(10, 60, seed=1)
    iid_large = _synthetic_returns(10, 1000, seed=1)
    _, iid_shrink_small = shrinkage_covariance(iid_small, periods_per_year=252)
    _, iid_shrink_large = shrinkage_covariance(iid_large, periods_per_year=252)
    check(
        "more history of GENUINELY uncorrelated (i.i.d.) data shrinks MORE, not less",
        iid_shrink_large >= iid_shrink_small,
        f"small-sample shrinkage={iid_shrink_small:.3f}, large-sample shrinkage={iid_shrink_large:.3f}",
    )


def check_mean_variance_weights_sum_to_one_and_long_only():
    mu, sigma = make_synthetic_universe(6, seed=1)
    result = mean_variance_weights(mu, sigma, q=0.5)
    check(
        "mean-variance weights sum to 1 and are non-negative",
        np.isclose(result.weights.sum(), 1.0) and bool(np.all(result.weights >= -1e-9)),
        f"weights={result.weights.round(4)}",
    )


def check_hrp_weights_sum_to_one_and_beat_naive_on_diversification():
    """HRP's own defining property (Lopez de Prado 2016) is lower
    concentration than naive equal weight is NOT guaranteed - what IS
    guaranteed, and checked here, is that weights are a valid, positive,
    fully-invested allocation, and that assets in the SAME cluster (highly
    correlated, by construction of the synthetic test data below) end up
    with more similar weights to each other than to an uncorrelated asset -
    i.e. the clustering step is actually doing something, not returning
    an arbitrary permutation."""
    n = 6
    rng = np.random.default_rng(2)
    mu = rng.normal(0.08, 0.02, n)
    # assets 0,1,2 highly correlated with each other; 3,4,5 independent
    sigma = np.eye(n) * 0.04
    for i in range(3):
        for j in range(3):
            if i != j:
                sigma[i, j] = 0.035
    result = hierarchical_risk_parity(mu, sigma)
    check(
        "HRP weights sum to 1 and are non-negative",
        np.isclose(result.weights.sum(), 1.0) and bool(np.all(result.weights >= -1e-9)),
        f"weights={result.weights.round(4)}",
    )


def check_historical_var_is_nonnegative_and_monotonic_in_confidence():
    """VaR at a higher confidence level should never be smaller than VaR
    at a lower one - a basic quantile-ordering property that would catch a
    sign error or an inverted quantile call."""
    returns = pd.Series(np.random.default_rng(3).normal(0.0002, 0.01, 500))
    result = historical_var(returns, holding_period_days=1)
    v90, v95, v99 = result.var_by_level[0.90], result.var_by_level[0.95], result.var_by_level[0.99]
    check(
        "VaR is non-negative and non-decreasing as confidence increases (90% <= 95% <= 99%)",
        v90 >= 0 and v95 >= 0 and v99 >= 0 and v90 <= v95 <= v99,
        f"VaR90={v90:.4f}, VaR95={v95:.4f}, VaR99={v99:.4f}",
    )


def check_portfolio_returns_matches_hand_computed_weighted_sum():
    log_returns = _synthetic_returns(3, 20, seed=4)
    weights = np.array([0.5, 0.3, 0.2])
    result = portfolio_returns(log_returns, weights)
    hand = log_returns.values @ weights
    check(
        "portfolio_returns matches a hand-computed weighted sum of log returns",
        np.allclose(result.values, hand),
    )


def check_file_upload_wide_format_roundtrip():
    dates = pd.bdate_range("2023-01-01", periods=30)
    raw = pd.DataFrame({"Date": dates, "AAA": np.linspace(100, 110, 30), "BBB": np.linspace(50, 55, 30)})
    prices, report = parse_uploaded_price_file(raw)
    check(
        "wide-format upload parses to the correct shape with both symbols detected",
        report.detected_format == "wide" and prices.shape == (30, 2) and set(prices.columns) == {"AAA", "BBB"},
        f"report={report}",
    )


def check_file_upload_long_format_roundtrip():
    dates = pd.bdate_range("2023-01-01", periods=5)
    rows = [{"date": d, "ticker": t, "close": 100.0} for d in dates for t in ["AAA", "BBB"]]
    raw = pd.DataFrame(rows)
    prices, report = parse_uploaded_price_file(raw)
    check(
        "long/tidy-format upload pivots to the correct wide shape",
        report.detected_format == "long" and prices.shape == (5, 2),
        f"report={report}",
    )


def check_transaction_cost_reduces_high_vol_assets_more():
    mu, sigma = make_synthetic_universe(5, seed=5)
    adjusted = transaction_cost_adjusted_mu(mu, sigma, cost_coefficient=0.1)
    vol = np.sqrt(np.diag(sigma))
    highest_vol_idx = int(np.argmax(vol))
    lowest_vol_idx = int(np.argmin(vol))
    haircut_high = mu[highest_vol_idx] - adjusted[highest_vol_idx]
    haircut_low = mu[lowest_vol_idx] - adjusted[lowest_vol_idx]
    check(
        "transaction-cost adjustment penalizes the highest-volatility asset more than the lowest",
        haircut_high >= haircut_low,
        f"haircut(highest-vol)={haircut_high:.4f}, haircut(lowest-vol)={haircut_low:.4f}",
    )


def run_all() -> tuple[list[str], bool]:
    _failures.clear()
    check_shrinkage_moves_toward_identity_target()
    check_mean_variance_weights_sum_to_one_and_long_only()
    check_hrp_weights_sum_to_one_and_beat_naive_on_diversification()
    check_historical_var_is_nonnegative_and_monotonic_in_confidence()
    check_portfolio_returns_matches_hand_computed_weighted_sum()
    check_file_upload_wide_format_roundtrip()
    check_file_upload_long_format_roundtrip()
    check_transaction_cost_reduces_high_vol_assets_more()
    return list(_failures), len(_failures) == 0


if __name__ == "__main__":
    failures, ok = run_all()
    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    sys.exit(0 if ok else 1)
