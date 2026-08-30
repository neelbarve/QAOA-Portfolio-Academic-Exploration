r"""
Generates the real example runs (one equity, one crypto) and the
comparative plots the portfolio README references - the Part 2 analogue
of scripts/make_readme_plots.py, kept in its own script/output pair per
the "keep processing/result files separate" instruction.

Usage:
    python scripts/run_portfolio_examples.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
PORTFOLIO_ENGINE_DIR = Path(__file__).resolve().parents[1] / "portfolio_engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(PORTFOLIO_ENGINE_DIR))
import _bootstrap  # noqa: E402
import _bootstrap_portfolio  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_sources import UniverseSelection  # noqa: E402
from portfolio_pipeline import run_portfolio_pipeline  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "portfolio_results"
PLOTS_DIR = RESULTS_DIR / "plots"


def _summarize(result) -> dict:
    def wp_summary(wp):
        if wp is None:
            return None
        return {
            "method": wp.selection.method, "symbols": wp.selection.selected_symbols,
            "feasible": wp.selection.feasible, "converged": wp.selection.converged,
            "elapsed_s": wp.selection.elapsed_s,
            "weights": wp.weights.weights.tolist(), "weight_method": wp.weights.method,
            "expected_return": wp.weights.expected_return, "expected_variance": wp.weights.expected_variance,
            "portfolio_var": wp.portfolio_var.var_by_level,
        }
    return {
        "symbols": result.symbols, "n_assets": result.n_assets, "k": result.k,
        "shrinkage_intensity": result.sigma_shrinkage_intensity,
        "quantum": wp_summary(result.quantum_portfolio),
        "classical": wp_summary(result.classical_portfolio),
        "quantum_skipped_reason": result.quantum_skipped_reason,
    }


def _plot(result, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, wp in [("QAOA", result.quantum_portfolio), ("Classical", result.classical_portfolio)]:
        if wp is None:
            continue
        idx = wp.selection.selected_indices
        rets = result.log_returns.iloc[:, idx].values @ wp.weights.weights
        cum = np.exp(np.cumsum(rets))
        ax.plot(result.log_returns.index, cum, label=f"{label} ({wp.weights.method})")
    equal_w = np.full(result.n_assets, 1.0 / result.n_assets)
    cum_all = np.exp(np.cumsum(result.log_returns.values @ equal_w))
    ax.plot(result.log_returns.index, cum_all, "--", label="Equal-weight, all assets")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative value")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    print("=== Equity example: tech sector, n=8, 3 months holding, mean-variance weights ===")
    equity_sel = UniverseSelection(engine="equity", sector="tech", n_assets=8, holding_period="3 Months", q=0.5)
    equity_result = run_portfolio_pipeline(equity_sel, weight_method="mean_variance", reps=2)
    with open(RESULTS_DIR / "example_equity.json", "w") as f:
        json.dump(_summarize(equity_result), f, indent=2)
    _plot(equity_result, "Equity example (tech, n=8): portfolio dynamics", PLOTS_DIR / "equity_example_dynamics.png")
    print(json.dumps(_summarize(equity_result), indent=2)[:2000])

    print("\n=== Crypto example: diversified, n=6, 1 month holding, HRP weights ===")
    crypto_sel = UniverseSelection(engine="crypto", sector="diversified", n_assets=6, holding_period="1 Month", q=0.5)
    crypto_result = run_portfolio_pipeline(crypto_sel, weight_method="hrp", reps=2)
    with open(RESULTS_DIR / "example_crypto.json", "w") as f:
        json.dump(_summarize(crypto_result), f, indent=2)
    _plot(crypto_result, "Crypto example (diversified, n=6): portfolio dynamics", PLOTS_DIR / "crypto_example_dynamics.png")
    print(json.dumps(_summarize(crypto_result), indent=2)[:2000])

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
