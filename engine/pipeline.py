r"""
Main pipeline orchestrator (stage flow: 01 -> 02 -> 03 -> 04 -> 05 -> 06 ->
07 -> 09, with stage 10 available on request).
==========================================================================
Ties every numbered stage together for ONE (engine, universe) run:

  01 data           -> raw price panel for the requested crypto/equity universe
  02 preprocessing  -> generic cleaning checklist, with a full audit report
  03 EDA            -> descriptive stats + QAOA-hardness proxy on log returns
  04 classical      -> brute-force exact optimum + relaxation baseline
  05 QUBO           -> qiskit-finance QUBO (+ the by-hand Ising derivation)
  06 quantum        -> QAOA solve + exact-QUBO control solve
  07 comparison     -> packages 04+05+06 into one InstanceComparison
  09 statistics     -> (only meaningful across many runs - see scripts/run_benchmark.py)

This is the "same solver core, real data instead of synthetic" pipeline -
directly the successor to qaoa_v1_update1/run_pipeline.py, extended with
the preprocessing/EDA stages and returning a single structured
PipelineRunResult the Streamlit dashboard renders from, instead of printing
to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402  (must run before the bare-name imports below)

from dataclasses import dataclass
from typing import Optional

import numpy as np

from base_adapter import RunConfig, compute_log_returns
from universe_builder import fetch_raw_prices
from cleaning import preprocess_price_panel, PreprocessingReport
from eda import run_eda, EDAReport
from comparison import run_comparison, InstanceComparison


@dataclass
class PipelineRunResult:
    config: RunConfig
    symbols: list
    budget: int
    preprocessing: PreprocessingReport
    eda: EDAReport
    comparison: InstanceComparison


def run(config: RunConfig, reps: int = 2, seed: int = 42) -> PipelineRunResult:
    prices, adapter, _requested_symbols = fetch_raw_prices(config)

    cleaned_prices, prep_report = preprocess_price_panel(prices)

    aligned_prices = adapter._align_and_clean(cleaned_prices)
    log_returns = compute_log_returns(aligned_prices, config.return_freq)

    periods_per_year = {"D": adapter.trading_days_per_year, "W": 52, "M": 12}[config.return_freq]
    mu = log_returns.mean().values * periods_per_year
    sigma = log_returns.cov().values * periods_per_year
    symbols = list(log_returns.columns)
    budget = config.budget or len(symbols) // 2

    eda_report = run_eda(log_returns, mu)

    cmp = run_comparison(mu, sigma, budget, config.risk_factor, reps=reps, seed=seed)

    return PipelineRunResult(
        config=config, symbols=symbols, budget=budget,
        preprocessing=prep_report, eda=eda_report, comparison=cmp,
    )


if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["crypto", "equity"], required=True)
    parser.add_argument("--n-assets", type=int, default=8)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--freq", type=str, default="D", choices=["D", "W", "M"])
    parser.add_argument("--risk-factor", type=float, default=0.5)
    args = parser.parse_args()

    cfg = RunConfig(
        engine=args.engine, n_assets=args.n_assets, budget=args.budget,
        start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
        return_freq=args.freq, risk_factor=args.risk_factor,
    )
    out = run(cfg)
    print(f"[{cfg.engine}] symbols: {out.symbols}")
    print(f"[{cfg.engine}] budget k={out.budget}")
    print("\n--- Preprocessing ---")
    print(out.preprocessing.as_markdown())
    print(f"\n--- EDA hardness proxy: {out.eda.predicted_hardness} "
          f"(s2_ret={out.eda.return_variance_s2_ret:.6f}, s2_cor={out.eda.correlation_variance_s2_cor:.6f}) ---")
    c = out.comparison
    print(f"\n--- Results ---")
    print(f"brute-force optimum : {c.brute_force.value:.4f}")
    print(f"relaxation+round    : {c.relaxation.value:.4f}" if c.relaxation else "relaxation+round    : unavailable")
    print(f"exact-QUBO          : {c.exact_qubo_true_val:.4f} (feasible={c.exact_qubo_feasible})")
    print(f"QAOA                : {c.qaoa_true_val:.4f} (feasible={c.qaoa_feasible}, {c.qaoa.elapsed_s:.2f}s)")
    print(f"approx ratio QAOA   : {c.approx_ratio_qaoa:.4f}")
