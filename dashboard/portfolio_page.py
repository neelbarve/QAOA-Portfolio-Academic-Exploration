r"""
Page 2/2 - Portfolio dashboard (Part 2 of the project).
================================================================
The full "select engine -> select sector -> select n stocks -> select
holding period -> run" flow from the instructions doc, plus a generic
file-upload path. Reuses the academic engine's stages 01-07 as a library
(same adapters, same solvers, same data_cache/) and portfolio_engine/'s
own stages on top (sector universes + holding periods, shrinkage
covariance + weighting, historical VaR, transaction-cost adjustment) -
see portfolio_engine/portfolio_pipeline.py for the orchestration.

Every result is produced by BOTH QAOA and a classical method on the same
real data (brute force or simulated annealing, chosen automatically by
size - see portfolio_pipeline.QAOA_MAX_N) and shown side by side, per the
instruction to "use both qaoa and classic optimizers... and compare
effects on actual data."
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import UniverseSelection, SECTOR_CHOICES, HOLDING_PERIODS
from file_upload import parse_uploaded_price_file
from portfolio_pipeline import (
    run_portfolio_pipeline, run_portfolio_pipeline_from_returns,
    PortfolioRunResult, WeightedPortfolio, QAOA_MAX_N,
)
from base_adapter import compute_log_returns
from sector_constrained import assign_sectors, brute_force_sector_constrained
from markowitz import brute_force_exact

CONSTRAINT_STACK_MAX_N = 20  # brute force at every sector count must stay a dashboard-click, not a coffee break


# ---------------------------------------------------------------------
# Cached pipeline calls - same reasoning as academic_page.py's
# _cached_run: Streamlit re-executes this whole module on every widget
# interaction, so caching is what keeps "click Run" from re-fetching data
# or re-solving QAOA on an unrelated control change elsewhere on the page.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_live_run(engine, sector, n_assets, holding_period, q, weight_method,
                      use_shrinkage, tx_cost, reps):
    sel = UniverseSelection(engine=engine, sector=sector, n_assets=n_assets,
                             holding_period=holding_period, q=q)
    return run_portfolio_pipeline(
        sel, weight_method=weight_method, reps=reps, use_shrinkage=use_shrinkage,
        transaction_cost_coefficient=tx_cost,
    )


@st.cache_data(show_spinner=False)
def _cached_upload_run(file_bytes, file_name, n_assets, holding_period, q,
                        weight_method, use_shrinkage, tx_cost, reps):
    if file_name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    prices, report = parse_uploaded_price_file(raw)
    prices = prices.iloc[:, :n_assets]
    log_returns = compute_log_returns(prices, return_freq="D")
    symbols = list(log_returns.columns)
    result = run_portfolio_pipeline_from_returns(
        log_returns, symbols, q, holding_period, periods_per_year=252,
        weight_method=weight_method, reps=reps, use_shrinkage=use_shrinkage,
        transaction_cost_coefficient=tx_cost,
    )
    return result, report


def _config_panel():
    st.subheader("Configure a portfolio")
    source = st.radio("Data source", ["Live fetch", "Upload your own file"], horizontal=True, key="pf_source")

    engine = sector = None
    uploaded_file = None
    if source == "Live fetch":
        c1, c2, c3 = st.columns(3)
        with c1:
            engine = st.selectbox("Engine (asset class)", ["equity", "crypto"], key="pf_engine")
        with c2:
            sector_options = ["(no filter - diversified default)"] + SECTOR_CHOICES[engine]
            sector_choice = st.selectbox("Sector filter", sector_options, key="pf_sector")
            sector = None if sector_choice.startswith("(no filter") else sector_choice
        with c3:
            n_assets = st.slider("Number of stocks (n)", min_value=2, max_value=50, value=8, key="pf_n")
    else:
        uploaded_file = st.file_uploader("Upload a price history file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        n_assets = st.slider("Number of columns to use as assets (n)", min_value=2, max_value=50, value=8, key="pf_n_upload")

    c4, c5, c6 = st.columns(3)
    with c4:
        holding_period = st.selectbox("Holding period", list(HOLDING_PERIODS.keys()),
                                       index=2, key="pf_holding")
    with c5:
        q = st.slider("Risk aversion q", 0.0, 1.0, 0.5, 0.05, key="pf_q")
    with c6:
        weight_method = st.selectbox(
            "Weighting method", ["mean_variance", "hrp", "equal"],
            format_func=lambda m: {"mean_variance": "Mean-variance (continuous QP)",
                                    "hrp": "Hierarchical Risk Parity",
                                    "equal": "Equal weight"}[m],
            key="pf_weight_method",
        )

    c7, c8, c9 = st.columns(3)
    with c7:
        use_shrinkage = st.checkbox("Use Ledoit-Wolf shrinkage covariance", value=True, key="pf_shrinkage")
    with c8:
        use_tx_cost = st.checkbox("Apply transaction-cost adjustment", value=False, key="pf_tx_cost_on")
        tx_cost = st.slider("Cost coefficient", 0.0, 0.5, 0.1, 0.05, key="pf_tx_cost",
                             disabled=not use_tx_cost) if use_tx_cost else None
    with c9:
        reps = st.slider("QAOA layers (p)", 1, 5, 2, key="pf_reps",
                          help=f"Only used when n <= {QAOA_MAX_N}; see the note below for why.")

    if n_assets > QAOA_MAX_N:
        st.caption(
            f"You requested n={n_assets} > {QAOA_MAX_N}: if the resolved universe (after any sector-list "
            f"or data-cleaning truncation) is still above {QAOA_MAX_N}, QAOA will be skipped for this run "
            f"in favor of simulated annealing - see the 'Is There An Edge' tab on the Academic page for "
            f"why, and the 'Universe' line after running for the actual count used."
        )

    run_clicked = st.button("Run portfolio construction", type="primary")
    return dict(
        source=source, engine=engine, sector=sector, n_assets=n_assets,
        holding_period=holding_period, q=q, weight_method=weight_method,
        use_shrinkage=use_shrinkage, tx_cost=tx_cost, reps=reps,
        uploaded_file=uploaded_file, run_clicked=run_clicked,
    )


def _results_table(wp: WeightedPortfolio) -> pd.DataFrame:
    rows = []
    for i, sym in enumerate(wp.selection.selected_symbols):
        var = wp.per_asset_var[sym]
        rows.append({
            "Symbol": sym,
            "Weight": wp.weights.weights[i],
            "Expected return (annualized)": wp.weights.expected_return,
            f"VaR {int(90)}% ({var.holding_period_days}d)": var.var_by_level[0.90],
            f"VaR {int(95)}% ({var.holding_period_days}d)": var.var_by_level[0.95],
            f"VaR {int(99)}% ({var.holding_period_days}d)": var.var_by_level[0.99],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def _render_portfolio_block(label: str, wp: Optional[WeightedPortfolio], skip_reason: Optional[str], k: int):
    st.markdown(f"#### {label}")
    if wp is None:
        st.info(skip_reason, icon="ℹ️")
        return

    n_selected = len(wp.selection.selected_symbols)
    if not wp.selection.feasible or n_selected != k:
        st.error(
            f"**INFEASIBLE selection**: {n_selected} assets selected against a requested budget of "
            f"k={k}. Shown anyway rather than hidden - this is the standard QAOA failure mode "
            f"documented on the Academic page's 'Is There An Edge' tab (the classical optimizer "
            f"inside QAOA didn't concentrate probability on a budget-satisfying bitstring within its "
            f"iteration budget), now reproduced on real portfolio data. Weights and VaR below are "
            f"computed on the {n_selected} assets actually returned, not a corrected k-asset set.",
            icon="🚫",
        )

    method_label = {
        "qaoa": "QAOA", "brute_force": "Brute force (exact)",
        "simulated_annealing": "Simulated annealing",
    }.get(wp.selection.method, wp.selection.method)
    conv_str = "" if wp.selection.converged is None else f", converged={wp.selection.converged}"
    st.caption(
        f"Selection method: **{method_label}** ({wp.selection.elapsed_s:.2f}s{conv_str}, "
        f"feasible={wp.selection.feasible}) | "
        f"Weighting: **{wp.weights.method}** | "
        f"Portfolio expected return: **{wp.weights.expected_return:.2%}**, "
        f"variance: **{wp.weights.expected_variance:.4f}**"
    )

    table = _results_table(wp)
    min_weight = st.slider(f"Filter: minimum weight ({label})", 0.0, float(table["Weight"].max()), 0.0,
                            0.01, key=f"pf_min_weight_{label}")
    filtered = table[table["Weight"] >= min_weight]
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        f"Download {label} table as CSV", data=filtered.to_csv(index=False),
        file_name=f"portfolio_{label.lower().replace(' ', '_')}.csv", mime="text/csv",
        key=f"pf_csv_{label}",
    )

    st.markdown("**Portfolio-level VaR**")
    pv = wp.portfolio_var
    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR 90% ({pv.holding_period_days}d)", f"{pv.var_by_level[0.90]:.2%}")
    c2.metric(f"VaR 95% ({pv.holding_period_days}d)", f"{pv.var_by_level[0.95]:.2%}")
    c3.metric(f"VaR 99% ({pv.holding_period_days}d)", f"{pv.var_by_level[0.99]:.2%}")
    if not pv.reliable:
        st.caption(
            f"Only {pv.n_windows} historical {pv.holding_period_days}-day windows available - "
            f"below the ~100-window floor this project treats as reliable for a 99% quantile. "
            f"Shown anyway, flagged rather than hidden."
        )


def _dynamics_chart(result: PortfolioRunResult) -> "go.Figure":
    """Cumulative growth of each portfolio's weighted historical returns,
    plus an equal-weight-of-everything baseline - the "plot of portfolio
    dynamics/graph" the instructions ask for."""
    fig = go.Figure()
    for label, wp in [("QAOA", result.quantum_portfolio), ("Classical", result.classical_portfolio)]:
        if wp is None:
            continue
        idx = wp.selection.selected_indices
        rets = result.log_returns.iloc[:, idx].values @ wp.weights.weights
        cum = np.exp(np.cumsum(rets))
        fig.add_scatter(x=result.log_returns.index, y=cum, mode="lines", name=f"{label} ({wp.weights.method})")

    equal_w = np.full(result.n_assets, 1.0 / result.n_assets)
    cum_all = np.exp(np.cumsum(result.log_returns.values @ equal_w))
    fig.add_scatter(x=result.log_returns.index, y=cum_all, mode="lines", name="Equal-weight, all assets",
                     line=dict(dash="dot"))

    fig.update_layout(title="Portfolio dynamics: cumulative growth of $1",
                       xaxis_title="date", yaxis_title="cumulative value")
    return fig


def _dynamics_chart_png(result: PortfolioRunResult) -> bytes:
    """Same chart, rendered with matplotlib for the downloadable-image
    button - plotly's own PNG export (kaleido) hung/spawned runaway
    processes in this project's dev environment (a known flaky dependency
    on some Windows setups); matplotlib's savefig has no subprocess and
    is already a project dependency, so it is used here instead, rather
    than fighting a fragile optional dependency for one button."""
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

    ax.set_title("Portfolio dynamics: cumulative growth of $1")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative value")
    ax.legend()
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_constraint_stacking_section(result: PortfolioRunResult):
    """"Consider specific constraints (transaction costs, cardinality
    limits) [that] are causing your classical solvers to slow down"
    (instructions) - reuses the academic engine's ALREADY-VALIDATED
    stage 11 sector-constrained QUBO machinery directly (not
    reimplemented) rather than the synthetic-only sweep on the Academic
    page's "Is There An Edge" tab, and runs it on THIS run's actual
    selected real universe: a second penalty term (a per-sector cap,
    labels assigned deterministically here since a curated ticker list
    has no true GICS metadata attached) stacked on top of the budget
    constraint, exactly as stage 11 does, timing brute force as the
    number of sector groups increases."""
    st.divider()
    st.subheader("Does an extra constraint slow the classical solver down here too?")
    st.caption(
        "Reuses the Academic page's own sector-constrained QUBO machinery (stage 11), on THIS "
        "run's real selected universe instead of a synthetic one: an extra per-sector cap layered "
        "on top of the budget constraint, at increasing sector counts (tighter cap, harder search)."
    )
    if result.n_assets > CONSTRAINT_STACK_MAX_N:
        st.info(f"Skipped: n={result.n_assets} > {CONSTRAINT_STACK_MAX_N} - brute force at every "
                f"sector count would take too long for a dashboard click at this size.", icon="ℹ️")
        return

    with st.expander("Run the constraint-stacking timing check", expanded=False):
        if st.button("Run", key="pf_run_constraint_stack"):
            mu, sigma, k, q = result.mu, result.sigma, result.k, 0.5
            unconstrained = brute_force_exact(mu, sigma, k, q)
            rows = []
            for n_sectors in [1, 2, 3, min(4, result.n_assets)]:
                sectors = assign_sectors(result.n_assets, n_sectors, seed=42)
                cap = max(1, k // n_sectors + 1)
                bf = brute_force_sector_constrained(mu, sigma, k, q, sectors, cap)
                rows.append({"Sectors": n_sectors, "Sector cap": cap, "Feasible": bf.feasible,
                             "Value": bf.value if bf.feasible else float("nan"),
                             "Time (s)": bf.elapsed_s})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption(
                f"Unconstrained optimum (no sector cap): {unconstrained.value:.4f} in "
                f"{unconstrained.elapsed_s:.4f}s. Each additional sector shrinks the FEASIBLE "
                f"region (never improves the value, per the same proof as stage 11's sanity "
                f"check) - watch the 'Time (s)' column: exact search still visits every C(n,k) "
                f"subset regardless of the cap, so timing differences here come from Python-level "
                f"overhead noise at this small n, not from the search space itself shrinking the "
                f"WORK, only the answer. This is stated explicitly rather than implied otherwise: "
                f"the QAOA-side slowdown from added constraints (documented on the Academic page) "
                f"is the real, robust finding; classical exact search's per-instance timing at "
                f"this scale is noisy and not the interesting part of the constraint-stacking story."
            )


def render():
    st.title("Portfolio Dashboard")
    st.caption(
        "Select an engine, an optional sector, how many stocks, and a holding period - "
        "both QAOA and a classical method build a portfolio from the same real data, "
        "with continuous weights, historical VaR, and a downloadable results table."
    )

    cfg = _config_panel()

    if cfg["run_clicked"]:
        if cfg["source"] == "Upload your own file" and cfg["uploaded_file"] is None:
            st.warning("Upload a file first.")
            return

        with st.spinner("Fetching data, building the QUBO, solving QAOA and the classical baseline..."):
            try:
                if cfg["source"] == "Live fetch":
                    result = _cached_live_run(
                        cfg["engine"], cfg["sector"], cfg["n_assets"], cfg["holding_period"],
                        cfg["q"], cfg["weight_method"], cfg["use_shrinkage"], cfg["tx_cost"], cfg["reps"],
                    )
                    upload_report = None
                else:
                    file_bytes = cfg["uploaded_file"].getvalue()
                    result, upload_report = _cached_upload_run(
                        file_bytes, cfg["uploaded_file"].name, cfg["n_assets"], cfg["holding_period"],
                        cfg["q"], cfg["weight_method"], cfg["use_shrinkage"], cfg["tx_cost"], cfg["reps"],
                    )
            except Exception as e:
                st.error(f"Run failed: {e}")
                return

        st.session_state["pf_last_result"] = result
        st.session_state["pf_upload_report"] = upload_report
        st.session_state["pf_requested_n"] = cfg["n_assets"]

    result: Optional[PortfolioRunResult] = st.session_state.get("pf_last_result")
    if result is None:
        st.info("Configure a run above and click **Run portfolio construction**.", icon="ℹ️")
        return

    upload_report = st.session_state.get("pf_upload_report")
    if upload_report is not None:
        st.caption(
            f"Uploaded file parsed as **{upload_report.detected_format}** format - "
            f"date column '{upload_report.date_column}', {upload_report.n_symbols} symbols detected, "
            f"{upload_report.n_rows} rows. "
            + (f"Dropped non-numeric columns: {upload_report.dropped_columns}." if upload_report.dropped_columns else "")
        )

    requested_n = st.session_state.get("pf_requested_n", result.n_assets)
    st.divider()
    st.subheader(f"Universe: {result.n_assets} assets, budget k={result.k}"
                 + (f" (shrinkage intensity {result.sigma_shrinkage_intensity:.3f})"
                    if result.sigma_shrinkage_intensity is not None else ""))
    if result.n_assets != requested_n:
        st.warning(
            f"You requested n={requested_n}, but only {result.n_assets} assets were actually used. "
            f"This happens when a sector's curated ticker list, or the uploaded file's column count, "
            f"is smaller than the request, or when data cleaning (stage 02) dropped a symbol for "
            f"missing history - reported here rather than silently running a smaller portfolio than "
            f"asked for.",
            icon="⚠️",
        )

    col1, col2 = st.columns(2)
    with col1:
        _render_portfolio_block("QAOA", result.quantum_portfolio, result.quantum_skipped_reason, result.k)
    with col2:
        _render_portfolio_block("Classical", result.classical_portfolio, None, result.k)

    st.divider()
    st.subheader("Portfolio dynamics")
    st.plotly_chart(_dynamics_chart(result), use_container_width=True)
    png_bytes = _dynamics_chart_png(result)
    st.download_button("Download dynamics chart as PNG", data=png_bytes,
                        file_name="portfolio_dynamics.png", mime="image/png")

    if result.quantum_portfolio is not None and result.classical_portfolio is not None:
        overlap = set(result.quantum_portfolio.selection.selected_symbols) & \
                  set(result.classical_portfolio.selection.selected_symbols)
        st.caption(
            f"QAOA and classical selections overlap on {len(overlap)}/{result.k} assets: "
            f"{sorted(overlap)}."
        )

    _render_constraint_stacking_section(result)
