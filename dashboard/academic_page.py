r"""
Page 1/2 - Academic page (Part 1 of the project).
======================================================
Everything the project instructions ask for on this page, in one module
with five tabs:

  1. Run & Results       - configure + run the live pipeline (stages 01-07)
                            on real crypto/equity data; benchmarking against
                            classical Markowitz + the exact-QUBO control.
  2. Scaling Benchmark    - PhD-level statistical findings (stage 08/09):
                            bootstrap CIs, a fitted scaling exponent, and a
                            significance test, from scripts/run_benchmark.py.
  3. Fama-French -> Hamiltonian - the mathematical formulation page, run for
                            real on the current universe (stage 10).
  4. Sanity Checks        - the tests/sanity_checks.py checks, run live.
  5. Portfolio Dynamics   - cumulative-return plot of the QAOA-selected
                            portfolio vs. brute-force-optimal vs. equal-weight.

Nothing here hardcodes n_assets or k - every control is a widget, and stage
01's adapters already scale to any n via RunConfig.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from base_adapter import RunConfig
from pipeline import run as run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------
# Cached pipeline call - Streamlit re-runs the whole script on every
# widget interaction, so this is what keeps "click Run" from re-fetching
# data / re-solving QAOA on unrelated UI changes elsewhere on the page.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_run(engine, n_assets, budget, start, end, freq, risk_factor, reps, seed):
    cfg = RunConfig(
        engine=engine, n_assets=n_assets, budget=budget,
        start=start, end=end, return_freq=freq, risk_factor=risk_factor,
    )
    return run_pipeline(cfg, reps=reps, seed=seed)


def _config_panel():
    st.subheader("Configure a run")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        engine = st.selectbox("Engine (asset class)", ["equity", "crypto"], key="engine")
    with c2:
        n_assets = st.slider("Universe size (n)", min_value=4, max_value=14, value=8, key="n_assets")
    with c3:
        default_k = n_assets // 2
        budget = st.slider("Budget k (assets to pick)", min_value=1, max_value=n_assets - 1,
                            value=min(default_k, n_assets - 1), key="budget")
    with c4:
        risk_factor = st.slider("Risk aversion q", 0.0, 1.0, 0.5, 0.05, key="risk_factor")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        start = st.date_input("Start", value=date(2023, 1, 1), key="start")
    with c6:
        end = st.date_input("End", value=date(2024, 1, 1), key="end")
    with c7:
        freq = st.selectbox("Holding period", ["D", "W", "M"], format_func=lambda f: {
            "D": "Daily", "W": "Weekly", "M": "Monthly"}[f], key="freq")
    with c8:
        reps = st.slider("QAOA layers (p)", 1, 5, 2, key="reps")

    run_clicked = st.button("Run pipeline", type="primary")
    return engine, n_assets, budget, start, end, freq, risk_factor, reps, run_clicked


def _render_run_results_tab():
    engine, n_assets, budget, start, end, freq, risk_factor, reps, run_clicked = _config_panel()

    if run_clicked:
        with st.spinner(f"Fetching {engine} data, preprocessing, solving classical/QUBO/QAOA..."):
            try:
                result = _cached_run(engine, n_assets, budget, start, end, freq, risk_factor, reps, 42)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Run failed: {e}")
                return

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Configure a run above and click **Run pipeline**.")
        return

    st.divider()
    st.subheader(f"Universe: {', '.join(result.symbols)}  (n={len(result.symbols)}, k={result.budget})")

    tab_data, tab_results = st.tabs(["Data & preprocessing", "Optimization results"])

    with tab_data:
        st.markdown("**Preprocessing report (stage 02)**")
        st.code(result.preprocessing.as_markdown(), language="markdown")

        st.markdown("**Exploratory data analysis (stage 03)**")
        e = result.eda
        colA, colB = st.columns(2)
        with colA:
            st.metric("Return-variance hardness proxy (s2_ret)", f"{e.return_variance_s2_ret:.5f}")
            st.metric("Correlation-variance hardness proxy (s2_cor)", f"{e.correlation_variance_s2_cor:.5f}")
            st.caption(f"Predicted QAOA difficulty: **{e.predicted_hardness}** "
                       "(per Brandhofer et al. 2023's hardness proxy - broadly spread "
                       "returns/correlations tend to be easier for the optimizer).")
        with colB:
            fig = px.imshow(e.correlation_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Asset return correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Summary statistics (log returns, per period)**")
        st.dataframe(e.summary_stats, use_container_width=True)

    with tab_results:
        c = result.comparison
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Brute-force optimum", f"{c.brute_force.value:.4f}")
        col2.metric("Relaxation + round", f"{c.relaxation.value:.4f}" if c.relaxation else "unavailable")
        col3.metric("Exact-QUBO diagonalization", f"{c.exact_qubo_true_val:.4f}",
                    delta="feasible" if c.exact_qubo_feasible else "INFEASIBLE",
                    delta_color="normal" if c.exact_qubo_feasible else "inverse")
        col4.metric("QAOA", f"{c.qaoa_true_val:.4f}",
                    delta="feasible" if c.qaoa_feasible else "INFEASIBLE",
                    delta_color="normal" if c.qaoa_feasible else "inverse")

        st.metric("QAOA approximation ratio (vs. brute-force optimum)", f"{c.approx_ratio_qaoa:.4f}")
        st.caption(f"QAOA wall-clock: {c.qaoa.elapsed_s:.2f}s, {c.qaoa.reps} layers, "
                   f"initial point (linear ansatz): {np.round(c.qaoa.initial_point, 3).tolist()}")

        fig = go.Figure()
        methods = ["Brute-force\n(exact)", "Relaxation\n+ round", "Exact-QUBO\n(control)", "QAOA"]
        values = [c.brute_force.value, c.relaxation.value if c.relaxation else None,
                  c.exact_qubo_true_val, c.qaoa_true_val]
        colors = ["#2E7D32", "#757575", "#1565C0", "#C62828" if not c.qaoa_feasible else "#6A1B9A"]
        fig.add_bar(x=methods, y=values, marker_color=colors,
                    text=[f"{v:.4f}" if v is not None else "n/a" for v in values], textposition="outside")
        fig.update_layout(title="Objective value by solver", yaxis_title="mu^T x - q * x^T Sigma x")
        st.plotly_chart(fig, use_container_width=True)

        selected_assets = [result.symbols[i] for i in c.brute_force.selection]
        st.success(f"Brute-force-optimal selection: {selected_assets}")
        if c.qaoa_feasible:
            qaoa_selected = [result.symbols[i] for i in np.nonzero(c.qaoa.x)[0]]
            match = set(qaoa_selected) == set(selected_assets)
            st.write(f"QAOA selection: {qaoa_selected} " + ("(matches optimum)" if match else "(differs from optimum)"))


def _render_scaling_tab():
    st.subheader("Scaling benchmark: solution quality and wall-clock vs. problem size")
    st.caption(
        "Generated by scripts/run_benchmark.py on synthetic instances (multiple random seeds per "
        "problem size, so the numbers below are distributions, not single anecdotal runs). "
        "Statistical methodology follows Yalovetzky et al. (2026): percentile-bootstrap confidence "
        "intervals, an OLS log-linear fit for the scaling exponent, and a paired Wilcoxon "
        "signed-rank test between QAOA and the exact-QUBO control."
    )

    results_path, summary_path = RESULTS_DIR / "scaling_results.json", RESULTS_DIR / "scaling_summary.json"
    if not results_path.exists() or not summary_path.exists():
        st.warning(
            "No benchmark results yet. Run:\n\n"
            "`python scripts/run_benchmark.py --preset quick`\n\nfrom the project root, then reload this page."
        )
        return

    with open(results_path) as f:
        raw = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    rows = pd.DataFrame(raw["rows"])
    st.caption(f"Config: sizes={raw['config']['sizes']}, seeds per size={raw['config']['n_seeds']}, "
               f"q={raw['config']['q']}, reps={raw['config']['reps']}")

    sizes_sorted = sorted(int(n) for n in summary["per_size"].keys())
    qaoa_means = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["mean"] for n in sizes_sorted]
    qaoa_lo = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["ci_low"] for n in sizes_sorted]
    qaoa_hi = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["ci_high"] for n in sizes_sorted]
    exact_means = [summary["per_size"][str(n)]["approx_ratio_exact_qubo_ci"]["mean"] for n in sizes_sorted]

    fig = go.Figure()
    fig.add_scatter(x=sizes_sorted, y=qaoa_means, mode="lines+markers", name="QAOA",
                     error_y=dict(type="data", symmetric=False,
                                   array=[h - m for h, m in zip(qaoa_hi, qaoa_means)],
                                   arrayminus=[m - l for m, l in zip(qaoa_means, qaoa_lo)]))
    fig.add_scatter(x=sizes_sorted, y=exact_means, mode="lines+markers", name="Exact-QUBO (control)")
    fig.add_hline(y=1.0, line_dash="dot", annotation_text="optimal")
    fig.update_layout(title="Approximation ratio vs. problem size (95% bootstrap CI)",
                       xaxis_title="n (qubits)", yaxis_title="approximation ratio")
    st.plotly_chart(fig, use_container_width=True)

    qaoa_times = rows.groupby("n_assets")["qaoa_time_s"].mean()
    bf_times = rows.groupby("n_assets")["bf_time_s"].mean()
    fig2 = go.Figure()
    fig2.add_scatter(x=qaoa_times.index, y=qaoa_times.values, mode="lines+markers", name="QAOA (mean)")
    fig2.add_scatter(x=bf_times.index, y=bf_times.values, mode="lines+markers", name="Brute-force (mean)")
    fig2.update_layout(title="Wall-clock time vs. problem size", xaxis_title="n (qubits)",
                        yaxis_title="seconds", yaxis_type="log")
    st.plotly_chart(fig2, use_container_width=True)

    qfit, bfit = summary["qaoa_time_scaling_fit"], summary["brute_force_time_scaling_fit"]
    diff = summary["slope_difference_test"]
    st.markdown("**Fitted scaling exponents** (log2(time) = intercept + slope * n):")
    fit_df = pd.DataFrame([
        {"method": "QAOA", "slope": qfit["slope"], "slope_se": qfit["slope_se"], "R2": qfit["r_squared"], "p_value": qfit["p_value"]},
        {"method": "Brute force", "slope": bfit["slope"], "slope_se": bfit["slope_se"], "R2": bfit["r_squared"], "p_value": bfit["p_value"]},
    ])
    st.dataframe(fit_df, use_container_width=True)
    st.caption(f"Two-sample slope-difference test: delta={diff['delta_slope']:.4f}, "
               f"t={diff['t_statistic']:.3f}, p={diff['p_value']:.4g}")

    st.markdown("**Per-size paired test: QAOA vs. exact-QUBO approximation ratio**")
    paired_rows = []
    for n in sizes_sorted:
        pt = summary["per_size"][str(n)]["paired_qaoa_vs_exact"]
        paired_rows.append({"n": n, "test": pt["test"], "statistic": pt["statistic"],
                             "p_value": pt["p_value"], "mean_diff": pt["mean_difference"]})
    st.dataframe(pd.DataFrame(paired_rows), use_container_width=True)

    st.info(
        "Honest read (carried over from the qaoa_v1 pilot study): classical brute force is faster "
        "than QAOA at every size tested here, and there is no reason to expect otherwise at this "
        "qubit count on a noiseless simulator - COBYLA's classical outer loop, not quantum "
        "execution, dominates wall-clock time. The scaling *exponents* fitted above, not a single "
        "size's raw runtime, are the meaningful comparison, and the honest classical bar past small "
        "n is a real MIQP solver, not brute force.",
        icon="🔎",
    )


def _render_fama_french_tab():
    st.subheader("Fama-French factor model to QAOA Hamiltonian")
    st.markdown(
        r"""
Runs the derivation for real, on the currently loaded universe (run the **Run & Results** tab first).

**1. Three-factor model.** For each asset $i$, regress excess return on the market, size (SMB) and
value (HML) factors:

$$ R_i(t) - RF(t) = \alpha_i + \beta_{i1} \cdot \text{MktRF}(t) + \beta_{i2} \cdot \text{SMB}(t) + \beta_{i3} \cdot \text{HML}(t) + \epsilon_i(t) $$

**2. Factor covariance decomposition.** With $B$ the $(n \times 3)$ loadings matrix, $F$ the $(3\times3)$
factor covariance, and $D = \text{diag}(\text{var}(\epsilon_i))$ the idiosyncratic variances:

$$ \Sigma_{FF} = B F B^{\top} + D, \qquad \mu_{FF,i} = \alpha_i + \beta_i \cdot \overline{\text{factors}} $$

**3. QUBO / Ising conversion.** $(\mu_{FF}, \Sigma_{FF})$ feed into the *same* by-hand Ising derivation
(stage 05b) used everywhere else in this project. Penalized objective:

$$ F(x) = q \cdot x^{\top}\Sigma_{FF} x - \mu_{FF}^{\top} x + A(\sum_i x_i - k)^2 $$

Mapped to spin variables $x_i = (1-z_i)/2$, this becomes the Ising cost Hamiltonian QAOA's cost
unitary exponentiates:

$$ H(z) = \sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j + \text{const} $$
        """
    )

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Run the **Run & Results** tab first - this page reuses that universe's return data.")
        return

    if st.button("Fit Fama-French model on current universe"):
        with st.spinner("Fetching Fama-French factors and fitting OLS per asset..."):
            from fama_french_hamiltonian import fetch_fama_french_factors, fit_fama_french, build_hamiltonian_from_fama_french
            from base_adapter import compute_log_returns
            from universe_builder import fetch_raw_prices

            cfg = result.config
            prices, adapter, _ = fetch_raw_prices(cfg)
            cleaned = adapter._align_and_clean(prices)
            log_returns = compute_log_returns(cleaned, cfg.return_freq)

            factors, source = fetch_fama_french_factors(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
            model = fit_fama_french(log_returns, factors, trading_days_per_year=adapter.trading_days_per_year)
            H = build_hamiltonian_from_fama_french(model, result.budget, cfg.risk_factor)

            st.session_state["ff_model"] = model
            st.session_state["ff_hamiltonian"] = H
            st.session_state["ff_source"] = source

    model = st.session_state.get("ff_model")
    H = st.session_state.get("ff_hamiltonian")
    if model is None or H is None:
        return

    st.caption(f"Factor data source: {st.session_state.get('ff_source')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factor loadings (beta) and fit quality**")
        beta_df = pd.DataFrame(model.beta, index=model.symbols, columns=["MktRF", "SMB", "HML"])
        beta_df["R2"] = model.r_squared
        st.dataframe(beta_df, use_container_width=True)
    with col2:
        st.markdown("**Annualized factor covariance F**")
        st.dataframe(pd.DataFrame(model.factor_cov, index=["MktRF", "SMB", "HML"], columns=["MktRF", "SMB", "HML"]),
                     use_container_width=True)

    st.markdown(f"**Resulting Ising Hamiltonian** ({H.n_qubits} qubits, penalty A={H.penalty:.5f})")
    h_df = pd.DataFrame({"asset": model.symbols, "h_i (linear coeff.)": H.h})
    st.dataframe(h_df, use_container_width=True)

    n = H.n_qubits
    J_matrix = np.zeros((n, n))
    for (i, j), v in H.J.items():
        J_matrix[i, j] = J_matrix[j, i] = v
    fig = px.imshow(J_matrix, x=model.symbols, y=model.symbols, text_auto=".3f",
                     color_continuous_scale="RdBu_r", title="J_ij (pairwise ZZ coupling strengths)")
    st.plotly_chart(fig, use_container_width=True)


def _render_sanity_tab():
    st.subheader("Sanity checks")
    st.caption("Same checks as tests/sanity_checks.py, run live from the dashboard.")
    if st.button("Run sanity checks"):
        import sanity_runner  # local helper module, see dashboard/sanity_runner.py
        with st.spinner("Running..."):
            output, ok = sanity_runner.run_all()
        st.code(output, language="text")
        if ok:
            st.success("All checks passed.")
        else:
            st.error("One or more checks failed - see output above.")


def _render_portfolio_dynamics_tab():
    st.subheader("Portfolio dynamics")
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Run the **Run & Results** tab first.")
        return

    with st.spinner("Rebuilding the cleaned price history for a cumulative-return backtest..."):
        from universe_builder import fetch_raw_prices
        cfg = result.config
        prices, adapter, _ = fetch_raw_prices(cfg)
        cleaned = adapter._align_and_clean(prices)
        cleaned = cleaned[result.symbols]

    c = result.comparison
    bf_selection = [result.symbols[i] for i in c.brute_force.selection]
    qaoa_selection = [result.symbols[i] for i in np.nonzero(c.qaoa.x)[0]] if c.qaoa_feasible else []

    def equal_weight_cum_return(cols):
        if not cols:
            return None
        sub = cleaned[cols]
        rets = sub.pct_change().fillna(0.0).mean(axis=1)
        return (1 + rets).cumprod()

    fig = go.Figure()
    bf_curve = equal_weight_cum_return(bf_selection)
    fig.add_scatter(x=bf_curve.index, y=bf_curve.values, name=f"Brute-force optimal {bf_selection}")
    if qaoa_selection:
        qaoa_curve = equal_weight_cum_return(qaoa_selection)
        fig.add_scatter(x=qaoa_curve.index, y=qaoa_curve.values, name=f"QAOA selection {qaoa_selection}")
    ew_curve = equal_weight_cum_return(result.symbols)
    fig.add_scatter(x=ew_curve.index, y=ew_curve.values, name="Equal-weight, full universe", line=dict(dash="dot"))

    fig.update_layout(title="Cumulative return (equal-weight within each selection)",
                       xaxis_title="date", yaxis_title="growth of $1")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Illustrative only: equal weighting within a selection, no transaction costs, no rebalancing. "
        "The optimizer itself only ever chooses WHICH k assets, not their weights - see the README's "
        "assumptions section."
    )


def _render_edge_search_tab():
    st.subheader("Is there a QAOA edge?")
    st.markdown(
        "The scaling benchmark above shows QAOA matching brute force's solution quality "
        "when it converges, but never beating it, and being far slower. That raises the "
        "obvious question: **why use QAOA at all?** This tab answers it two ways - "
        "what the reference literature says, and what this project's own pipeline finds "
        "when pushed at two axes brute force can't reach: harder constraints, and larger n."
    )

    with st.expander("Literature review: does any reference paper find a QAOA edge?", expanded=True):
        st.markdown(
            """
**Short answer: no paper in this project's reference set demonstrates a QAOA edge over
classical methods on portfolio optimization, at any tested size, on any axis (quality,
speed, or robustness) - and the more careful papers say so explicitly, unprompted.**

- **Brandhofer et al. (2023)**: *"a quantum advantage of QAOA has not yet been rigorously
  proven."* Never even ran a classical baseline. Cites a literature extrapolation that a
  Max-Cut QAOA speedup would need **several hundred qubits** - this project (and every
  paper in its reference set) tests 4-22 qubits.
- **Yalovetzky et al. (2026)**, the one real-hardware paper: *"the primary contribution of
  this work is not to claim advantage over state-of-the-art classical solvers - indeed, SA
  is exceptionally effective on the instance sizes accessible to current quantum
  hardware."* Their real Quantinuum-hardware QAOA run scored **0/100 successes** on the two
  largest test cases.
- **Uotila et al. (QCE 2025)**, the HUBO paper: ran QAOA against three other methods on 100
  instances - QAOA finished **last** (3/100 vs. 46-53/100 for exact/classical methods).
  *"solving higher-order portfolio optimization problems with QAOA proved challenging."*

Where these authors say an edge **might eventually** appear: much larger, harder instances
than current hardware reaches (Yalovetzky), several hundred qubits *and* much lower
gate-error rates simultaneously (Brandhofer), or classically-hard non-quadratic objectives
where even weak QAOA has a lower bar to clear (Uotila's more speculative argument - though
their own QAOA results on that exact idea were the weakest of the four methods tested).
Full citations in the README's References section.
            """
        )

    st.divider()
    st.markdown("### This project's own search: harder constraints (stage 11)")
    st.caption(
        "Fixed problem size, increasing constraint complexity - a second penalty term "
        "(per-sector caps) on top of the budget constraint, testing whether QAOA's relative "
        "standing changes as the QUBO landscape gets more rugged (Uotila et al.'s argument)."
    )
    hardness_path = RESULTS_DIR / "hardness_sweep.json"
    if not hardness_path.exists():
        st.warning("No hardness-sweep results yet. Run:\n\n`python scripts/run_edge_search.py --preset quick`")
    else:
        with open(hardness_path) as f:
            hs = json.load(f)
        hdf = pd.DataFrame(hs["rows"])
        st.caption(f"n_assets={hs['config'].get('hardness_n', hs['config'].get('n_assets'))}, "
                   f"q={hs['config']['q']}, reps={hs['config']['reps']} - single seed, illustrative not statistical.")

        fig = go.Figure()
        fig.add_scatter(x=hdf["n_sectors"], y=hdf["qaoa_val"] / hdf["bf_val"], mode="lines+markers",
                         name="QAOA / brute-force-optimal",
                         marker=dict(color=["#C62828" if not c else "#6A1B9A" for c in hdf["qaoa_converged"]]))
        fig.add_hline(y=1.0, line_dash="dot", annotation_text="optimal")
        fig.update_layout(title="Solution quality vs. constraint hardness (more sectors = tighter, harder constraints)",
                           xaxis_title="number of sectors (constraint complexity)",
                           yaxis_title="QAOA value / true optimum")
        st.plotly_chart(fig, use_container_width=True)

        show_df = hdf[["n_sectors", "sector_cap", "bf_val", "qaoa_val", "qaoa_feasible", "qaoa_converged", "qaoa_time_s"]]
        st.dataframe(show_df, use_container_width=True)
        if not hdf["qaoa_converged"].all():
            st.info(
                "Rows marked `qaoa_converged=False` mean QAOA's sampled output stayed near-uniform - "
                "the classical optimizer failed to concentrate probability on ANYTHING within its "
                "iteration budget, not just a wrong answer. This is a stronger version of the same "
                "finding as the main scaling benchmark: harder landscapes break QAOA's classical "
                "optimizer loop before they break the QUBO formulation itself.",
                icon="🔎",
            )

    st.divider()
    st.markdown("### This project's own search: pushing past exact-solver range (stage 12)")
    st.caption(
        "Fixed constraint structure, increasing n past where brute force stays practical - "
        "QAOA compared against simulated annealing (a realistic classical heuristic, matching "
        "Yalovetzky et al.'s own methodology), not an exact solver that stops being available."
    )
    extended_path = RESULTS_DIR / "extended_scaling.json"
    if not extended_path.exists():
        st.warning("No extended-scaling results yet. Run:\n\n`python scripts/run_edge_search.py --preset quick`")
    else:
        with open(extended_path) as f:
            ex = json.load(f)
        edf = pd.DataFrame(ex["rows"])
        st.caption(f"sizes={ex['config']['extended_sizes']}, exact_cutoff_n={ex['config']['exact_cutoff_n']} "
                   f"- single seed, illustrative not statistical.")

        fig2 = go.Figure()
        fig2.add_scatter(x=edf["n_assets"], y=edf["sa_time_s"], mode="lines+markers", name="Simulated annealing")
        fig2.add_scatter(x=edf["n_assets"], y=edf["qaoa_time_s"], mode="lines+markers", name="QAOA")
        fig2.update_layout(title="Wall-clock time: SA vs. QAOA", xaxis_title="n (qubits)",
                            yaxis_title="seconds", yaxis_type="log")
        st.plotly_chart(fig2, use_container_width=True)

        show_df2 = edf[["n_assets", "bf_val", "sa_val", "qaoa_val", "qaoa_feasible", "qaoa_converged",
                         "sa_time_s", "qaoa_time_s", "ratio_qaoa_vs_sa"]]
        st.dataframe(show_df2, use_container_width=True)
        st.info(
            "Simulated annealing is feasible by construction at every size (cardinality-preserving "
            "swap moves), reaches the true optimum whenever it's known, and runs in a fraction of a "
            "second - orders of magnitude faster than QAOA at every size tested, with no convergence "
            "failures. That gap is the honest headline number: at these sizes, the realistic classical "
            "competitor isn't just as good as QAOA, it's both better and dramatically cheaper.",
            icon="🔎",
        )

    st.divider()
    st.markdown("### Improving QAOA itself: warm-starting (stage 14)")
    st.caption(
        "The two searches above test whether QAOA's STANDING changes under harder conditions. "
        "This one asks a different question: is the standard implementation (uniform |+>^n start, "
        "plain X-mixer) actually the best this project's own QAOA can do, or is some of its failure "
        "an implementation artifact? Egger, Marecek & Woerner (2021) - not one of the original five "
        "reference papers - describe biasing QAOA's initial state and mixer toward a classical "
        "relaxation's fractional solution instead of starting from scratch every time."
    )
    warm_path = RESULTS_DIR / "warm_start_comparison.json"
    if not warm_path.exists():
        st.warning("No warm-start results yet. Run:\n\n`python scripts/run_warm_start_comparison.py`")
    else:
        with open(warm_path) as f:
            ws = json.load(f)
        wdf = pd.DataFrame(ws["rows"])
        st.caption(f"Same sizes/seed as the extended-scaling sweep above - "
                   f"sizes={ws['config']['sizes']}, q={ws['config']['q']}, reps={ws['config']['reps']}.")

        fig3 = go.Figure()
        fig3.add_bar(x=wdf["n_assets"].astype(str), y=wdf["ratio_standard_vs_bf"].fillna(0),
                     name="Standard QAOA / optimum")
        fig3.add_bar(x=wdf["n_assets"].astype(str), y=wdf["ratio_warm_vs_bf"].fillna(0),
                     name="Warm-start QAOA / optimum")
        fig3.update_layout(title="Approximation ratio: standard vs. warm-start QAOA (0 = did not converge/infeasible)",
                            xaxis_title="n (qubits)", yaxis_title="value / true optimum", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

        show_df3 = wdf[["n_assets", "bf_val", "standard_qaoa_converged", "standard_qaoa_val",
                         "warm_qaoa_converged", "warm_qaoa_feasible", "warm_qaoa_val", "warm_qaoa_time_s"]]
        st.dataframe(show_df3, use_container_width=True)

        st.success(
            "**This one actually moved the needle.** At n=16, standard QAOA fails completely "
            "(near-uniform output, not converged, 0 value) - warm-starting reaches the EXACT true "
            "optimum, feasible, converged. At n=20, standard QAOA again produces nothing usable; "
            "warm-starting at least returns a feasible answer (49% of optimal - not great, but not "
            "nothing). At n=24, both still fail - the fix has a reach, not an unlimited one. "
            "**This is a real, reproducible improvement to QAOA's own reliability, gained for free "
            "by using information the pipeline already computes (stage 04's continuous relaxation) "
            "rather than starting from a blank uniform superposition every time.**",
            icon="✅",
        )
        st.info(
            "What this is NOT: a quantum edge over classical methods. Simulated annealing above still "
            "solves every one of these instances exactly in well under a second - warm-start QAOA at "
            "n=16 takes 10.9 seconds to *match* what SA does in 0.19s. The finding is narrower and "
            "still genuinely useful: a smarter, still-standard-hardware-compatible way to RUN QAOA "
            "measurably delays where it breaks, which matters for anyone trying to push QAOA "
            "experiments as far as today's hardware/simulators allow - it just doesn't change the "
            "answer to 'does QAOA beat classical methods here' (still no).",
            icon="🔎",
        )

    st.divider()
    st.markdown("### Does warm-starting help simulated annealing too?")
    st.caption(
        "The obvious follow-up: if biasing QAOA's start with the relaxation helped, does the "
        "same trick help SA - the method already beating QAOA on every axis? Tracks each run's "
        "best-value-so-far trajectory and measures the iteration it first reaches its own final "
        "value, across 40 runs (sizes 16-32, 8 seeds each)."
    )
    sa_warm_path = RESULTS_DIR / "sa_warm_start_comparison.json"
    if not sa_warm_path.exists():
        st.warning("No SA warm-start results yet. Run:\n\n`python scripts/run_sa_warm_start_comparison.py`")
    else:
        with open(sa_warm_path) as f:
            saw = json.load(f)
        s = saw["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Converged instantly (0 iters)", f"{s['instant_count']}/{s['n_runs']}",
                   f"{100*s['instant_fraction']:.0f}%")
        c2.metric("Warm found a WORSE final value", f"{s['worse_final_count']}/{s['n_runs']}")
        if "non_instant_count" in s:
            c3.metric("Of non-instant runs, warm was slower", f"{s['non_instant_slower_count']}/{s['non_instant_count']}")

        sadf = pd.DataFrame(saw["rows"])
        show_sa = sadf[["n_assets", "seed", "random_iters_to_converge", "random_final_val",
                         "warm_iters_to_converge", "warm_final_val"]]
        st.dataframe(show_sa, use_container_width=True)

        st.warning(
            "A naive mean(random_iters / warm_iters) here comes out around 1,400x - real "
            "arithmetic, but a misleading headline: most of that comes from dividing by "
            "near-zero denominators when warm-start converges instantly. The truer picture: "
            "**72% of runs needed zero annealing at all** - the relaxation-rounded guess already "
            "matched what SA finds from scratch. But of the 28% that genuinely still needed "
            "annealing, warm-starting was NOT reliably faster - most of those took MORE "
            "iterations than a random start, and one run even landed on a strictly worse final "
            "answer. This is a different finding than QAOA's: it isn't 'smarter search,' it's "
            "'the starting guess is often already good enough that no search is needed' - and "
            "when search actually happens, the warm start is a wash at best.",
            icon="⚖️",
        )

    st.divider()
    st.markdown(
        "**Bottom line:** consistent with the literature review above, none of these experiments "
        "find a QAOA edge over classical methods - SA remains both more reliable and dramatically "
        "faster at every size tested. But the warm-start result shows QAOA's own performance is not "
        "fixed: a literature-backed change to *how* QAOA is run recovered the exact optimum at a size "
        "(n=16) where the standard construction produced nothing at all, and a usable-if-imperfect "
        "answer one size further (n=20) where it previously produced nothing. Warm-starting SA with "
        "the same relaxation, tested for completeness, shows the fix doesn't generalize "
        "automatically: it mostly just lets SA skip a search it would have won anyway. The honest "
        "picture is layered: no quantum computational advantage exists at this scale, "
        "QAOA-the-algorithm still has real, fixable headroom within that scale, that fix is specific "
        "to QAOA rather than universal, and classical brute force itself got faster "
        "too - a vectorized rewrite (`brute_force_exact_vectorized`, stage 04) reaches the exact "
        "optimum roughly 6-7x quicker at the same sizes (verified bit-identical to the original), "
        "pushing the exact-ground-truth frontier from ~n=22 to ~n=28 in a comparable time budget. "
        "See the README's 'Is there a QAOA edge' section for the full writeup."
    )


def render():
    st.title("Academic Page")
    st.caption(
        "QAOA for constrained portfolio optimization - research findings, benchmarking, "
        "and the underlying mathematics, run against real market data."
    )
    tabs = st.tabs([
        "Run & Results", "Scaling Benchmark", "Fama-French -> Hamiltonian",
        "Sanity Checks", "Portfolio Dynamics", "Is There An Edge?",
    ])
    with tabs[0]:
        _render_run_results_tab()
    with tabs[1]:
        _render_scaling_tab()
    with tabs[2]:
        _render_fama_french_tab()
    with tabs[3]:
        _render_sanity_tab()
    with tabs[4]:
        _render_portfolio_dynamics_tab()
    with tabs[5]:
        _render_edge_search_tab()
