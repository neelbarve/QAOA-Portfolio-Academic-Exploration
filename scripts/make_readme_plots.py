r"""
Generates static PNG plots for the README from results/scaling_*.json
(produced by scripts/run_benchmark.py). Run after run_benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    with open(RESULTS_DIR / "scaling_results.json") as f:
        raw = json.load(f)
    with open(RESULTS_DIR / "scaling_summary.json") as f:
        summary = json.load(f)

    sizes = sorted(int(n) for n in summary["per_size"].keys())
    qaoa_mean = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["mean"] for n in sizes]
    qaoa_lo = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["ci_low"] for n in sizes]
    qaoa_hi = [summary["per_size"][str(n)]["approx_ratio_qaoa_ci"]["ci_high"] for n in sizes]
    exact_mean = [summary["per_size"][str(n)]["approx_ratio_exact_qubo_ci"]["mean"] for n in sizes]
    feasible_rate = [summary["per_size"][str(n)]["qaoa_feasible_rate"] for n in sizes]

    rows = raw["rows"]
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_assets"], []).append(r)
    qaoa_times = [np.mean([r["qaoa_time_s"] for r in by_n[n]]) for n in sizes]
    bf_times = [np.mean([r["bf_time_s"] for r in by_n[n]]) for n in sizes]

    PLOTS_DIR.mkdir(exist_ok=True)

    # --- Side-by-side comparative figure: quality (left) vs. wall-clock (right) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    err_lo = [m - l for m, l in zip(qaoa_mean, qaoa_lo)]
    err_hi = [h - m for h, m in zip(qaoa_hi, qaoa_mean)]
    ax.errorbar(sizes, qaoa_mean, yerr=[err_lo, err_hi], marker="o", capsize=4, label="QAOA (mean, 95% bootstrap CI)")
    ax.plot(sizes, exact_mean, marker="s", label="Exact-QUBO (control)")
    ax.axhline(1.0, linestyle=":", color="gray", label="optimal")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("approximation ratio")
    ax.set_title("Solution quality vs. problem size")
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.bar(sizes, feasible_rate, alpha=0.15, width=0.8, color="tab:red")
    ax2.set_ylabel("QAOA feasible rate", color="tab:red")
    ax2.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(sizes, qaoa_times, marker="o", label="QAOA (mean)")
    ax.plot(sizes, bf_times, marker="s", label="Brute force (mean)")
    ax.set_yscale("log")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("wall-clock time (s, log scale)")
    ax.set_title("Wall-clock time vs. problem size")
    ax.legend(fontsize=8)

    fig.suptitle(f"QAOA vs. classical: {raw['config']['n_seeds']} random seeds per size, "
                 f"q={raw['config']['q']}, reps={raw['config']['reps']}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scaling_comparison.png", dpi=140)
    print(f"Saved {PLOTS_DIR / 'scaling_comparison.png'}")

    # --- Classical-only scaling, pushed further out ---
    classical_only = raw["classical_only"]
    c_n = [c["n_assets"] for c in classical_only]
    c_t = [c["bf_time_s"] for c in classical_only]
    fig2, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(c_n, c_t, marker="o", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("n (assets), k = n // 2")
    ax.set_ylabel("brute-force wall-clock (s, log scale)")
    ax.set_title("Classical combinatorial explosion: C(n, k) brute force")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "classical_scaling.png", dpi=140)
    print(f"Saved {PLOTS_DIR / 'classical_scaling.png'}")


if __name__ == "__main__":
    main()
