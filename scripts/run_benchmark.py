r"""
CLI entry point for the stage 08 scaling benchmark + stage 09 statistics.
==============================================================================
Runs the full multi-seed, multi-size QAOA-vs-classical sweep and writes:
  results/scaling_results.json    - raw per-(n, seed) rows (stage 08)
  results/scaling_summary.json    - bootstrap CIs, scaling-exponent fits,
                                     and the paired significance test
                                     (stage 09), consumed directly by the
                                     Streamlit academic page.

Usage:
    python scripts/run_benchmark.py --preset quick
    python scripts/run_benchmark.py --sizes 6 8 10 12 14 --n-seeds 10

Runtime note: each QAOA solve is a noiseless Aer statevector simulation
optimized by COBYLA over up to 250 circuit evaluations - a few tens of
seconds per instance on a laptop CPU (see REPORT.md-style discussion in the
README). A --preset full sweep can take well over an hour; --preset quick
is meant for a fast sanity pass while iterating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

from scaling_benchmark import run_scaling_benchmark, classical_scaling_only, results_to_dicts  # noqa: E402
from statistical_tests import (  # noqa: E402
    bootstrap_mean_ci, fit_scaling_exponent, paired_comparison,
    two_sample_slope_difference_test,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PRESETS = {
    "quick": dict(sizes=[6, 8, 10], n_seeds=3),
    "standard": dict(sizes=[6, 8, 10, 12], n_seeds=5),
    "full": dict(sizes=[6, 8, 10, 12, 14], n_seeds=10),
}


def summarize(rows) -> dict:
    by_size = {}
    for r in rows:
        by_size.setdefault(r.n_assets, []).append(r)

    per_size_summary = {}
    for n, group in by_size.items():
        qaoa_ratios = [g.approx_ratio_qaoa for g in group]
        exact_ratios = [g.approx_ratio_exact_qubo for g in group]
        qaoa_times = [g.qaoa_time_s for g in group]
        per_size_summary[n] = dict(
            n_seeds=len(group),
            qaoa_feasible_rate=sum(g.qaoa_feasible for g in group) / len(group),
            approx_ratio_qaoa_ci=vars(bootstrap_mean_ci(qaoa_ratios)),
            approx_ratio_exact_qubo_ci=vars(bootstrap_mean_ci(exact_ratios)),
            qaoa_time_s_ci=vars(bootstrap_mean_ci(qaoa_times)),
            paired_qaoa_vs_exact=vars(paired_comparison(qaoa_ratios, exact_ratios)),
        )

    sizes_sorted = sorted(by_size.keys())
    mean_qaoa_time = [sum(g.qaoa_time_s for g in by_size[n]) / len(by_size[n]) for n in sizes_sorted]
    mean_bf_time = [sum(g.bf_time_s for g in by_size[n]) / len(by_size[n]) for n in sizes_sorted]

    qaoa_fit = fit_scaling_exponent(sizes_sorted, mean_qaoa_time)
    bf_fit = fit_scaling_exponent(sizes_sorted, mean_bf_time)
    delta, t_stat, p_val = two_sample_slope_difference_test(qaoa_fit, bf_fit)

    return dict(
        per_size=per_size_summary,
        qaoa_time_scaling_fit=vars(qaoa_fit),
        brute_force_time_scaling_fit=vars(bf_fit),
        slope_difference_test=dict(delta_slope=delta, t_statistic=t_stat, p_value=p_val),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()

    if args.preset:
        cfg = PRESETS[args.preset]
        sizes, n_seeds = cfg["sizes"], cfg["n_seeds"]
    else:
        sizes = args.sizes or PRESETS["quick"]["sizes"]
        n_seeds = args.n_seeds or PRESETS["quick"]["n_seeds"]

    print(f"Running scaling benchmark: sizes={sizes}, n_seeds={n_seeds}, q={args.q}, reps={args.reps}")
    rows = run_scaling_benchmark(sizes, n_seeds=n_seeds, q=args.q, reps=args.reps)

    print("\nRunning classical-only scaling (pushed to larger n)...")
    classical_only_sizes = sorted(set(sizes) | {max(sizes) + 4, max(sizes) + 8, max(sizes) + 10})
    classical_only = classical_scaling_only(classical_only_sizes, q=args.q)

    summary = summarize(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "scaling_results.json", "w") as f:
        json.dump({"rows": results_to_dicts(rows), "classical_only": classical_only,
                    "config": {"sizes": sizes, "n_seeds": n_seeds, "q": args.q, "reps": args.reps}},
                   f, indent=2)
    with open(RESULTS_DIR / "scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {RESULTS_DIR / 'scaling_results.json'}")
    print(f"Saved {RESULTS_DIR / 'scaling_summary.json'}")


if __name__ == "__main__":
    main()
