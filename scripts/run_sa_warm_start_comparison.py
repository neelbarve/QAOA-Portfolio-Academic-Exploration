r"""
CLI entry point: does warm-starting simulated annealing (from the same
continuous relaxation stage 14 uses for QAOA) make it converge faster?

SA already reaches the true optimum reliably within the iteration budgets
used elsewhere in this project (results/extended_scaling.json shows exact
matches at n=12/16/20 with n_iters=6000), so "does it find a better answer"
isn't the interesting question - "does it get there in fewer iterations"
is. This script tracks each run's best-value-so-far trajectory
(track_history=True) and reports, for both a random start and a
relaxation-warm start, the iteration at which that run's OWN best-so-far
first reaches its own final value - a convergence-speed metric that works
whether or not the true global optimum is independently known, since it
doesn't require one.

Usage:
    python scripts/run_sa_warm_start_comparison.py
    python scripts/run_sa_warm_start_comparison.py --sizes 16 24 32 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

from markowitz import brute_force_exact_vectorized  # noqa: E402
from simulated_annealing import simulated_annealing, simulated_annealing_warm_started  # noqa: E402
from scaling_benchmark import make_synthetic_universe  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _iters_to_converge(history: List[float]) -> int:
    """First iteration index at which this run's best-so-far reaches the
    value it eventually ends up at - well-defined because best-so-far is
    monotone non-decreasing by construction (only updates on improvement)."""
    final = history[-1]
    for i, v in enumerate(history):
        if v >= final:
            return i
    return len(history) - 1  # unreachable given the property above


@dataclass
class SAConvergenceRow:
    n_assets: int
    seed: int
    bf_val: Optional[float]
    random_final_val: float
    random_iters_to_converge: int
    random_time_s: float
    warm_final_val: float
    warm_iters_to_converge: int
    warm_time_s: float
    random_ratio_to_bf: Optional[float]
    warm_ratio_to_bf: Optional[float]


def run_sa_convergence_comparison(
    sizes: List[int], n_seeds: int, n_iters: int, q: float, exact_cutoff_n: int,
) -> List[SAConvergenceRow]:
    rows: List[SAConvergenceRow] = []
    for n in sizes:
        k = n // 2
        for seed in range(n_seeds):
            instance_seed = 42 * 1000 + seed
            mu, sigma = make_synthetic_universe(n, seed=instance_seed)

            bf_val = None
            if n <= exact_cutoff_n:
                bf_val = brute_force_exact_vectorized(mu, sigma, k, q).value

            r_result, r_hist = simulated_annealing(
                mu, sigma, k, q, n_iters=n_iters, seed=seed, track_history=True,
            )
            w_result, w_hist = simulated_annealing_warm_started(
                mu, sigma, k, q, n_iters=n_iters, seed=seed, track_history=True,
            )

            rows.append(SAConvergenceRow(
                n_assets=n, seed=seed, bf_val=bf_val,
                random_final_val=r_result.value,
                random_iters_to_converge=_iters_to_converge(r_hist),
                random_time_s=r_result.elapsed_s,
                warm_final_val=w_result.value,
                warm_iters_to_converge=_iters_to_converge(w_hist),
                warm_time_s=w_result.elapsed_s,
                random_ratio_to_bf=(r_result.value / bf_val) if bf_val else None,
                warm_ratio_to_bf=(w_result.value / bf_val) if bf_val else None,
            ))
            r = rows[-1]
            print(f"n={n:2d} seed={seed} | random: converge@{r.random_iters_to_converge:4d} "
                  f"val={r.random_final_val:.4f} | warm: converge@{r.warm_iters_to_converge:4d} "
                  f"val={r.warm_final_val:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--exact-cutoff-n", type=int, default=28)
    args = parser.parse_args()

    rows = run_sa_convergence_comparison(
        args.sizes, args.seeds, args.n_iters, args.q, args.exact_cutoff_n,
    )

    # A naive mean(random_iters / warm_iters) is a misleading headline number
    # here: many warm-start runs converge at iteration 0 (the relaxation's
    # rounded guess already IS the value SA settles on with the full
    # budget), and dividing by a near-zero denominator produces enormous,
    # not-really-meaningful ratios that dominate a mean. Report the actual
    # shape of the result instead: how often warm-starting made annealing
    # unnecessary, and - restricted to the runs where annealing genuinely
    # still happened - whether it was faster, honestly, including the
    # inconvenient cases where it was not.
    instant = [r for r in rows if r.warm_iters_to_converge == 0]
    non_instant = [r for r in rows if r.warm_iters_to_converge > 0]
    worse_final = [r for r in rows if r.warm_final_val < r.random_final_val - 1e-9]
    slower_of_non_instant = [
        r for r in non_instant if r.warm_iters_to_converge > r.random_iters_to_converge
    ]

    print(f"\n=== Summary across {len(rows)} runs ===")
    print(f"warm-start converged INSTANTLY (0 annealing iterations needed): "
          f"{len(instant)}/{len(rows)} ({100*len(instant)/len(rows):.0f}%)")
    print(f"warm found a strictly WORSE final value than random start: {len(worse_final)}/{len(rows)}")

    summary = {
        "n_runs": len(rows),
        "instant_count": len(instant),
        "instant_fraction": len(instant) / len(rows),
        "worse_final_count": len(worse_final),
    }
    if non_instant:
        ri = np.array([r.random_iters_to_converge for r in non_instant])
        wi = np.array([r.warm_iters_to_converge for r in non_instant])
        summary.update({
            "non_instant_count": len(non_instant),
            "non_instant_slower_count": len(slower_of_non_instant),
            "non_instant_mean_random_iters": float(ri.mean()),
            "non_instant_mean_warm_iters": float(wi.mean()),
            "non_instant_median_ratio_random_over_warm": float(np.median(ri / wi)),
        })
        print(f"of the {len(non_instant)} runs where warm-start STILL needed annealing: "
              f"{len(slower_of_non_instant)} took MORE iterations than a random start "
              f"(median ratio random/warm = {summary['non_instant_median_ratio_random_over_warm']:.2f}x)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "sa_warm_start_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "rows": [asdict(r) for r in rows],
            "summary": summary,
            "config": {"sizes": args.sizes, "n_seeds": args.seeds, "n_iters": args.n_iters,
                       "q": args.q, "exact_cutoff_n": args.exact_cutoff_n},
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
