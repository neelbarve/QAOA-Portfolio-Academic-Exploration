r"""
CLI entry point: does warm-starting QAOA (stage 14) fix the non-convergence
found in stage 12's extended scaling sweep?

Runs solve_warm_start_qaoa at the same sizes/seed results/extended_scaling.json
already used, then merges the two into results/warm_start_comparison.json -
one table with brute force, simulated annealing, standard QAOA, and
warm-start QAOA side by side. Reported honestly regardless of outcome: if
warm-starting doesn't help, that's the result.

Usage:
    python scripts/run_warm_start_comparison.py
    python scripts/run_warm_start_comparison.py --sizes 12 16 20 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

from warm_start_scaling import run_warm_start_scaling, results_to_dicts  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extended_path = RESULTS_DIR / "extended_scaling.json"
    with open(extended_path) as f:
        extended = json.load(f)
    extended_by_n = {r["n_assets"]: r for r in extended["rows"]}

    print(f"=== Warm-start QAOA scaling: sizes={args.sizes} ===")
    warm_rows = run_warm_start_scaling(args.sizes, q=args.q, reps=args.reps, seed=args.seed)

    merged = []
    for r in warm_rows:
        base = extended_by_n.get(r.n_assets, {})
        row = {
            "n_assets": r.n_assets, "k": r.k,
            "bf_val": base.get("bf_val"),
            "sa_val": base.get("sa_val"), "sa_time_s": base.get("sa_time_s"),
            "standard_qaoa_val": base.get("qaoa_val"),
            "standard_qaoa_feasible": base.get("qaoa_feasible"),
            "standard_qaoa_converged": base.get("qaoa_converged"),
            "standard_qaoa_time_s": base.get("qaoa_time_s"),
            "warm_qaoa_val": r.warm_val,
            "warm_qaoa_feasible": r.warm_feasible,
            "warm_qaoa_converged": r.warm_converged,
            "warm_qaoa_time_s": r.warm_time_s,
        }
        bf_val = row["bf_val"]
        row["ratio_warm_vs_bf"] = (
            row["warm_qaoa_val"] / bf_val
            if bf_val and row["warm_qaoa_feasible"] and bf_val != 0 else None
        )
        row["ratio_standard_vs_bf"] = (
            row["standard_qaoa_val"] / bf_val
            if bf_val and row["standard_qaoa_feasible"] and bf_val != 0 else None
        )
        merged.append(row)
        print(f"n={r.n_assets:2d} | bf={bf_val} | standard-QAOA conv={row['standard_qaoa_converged']} "
              f"val={row['standard_qaoa_val']:.4f} | warm-QAOA conv={r.warm_converged} val={r.warm_val:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "warm_start_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "rows": merged,
            "warm_start_raw": results_to_dicts(warm_rows),
            "config": {"sizes": args.sizes, "q": args.q, "reps": args.reps, "seed": args.seed},
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
