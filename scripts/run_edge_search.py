r"""
CLI entry point for the "is there a QAOA edge" search: stage 11 (harder,
sector-constrained QUBO) + stage 12 (SA baseline, extended-n scaling).
==============================================================================
This exists because a direct question deserves a direct, run-it-yourself
answer, not just a literature citation: given the reference papers find no
demonstrated QAOA advantage over classical methods (see the dashboard's
"Is There An Edge" page and README section for the full literature review),
where - if anywhere - does this project's own pipeline show one?

Two independent experiments, run and reported honestly regardless of outcome:

  1. Hardness sweep (stage 11): fixed problem size, increasing constraint
     complexity (more sectors, tighter per-sector caps) - tests Uotila et
     al.'s speculative "harder constraints favor QAOA" argument directly.
  2. Extended scaling (stage 12): fixed constraint structure, increasing n
     past where brute force stays practical, comparing QAOA against
     simulated annealing (the realistic classical baseline, matching
     Yalovetzky et al.'s own methodology) rather than an exact solver that
     stops being available.

Writes:
  results/hardness_sweep.json
  results/extended_scaling.json

Usage:
    python scripts/run_edge_search.py --preset quick
    python scripts/run_edge_search.py --preset standard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

from hardness_sweep import run_hardness_sweep, results_to_dicts as hardness_to_dicts  # noqa: E402
from extended_scaling import run_extended_scaling, results_to_dicts as extended_to_dicts  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PRESETS = {
    "quick": dict(
        hardness_n=10, n_sectors_list=[1, 2, 3, 5],
        extended_sizes=[10, 14, 18], exact_cutoff_n=16, sa_iters=3000,
    ),
    "standard": dict(
        hardness_n=12, n_sectors_list=[1, 2, 3, 4, 6],
        extended_sizes=[12, 16, 20, 24], exact_cutoff_n=20, sa_iters=6000,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(PRESETS), default="quick")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hardness", action="store_true")
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()
    cfg = PRESETS[args.preset]

    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.skip_hardness:
        print(f"=== Hardness sweep: n={cfg['hardness_n']}, n_sectors={cfg['n_sectors_list']} ===")
        hardness_rows = run_hardness_sweep(
            cfg["hardness_n"], cfg["n_sectors_list"], q=args.q, reps=args.reps, seed=args.seed,
        )
        with open(RESULTS_DIR / "hardness_sweep.json", "w") as f:
            json.dump({
                "rows": hardness_to_dicts(hardness_rows),
                "config": {**cfg, "q": args.q, "reps": args.reps, "seed": args.seed, "preset": args.preset},
            }, f, indent=2)
        print(f"Saved {RESULTS_DIR / 'hardness_sweep.json'}\n")

    if not args.skip_extended:
        print(f"=== Extended scaling: sizes={cfg['extended_sizes']}, exact_cutoff_n={cfg['exact_cutoff_n']} ===")
        extended_rows = run_extended_scaling(
            cfg["extended_sizes"], exact_cutoff_n=cfg["exact_cutoff_n"],
            q=args.q, reps=args.reps, seed=args.seed, sa_iters=cfg["sa_iters"],
        )
        with open(RESULTS_DIR / "extended_scaling.json", "w") as f:
            json.dump({
                "rows": extended_to_dicts(extended_rows),
                "config": {**cfg, "q": args.q, "reps": args.reps, "seed": args.seed, "preset": args.preset},
            }, f, indent=2)
        print(f"Saved {RESULTS_DIR / 'extended_scaling.json'}")


if __name__ == "__main__":
    main()
