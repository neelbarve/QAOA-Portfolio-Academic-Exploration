r"""
CLI entry point: run the same small QAOA instance on real IQM hardware
(stage 13) and on the Aer simulator (standard mixer, stage 06; warm-start
mixer, stage 14), and compare - the same kind of hardware-vs-simulator
check Yalovetzky et al. (2026) and the IQM summer-school notebook both run,
at a size small and cheap enough to be a deliberate, one-off use of shared
real QPU time rather than a sweep.

Connection is validated for free before this script is ever invoked
(IQMClient.get_about/get_health/get_static_quantum_architecture - no shots,
no cost) - this script itself DOES submit a real, metered job the moment it
runs, so it is never called automatically by anything else in this
project (not the dashboard, not any other script).

Usage:
    python scripts/run_iqm_hardware_comparison.py --n-assets 4 --shots 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE_DIR))
import _bootstrap  # noqa: E402

from markowitz import brute_force_exact, objective  # noqa: E402
from qiskit_qubo import build_qubo  # noqa: E402
from qaoa_solver import solve_qaoa  # noqa: E402
from warm_start_qaoa import solve_warm_start_qaoa  # noqa: E402
from iqm_hardware import iqm_available, run_qaoa_on_iqm_hardware, DEFAULT_DEVICE_INSTANCE  # noqa: E402
from scaling_benchmark import make_synthetic_universe  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-assets", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--device-instance", type=str, default=DEFAULT_DEVICE_INSTANCE)
    args = parser.parse_args()

    if not iqm_available():
        print("IQM_RESONANCE_TOKEN is not set - nothing to do. See README section 9 for setup.")
        sys.exit(1)

    n, k, q = args.n_assets, args.n_assets // 2, args.q
    mu, sigma = make_synthetic_universe(n, seed=args.seed)
    print(f"Instance: n={n}, k={k}, q={q}, seed={args.seed} (same synthetic-universe "
          f"construction used throughout this project's scaling studies)")

    bf = brute_force_exact(mu, sigma, k, q)
    print(f"Brute-force optimum: {bf.value:.4f}, selection={bf.selection}")

    qp = build_qubo(mu, sigma, k, q)
    sim_standard = solve_qaoa(qp, reps=args.reps, seed=args.seed, shots=args.shots)
    sim_std_val = objective(mu, sigma, sim_standard.x, q)
    sim_std_feasible = bool((sim_standard.x.sum() - k) == 0)
    print(f"Simulator, standard mixer: val={sim_std_val:.4f} feasible={sim_std_feasible} "
          f"converged={sim_standard.converged} ({sim_standard.elapsed_s:.2f}s)")

    sim_warm = solve_warm_start_qaoa(mu, sigma, k, q, reps=args.reps, seed=args.seed, shots=args.shots)
    sim_warm_val = objective(mu, sigma, sim_warm.qaoa.x, q)
    sim_warm_feasible = bool((sim_warm.qaoa.x.sum() - k) == 0)
    print(f"Simulator, warm-start mixer: val={sim_warm_val:.4f} feasible={sim_warm_feasible} "
          f"converged={sim_warm.qaoa.converged} ({sim_warm.qaoa.elapsed_s:.2f}s)")

    print(f"\nSubmitting to REAL IQM hardware ({args.device_instance}, reps={args.reps}, "
          f"shots={args.shots}) - this uses metered QPU time...")
    hw = run_qaoa_on_iqm_hardware(
        mu, sigma, k, q, reps=args.reps, shots=args.shots, device_instance=args.device_instance,
        known_optimal_selection=frozenset(bf.selection),
    )
    if hw is None:
        print("Hardware run returned None (token not available at call time).")
        sys.exit(1)

    print(f"Hardware result (top bitstring): val={hw.best_value:.4f} feasible={hw.best_feasible} "
          f"best_bitstring={hw.best_bitstring} n_qubits={hw.n_qubits} ({hw.elapsed_s:.2f}s)")
    top5 = sorted(hw.counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"Top 5 sampled bitstrings (of {sum(hw.counts.values())} shots): {top5}")
    print(f"Fraction of shots that were FEASIBLE (sum(x)=k): {hw.feasible_shot_fraction*100:.1f}%")
    print(f"Fraction of shots landing on the EXACT true optimum: "
          f"{hw.true_optimum_shot_fraction*100:.1f}%" if hw.true_optimum_shot_fraction is not None else "n/a")
    print(f"Best value among any FEASIBLE sampled shot: {hw.best_feasible_value} "
          f"(bitstring {hw.best_feasible_bitstring})")

    out = {
        "config": {"n_assets": n, "k": k, "q": q, "seed": args.seed, "reps": args.reps,
                   "shots": args.shots, "device_instance": args.device_instance},
        "bf_val": bf.value, "bf_selection": list(bf.selection),
        "sim_standard": {"val": sim_std_val, "feasible": sim_std_feasible,
                          "converged": sim_standard.converged, "elapsed_s": sim_standard.elapsed_s},
        "sim_warm_start": {"val": sim_warm_val, "feasible": sim_warm_feasible,
                            "converged": sim_warm.qaoa.converged, "elapsed_s": sim_warm.qaoa.elapsed_s},
        "hardware": {"val": hw.best_value, "feasible": hw.best_feasible,
                     "best_bitstring": hw.best_bitstring, "n_qubits": hw.n_qubits,
                     "elapsed_s": hw.elapsed_s, "counts": hw.counts,
                     "top5": top5,
                     "feasible_shot_fraction": hw.feasible_shot_fraction,
                     "true_optimum_shot_fraction": hw.true_optimum_shot_fraction,
                     "best_feasible_value": hw.best_feasible_value,
                     "best_feasible_bitstring": hw.best_feasible_bitstring},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "iqm_hardware_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
