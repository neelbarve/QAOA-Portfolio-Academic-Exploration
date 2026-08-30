"""
Stage 12a - Simulated annealing: a realistic, scalable classical heuristic.
=============================================================================
brute_force_exact (stage 04) is the right ground truth at small n, but it
is NOT the classical method QAOA should be measured against once n grows
past exact-search range - that comparison is against a classical HEURISTIC,
because that's what a practitioner would actually reach for. This is also
exactly the comparison the most rigorous of the five reference papers
(Yalovetzky et al. 2026) uses: simulated annealing (SA) as the classical
baseline, not brute force, once problem size grows.

Implementation choice: SA operates directly on cardinality-preserving
bitstrings via swap moves (flip one selected asset off, one unselected
asset on) rather than on a continuous relaxation. This means SA - like a
constraint-preserving quantum mixer, and UNLIKE the standard X-mixer QAOA
actually run in this project - can never produce an infeasible sum(x) != k
solution. That asymmetry is reported honestly wherever SA and QAOA are
compared: SA's "feasible rate" is always 100% by construction, which is an
advantage of the algorithm design, not evidence of a better search.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from markowitz import objective, continuous_relaxation_fractional, ClassicalResult


def _relaxation_top_k_start(mu: np.ndarray, sigma: np.ndarray, k: int, q: float) -> Optional[np.ndarray]:
    """Warm-start initial point for SA: the same continuous relaxation stage
    14 uses for QAOA (stage 04's continuous_relaxation_fractional), rounded
    to its top-k assets by fractional weight - the single most natural
    "good guess" already sitting in the pipeline, reused here instead of
    computed twice."""
    c = continuous_relaxation_fractional(mu, sigma, k, q)
    if c is None:
        return None
    n = len(mu)
    x = np.zeros(n, dtype=int)
    x[np.argsort(-c)[:k]] = 1
    return x


def simulated_annealing(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    n_iters: int = 4000, seed: int = 42, t_start: float = 1.0, t_end: float = 1e-3,
    initial_x: Optional[np.ndarray] = None,
    track_history: bool = False,
) -> ClassicalResult | Tuple[ClassicalResult, List[float]]:
    """initial_x lets a caller start from something other than a random
    k-subset - stage 14's simulated_annealing_warm_started passes the
    relaxation-rounded start computed above; every other caller in this
    project leaves it None (the original, unchanged random-start behavior).
    track_history=True additionally returns the best-value-so-far at every
    iteration, for plotting convergence speed rather than only the final
    answer - used by scripts/run_warm_start_comparison.py, not by stage
    08/12's scaling sweeps (which only need the final result)."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    t0 = time.time()

    if initial_x is not None:
        x = initial_x.copy()
    else:
        x = np.zeros(n, dtype=int)
        x[rng.choice(n, size=k, replace=False)] = 1
    cur_val = objective(mu, sigma, x, q)
    best_x, best_val = x.copy(), cur_val
    history = [best_val] if track_history else None

    for it in range(n_iters):
        # geometric cooling schedule from t_start down to t_end over the run
        t = t_start * (t_end / t_start) ** (it / max(n_iters - 1, 1))
        selected = np.flatnonzero(x)
        unselected = np.flatnonzero(1 - x)
        i = selected[rng.integers(len(selected))]
        j = unselected[rng.integers(len(unselected))]

        x_new = x.copy()
        x_new[i], x_new[j] = 0, 1
        new_val = objective(mu, sigma, x_new, q)
        delta = new_val - cur_val

        if delta > 0 or rng.random() < np.exp(min(delta / max(t, 1e-12), 0)):
            x, cur_val = x_new, new_val
            if new_val > best_val:
                best_x, best_val = x_new.copy(), new_val
        if track_history:
            history.append(best_val)

    elapsed = time.time() - t0
    selection = tuple(sorted(np.flatnonzero(best_x).tolist()))
    result = ClassicalResult("simulated_annealing", best_val, selection, True, elapsed)
    return (result, history) if track_history else result


def simulated_annealing_warm_started(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    n_iters: int = 4000, seed: int = 42, t_start: float = 1.0, t_end: float = 1e-3,
    track_history: bool = False,
) -> ClassicalResult | Tuple[ClassicalResult, List[float]]:
    """Same annealing loop, started from the relaxation-rounded guess
    instead of a random k-subset - the direct SA analogue of stage 14's
    warm-start QAOA, sharing the same relaxation call. Falls back to a
    random start (with a result.method flag reflecting that) if the
    relaxation solver is unavailable in this environment."""
    x0 = _relaxation_top_k_start(mu, sigma, k, q)
    out = simulated_annealing(
        mu, sigma, k, q, n_iters=n_iters, seed=seed, t_start=t_start, t_end=t_end,
        initial_x=x0, track_history=track_history,
    )
    result = out[0] if track_history else out
    result.method = "simulated_annealing_warm_started" if x0 is not None else "simulated_annealing_warm_start_unavailable"
    return out
