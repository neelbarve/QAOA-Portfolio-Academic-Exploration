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

import numpy as np

from markowitz import objective, ClassicalResult


def simulated_annealing(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    n_iters: int = 4000, seed: int = 42, t_start: float = 1.0, t_end: float = 1e-3,
) -> ClassicalResult:
    rng = np.random.default_rng(seed)
    n = len(mu)
    t0 = time.time()

    x = np.zeros(n, dtype=int)
    x[rng.choice(n, size=k, replace=False)] = 1
    cur_val = objective(mu, sigma, x, q)
    best_x, best_val = x.copy(), cur_val

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

    elapsed = time.time() - t0
    selection = tuple(sorted(np.flatnonzero(best_x).tolist()))
    return ClassicalResult("simulated_annealing", best_val, selection, True, elapsed)
