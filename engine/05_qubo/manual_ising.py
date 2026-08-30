r"""
Stage 05b - QUBO -> Ising Hamiltonian, derived by hand.
============================================================
This is the "actual learning" step the project instructions call out: the
translation from the constrained classical problem to the unconstrained
Ising Hamiltonian QAOA's cost unitary is built from. qiskit-finance
(stage 05a) does this automatically for the running pipeline; this module
reproduces the same derivation explicitly and symbolically, for the
dashboard's academic/math page and as a cross-check of stage 05a's output
(see tests/sanity_checks.py). The approach is inspired by, but not a
literal reproduction of, the penalty-tuning and global-scaling ideas in
Brandhofer et al. (2023) "Benchmarking the performance of portfolio
optimization with QAOA" (Quantum Information Processing) - only a
paraphrase of their derivation was available as source material, and the
project instructions explicitly say papers are reference material, not a
template to match exactly.

------------------------------------------------------------------
1. Start from the same MINIMIZE convention qiskit-finance uses:

    min   G(x) = q * x^T Sigma x - mu^T x,   x in {0,1}^n
    s.t.  sum_i x_i = k

2. Fold the equality constraint into the objective as a quadratic penalty:

    F(x) = G(x) + A * (sum_i x_i - k)^2,     A > 0

   Using x_i^2 = x_i (true for binary variables) and expanding:

    F(x) = sum_i c_i x_i + 2 * sum_{i<j} Q_ij x_i x_j + A*k^2

   where
    c_i   = q*sigma_ii - mu_i + A*(1 - 2k)
    Q_ij  = q*sigma_ij + A                      (i != j)

3. Map to spin variables x_i = (1 - z_i)/2, z_i in {-1, +1}, and collect
   terms (algebra in the module-level docstring of `to_ising_hamiltonian`
   below) to get the standard Ising cost Hamiltonian

    H(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const

   which is exactly the operator QAOA's cost unitary exp(-i*gamma*H)
   exponentiates, one ZZ-rotation per (i,j) pair with J_ij != 0 and one
   Z-rotation per qubit with h_i != 0.
------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class IsingHamiltonian:
    h: np.ndarray                       # linear coefficients, shape (n,)
    J: Dict[Tuple[int, int], float]      # quadratic coefficients, keys i<j
    constant: float
    penalty: float                       # A used to build this Hamiltonian
    n_qubits: int

    def energy(self, z: np.ndarray) -> float:
        """H(z) for a +-1 spin vector - used to sanity-check a sampled
        bitstring's cost without re-deriving anything."""
        val = float(self.h @ z) + self.constant
        for (i, j), coeff in self.J.items():
            val += coeff * z[i] * z[j]
        return val

    def spectral_width(self, exact_if_n_leq: int = 18) -> float:
        """max H(z) - min H(z) over all feasible spin assignments. Exact by
        enumeration for small n (used by the global scaling factor below);
        for larger n, an upper bound via sum of absolute coefficients."""
        if self.n_qubits <= exact_if_n_leq:
            energies = [
                self.energy(np.array(bits))
                for bits in itertools.product([-1, 1], repeat=self.n_qubits)
            ]
            return float(max(energies) - min(energies))
        bound = float(np.sum(np.abs(self.h)) + sum(abs(v) for v in self.J.values()))
        return 2 * bound  # loose upper bound: max <= sum|coeff|, min >= -sum|coeff|


def find_minimal_penalty(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    n_grid: int = 30, exact_if_n_leq: int = 18,
) -> Tuple[float, str]:
    """Smallest tested penalty A (from a log-spaced grid) for which the true
    constrained optimum still beats every infeasible bitstring under the
    penalized objective - i.e. the smallest A that doesn't let the QAOA
    optimizer "cheat" with an infeasible answer, without being so large it
    washes out the actual risk/return landscape (see REPORT.md's discussion
    of this exact failure mode in qaoa_v1).

    Exact (2^n enumeration) for n <= exact_if_n_leq; a conservative
    closed-form heuristic otherwise, since exhaustive enumeration is only
    tractable at small n - documented rather than silently swapped in.
    """
    n = len(mu)
    obj_scale = max(q * np.max(np.abs(sigma)) * n + np.max(np.abs(mu)), 1e-8)

    if n > exact_if_n_leq:
        A = 2.0 * obj_scale
        return A, f"heuristic (n={n} > exact search cutoff {exact_if_n_leq})"

    candidates = np.geomspace(0.05, 10.0, n_grid) * obj_scale
    for A in candidates:
        feasible_best, infeasible_best = np.inf, np.inf
        for bits in itertools.product([0, 1], repeat=n):
            x = np.array(bits)
            g = q * (x @ sigma @ x) - mu @ x
            f = g + A * (x.sum() - k) ** 2
            if x.sum() == k:
                feasible_best = min(feasible_best, f)
            else:
                infeasible_best = min(infeasible_best, f)
        if feasible_best < infeasible_best:
            return float(A), f"exact grid search over {2 ** n} states (n={n})"
    return float(candidates[-1]), f"exact grid search (n={n}) - no clean separation within search bound, using largest tested A"


def to_ising_hamiltonian(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float, penalty: float | None = None,
) -> IsingHamiltonian:
    """Build H(z) = sum h_i z_i + sum_{i<j} J_ij z_i z_j + const from
    (mu, sigma, k, q), auto-tuning the penalty A via find_minimal_penalty
    if not supplied explicitly."""
    n = len(mu)
    if penalty is None:
        penalty, _ = find_minimal_penalty(mu, sigma, k, q)
    A = penalty

    c = q * np.diag(sigma) - mu + A * (1 - 2 * k)
    Q = {}  # Q[i, j] = q*sigma_ij + A, for i < j
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = q * sigma[i, j] + A

    h = np.zeros(n)
    J: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        neighbor_sum = sum(Q[i, j] if i < j else Q[j, i] for j in range(n) if j != i)
        h[i] = -0.5 * c[i] - 0.5 * neighbor_sum
    for (i, j), q_ij in Q.items():
        J[i, j] = q_ij / 2.0

    const = 0.5 * float(c.sum()) + 0.5 * sum(Q.values()) + A * k * k

    return IsingHamiltonian(h=h, J=J, constant=const, penalty=A, n_qubits=n)


def global_scaling_factor(hamiltonian: IsingHamiltonian) -> float:
    """lambda = (mixer spectral width) / (cost Hamiltonian spectral width).
    Rescaling H -> lambda*H before building the QAOA cost unitary keeps
    gamma and beta on comparable numeric scales, which is what actually
    stabilizes the classical outer-loop optimizer (COBYLA/SLSQP) - without
    it, gamma and beta can differ by orders of magnitude and the optimizer
    wastes iterations on the wrong scale. Standard X-mixer H_mix = sum X_i
    has spectrum [-n, n], width 2n."""
    n = hamiltonian.n_qubits
    mixer_width = 2.0 * n
    cost_width = hamiltonian.spectral_width()
    if cost_width <= 0:
        return 1.0
    return mixer_width / cost_width
