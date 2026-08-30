"""
Stage 06c - Exact QUBO diagonalization (control condition).
================================================================
Solves the SAME penalized QUBO QAOA gets (not the original constrained
problem) via NumPyMinimumEigensolver - exact statevector diagonalization,
no variational/classical-optimizer loop involved. This is the control
that separates two very different failure modes when QAOA underperforms:

  - if this control ALSO fails to recover the true constrained optimum,
    the QUBO formulation itself (the penalty term, stage 05) is the
    problem, not QAOA;
  - if this control succeeds but QAOA doesn't, the classical variational
    optimizer (COBYLA) is the bottleneck, not the formulation or the
    quantum device.

qaoa_v1's REPORT.md found exactly the second failure mode at n=14 - this
control is what made that diagnosis possible, so it's carried forward here
unchanged rather than dropped as "redundant" with QAOA.
"""

from __future__ import annotations

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, OptimizationResult


def solve_exact_qubo(qp: QuadraticProgram) -> OptimizationResult:
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    return exact.solve(qp)
