r"""
Stage 06b - QAOA solver.
===========================
Solves the QUBO built in stage 05 with Qiskit's QAOA
(qiskit_algorithms.QAOA + MinimumEigenOptimizer), on Aer's noiseless
statevector-backed sampler via the TranspilingAerSampler shim (06a).

Mixer: standard transverse-field X-mixer (H_mix = sum_i X_i), initial state
|+>^n. This is the mixer every one of the five reference papers treats as
the baseline, and the only one that needs no extra state-preparation
circuit. Brandhofer et al. (2023) show "hard-constraint" Dicke-state XY
-mixers (ring / full / QAMPA) reach higher approximation ratios in the
noise-free case but are LESS noise-robust past a moderate depolarizing
error threshold, and require preparing a Dicke state up front - a genuinely
separate, more involved implementation. That comparison is documented on
the academic dashboard page as a benchmarked design space from the
literature; only the standard mixer is implemented and run here, so the
project's own benchmarking claims stay backed by code that actually ran,
per the "honest benchmarking" principle carried over from qaoa_v1/REPORT.md.

Initial parameter point: a linear ramp (gamma increasing across layers,
beta decreasing), motivated by the adiabatic-annealing interpretation of
QAOA (gamma controls "how much cost Hamiltonian", beta "how much mixer" -
an adiabatic path ramps the former up and the latter down). This is a
simplified version of the linear-ansatz initialization in Brandhofer et al.
(who additionally grid-search the two slope parameters per instance,
~10,000 circuit evaluations just for the p=1 starting point) - the fixed
canonical fractions of pi used here are the standard simplified form seen
throughout the QAOA literature (e.g. Zhou et al. 2020), traded off against
runtime cost. reps > 1 extends the p=1 ramp via linear interpolation to the
new depth, which is one of the four strategies from the same paper (their
"re-fit linear ansatz" and "extrapolate from p-1" alternatives are not
implemented here).

Non-convergence handling: on a hard enough landscape (see stage 11's
sector-constrained QUBOs), COBYLA's fixed iteration budget can fail to
concentrate the sampled distribution on anything - every bitstring stays
near the uniform 1/shots probability. qiskit-optimization 0.7.0's
MinimumEigenOptimizer has a real bug in this exact situation: its
`_eigenvector_to_solutions` unconditionally squares dict-valued eigenstates
(assuming amplitude-like values), but qiskit-algorithms 0.4.0's QAOA
returns an eigenstate dict of already-normalized probabilities through this
sampler path - squaring an already-tiny near-uniform probability a second
time pushes every entry below the library's fixed 1e-6 cutoff, so NO
samples survive and MinimumEigenOptimizer crashes with an IndexError deep
in its internals instead of returning an infeasible-but-present result.
That crash is caught here and turned into an explicit, honestly-labelled
non-convergence result (`converged=False`) rather than letting a real
software bug masquerade as either a silent success or an unexplained
pipeline crash - the near-uniform output it corresponds to is itself a
genuine finding (QAOA's optimizer failing to learn anything on a hard
landscape), independent of the library bug that happens to surface it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, OptimizationResult

from aer_sampler import TranspilingAerSampler


@dataclass
class QAOARunResult:
    result: Optional[OptimizationResult]
    elapsed_s: float
    x: np.ndarray
    fval: float
    reps: int
    initial_point: np.ndarray
    converged: bool = True


def linear_ansatz_initial_point(
    reps: int, gamma_max: float = np.pi / 2, beta_max: float = np.pi / 4
) -> np.ndarray:
    """[gamma_1..gamma_p, beta_1..beta_p], gamma ramping 0 -> gamma_max,
    beta ramping beta_max -> 0 across the p layers - see module docstring."""
    i = np.arange(1, reps + 1)
    x = (2 * i - 1) / (2 * reps)   # Brandhofer et al.'s x_i^(p) schedule
    gammas = gamma_max * x
    betas = beta_max * (1 - x)
    return np.concatenate([gammas, betas])


def solve_qaoa(
    qp: QuadraticProgram,
    reps: int = 2,
    seed: int = 42,
    shots: int = 4096,
    maxiter: int = 250,
    initial_point: Optional[np.ndarray] = None,
    mixer: Optional[QuantumCircuit] = None,
    initial_state: Optional[QuantumCircuit] = None,
) -> QAOARunResult:
    """mixer / initial_state default to Qiskit's standard X-mixer / |+>^n
    when left as None (the module-docstring behavior). Stage 14 passes its
    own warm-start mixer/initial_state pair through these same arguments so
    the two variants share every other piece of the pipeline - only the
    mixer changes, isolating it as the one manipulated variable."""
    sampler = TranspilingAerSampler(seed=seed, default_shots=shots)
    optimizer = COBYLA(maxiter=maxiter)
    init_pt = linear_ansatz_initial_point(reps) if initial_point is None else initial_point

    qaoa = QAOA(
        sampler=sampler, optimizer=optimizer, reps=reps, initial_point=init_pt,
        mixer=mixer, initial_state=initial_state,
    )
    meo = MinimumEigenOptimizer(qaoa)

    n = qp.get_num_binary_vars()
    t0 = time.time()
    try:
        result = meo.solve(qp)
    except IndexError:
        # See the module docstring: qiskit-optimization 0.7.0's sample
        # interpretation crashes (rather than returning an infeasible
        # result) when QAOA's sampled distribution stays near-uniform -
        # i.e. the classical optimizer failed to concentrate probability
        # on anything within its iteration budget. Reported honestly as
        # non-convergence, not silently retried or hidden.
        elapsed = time.time() - t0
        return QAOARunResult(
            result=None, elapsed_s=elapsed, x=np.zeros(n), fval=float("nan"),
            reps=reps, initial_point=init_pt, converged=False,
        )
    elapsed = time.time() - t0

    return QAOARunResult(
        result=result, elapsed_s=elapsed, x=np.array(result.x), fval=result.fval,
        reps=reps, initial_point=init_pt,
    )
