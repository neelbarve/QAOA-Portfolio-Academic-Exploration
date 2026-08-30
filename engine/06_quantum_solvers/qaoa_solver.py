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
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, OptimizationResult

from aer_sampler import TranspilingAerSampler


@dataclass
class QAOARunResult:
    result: OptimizationResult
    elapsed_s: float
    x: np.ndarray
    fval: float
    reps: int
    initial_point: np.ndarray


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
) -> QAOARunResult:
    sampler = TranspilingAerSampler(seed=seed, default_shots=shots)
    optimizer = COBYLA(maxiter=maxiter)
    init_pt = linear_ansatz_initial_point(reps) if initial_point is None else initial_point

    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, initial_point=init_pt)
    meo = MinimumEigenOptimizer(qaoa)

    t0 = time.time()
    result = meo.solve(qp)
    elapsed = time.time() - t0

    return QAOARunResult(
        result=result, elapsed_s=elapsed, x=np.array(result.x), fval=result.fval,
        reps=reps, initial_point=init_pt,
    )
