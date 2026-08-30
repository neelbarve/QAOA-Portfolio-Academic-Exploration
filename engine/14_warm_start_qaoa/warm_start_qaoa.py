r"""
Stage 14 - Warm-started QAOA: a hybrid attempt to fix the non-convergence
found in stage 12's extended scaling run.
==============================================================================
Motivation. Section 9 of the README (and the dashboard's "Is There An Edge"
tab) documents that standard QAOA (stage 06's H-gate initial state, X-mixer)
stops converging entirely from n=16 upward - the sampler's output stays
near-uniform, COBYLA has nothing to climb. That is not a fundamental limit of
QAOA itself; it is a known, named failure mode of the SPECIFIC construction
used - starting from a uniform superposition over all 2^n bitstrings and
relying on a penalty term to eventually discourage the C(n,k)-infeasible
majority of them - and the literature has a concrete, implementable fix:
Egger, Marecek & Woerner, "Warm-starting quantum optimization" (2021).

The idea: instead of starting QAOA from |+>^n (uniform over all bitstrings)
and a plain X-mixer, start it from a state biased toward a good classical
guess, with a mixer that keeps that bias as its natural resting point.

  1. Solve the CONTINUOUS relaxation of the same QUBO (already implemented
     in stage 04's continuous_relaxation_fractional - the box-constrained QP
     this project already runs as its "practitioner's first guess" classical
     baseline). This gives a fractional c_i in [0, 1] per asset - "how much"
     of asset i the relaxed, unconstrained-integrality problem wants.

  2. Regularize away from the extremes: c_i is clipped to
     [epsilon, 1 - epsilon] (epsilon = 0.25 here, Egger et al.'s recommended
     default). Without this, an asset the relaxation is fully confident about
     (c_i = 0 or 1 exactly) gets a mixer that CANNOT explore away from that
     bit at all - a warm start with zero ability to correct a wrong guess is
     worse than no warm start.

  3. Convert each c_i to a Bloch-sphere angle theta_i = 2*arcsin(sqrt(c_i)),
     chosen so that RY(theta_i)|0> measures 1 with probability exactly c_i.

  4. Initial state: apply RY(theta_i) to qubit i instead of the standard
     H gate - biases the starting superposition toward the relaxation's
     guess without collapsing it to a single bitstring.

  5. Mixer: for each qubit, RY(theta_i) . RZ(-2*beta) . RY(-theta_i) - this
     conjugates the standard Z-rotation mixer into the tilted basis where the
     warm-start state sits at the "north pole," so it interpolates smoothly
     around the SAME initial bias rather than uniformly mixing toward the
     computational basis states the way the standard X-mixer does. At
     theta_i = pi/2 (c_i = 0.5, i.e. "the relaxation has no opinion") this
     reduces exactly to the standard X-mixer - the warm-start construction
     strictly generalizes stage 06's baseline rather than replacing it with
     something unrelated.

This changes ONLY the mixer/initial-state pair passed into stage 06's
solve_qaoa - everything else (the QUBO from stage 05, the optimizer, the
sampler, the shot count) is identical, so any quality/convergence difference
observed is attributable to the warm start specifically, not a confound.

Honesty note, stated up front rather than after seeing results: this is a
real, published technique, not a guaranteed fix. Egger et al.'s own paper is
careful to frame it as improving the APPROXIMATION RATIO and reducing
variance across random restarts, not as a proof that warm-starting closes
the gap to classical heuristics identified in section 9 - it may not. The
scaling-comparison script (scripts/run_warm_start_comparison.py) reports
whatever actually happens, including a null result, exactly as stage 08-12
did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit

from markowitz import continuous_relaxation_fractional
from qiskit_qubo import build_qubo
from qaoa_solver import solve_qaoa, linear_ansatz_initial_point, QAOARunResult

EPSILON = 0.25  # Egger et al.'s recommended regularization margin


@dataclass
class WarmStartInfo:
    available: bool
    c_fractional: Optional[np.ndarray]  # raw relaxation output, pre-regularization
    thetas: Optional[np.ndarray]


def compute_warm_start_angles(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float, epsilon: float = EPSILON,
) -> WarmStartInfo:
    c = continuous_relaxation_fractional(mu, sigma, k, q)
    if c is None:
        return WarmStartInfo(available=False, c_fractional=None, thetas=None)
    c_reg = np.clip(c, epsilon, 1 - epsilon)
    thetas = 2 * np.arcsin(np.sqrt(c_reg))
    return WarmStartInfo(available=True, c_fractional=c, thetas=thetas)


def build_warm_start_initial_state(thetas: np.ndarray) -> QuantumCircuit:
    n = len(thetas)
    qc = QuantumCircuit(n)
    for i, theta in enumerate(thetas):
        qc.ry(float(theta), i)
    return qc


def build_warm_start_mixer(thetas: np.ndarray) -> QuantumCircuit:
    """One free Parameter 'beta', applied identically to every rep - the
    same contract stage 06's standard X-mixer (built implicitly by Qiskit)
    follows, so QAOA(reps=p) repeats this block p times with independent
    beta values as usual."""
    n = len(thetas)
    beta = Parameter("beta")
    qc = QuantumCircuit(n)
    for i, theta in enumerate(thetas):
        qc.ry(-float(theta), i)
        qc.rz(-2 * beta, i)
        qc.ry(float(theta), i)
    return qc


@dataclass
class WarmStartQAOARunResult:
    qaoa: QAOARunResult
    warm_start: WarmStartInfo


def solve_warm_start_qaoa(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    reps: int = 2, seed: int = 42, shots: int = 4096, maxiter: int = 250,
    epsilon: float = EPSILON,
) -> WarmStartQAOARunResult:
    warm = compute_warm_start_angles(mu, sigma, k, q, epsilon=epsilon)
    qp = build_qubo(mu, sigma, k, q)

    if not warm.available:
        # Relaxation solver unavailable in this environment - fall back to
        # the standard mixer/initial state rather than crashing; the caller
        # can see warm.available=False and report this honestly.
        run = solve_qaoa(qp, reps=reps, seed=seed, shots=shots, maxiter=maxiter)
        return WarmStartQAOARunResult(qaoa=run, warm_start=warm)

    mixer = build_warm_start_mixer(warm.thetas)
    initial_state = build_warm_start_initial_state(warm.thetas)
    # beta_max halved relative to stage 06's default linear ramp: the warm
    # -start mixer's rotation is centered on the biased state rather than on
    # |+>^n, so the same beta sweeps a proportionally larger effective angle
    # away from a (now informative) starting point - Egger et al. section
    # IV.A uses a similarly reduced mixer angle range for this reason.
    init_pt = linear_ansatz_initial_point(reps, beta_max=np.pi / 8)

    run = solve_qaoa(
        qp, reps=reps, seed=seed, shots=shots, maxiter=maxiter,
        initial_point=init_pt, mixer=mixer, initial_state=initial_state,
    )
    return WarmStartQAOARunResult(qaoa=run, warm_start=warm)
