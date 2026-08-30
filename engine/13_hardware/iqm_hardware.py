"""
Stage 13 - Real IQM quantum hardware execution (optional, gated).
=====================================================================
Everything else in this project runs QAOA on Aer's noiseless statevector
simulator. That's a deliberate limitation, not an oversight: Brandhofer et
al. (2023) show that under realistic hardware noise, the theoretically
best-performing mixers degrade fastest, and current per-gate error rates
combined with today's qubit counts are exactly the regime the literature
review (README, "Is there a QAOA edge" section) says is too far from where
any advantage would plausibly appear. This stage exists to actually
CHARACTERIZE that gap on real hardware at a small, cheap instance size -
not to chase an advantage this project's own literature review says isn't
there yet, but to see concretely how far real noisy output drifts from the
simulator's idealized picture, the same kind of comparison Yalovetzky et
al. (2026) run at much larger scale on trapped-ion hardware.

Gated on an IQM Resonance API token, following the exact same pattern as
Tiingo (engine/01_data/tiingo_source.py): the token is read from an
environment variable, never hardcoded, and this module's functions no-op
with a clear message if it isn't set - nothing here calls out to real,
metered hardware unless IQM_RESONANCE_TOKEN is explicitly present AND the
caller explicitly asks to run.

Setup (mirrors the Tiingo instructions in the README):
    setx IQM_RESONANCE_TOKEN "your-token-here"      (Windows, permanent)
    $env:IQM_RESONANCE_TOKEN = "your-token-here"     (Windows, current session)

Uses qrisp's IQMBackend (qrisp.interface), which wraps IQM Resonance's
cloud API - already a project dependency (the instructions doc calls out
qrisp by name for IQM hardware access) and considerably simpler than
driving iqm-client/qiskit-on-iqm directly for a single small circuit.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

DEFAULT_DEVICE_INSTANCE = "garnet"


def iqm_available() -> bool:
    return bool(os.environ.get("IQM_RESONANCE_TOKEN"))


@dataclass
class HardwareRunResult:
    device_instance: str
    n_qubits: int
    shots: int
    elapsed_s: float
    counts: Dict[str, int]              # raw bitstring -> shot count
    best_bitstring: str
    best_feasible: bool
    best_value: float


def run_qaoa_on_iqm_hardware(
    mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
    reps: int = 1, shots: int = 1000, device_instance: str = DEFAULT_DEVICE_INSTANCE,
) -> Optional[HardwareRunResult]:
    """Builds the SAME QUBO (stage 05) and the SAME linear-ramp QAOA circuit
    construction (stage 06's ansatz choice) used everywhere else in this
    project, but submits it to real IQM hardware via qrisp's IQMBackend
    instead of Aer. Returns None immediately (no network call, no cost) if
    IQM_RESONANCE_TOKEN isn't set - callers must check iqm_available() or
    handle a None return, exactly like the Tiingo adapters do.

    Deliberately capped defaults (reps=1, shots=1000, small n expected from
    the caller) - real QPU time is a shared, metered resource, unlike the
    free simulator everywhere else in this project."""
    if not iqm_available():
        return None

    from qiskit_optimization.converters import (
        IntegerToBinary, LinearEqualityToPenalty,
    )
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qrisp.interface import IQMBackend

    from qiskit_qubo import build_qubo
    from qaoa_solver import linear_ansatz_initial_point
    from markowitz import objective

    n = len(mu)
    qp = build_qubo(mu, sigma, k, q)
    qp = IntegerToBinary().convert(qp)
    qp = LinearEqualityToPenalty().convert(qp)
    operator, _offset = qp.to_ising()

    ansatz = QAOAAnsatz(cost_operator=operator, reps=reps)
    init_pt = linear_ansatz_initial_point(reps)
    bound_circuit = ansatz.assign_parameters(init_pt)
    bound_circuit.measure_all()

    backend = IQMBackend(
        api_token=os.environ["IQM_RESONANCE_TOKEN"], device_instance=device_instance,
    )

    t0 = time.time()
    job_result = backend.run(bound_circuit, shots=shots)
    elapsed = time.time() - t0

    counts: Dict[str, int] = dict(job_result)
    best_bitstring = max(counts, key=counts.get)
    x = np.array([int(b) for b in best_bitstring[::-1][:n]])
    best_feasible = bool(np.isclose(x.sum(), k))
    best_value = objective(mu, sigma, x, q)

    return HardwareRunResult(
        device_instance=device_instance, n_qubits=operator.num_qubits, shots=shots,
        elapsed_s=elapsed, counts=counts, best_bitstring=best_bitstring,
        best_feasible=best_feasible, best_value=best_value,
    )
