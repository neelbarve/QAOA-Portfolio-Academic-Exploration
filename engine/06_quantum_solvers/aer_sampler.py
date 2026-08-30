"""
Stage 06a - Aer sampler shim.
===============================
qiskit_algorithms' QAOA builds a QAOAAnsatz circuit whose PauliEvolutionGate
blocks Aer's low-level circuit assembler doesn't know how to unroll on its
own. This transpiles each pub's circuit to Aer's basis gates right before
execution, bridging the two APIs. Without it: `AerError: unknown
instruction: QAOA`. Carried over verbatim from qaoa_v1 (portfolio_qaoa.py),
where it was already diagnosed and fixed - a version-compatibility seam in
the current Qiskit ecosystem, not a portfolio-optimization issue.
"""

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as _AerSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub


class TranspilingAerSampler(_AerSamplerV2):
    def run(self, pubs, *, shots=None):
        backend = AerSimulator()
        coerced = [SamplerPub.coerce(p, shots) for p in pubs]
        new_pubs = []
        for pub in coerced:
            t_circ = transpile(pub.circuit, backend, optimization_level=1)
            new_pubs.append((t_circ, pub.parameter_values, pub.shots))
        return super().run(new_pubs, shots=shots)
