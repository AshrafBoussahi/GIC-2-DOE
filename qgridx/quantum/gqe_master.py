"""PCE-GQE quantum master: the default QuantumMasterBase implementation.

Combines PCE assignment, brickwork ansatz, backend execution, and readout
into a single ``propose()`` call.  Also provides a random-baseline master
for comparison.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from qgridx.config import QuantumMasterConfig
from qgridx.problems.ising import IsingSpec
from qgridx.quantum.ansatz import build_brickwork_circuit
from qgridx.quantum.backend import build_backend
from qgridx.quantum.base import QuantumMasterBase, QuantumResult
from qgridx.quantum.pce import PCEAssignment
from qgridx.quantum.readout import make_quantum_result
from qgridx.registry import register_component


@register_component("quantum_master", "pce_gqe")
class PCEGQEMaster(QuantumMasterBase):
    """Generative quantum master using PCE-brickwork circuits.

    At each call to :meth:`propose`:
    1. Build the PCE assignment for the current Ising size.
    2. Generate ``n_samples`` circuits via the brickwork ansatz.
    3. Execute each circuit using the configured backend.
    4. Convert correlations → bitstrings via readout.
    """

    def __init__(self, cfg: QuantumMasterConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed
        self._backend = build_backend(cfg.backend)

    def propose(
        self,
        ising: IsingSpec,
        n_samples: int,
        context: Any = None,
    ) -> list[QuantumResult]:
        pce = PCEAssignment(self.cfg.pce, m=ising.m, seed=self.seed)
        results: list[QuantumResult] = []
        rng = np.random.default_rng(self.seed)

        for cid in range(n_samples):
            gates, params = build_brickwork_circuit(
                n=pce.n,
                n_layers=self.cfg.ansatz.n_layers,
                seed=self.seed + cid,
            )
            correlations = self._backend.run_circuit(
                gates=gates,
                n_qubits=pce.n,
                pauli_strings=pce.pauli_strings,
                shots=self.cfg.backend.shots,
            )
            # Pad/truncate correlations to length m
            if len(correlations) < ising.m:
                correlations = np.pad(correlations, (0, ising.m - len(correlations)))
            elif len(correlations) > ising.m:
                correlations = correlations[: ising.m]

            results.append(make_quantum_result(correlations, circuit_id=cid))

        return results


@register_component("quantum_master", "random_baseline")
class RandomBaselineMaster(QuantumMasterBase):
    """Random bitstring generator — used as a comparison baseline.

    Produces uniformly random bitstrings with uniform confidence = 0.5.
    This is intentionally weaker than PCE-GQE.
    """

    def __init__(self, cfg: QuantumMasterConfig | None = None, seed: int = 0) -> None:
        self.seed = seed

    def propose(
        self,
        ising: IsingSpec,
        n_samples: int,
        context: Any = None,
    ) -> list[QuantumResult]:
        rng = np.random.default_rng(self.seed)
        results: list[QuantumResult] = []
        for cid in range(n_samples):
            correlations = rng.uniform(-1.0, 1.0, size=ising.m)
            results.append(make_quantum_result(correlations, circuit_id=cid))
        return results
