"""Abstract base class for quantum master components."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qgridx.problems.ising import IsingSpec


@dataclasses.dataclass
class QuantumResult:
    """Output of a single circuit execution.

    Attributes:
        correlations: Array of shape (m,) — estimated <P_i> for each variable.
        bitstring:    Raw bitstring x_i = sign(<P_i>), shape (m,).
        confidence:   |<P_i>| — readout confidence per variable, shape (m,).
        circuit_id:   Index within this iteration's M circuits.
        metadata:     Free-form dict for extra diagnostics.
    """
    correlations: np.ndarray
    bitstring: np.ndarray
    confidence: np.ndarray
    circuit_id: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class QuantumMasterBase(ABC):
    """Abstract quantum master: generates M candidate plans from an Ising."""

    @abstractmethod
    def propose(
        self,
        ising: IsingSpec,
        n_samples: int,
        context: Any = None,
    ) -> list[QuantumResult]:
        """Sample ``n_samples`` candidate plans from the Ising.

        Args:
            ising:     Current Ising specification (rescaled, R2-compliant).
            n_samples: Number of circuits / candidate plans to generate (M).
            context:   Optional context vector from the GNN encoder.

        Returns:
            List of :class:`QuantumResult` objects, one per circuit.
        """
