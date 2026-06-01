"""Abstract base class for feasibility projectors."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from qgridx.config import ProblemSpec
from qgridx.quantum.base import QuantumResult


@dataclasses.dataclass
class InvestmentPlan:
    """A feasible (post-projection) investment plan.

    Attributes:
        bitstring:   Binary array of shape (m,) — projected decision variables.
        build_mask:  Boolean array length n_buses — which buses get a battery.
        size_mwh:    Float array length n_buses — assigned MWh capacity per bus.
        microgrids:  Boolean array — which microgrid zones are activated.
        total_capex: Total investment cost in $.
        circuit_id:  Index of the originating quantum circuit.
    """
    bitstring: np.ndarray
    build_mask: np.ndarray
    size_mwh: np.ndarray
    microgrids: np.ndarray
    total_capex: float
    circuit_id: int


class FeasibilityProjector(ABC):
    """Projects a raw quantum bitstring to the nearest feasible investment plan."""

    @abstractmethod
    def project(
        self,
        result: QuantumResult,
        spec: ProblemSpec,
    ) -> InvestmentPlan:
        """Project a single :class:`QuantumResult` to a feasible plan.

        Args:
            result: Output of a single circuit execution (bitstring + confidence).
            spec:   Problem specification providing the feasibility constraints.

        Returns:
            A feasible :class:`InvestmentPlan`.
        """
