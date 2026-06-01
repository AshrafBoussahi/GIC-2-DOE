"""Abstract base class for Ising encoders."""
from __future__ import annotations

from abc import ABC, abstractmethod

from qgridx.config import ProblemSpec
from qgridx.problems.ising import IsingSpec


class IsingEncoder(ABC):
    """Converts a :class:`ProblemSpec` (plus optional cuts) into an :class:`IsingSpec`."""

    @abstractmethod
    def encode(
        self,
        spec: ProblemSpec,
        cut_pool: list[dict] | None = None,
    ) -> IsingSpec:
        """Encode the problem into an Ising Hamiltonian.

        Args:
            spec:      Validated problem specification.
            cut_pool:  Optional list of cut dictionaries from the subproblem.

        Returns:
            :class:`IsingSpec` ready for the quantum master.
        """
