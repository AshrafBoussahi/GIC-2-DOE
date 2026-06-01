"""ProblemSpec re-export for convenience.

The canonical definition lives in :mod:`qgridx.config` to keep the single
source of truth. This module re-exports it so pipeline code can import from
``qgridx.problems.base`` without circular dependencies.
"""
from qgridx.config import (  # noqa: F401
    Bus,
    BudgetSpec,
    CandidatesSpec,
    Contingency,
    CostsSpec,
    EncodingSpec,
    Generator,
    Line,
    ProblemSpec,
    ScenariosSpec,
    SystemSpec,
)
