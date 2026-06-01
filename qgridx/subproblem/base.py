"""Abstract base class for classical subproblems."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

from qgridx.config import ProblemSpec
from qgridx.problems.scenarios import Scenario
from qgridx.projection.base import InvestmentPlan


@dataclasses.dataclass
class SubproblemResult:
    """Result from evaluating an investment plan across one scenario.

    Attributes:
        plan_id:         Index of the investment plan.
        scenario_id:     Scenario id.
        feasible:        Whether the scenario is operationally feasible.
        operational_cost:Total operational cost (fuel + curtailment penalties).
        dual_info:       Optional dual variables / sensitivity information
                         used to generate Benders-inspired cuts.
    """
    plan_id: int
    scenario_id: int
    feasible: bool
    operational_cost: float
    dual_info: dict[str, Any] = dataclasses.field(default_factory=dict)


class SubproblemBase(ABC):
    """Evaluates investment plans across a scenario set and returns cuts."""

    @abstractmethod
    def solve_scenario(
        self,
        plan: InvestmentPlan,
        scenario: Scenario,
        spec: ProblemSpec,
        plan_id: int = 0,
    ) -> SubproblemResult:
        """Solve the operational subproblem for one scenario.

        Args:
            plan:      Feasible investment plan.
            scenario:  Operating scenario to evaluate.
            spec:      Problem specification.
            plan_id:   Index of this plan in the current iteration.

        Returns:
            :class:`SubproblemResult`.
        """

    def solve_all(
        self,
        plan: InvestmentPlan,
        scenarios: list[Scenario],
        spec: ProblemSpec,
        plan_id: int = 0,
        n_workers: int = 1,
    ) -> list[SubproblemResult]:
        """Solve all scenarios for a plan.

        Default: sequential. Override for parallel execution.
        """
        return [
            self.solve_scenario(plan, sc, spec, plan_id=plan_id)
            for sc in scenarios
        ]
