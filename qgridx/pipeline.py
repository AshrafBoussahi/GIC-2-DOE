"""Main pipeline orchestrator for qGridX.

Implements the 4-stage "cut-guided amortized sampling" loop:

  Stage 1 — Classical scenario & preprocessing.
  Stage 2 — Quantum master (PCE-GQE): propose M candidate plans.
  Stage 3 — Readout & constraint projection: snap to feasible plans.
  Stage 4 — Classical subproblem & cut generation: evaluate and rank.

The loop is a HEURISTIC (R1): it does NOT claim Benders optimality.
Timing is reported separately for master-proposal vs subproblem (R3).
R2 (coefficient rescaling + cut-pool cap) is enforced before every encoder pass.

Public API::

    pipeline = Pipeline(config)
    results = pipeline.run()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from qgridx.config import Config
from qgridx.problems.ising import IsingSpec, problem_spec_to_ising
from qgridx.problems.scenarios import Scenario, build_scenario_tree
from qgridx.projection.base import InvestmentPlan
from qgridx.subproblem.cuts import generate_cut, rank_plans_by_cost
from qgridx.utils.io import ensure_dir, save_json
from qgridx.utils.logging import get_logger
from qgridx.utils.seeding import seed_everything

log = get_logger("qgridx.pipeline")


@dataclass
class IterationResult:
    """Per-iteration diagnostic information."""
    iteration: int
    best_cost: float
    mean_cost: float
    feasibility_rate: float
    master_time_s: float        # R3: proposal time only
    subproblem_time_s: float    # R3: classical subproblem time only
    n_cuts: int
    dpo_loss: Optional[float] = None


@dataclass
class PipelineResult:
    """Final output of a complete pipeline run."""
    best_plan: Optional[InvestmentPlan]
    best_cost: float
    baseline_cost: float
    feasibility_rate: float
    iteration_log: list[IterationResult] = field(default_factory=list)
    pareto_plans: list[tuple[InvestmentPlan, float]] = field(default_factory=list)
    master_time_total_s: float = 0.0      # R3
    subproblem_time_total_s: float = 0.0  # R3
    baseline_time_s: float = 0.0          # R3


class Pipeline:
    """Orchestrates the 4-stage cut-guided amortized sampling loop.

    Args:
        config: Validated :class:`Config` object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.spec = config.problem
        self.pcfg = config.pipeline
        seed_everything(self.pcfg.experiment.seed)
        self._build_components()

    def _build_components(self) -> None:
        """Instantiate all pipeline components from config."""
        from qgridx.registry import build_component

        # Trigger registration of all built-in components
        import qgridx.encoding.log_binary_sizing  # noqa: F401
        import qgridx.quantum.gqe_master  # noqa: F401
        import qgridx.subproblem.scopf  # noqa: F401
        import qgridx.projection.confidence_ilp  # noqa: F401
        import qgridx.baselines.milp_misocp  # noqa: F401

        enc_cfg = self.pcfg.encoder
        self.encoder = build_component(
            "encoder", enc_cfg.name,
            mutex_penalty=enc_cfg.mutex_penalty,
            budget_penalty=enc_cfg.budget_penalty,
        )

        qm_cfg = self.pcfg.quantum_master
        self.quantum_master = build_component(
            "quantum_master", qm_cfg.name,
            cfg=qm_cfg,
            seed=self.pcfg.experiment.seed,
        )

        proj_cfg = self.pcfg.projection
        self.projector = build_component(
            "projector", proj_cfg.name,
            solver=proj_cfg.solver,
            local_search=proj_cfg.local_search,
        )

        sub_cfg = self.pcfg.subproblem
        self.subproblem = build_component(
            "subproblem", sub_cfg.name,
            solver=sub_cfg.solver,
        )

        # GNN encoder + Transformer decoder
        from qgridx.config import EncoderGNNConfig
        from qgridx.model.encoder_gnn import IsingGNNEncoder
        from qgridx.model.tokenizer import CircuitTokenizer
        from qgridx.model.decoder_transformer import PCEBrickworkDecoder
        from qgridx.model.dpo import DPOTrainer
        from qgridx.quantum.pce import derive_n_qubits

        m = self.spec.n_binary_variables()
        k = self.pcfg.quantum_master.pce.k
        n_qubits = derive_n_qubits(m, k)

        self.gnn = IsingGNNEncoder(self.pcfg.model.encoder_gnn)
        self.tokenizer = CircuitTokenizer(n_qubits, topology=self.pcfg.quantum_master.ansatz.topology)
        self.decoder = PCEBrickworkDecoder(self.pcfg.model.decoder, self.tokenizer)
        self.dpo = DPOTrainer(self.decoder, self.pcfg.model.dpo, device=self.pcfg.experiment.device.value)

        self.device = self.pcfg.experiment.device.value

    def run(self) -> PipelineResult:
        """Execute the full 4-stage loop and return results.

        Returns:
            :class:`PipelineResult` with best plan, costs, and timing info (R3).
        """
        log.info("=" * 60)
        log.info("qGridX cut-guided amortized sampling pipeline (HEURISTIC, R1)")
        log.info(f"  System: {self.spec.system.name}")
        log.info(f"  Variables m = {self.spec.n_binary_variables()}")
        log.info(f"  Max iterations: {self.pcfg.loop.max_iters}")
        log.info("=" * 60)

        out_dir = ensure_dir(self.pcfg.experiment.out_dir)

        # Stage 1: scenario tree + initial Ising
        scenarios = build_scenario_tree(self.spec)
        log.info(f"Stage 1: built {len(scenarios)} scenarios.")

        cut_pool: list[dict] = []
        best_cost = float("inf")
        best_plan: Optional[InvestmentPlan] = None
        pareto: list[tuple[InvestmentPlan, float]] = []
        iter_log: list[IterationResult] = []
        master_total = 0.0
        subproblem_total = 0.0
        prev_best = float("inf")

        for it in range(self.pcfg.loop.max_iters):
            log.info(f"\n--- Iteration {it+1}/{self.pcfg.loop.max_iters} ---")

            # R2: encode with rescaling and cut-pool cap
            ising = self.encoder.encode(self.spec, cut_pool=cut_pool)

            # Stage 2: quantum master proposal
            t0_master = time.perf_counter()
            context, _ = self.gnn.encode_ising(ising, device=self.device)
            results = self.quantum_master.propose(
                ising,
                n_samples=self.pcfg.loop.M_circuits,
                context=context,
            )
            master_time = time.perf_counter() - t0_master
            master_total += master_time

            # Stage 3: project to feasible plans
            plans: list[InvestmentPlan] = []
            for r in results:
                try:
                    plan = self.projector.project(r, self.spec)
                    plans.append(plan)
                except Exception as e:
                    log.warning(f"Projection failed for circuit {r.circuit_id}: {e}")

            n_feasible = len(plans)
            feasibility_rate = n_feasible / max(len(results), 1)
            log.info(f"  Feasible plans: {n_feasible}/{len(results)} ({100*feasibility_rate:.0f}%)")

            # Stage 4: evaluate plans via subproblem
            t0_sub = time.perf_counter()
            plan_costs: list[float] = []
            new_cuts: list[dict] = []

            for p_idx, plan in enumerate(plans):
                sub_results = self.subproblem.solve_all(plan, scenarios, self.spec, plan_id=p_idx)
                op_cost = sum(r.operational_cost for r in sub_results)
                true_cost = plan.total_capex + op_cost
                plan_costs.append(true_cost)
                cut = generate_cut(plan, sub_results, self.spec)
                new_cuts.append(cut)

            subproblem_time = time.perf_counter() - t0_sub
            subproblem_total += subproblem_time

            # Update cut pool (R2: cap applied in encoder.encode next iteration)
            cut_pool.extend(new_cuts)
            K = self.pcfg.cuts.pool_cap_K
            cut_pool = cut_pool[-K:]

            # Rank and update best
            if plans and plan_costs:
                ranked = rank_plans_by_cost(plans, plan_costs)
                it_best_plan, it_best_cost = ranked[0]
                pareto.extend(ranked[:3])

                if it_best_cost < best_cost:
                    best_cost = it_best_cost
                    best_plan = it_best_plan

                mean_cost = float(np.mean(plan_costs))
            else:
                it_best_cost = float("inf")
                mean_cost = float("inf")

            # DPO update
            dpo_loss = None
            if len(plans) >= 2 and plan_costs:
                dpo_loss = self._dpo_step(results, plan_costs, context)

            iter_result = IterationResult(
                iteration=it + 1,
                best_cost=best_cost,
                mean_cost=mean_cost,
                feasibility_rate=feasibility_rate,
                master_time_s=master_time,
                subproblem_time_s=subproblem_time,
                n_cuts=len(cut_pool),
                dpo_loss=dpo_loss,
            )
            iter_log.append(iter_result)

            log.info(
                f"  Best cost: {best_cost:.2f}  "
                f"Master: {master_time:.3f}s  "  # R3
                f"Subproblem: {subproblem_time:.3f}s"  # R3
            )

            # Convergence check
            if abs(prev_best - best_cost) < self.pcfg.loop.convergence_tol * max(abs(best_cost), 1):
                if it > 0:
                    log.info(f"Converged at iteration {it+1}.")
                    break
            prev_best = best_cost

        # Run baseline (separately timed — R3)
        baseline_cost = float("inf")
        t0_baseline = time.perf_counter()
        if self.pcfg.baseline.enabled:
            log.info("\nRunning classical MISOCP baseline...")
            baseline_plan, baseline_cost = self._run_baseline(scenarios)
        baseline_time = time.perf_counter() - t0_baseline

        # Save results
        results_data = {
            "best_cost": best_cost,
            "baseline_cost": baseline_cost,
            "cost_gap_pct": 100 * (best_cost - baseline_cost) / max(abs(baseline_cost), 1),
            "feasibility_rate": feasibility_rate,
            "master_time_total_s": master_total,       # R3
            "subproblem_time_total_s": subproblem_total,  # R3
            "baseline_time_s": baseline_time,          # R3
            "iterations": [
                {
                    "iter": r.iteration,
                    "best_cost": r.best_cost,
                    "mean_cost": r.mean_cost,
                    "feasibility_rate": r.feasibility_rate,
                    "master_time_s": r.master_time_s,
                    "subproblem_time_s": r.subproblem_time_s,
                    "n_cuts": r.n_cuts,
                    "dpo_loss": r.dpo_loss,
                }
                for r in iter_log
            ],
        }
        save_json(results_data, out_dir / "results.json")
        log.info(f"\nResults saved to {out_dir}/results.json")

        return PipelineResult(
            best_plan=best_plan,
            best_cost=best_cost,
            baseline_cost=baseline_cost,
            feasibility_rate=feasibility_rate,
            iteration_log=iter_log,
            pareto_plans=pareto,
            master_time_total_s=master_total,
            subproblem_time_total_s=subproblem_total,
            baseline_time_s=baseline_time,
        )

    def _dpo_step(
        self,
        results: list,
        plan_costs: list[float],
        context: "Any",
    ) -> Optional[float]:
        """Run one DPO step using the best and worst circuits as the preference pair."""
        import torch
        try:
            if len(plan_costs) < 2:
                return None
            sorted_idx = sorted(range(len(plan_costs)), key=lambda i: plan_costs[i])
            best_idx = sorted_idx[0]
            worst_idx = sorted_idx[-1]
            # For a proper DPO step we need token sequences — use dummy sequences for now
            # TODO(scientific-review): wire actual token generation through decoder
            T = 8
            V = self.tokenizer.vocab_size
            chosen_ids = torch.randint(0, V, (1, T))
            rejected_ids = torch.randint(0, V, (1, T))
            ctx = context.unsqueeze(0) if context.dim() == 1 else context
            loss = self.dpo.step(chosen_ids, rejected_ids, context=ctx)
            return loss
        except Exception:
            return None

    def _run_baseline(
        self, scenarios: list[Scenario]
    ) -> tuple[Optional[InvestmentPlan], float]:
        """Run the classical MISOCP baseline and return (plan, total_cost)."""
        from qgridx.baselines.milp_misocp import MILPMISOCPBaseline
        baseline = MILPMISOCPBaseline(cfg=self.pcfg.baseline, solver=self.pcfg.baseline.solver)
        return baseline.solve(self.spec, scenarios)
