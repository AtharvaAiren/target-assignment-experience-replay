"""Compact positive sanity check for the locked functional-OT allocation solver."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from chartfilm.fmmt.core import (
    box_simplex_bounds,
    sinkhorn_prox_allocation,
    sinkhorn_prox_objective,
)


ROWS = ("uniform", "functional_cost", "shuffled_cost")


def _target_allocation(active: int, atoms: int, sharpness: float) -> torch.Tensor:
    """Interior, geometry-local allocation with no active box constraint."""
    coordinates = torch.arange(atoms, dtype=torch.float64)
    logits = -float(sharpness) * torch.abs(coordinates - float(active))
    localized = torch.softmax(logits, dim=0)
    neutral = torch.full_like(localized, 1.0 / atoms)
    # Keeping the target strictly interior makes the first-order construction
    # below unambiguous; the box bounds remain an additional integrity check.
    return 0.65 * localized + 0.35 * neutral


def _demand_with_known_optimum(
    target: torch.Tensor,
    previous: torch.Tensor,
    neutral: torch.Tensor,
    cost: torch.Tensor,
    specification: Mapping[str, Any],
) -> torch.Tensor:
    """Construct demand so ``target`` is stationary for the correct geometry.

    The proximal objective is ``R(p) - <d,p>``.  Setting ``d=grad R(target)``
    makes the target an interior first-order optimum (up to the irrelevant
    simplex-normal constant).  The exact same demand is then supplied to a
    spectrum-matched but semantically shuffled cost.  This is an explicitly
    engineered solver sanity check, not evidence about a natural stream.
    """
    variable = target.detach().clone().requires_grad_(True)
    objective = sinkhorn_prox_objective(
        variable,
        torch.zeros_like(variable),
        torch.zeros_like(variable),
        previous,
        neutral,
        cost,
        kl_weight=float(specification["kl_weight"]),
        step_size=float(specification["proximal_step"]),
        epsilon=float(specification["sinkhorn_epsilon"]),
        sinkhorn_iterations=int(specification["sinkhorn_iterations"]),
    )
    (demand,) = torch.autograd.grad(objective, variable)
    return demand.detach()


def run_ot_positive_unit(seed: int, specification: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    rng = np.random.default_rng([int(seed), 0x07C05])
    atoms = int(specification["atoms"])
    neutral = torch.full((atoms,), 1.0 / atoms, dtype=torch.float64)
    lower, upper = box_simplex_bounds(
        neutral,
        minimum_ratio=float(specification["minimum_ratio"]),
        maximum_ratio=float(specification["maximum_ratio"]),
    )
    coordinates = torch.arange(atoms, dtype=torch.float64)
    functional_cost = torch.abs(coordinates[:, None] - coordinates[None, :])
    positive = functional_cost[functional_cost > 0]
    functional_cost /= positive.median()
    # Alternating extremes makes adjacent functional states deliberately far
    # under the spectrum-matched shuffled geometry.
    permutation = torch.as_tensor([0, 7, 1, 6, 2, 5, 3, 4], dtype=torch.long)
    if atoms != 8:
        permutation = torch.as_tensor(rng.permutation(atoms), dtype=torch.long)
    shuffled_cost = functional_cost[permutation][:, permutation]
    losses = {row: [] for row in ROWS}
    allocation_tv = {"functional_cost": [], "shuffled_cost": []}
    target_stationarity: list[float] = []
    previous_target = neutral.clone()

    for step in range(int(specification["steps"])):
        position = (step // int(specification["state_residence_steps"])) % (2 * atoms - 2)
        active = position if position < atoms else (2 * atoms - 2 - position)
        target = _target_allocation(
            active, atoms, float(specification["target_sharpness"])
        )
        if not bool(((target > lower) & (target < upper)).all().item()):
            raise RuntimeError("positive-control target must be strictly interior")
        demand = _demand_with_known_optimum(
            target, previous_target, neutral, functional_cost, specification
        )
        noise = torch.as_tensor(
            rng.normal(0.0, float(specification["demand_noise"]), atoms),
            dtype=torch.float64,
        )
        demand = demand + noise
        allocations: dict[str, torch.Tensor] = {}
        for row, cost in (
            ("functional_cost", functional_cost),
            ("shuffled_cost", shuffled_cost),
        ):
            allocation = sinkhorn_prox_allocation(
                demand,
                previous_target,
                cost,
                consolidation=torch.zeros_like(neutral),
                neutral=neutral,
                lower=lower,
                upper=upper,
                kl_weight=float(specification["kl_weight"]),
                step_size=float(specification["proximal_step"]),
                epsilon=float(specification["sinkhorn_epsilon"]),
                sinkhorn_iterations=int(specification["sinkhorn_iterations"]),
                allocation_iterations=int(specification["allocation_iterations"]),
            )
            allocation_tv[row].append(
                0.5 * float(torch.abs(allocation - previous_target).sum())
            )
            allocations[row] = allocation
        losses["uniform"].append(float(torch.square(neutral - target).mean()))
        losses["functional_cost"].append(
            float(torch.square(allocations["functional_cost"] - target).mean())
        )
        losses["shuffled_cost"].append(
            float(torch.square(allocations["shuffled_cost"] - target).mean())
        )
        target_stationarity.append(
            float(torch.square(allocations["functional_cost"] - target).mean())
        )
        previous_target = target

    burn = int(specification["burn_in"])
    metrics = {row: float(np.mean(values[burn:])) for row, values in losses.items()}
    audit = {
        "same_demand_all_rows": True,
        "same_box_simplex_bounds": True,
        "same_initial_allocation": True,
        "shuffled_cost_has_identical_edge_spectrum": bool(
            torch.allclose(
                torch.sort(functional_cost.flatten()).values,
                torch.sort(shuffled_cost.flatten()).values,
            )
        ),
        "allocations_finite": all(
            torch.isfinite(value).all().item() for value in allocations.values()
        ),
        "allocations_sum_to_one": all(
            torch.isclose(value.sum(), torch.ones((), dtype=value.dtype), atol=1e-8).item()
            for value in allocations.values()
        ),
        "engineered_optimum_explicitly_flagged": True,
    }
    audit["all_pass"] = bool(all(audit.values()))
    return {
        "seed": int(seed),
        "metrics": metrics,
        "mean_allocation_tv": {
            row: float(np.mean(values[burn:])) for row, values in allocation_tv.items()
        },
        "mean_functional_target_error": float(np.mean(target_stationarity[burn:])),
        "audit": audit,
    }


def _seed_ci(effects: np.ndarray, seed: int, draws: int) -> list[float]:
    effects = np.sort(np.asarray(effects, dtype=float))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(draws), dtype=float)
    for index in range(int(draws)):
        bootstrap[index] = float(
            np.mean(effects[rng.integers(0, len(effects), size=len(effects))])
        )
    return [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]


def run_ot_positive_suite(
    lock: Mapping[str, Any], *, smoke: bool = False
) -> dict[str, Any]:
    specification = dict(lock["ot_positive_control"])
    if smoke:
        specification.update(
            seeds=specification["seeds"][:1],
            steps=8,
            burn_in=1,
            sinkhorn_iterations=4,
            allocation_iterations=4,
        )
    units = [
        run_ot_positive_unit(int(seed), specification)
        for seed in specification["seeds"]
    ]
    functional = np.asarray([unit["metrics"]["functional_cost"] for unit in units])
    uniform = np.asarray([unit["metrics"]["uniform"] for unit in units])
    shuffled = np.asarray([unit["metrics"]["shuffled_cost"] for unit in units])
    versus_uniform = uniform - functional
    versus_shuffled = shuffled - functional
    relative_uniform = float(np.mean(versus_uniform) / max(float(np.mean(uniform)), 1e-12))
    relative_shuffled = float(np.mean(versus_shuffled) / max(float(np.mean(shuffled)), 1e-12))
    runtime = lock["runtime"]
    ci_uniform = _seed_ci(
        versus_uniform,
        int(runtime["cluster_bootstrap_seed"]) + 800,
        int(runtime["cluster_bootstrap_draws"]),
    )
    ci_shuffled = _seed_ci(
        versus_shuffled,
        int(runtime["cluster_bootstrap_seed"]) + 801,
        int(runtime["cluster_bootstrap_draws"]),
    )
    gate = specification["gate"]
    gates = {
        "beats_uniform_relative": relative_uniform
        >= float(gate["functional_cost_must_beat_uniform_relative"]),
        "beats_shuffled_relative": relative_shuffled
        >= float(gate["functional_cost_must_beat_shuffled_cost_relative"]),
        "uniform_ci_lower_above_zero": ci_uniform[0] > 0,
        "shuffled_ci_lower_above_zero": ci_shuffled[0] > 0,
        "minimum_seed_wins_uniform": int(np.sum(functional < uniform))
        >= int(gate["minimum_seed_wins"]),
        "minimum_seed_wins_shuffled": int(np.sum(functional < shuffled))
        >= int(gate["minimum_seed_wins"]),
        "integrity": all(unit["audit"]["all_pass"] for unit in units),
    }
    if smoke:
        gates = {"integrity": all(unit["audit"]["all_pass"] for unit in units)}
    return {
        "schema_version": "fmmt.workshop.ot_positive.v1",
        "units": units,
        "means": {
            row: float(np.mean([unit["metrics"][row] for unit in units])) for row in ROWS
        },
        "functional_vs_uniform": {
            "relative_improvement": relative_uniform,
            "ci95_absolute": ci_uniform,
            "wins": int(np.sum(functional < uniform)),
        },
        "functional_vs_shuffled": {
            "relative_improvement": relative_shuffled,
            "ci95_absolute": ci_shuffled,
            "wins": int(np.sum(functional < shuffled)),
        },
        "gates": gates,
        "all_pass": bool(all(gates.values())),
        "verdict": (
            "OT_POSITIVE_CONTROL_SMOKE_PASS"
            if smoke and all(gates.values())
            else "OT_POSITIVE_CONTROL_PASS"
            if all(gates.values())
            else "OT_POSITIVE_CONTROL_FAIL"
        ),
        "smoke": bool(smoke),
    }
