"""Fixed-trajectory multi-permutation sensitivity for the frozen K4 control.

This module is intentionally outside the source-frozen ``fmmt_workshop``
namespace.  It replays the unchanged primary learner once, evaluates several
independent permuted-target directions as discarded branches, and performs
exactly one true center update per packet.  It never changes an existing unit,
summary, lock, or verdict.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from chartfilm.fmmt.core import FunctionalUnitRegressor
from experiments.fmmt import study_a as A
from experiments.fmmt.common import metric_by_month, packetize_stream, seed_everything
from experiments.fmmt_workshop.specificity import LinearRegressor, _model, _optimizer


K0 = "K0_ZERO"
K1 = "K1_PREVIOUS_GRADIENT"
K4_LEGACY = "K4_LEGACY_SINGLE_PERMUTATION"


def _permutation(
    size: int, *, initialization_seed: int, packet_number: int,
    replica: int, salt: int,
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(initialization_seed), int(packet_number), int(replica), int(salt)]
    )
    return np.random.default_rng(sequence).permutation(int(size)).astype(np.int64)


def _batched_mse_gradients(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """Analytic MSE gradients for many target vectors at one frozen center.

    ``targets`` is shaped ``[replicas,batch]``.  The formula matches
    ``torch.nn.functional.mse_loss(..., reduction='mean')`` including its
    factor of two.  Computing replicas together avoids repeating autograd and
    leaves the live model and global RNG untouched.
    """
    if inputs.ndim != 2 or targets.ndim != 2 or targets.shape[1] != inputs.shape[0]:
        raise ValueError("expected inputs [batch,features] and targets [replicas,batch]")
    replicas, batch = int(targets.shape[0]), int(targets.shape[1])
    if replicas < 1 or batch < 1:
        raise ValueError("gradient batch and replica dimensions must be nonempty")
    factor = 2.0 / float(batch)
    with torch.no_grad():
        if isinstance(model, FunctionalUnitRegressor):
            base_weight = parameters["base_weight"]
            base_bias = parameters["base_bias"]
            atom_input_weight = parameters["atom_input_weight"]
            atom_input_bias = parameters["atom_input_bias"]
            atom_output_weight = parameters["atom_output_weight"]
            preactivation = torch.nn.functional.linear(
                inputs, atom_input_weight, atom_input_bias
            )
            hidden = torch.relu(preactivation)
            scale = float(model.contribution_scale)
            prediction = (
                torch.nn.functional.linear(
                    inputs, base_weight.unsqueeze(0), base_bias.reshape(1)
                ).squeeze(-1)
                + (hidden * atom_output_weight.unsqueeze(0) * scale).sum(dim=-1)
            )
            residual = prediction.unsqueeze(0) - targets
            gate = (preactivation > 0).to(dtype=inputs.dtype)
            atom_coefficient = (
                residual.unsqueeze(-1)
                * gate.unsqueeze(0)
                * atom_output_weight.reshape(1, 1, -1)
                * scale
            )
            stacked = {
                "base_weight": factor * torch.einsum("pn,nd->pd", residual, inputs),
                "base_bias": factor * residual.sum(dim=1),
                "atom_input_weight": factor * torch.einsum(
                    "pna,nd->pad", atom_coefficient, inputs
                ),
                "atom_input_bias": factor * atom_coefficient.sum(dim=1),
                "atom_output_weight": factor * scale * torch.einsum(
                    "pn,na->pa", residual, hidden
                ),
            }
        elif isinstance(model, LinearRegressor):
            weight = parameters["linear.weight"]
            bias = parameters["linear.bias"]
            prediction = torch.nn.functional.linear(inputs, weight, bias).squeeze(-1)
            residual = prediction.unsqueeze(0) - targets
            stacked = {
                "linear.weight": (
                    factor * torch.einsum("pn,nd->pd", residual, inputs)
                ).unsqueeze(1),
                "linear.bias": (factor * residual.sum(dim=1)).unsqueeze(1),
            }
        else:
            raise TypeError(f"unsupported post-hoc model: {type(model).__name__}")
    if tuple(stacked) != tuple(parameters):
        raise RuntimeError("analytic gradient keys differ from the frozen model")
    return [
        {name: values[index].detach().clone() for name, values in stacked.items()}
        for index in range(replicas)
    ]


def _stacked_branch_predictions(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    directions: Sequence[Mapping[str, torch.Tensor] | None],
    radius: float,
    inputs: torch.Tensor,
) -> np.ndarray:
    """Vectorized predictions for equal-radius discarded branches."""
    replicas = len(directions)
    if replicas < 1:
        raise ValueError("at least one direction is required")
    with torch.no_grad():
        norms = torch.as_tensor(
            [A._dict_norm(direction) for direction in directions],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        scales = torch.zeros_like(norms)
        valid = norms > 1.0e-20
        if float(radius) > 0:
            scales[valid] = -float(radius) / norms[valid]
        stacked: dict[str, torch.Tensor] = {}
        for name, parameter in parameters.items():
            values = []
            for direction in directions:
                values.append(
                    torch.zeros_like(parameter)
                    if direction is None else direction[name].to(parameter)
                )
            delta = torch.stack(values, dim=0)
            shape = (replicas,) + (1,) * parameter.ndim
            stacked[name] = parameter.unsqueeze(0) + scales.reshape(shape) * delta

        if isinstance(model, FunctionalUnitRegressor):
            base = torch.einsum("nd,pd->pn", inputs, stacked["base_weight"])
            base = base + stacked["base_bias"].reshape(replicas, 1)
            preactivation = torch.einsum(
                "nd,pad->pna", inputs, stacked["atom_input_weight"]
            ) + stacked["atom_input_bias"].reshape(replicas, 1, -1)
            output = base + (
                torch.relu(preactivation)
                * stacked["atom_output_weight"].reshape(replicas, 1, -1)
                * float(model.contribution_scale)
            ).sum(dim=-1)
        elif isinstance(model, LinearRegressor):
            output = torch.einsum(
                "nd,pd->pn", inputs, stacked["linear.weight"].squeeze(1)
            ) + stacked["linear.bias"].reshape(replicas, 1)
        else:
            raise TypeError(f"unsupported post-hoc model: {type(model).__name__}")
    return output.detach().cpu().numpy().astype(np.float32)


def _tensor_state_sha256(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is not None:
            digest.update(repr(value).encode())

    visit({name: value for name, value in model.state_dict().items()})
    visit(optimizer.state_dict())
    return digest.hexdigest()


def run_multipermutation_unit(
    data: dict[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    n_permutations: int,
    permutation_salt: int,
    model_type: str = "functional_units",
    optimizer_type: str = "adamw",
    packet_size: int = 32,
) -> dict[str, Any]:
    """Replay one immutable live trajectory with several discarded K4 rows."""
    if int(n_permutations) < 2:
        raise ValueError("multi-permutation sensitivity requires at least two replicas")
    torch.set_num_threads(int(config.get("torch_threads_per_worker", 1)))
    seed_everything(int(seed))
    packets = packetize_stream(data, int(packet_size))
    model = _model(data["X"].shape[1], model_type, int(config["functional_units"]))
    model.to(torch.device("cpu")).train()
    optimizer = _optimizer(model, optimizer_type, config)

    n = len(data["X"])
    core_predictions = {
        K0: np.full(n, np.nan, dtype=np.float32),
        K1: np.full(n, np.nan, dtype=np.float32),
        K4_LEGACY: np.full(n, np.nan, dtype=np.float32),
    }
    permutation_predictions = np.full(
        (int(n_permutations), n), np.nan, dtype=np.float32
    )
    previous_gradient: dict[str, torch.Tensor] | None = None
    previous_legacy: dict[str, torch.Tensor] | None = None
    previous_permutations: list[dict[str, torch.Tensor] | None] = [
        None for _ in range(int(n_permutations))
    ]
    radii: deque[float] = deque(maxlen=int(config["branch_radius_history"]))
    legacy_rng = np.random.default_rng([int(seed), 0x5A17])
    isolation_checks = 0

    for packet_number, packet in enumerate(packets):
        index = np.asarray(packet["index"], dtype=np.int64)
        inputs = torch.as_tensor(packet["x"], dtype=torch.float32)
        parameters = A._named_parameters(model)
        radius = float(np.median(radii)) if radii else 0.0
        snapshot = A._live_snapshot(model, optimizer)

        core_predictions[K0][index] = A._predict(model, parameters, inputs)
        core_predictions[K1][index] = A._predict(
            model, A._shift(parameters, A._delta(previous_gradient, radius)), inputs
        )
        core_predictions[K4_LEGACY][index] = A._predict(
            model, A._shift(parameters, A._delta(previous_legacy, radius)), inputs
        )
        permutation_predictions[:, index] = _stacked_branch_predictions(
            model, parameters, previous_permutations, radius, inputs
        )

        # First materialization of packet-t labels.  Everything above is causal.
        targets = torch.as_tensor(packet["y"], dtype=torch.float32)
        full_gradient, _ = A._gradient_dict(
            model, parameters, inputs, targets
        )

        # Reproduce the exact legacy RNG consumption: the frozen runner drew
        # two fold permutations for K8 before drawing the full K4 permutation.
        fold_a, fold_b = A._folds(index, np.asarray(packet["event_id"], dtype=np.int64))
        legacy_rng.permutation(len(fold_a))
        legacy_rng.permutation(len(fold_b))
        legacy_index = torch.as_tensor(
            legacy_rng.permutation(len(index)), dtype=torch.long
        )
        legacy_gradient, _ = A._gradient_dict(
            model, parameters, inputs, targets[legacy_index]
        )

        indices = np.stack(
            [
                _permutation(
                    len(index), initialization_seed=int(seed),
                    packet_number=packet_number, replica=replica,
                    salt=int(permutation_salt),
                )
                for replica in range(int(n_permutations))
            ],
            axis=0,
        )
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        permuted_targets = targets[index_tensor]
        permutation_gradients = _batched_mse_gradients(
            model, parameters, inputs, permuted_targets
        )

        if not A._live_unchanged(snapshot, model, optimizer):
            raise RuntimeError("post-hoc branch mutated the fixed live trajectory")
        isolation_checks += 1

        optimizer.zero_grad(set_to_none=True)
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        for name, value in model.named_parameters():
            value.grad = full_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        radii.append(A._difference_norm(before, after))
        previous_gradient = {
            name: value.detach().clone() for name, value in full_gradient.items()
        }
        previous_legacy = {
            name: value.detach().clone() for name, value in legacy_gradient.items()
        }
        previous_permutations = permutation_gradients

    scored = np.asarray(data["scored_mask"], dtype=bool)
    if not all(np.isfinite(value[scored]).all() for value in core_predictions.values()):
        raise RuntimeError("non-finite core replay prediction")
    if not np.isfinite(permutation_predictions[:, scored]).all():
        raise RuntimeError("non-finite multi-permutation prediction")
    target = np.asarray(data["y"], dtype=np.float32)
    month = np.asarray(data["month_id"])
    core_metrics = {
        row: metric_by_month(prediction, target, month, scored)
        for row, prediction in core_predictions.items()
    }
    permutation_metrics = [
        metric_by_month(prediction, target, month, scored)
        for prediction in permutation_predictions
    ]
    packet_signature = hashlib.sha256(
        np.asarray(
            [
                [len(packet["index"]), packet["n_acquisition_events"]]
                for packet in packets
            ],
            dtype=np.int64,
        ).tobytes()
    ).hexdigest()
    audit = {
        "posthoc_labeled": True,
        "chronology_preserved": bool(data.get("chronology_preserved")),
        "all_rows_retained": bool(data.get("all_preprocessed_rows_retained")),
        "all_branches_before_current_labels": True,
        "branches_state_isolated": isolation_checks == len(packets),
        "one_true_center_update_per_packet": True,
        "equal_branch_radius": True,
        "counter_based_permutation_streams": True,
        "no_raw_replay": True,
        "cpu_only": True,
    }
    audit["all_pass"] = bool(all(audit.values()))
    return {
        "schema_version": "fmmt.posthoc.k4_multipermutation.unit.v1",
        "posthoc": True,
        "dataset": data["name"],
        "seed": int(seed),
        "model": model_type,
        "optimizer": optimizer_type,
        "packet_size": int(packet_size),
        "n_permutations": int(n_permutations),
        "n_examples": int(n),
        "n_packets": int(len(packets)),
        "n_scored": int(scored.sum()),
        "packet_signature_sha256": packet_signature,
        "final_live_state_sha256": _tensor_state_sha256(model, optimizer),
        "core_metrics": core_metrics,
        "permutation_metrics": permutation_metrics,
        "data_manifest": data.get("data_manifest"),
        "chronology_audit": data.get("chronology_audit"),
        "audit": audit,
        "resources": {
            "center_updates": int(len(packets)),
            "permuted_gradient_replicas_per_packet": int(n_permutations),
            "legacy_gradient_replicas_per_packet": 1,
            "raw_examples_persistently_stored": 0,
            "live_trajectory_count": 1,
        },
    }


def _common_clusters(units: Sequence[Mapping[str, Any]]) -> np.ndarray:
    shared: set[int] | None = None
    for unit in units:
        rows = [*unit["core_metrics"].values(), *unit["permutation_metrics"]]
        for metric in rows:
            observed = {int(value) for value in metric["per_month"]}
            shared = observed if shared is None else shared & observed
    if not shared or len(shared) < 2:
        raise RuntimeError("fewer than two common temporal clusters")
    return np.asarray(sorted(shared), dtype=np.int64)


def _metric_vector(metric: Mapping[str, Any], clusters: np.ndarray) -> np.ndarray:
    return np.asarray(
        [metric["per_month"][str(int(cluster))]["rmse"] for cluster in clusters],
        dtype=np.float64,
    )


def _circular_columns(
    rng: np.random.Generator, n_columns: int, block_length: int
) -> np.ndarray:
    length = min(max(int(block_length), 1), int(n_columns))
    blocks = int(np.ceil(n_columns / length))
    starts = rng.integers(0, n_columns, size=blocks)
    sampled = np.concatenate(
        [(start + np.arange(length, dtype=np.int64)) % n_columns for start in starts]
    )
    return sampled[:n_columns]


def _nested_bootstrap(
    effects: np.ndarray,
    *,
    baseline: np.ndarray,
    seed: int,
    draws: int,
    block_length: int,
) -> dict[str, Any]:
    """Bootstrap seed, temporal block, and permutation-within-seed axes."""
    effects = np.asarray(effects, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if effects.ndim != 3:
        raise ValueError("effects must have shape [seed,permutation,cluster]")
    if baseline.shape != effects.shape:
        raise ValueError("baseline and effects must have identical shapes")
    if not np.isfinite(effects).all() or not np.isfinite(baseline).all():
        raise ValueError("nested bootstrap contains non-finite values")
    n_seed, n_permutation, n_cluster = effects.shape
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    relative = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        seed_index = rng.integers(0, n_seed, size=n_seed)
        cluster_index = _circular_columns(rng, n_cluster, int(block_length))
        draw_effects: list[np.ndarray] = []
        draw_baselines: list[np.ndarray] = []
        for selected_seed in seed_index:
            permutation_index = rng.integers(
                0, n_permutation, size=n_permutation
            )
            draw_effects.append(
                effects[selected_seed][np.ix_(permutation_index, cluster_index)]
            )
            draw_baselines.append(
                baseline[selected_seed][np.ix_(permutation_index, cluster_index)]
            )
        effect = float(np.mean(draw_effects))
        denominator = max(float(np.mean(draw_baselines)), 1.0e-12)
        samples[draw] = effect
        relative[draw] = effect / denominator
    return {
        "improvement": float(effects.mean()),
        "relative_improvement": float(effects.mean() / max(baseline.mean(), 1.0e-12)),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "ci95_relative": [
            float(value) for value in np.quantile(relative, [0.025, 0.975])
        ],
        "bootstrap": {
            "method": "nested_seed_circular_time_permutation",
            "draws": int(draws),
            "temporal_block_length": min(int(block_length), int(n_cluster)),
            "initialization_seeds_resampled": True,
            "permutation_replicas_resampled_within_seed": True,
            "ordered_temporal_blocks": True,
        },
    }


def summarize_multipermutation(
    units: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate fixed-trajectory K4 randomization sensitivity."""
    if not units:
        raise ValueError("no multi-permutation units")
    units = sorted(units, key=lambda value: int(value["seed"]))
    identity = {
        (
            unit["dataset"], unit["model"], unit["optimizer"],
            int(unit["packet_size"]), int(unit["n_permutations"]),
        )
        for unit in units
    }
    if len(identity) != 1:
        raise RuntimeError(f"mixed multi-permutation identities: {sorted(identity)}")
    n_permutations = int(units[0]["n_permutations"])
    clusters = _common_clusters(units)
    k0 = np.stack(
        [_metric_vector(unit["core_metrics"][K0], clusters) for unit in units]
    )
    k1 = np.stack(
        [_metric_vector(unit["core_metrics"][K1], clusters) for unit in units]
    )
    legacy = np.stack(
        [_metric_vector(unit["core_metrics"][K4_LEGACY], clusters) for unit in units]
    )
    permuted = np.stack(
        [
            np.stack(
                [_metric_vector(metric, clusters) for metric in unit["permutation_metrics"]]
            )
            for unit in units
        ]
    )
    if permuted.shape != (len(units), n_permutations, len(clusters)):
        raise RuntimeError("unexpected multi-permutation metric tensor shape")

    inference = lock["inference"]
    k1_expanded = np.broadcast_to(k1[:, None, :], permuted.shape)
    k0_expanded = np.broadcast_to(k0[:, None, :], permuted.shape)
    pair_effects = permuted - k1_expanded
    nested = _nested_bootstrap(
        pair_effects,
        baseline=permuted,
        seed=int(inference["bootstrap_seed"]),
        draws=int(inference["bootstrap_draws"]),
        block_length=int(inference["temporal_block_length"]),
    )
    gross = float((k0 - k1).mean())
    decoupled = float((k0_expanded - permuted).mean())
    pairing = float(pair_effects.mean())
    if not np.isclose(gross, decoupled + pairing, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("common-baseline prospective-value identity failed")

    per_replica: list[dict[str, Any]] = []
    for replica in range(n_permutations):
        baseline_mean = float(permuted[:, replica, :].mean())
        effect = float(pair_effects[:, replica, :].mean())
        dec = float((k0 - permuted[:, replica, :]).mean())
        per_replica.append(
            {
                "replica": replica,
                "pairing_increment": effect,
                "pairing_increment_relative_to_K4": effect / max(baseline_mean, 1.0e-12),
                "decoupled_value": dec,
                "retention_fraction": dec / max(gross, 1.0e-12),
            }
        )
    relative_values = np.asarray(
        [value["pairing_increment_relative_to_K4"] for value in per_replica]
    )
    retention_values = np.asarray(
        [value["retention_fraction"] for value in per_replica]
    )
    rule = lock["descriptive_sensitivity_rule"]
    threshold = float(rule["pairing_increment_relative_to_K4_below"])
    below = int(np.sum(relative_values < threshold))
    sensitivity_pass = bool(
        nested["relative_improvement"] < threshold
        and nested["ci95_relative"][1] < float(rule["nested_ci_upper_below"])
        and below >= int(rule["minimum_aggregate_permutation_replicas_below"])
        and all(unit["audit"]["all_pass"] for unit in units)
    )
    dataset, model, optimizer, packet_size, _ = next(iter(identity))
    return {
        "schema_version": "fmmt.posthoc.k4_multipermutation.summary.v1",
        "posthoc": True,
        "identity": {
            "dataset": dataset,
            "model": model,
            "optimizer": optimizer,
            "packet_size": packet_size,
        },
        "n_initialization_seeds": len(units),
        "n_permutations_per_seed": n_permutations,
        "n_temporal_clusters": int(len(clusters)),
        "macro_temporal_nrmse": {
            K0: float(k0.mean()),
            K1: float(k1.mean()),
            K4_LEGACY: float(legacy.mean()),
            "K4_MULTIPERMUTATION_MEAN": float(permuted.mean()),
        },
        "common_K0_value_decomposition": {
            "gross_K1_value": gross,
            "decoupled_K4_value": decoupled,
            "pairing_increment": pairing,
            "identity_residual": gross - decoupled - pairing,
            "retention_fraction": decoupled / max(gross, 1.0e-12),
            "pairing_share_of_gross": pairing / max(gross, 1.0e-12),
        },
        "pairing_increment_K1_vs_expected_K4": nested,
        "legacy_single_permutation": {
            "pairing_increment": float((legacy - k1).mean()),
            "pairing_increment_relative_to_K4": float(
                (legacy - k1).mean() / max(legacy.mean(), 1.0e-12)
            ),
            "difference_from_multipermutation_mean": float(
                legacy.mean() - permuted.mean()
            ),
        },
        "permutation_sensitivity": {
            "per_replica": per_replica,
            "pairing_relative_min_median_max": [
                float(relative_values.min()),
                float(np.median(relative_values)),
                float(relative_values.max()),
            ],
            "retention_min_median_max": [
                float(retention_values.min()),
                float(np.median(retention_values)),
                float(retention_values.max()),
            ],
            "replicas_below_one_percent": below,
            "replicas_total": n_permutations,
        },
        "descriptive_rule_pass": sensitivity_pass,
        "verdict": (
            "POSTHOC_SINGLE_PERMUTATION_CONCLUSION_ROBUST"
            if sensitivity_pass
            else "POSTHOC_SINGLE_PERMUTATION_SENSITIVE"
        ),
        "claim_boundary": (
            "Decision-neutral post-hoc sensitivity; the frozen workshop verdict is unchanged."
        ),
    }
