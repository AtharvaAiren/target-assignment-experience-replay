"""Target-specific prospective-gradient decomposition on a fixed trajectory."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from experiments.fmmt.common import metric_by_month, packetize_stream, paired_summary, seed_everything
from experiments.fmmt import study_a as A
from chartfilm.fmmt.core import FunctionalUnitRegressor


ROWS = (
    "K0_ZERO",
    "K1_PREVIOUS_GRADIENT",
    "K2_RADIAL_SHRINKAGE",
    "K3_PREVIOUS_OUTPUT_BIAS",
    "K4_PREVIOUS_SHUFFLED_LABEL_GRADIENT",
    "K5_PREVIOUS_GRADIENT_RESIDUAL",
    "K6_MATCHED_RANDOM_DIRECTION",
    "K7_ORACLE_XFIT",
    "K8_CURRENT_SHUFFLED_TARGET_XFIT",
    "K9_ANTI_ORACLE_XFIT",
)

CAUSAL_ROWS = set(ROWS[:7])
NONCAUSAL_ROWS = set(ROWS[7:])


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs).squeeze(-1)


def _model(input_dim: int, model_type: str, n_atoms: int) -> nn.Module:
    if model_type == "linear":
        return LinearRegressor(input_dim)
    if model_type == "functional_units":
        return FunctionalUnitRegressor(input_dim=input_dim, n_atoms=int(n_atoms))
    raise ValueError(f"unknown model type: {model_type!r}")


def _optimizer(model: nn.Module, optimizer_type: str, config: Mapping[str, Any]):
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=tuple(float(value) for value in config["adamw_betas"]),
            weight_decay=weight_decay,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config["sgd_learning_rate"]),
            momentum=float(config["sgd_momentum"]),
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer type: {optimizer_type!r}")


def _zero_like(parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in parameters.items()}


def _output_bias_component(
    direction: Mapping[str, torch.Tensor] | None,
    parameters: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    if direction is None:
        return None
    output = _zero_like(parameters)
    selected = [
        name for name in direction
        if name == "base_bias" or name == "linear.bias"
    ]
    for name in selected:
        output[name] = direction[name].detach().clone()
    return output if selected else None


def _radial_direction(parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in parameters.items()}


def _residual_direction(
    direction: Mapping[str, torch.Tensor] | None,
    parameters: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Remove output-bias coordinates and the current radial component."""
    if direction is None:
        return None
    residual = {name: value.detach().clone() for name, value in direction.items()}
    for name in residual:
        if name == "base_bias" or name == "linear.bias":
            residual[name].zero_()
    radial = _radial_direction(parameters)
    for name in radial:
        if name == "base_bias" or name == "linear.bias":
            radial[name].zero_()
    denominator = sum(
        float(value.detach().double().square().sum().item()) for value in radial.values()
    )
    if denominator > 1e-20:
        numerator = sum(
            float((residual[name].double() * radial[name].double()).sum().item())
            for name in residual
        )
        coefficient = numerator / denominator
        residual = {
            name: residual[name] - coefficient * radial[name] for name in residual
        }
    return residual


def _random_direction(
    parameters: Mapping[str, torch.Tensor], *, seed: int, packet_number: int
) -> dict[str, torch.Tensor]:
    sequence = np.random.SeedSequence([int(seed), int(packet_number), 0xD1EC710])
    rng = np.random.default_rng(sequence)
    return {
        name: torch.as_tensor(
            rng.standard_normal(tuple(value.shape)),
            dtype=value.dtype,
            device=value.device,
        )
        for name, value in parameters.items()
    }


def _cosine(
    left: Mapping[str, torch.Tensor] | None,
    right: Mapping[str, torch.Tensor] | None,
) -> float | None:
    left_norm, right_norm = A._dict_norm(left), A._dict_norm(right)
    if left is None or right is None or left_norm <= 1e-20 or right_norm <= 1e-20:
        return None
    dot = sum(
        float((left[name].double() * right[name].double()).sum().item())
        for name in left
    )
    return dot / (left_norm * right_norm)


def run_specificity_unit(
    data: dict[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    model_type: str,
    optimizer_type: str,
    packet_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run all virtual rows without allowing any branch to alter the center."""
    if str(device) != "cpu":
        raise ValueError("the workshop program is locked to CPU")
    torch.set_num_threads(int(config.get("torch_threads_per_worker", 1)))
    seed_everything(seed)
    torch_device = torch.device("cpu")
    packets = packetize_stream(data, int(packet_size))
    model = _model(data["X"].shape[1], model_type, int(config["functional_units"]))
    model.to(torch_device).train()
    optimizer = _optimizer(model, optimizer_type, config)

    n = len(data["X"])
    predictions = {row: np.full(n, np.nan, dtype=np.float32) for row in ROWS}
    previous_gradient: dict[str, torch.Tensor] | None = None
    previous_shuffled: dict[str, torch.Tensor] | None = None
    radii: deque[float] = deque(maxlen=int(config["branch_radius_history"]))
    shuffle_rng = np.random.default_rng([int(seed), 0x5A17])
    cosines: dict[str, list[float]] = {
        "previous_to_current": [],
        "radial_to_current": [],
        "shuffled_previous_to_current": [],
        "residual_previous_to_current": [],
    }
    residual_energy: list[float] = []
    bias_energy: list[float] = []
    isolation_checks = 0

    for packet_number, packet in enumerate(packets):
        index = np.asarray(packet["index"], dtype=np.int64)
        x = torch.as_tensor(packet["x"], dtype=torch.float32, device=torch_device)
        parameters = A._named_parameters(model)
        radius = float(np.median(radii)) if radii else 0.0
        radial = _radial_direction(parameters)
        bias = _output_bias_component(previous_gradient, parameters)
        residual = _residual_direction(previous_gradient, parameters)
        random_direction = _random_direction(
            parameters, seed=seed, packet_number=packet_number
        )
        snapshot = A._live_snapshot(model, optimizer)

        causal_directions = {
            "K0_ZERO": None,
            "K1_PREVIOUS_GRADIENT": previous_gradient,
            "K2_RADIAL_SHRINKAGE": radial,
            "K3_PREVIOUS_OUTPUT_BIAS": bias,
            "K4_PREVIOUS_SHUFFLED_LABEL_GRADIENT": previous_shuffled,
            "K5_PREVIOUS_GRADIENT_RESIDUAL": residual,
            "K6_MATCHED_RANDOM_DIRECTION": random_direction,
        }
        for row, direction in causal_directions.items():
            predictions[row][index] = A._predict(
                model,
                A._shift(parameters, A._delta(direction, radius)),
                x,
            )

        # This is the only point at which packet-t labels are materialized.
        y = torch.as_tensor(packet["y"], dtype=torch.float32, device=torch_device)
        event_ids = np.asarray(packet["event_id"], dtype=np.int64)
        fold_a, fold_b = A._folds(index, event_ids)
        ta = torch.as_tensor(fold_a, dtype=torch.long)
        tb = torch.as_tensor(fold_b, dtype=torch.long)
        grad_a, _ = A._gradient_dict(model, parameters, x[ta], y[ta])
        grad_b, _ = A._gradient_dict(model, parameters, x[tb], y[tb])
        full_gradient, _ = A._gradient_dict(model, parameters, x, y)

        oracle = np.empty(len(index), dtype=np.float32)
        oracle[fold_b] = A._predict(
            model, A._shift(parameters, A._delta(grad_a, radius)), x[tb]
        )
        oracle[fold_a] = A._predict(
            model, A._shift(parameters, A._delta(grad_b, radius)), x[ta]
        )
        predictions["K7_ORACLE_XFIT"][index] = oracle

        anti = np.empty(len(index), dtype=np.float32)
        anti[fold_b] = A._predict(
            model,
            A._shift(parameters, A._delta(grad_a, radius, descent=False)),
            x[tb],
        )
        anti[fold_a] = A._predict(
            model,
            A._shift(parameters, A._delta(grad_b, radius, descent=False)),
            x[ta],
        )
        predictions["K9_ANTI_ORACLE_XFIT"][index] = anti

        permutation_a = torch.as_tensor(shuffle_rng.permutation(len(fold_a)), dtype=torch.long)
        permutation_b = torch.as_tensor(shuffle_rng.permutation(len(fold_b)), dtype=torch.long)
        shuffled_a, _ = A._gradient_dict(
            model, parameters, x[ta], y[ta][permutation_a]
        )
        shuffled_b, _ = A._gradient_dict(
            model, parameters, x[tb], y[tb][permutation_b]
        )
        current_shuffled = np.empty(len(index), dtype=np.float32)
        current_shuffled[fold_b] = A._predict(
            model, A._shift(parameters, A._delta(shuffled_a, radius)), x[tb]
        )
        current_shuffled[fold_a] = A._predict(
            model, A._shift(parameters, A._delta(shuffled_b, radius)), x[ta]
        )
        predictions["K8_CURRENT_SHUFFLED_TARGET_XFIT"][index] = current_shuffled

        full_permutation = torch.as_tensor(
            shuffle_rng.permutation(len(index)), dtype=torch.long
        )
        shuffled_full, _ = A._gradient_dict(
            model, parameters, x, y[full_permutation]
        )

        if not A._live_unchanged(snapshot, model, optimizer):
            raise RuntimeError("a specificity branch mutated the live trajectory")
        isolation_checks += 1
        for key, direction in (
            ("previous_to_current", previous_gradient),
            ("radial_to_current", radial),
            ("shuffled_previous_to_current", previous_shuffled),
            ("residual_previous_to_current", residual),
        ):
            value = _cosine(direction, full_gradient)
            if value is not None and bool(np.any(packet["scored"])):
                cosines[key].append(value)
        if previous_gradient is not None and bool(np.any(packet["scored"])):
            total = max(A._dict_norm(previous_gradient) ** 2, 1e-20)
            residual_energy.append(A._dict_norm(residual) ** 2 / total)
            bias_energy.append(A._dict_norm(bias) ** 2 / total)

        optimizer.zero_grad(set_to_none=True)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for name, value in model.named_parameters():
            value.grad = full_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = {name: value.detach().clone() for name, value in model.named_parameters()}
        radii.append(A._difference_norm(before, after))
        previous_gradient = {name: value.detach().clone() for name, value in full_gradient.items()}
        previous_shuffled = {
            name: value.detach().clone() for name, value in shuffled_full.items()
        }

    covered = np.concatenate([np.asarray(packet["index"]) for packet in packets])
    if not np.array_equal(covered, np.arange(n)):
        raise RuntimeError("specificity packets do not cover the stream exactly once")
    scored = np.asarray(data["scored_mask"], dtype=bool)
    for row in ROWS:
        if not np.isfinite(predictions[row][scored]).all():
            raise RuntimeError(f"{row}: missing or non-finite scored prediction")

    metrics = {
        row: metric_by_month(
            prediction,
            np.asarray(data["y"], dtype=np.float32),
            np.asarray(data["month_id"]),
            scored,
        )
        for row, prediction in predictions.items()
    }
    audit = {
        "chronology_preserved": bool(data.get("chronology_preserved")),
        "all_rows_retained": bool(data.get("all_preprocessed_rows_retained")),
        "all_causal_rows_snapshotted_before_current_labels": True,
        "noncausal_rows_explicit": NONCAUSAL_ROWS == set(ROWS) - CAUSAL_ROWS,
        "branches_state_isolated": isolation_checks == len(packets),
        "one_center_update_per_packet": True,
        "equal_branch_radius": True,
        "no_raw_replay": True,
        "scored_predictions_complete": True,
        "cpu_only": True,
    }
    audit["all_pass"] = bool(all(audit.values()))
    signature = hashlib.sha256(
        np.asarray(
            [[len(packet["index"]), packet["n_acquisition_events"]] for packet in packets],
            dtype=np.int64,
        ).tobytes()
    ).hexdigest()
    return {
        "schema_version": "fmmt.workshop.specificity.unit.v1",
        "dataset": data["name"],
        "seed": int(seed),
        "model": model_type,
        "optimizer": optimizer_type,
        "packet_size": int(packet_size),
        "n_examples": int(n),
        "n_packets": int(len(packets)),
        "n_scored": int(scored.sum()),
        "metrics": metrics,
        "direction_diagnostics": {
            key: (float(np.mean(values)) if values else None)
            for key, values in cosines.items()
        }
        | {
            "previous_residual_energy_fraction": (
                float(np.mean(residual_energy)) if residual_energy else None
            ),
            "previous_output_bias_energy_fraction": (
                float(np.mean(bias_energy)) if bias_energy else None
            ),
        },
        "data_manifest": data.get("data_manifest"),
        "chronology_audit": data.get("chronology_audit"),
        "packet_signature_sha256": signature,
        "audit": audit,
        "resources": {
            "center_updates": int(len(packets)),
            "raw_examples_persistently_stored": 0,
            "parameter_sized_past_gradient_states": 2,
            "virtual_forward_rows_per_packet": len(ROWS),
            "virtual_gradient_calls_per_packet": 6,
        },
    }


def _metric_matrix(units: list[dict[str, Any]], row: str) -> tuple[np.ndarray, np.ndarray]:
    shared = set(units[0]["metrics"][row]["per_month"])
    for unit in units[1:]:
        shared &= set(unit["metrics"][row]["per_month"])
    clusters = np.asarray(sorted(shared, key=int), dtype=np.int64)
    if len(clusters) < 2:
        raise RuntimeError("fewer than two shared scored time clusters")
    values = np.asarray(
        [
            [unit["metrics"][row]["per_month"][str(int(cluster))]["rmse"] for cluster in clusters]
            for unit in units
        ],
        dtype=np.float64,
    )
    return values, clusters


def _contrast(
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    inference: Mapping[str, Any],
    *,
    offset: int,
) -> dict[str, Any]:
    result = paired_summary(
        baseline,
        candidate,
        clusters,
        seed=int(inference["cluster_bootstrap_seed"]) + int(offset),
        n_boot=int(inference["cluster_bootstrap_draws"]),
        block_length=int(inference["cluster_bootstrap_block_length"]),
    )
    denominator = max(float(np.mean(baseline)), 1e-12)
    result["baseline_mean"] = float(np.mean(baseline))
    result["candidate_mean"] = float(np.mean(candidate))
    result["relative_improvement"] = float(result["improvement"] / denominator)
    result["ci95_relative"] = [float(value / denominator) for value in result["ci95"]]
    return result


def summarize_specificity(
    units: list[dict[str, Any]], lock: Mapping[str, Any], *, offset: int = 0
) -> dict[str, Any]:
    if not units:
        raise ValueError("no units to summarize")
    identity = {
        (unit["dataset"], unit["model"], unit["optimizer"], unit["packet_size"])
        for unit in units
    }
    if len(identity) != 1:
        raise RuntimeError(f"cannot mix specificity configurations: {sorted(identity)}")
    matrices: dict[str, np.ndarray] = {}
    clusters = None
    for row in ROWS:
        matrices[row], observed = _metric_matrix(units, row)
        if clusters is None:
            clusters = observed
        elif not np.array_equal(clusters, observed):
            raise RuntimeError("row time-cluster supports differ")
    assert clusters is not None
    inference = lock["runtime"]
    contrasts_vs_zero = {
        row: _contrast(
            matrices["K0_ZERO"], matrices[row], clusters, inference,
            offset=offset + index,
        )
        for index, row in enumerate(ROWS)
        if row != "K0_ZERO"
    }
    trivial_rows = list(lock["specificity"]["trivial_explanation_rows"])
    k1_vs_trivial = {
        row: _contrast(
            matrices[row], matrices["K1_PREVIOUS_GRADIENT"], clusters, inference,
            offset=offset + 30 + index,
        )
        for index, row in enumerate(trivial_rows)
    }
    oracle_vs_shuffled = _contrast(
        matrices["K8_CURRENT_SHUFFLED_TARGET_XFIT"],
        matrices["K7_ORACLE_XFIT"],
        clusters,
        inference,
        offset=offset + 50,
    )
    gatespec = lock["specificity"]["target_specific_gate"]
    required = float(gatespec["K1_must_beat_every_trivial_control_relative"])
    residual_required = float(gatespec["K5_residual_vs_K0_relative"])
    seed_wins = sum(
        unit["metrics"]["K1_PREVIOUS_GRADIENT"]["macro_month_nrmse"]
        < unit["metrics"]["K0_ZERO"]["macro_month_nrmse"]
        for unit in units
    )
    gates = {
        "K1_beats_every_trivial_by_required_margin": all(
            value["relative_improvement"] >= required for value in k1_vs_trivial.values()
        ),
        "K1_beats_every_trivial_with_positive_ci": all(
            value["ci95_relative"][0] > 0 for value in k1_vs_trivial.values()
        ),
        "K5_residual_beats_zero_by_required_margin": (
            contrasts_vs_zero["K5_PREVIOUS_GRADIENT_RESIDUAL"]["relative_improvement"]
            >= residual_required
        ),
        "K5_residual_beats_zero_with_positive_ci": (
            contrasts_vs_zero["K5_PREVIOUS_GRADIENT_RESIDUAL"]["ci95_relative"][0] > 0
        ),
        "minimum_seed_wins": seed_wins >= min(
            int(gatespec["minimum_seed_wins"]), len(units)
        ),
        "all_integrity_audits": all(unit["audit"]["all_pass"] for unit in units),
    }
    verdict = (
        "TARGET_SPECIFIC_TEMPORAL_SIGNAL"
        if all(gates.values())
        else "TRIVIAL_CORRECTION_EXPLAINS_GAIN"
    )
    return {
        "schema_version": "fmmt.workshop.specificity.summary.v1",
        "identity": {
            "dataset": units[0]["dataset"],
            "model": units[0]["model"],
            "optimizer": units[0]["optimizer"],
            "packet_size": units[0]["packet_size"],
        },
        "n_initialization_seeds": len(units),
        "n_time_clusters": int(len(clusters)),
        "rows_macro_cluster_nrmse": {
            row: float(np.mean(values)) for row, values in matrices.items()
        },
        "contrasts_vs_K0": contrasts_vs_zero,
        "K1_vs_trivial_controls": k1_vs_trivial,
        "K7_vs_K8_oracle_target_specificity": oracle_vs_shuffled,
        "oracle_target_specificity_pass": bool(
            oracle_vs_shuffled["relative_improvement"]
            >= float(
                lock["specificity"]["oracle_specificity_diagnostic"][
                    "K7_must_beat_K8_relative_for_oracle_to_be_called_target_specific"
                ]
            )
            and oracle_vs_shuffled["ci95_relative"][0] > 0
        ),
        "seed_wins": int(seed_wins),
        "mean_direction_diagnostics": {
            key: float(np.mean([unit["direction_diagnostics"][key] for unit in units]))
            for key in units[0]["direction_diagnostics"]
            if all(unit["direction_diagnostics"][key] is not None for unit in units)
        },
        "gates": gates,
        "verdict": verdict,
    }
