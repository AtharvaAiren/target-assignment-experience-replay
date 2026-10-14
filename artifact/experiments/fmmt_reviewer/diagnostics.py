"""Fixed-trajectory permutation, actuator, and locality diagnostics.

This module is a new post-hoc campaign.  It imports the frozen workshop data
and learner definitions, but never writes into an earlier result directory.
All counterfactual states are discarded; exactly one ordinary center update is
performed for every chronological packet.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from chartfilm.fmmt.core import FunctionalUnitRegressor
from experiments.fmmt import study_a as A
from experiments.fmmt.common import metric_by_month, packetize_stream, seed_everything
from experiments.fmmt_posthoc.k4_multipermutation import _batched_mse_gradients
from experiments.fmmt_workshop.specificity import LinearRegressor, _model, _optimizer


K0 = "K0_ZERO"


def _seed_for_device(seed: int, device: torch.device) -> None:
    """Seed a CPU replay without consulting or initializing the CUDA runtime."""
    if device.type == "cuda":
        seed_everything(int(seed))
        return
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    # ``torch.manual_seed`` also dispatches to CUDA when CUDA is available.
    # Directly seed the CPU default generator so forked CPU workers remain
    # completely CUDA-agnostic while preserving the frozen CPU RNG sequence.
    torch.random.default_generator.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def _live_snapshot_for_device(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Snapshot only RNG domains on which this live trajectory can operate."""
    parameter_device_types = {
        parameter.device.type for parameter in model.parameters()
    }
    buffer_device_types = {buffer.device.type for buffer in model.buffers()}
    observed_device_types = parameter_device_types | buffer_device_types
    if observed_device_types != {device.type}:
        raise RuntimeError(
            "declared trajectory device differs from model tensor devices: "
            f"declared={device.type}, observed={sorted(observed_device_types)}"
        )
    track_cuda = observed_device_types == {"cuda"}
    return {
        "parameters": {
            name: value.detach().clone() for name, value in model.named_parameters()
        },
        "buffers": {
            name: value.detach().clone() for name, value in model.named_buffers()
        },
        "gradients": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "track_cuda_rng": track_cuda,
        "cuda_rng": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if track_cuda
            else None
        ),
    }


def _live_unchanged_for_device(
    snapshot: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> bool:
    track_cuda = bool(snapshot["track_cuda_rng"])
    return bool(
        A._tree_equal(
            snapshot["parameters"],
            {name: value.detach() for name, value in model.named_parameters()},
        )
        and A._tree_equal(
            snapshot["buffers"],
            {name: value.detach() for name, value in model.named_buffers()},
        )
        and A._tree_equal(
            snapshot["gradients"],
            {
                name: None if value.grad is None else value.grad.detach()
                for name, value in model.named_parameters()
            },
        )
        and A._tree_equal(snapshot["optimizer"], optimizer.state_dict())
        and torch.equal(snapshot["cpu_rng"], torch.random.get_rng_state())
        and (
            A._tree_equal(snapshot["cuda_rng"], torch.cuda.get_rng_state_all())
            if track_cuda
            else snapshot["cuda_rng"] is None
        )
    )


def _permutation(
    size: int,
    *,
    initialization_seed: int,
    packet_number: int,
    replica: int,
    salt: int,
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(initialization_seed), int(packet_number), int(replica), int(salt)]
    )
    return np.random.default_rng(sequence).permutation(int(size)).astype(np.int64)


def _flatten(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values.values()])


def _unflatten_like(
    vector: torch.Tensor, parameters: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    cursor = 0
    for name, parameter in parameters.items():
        count = int(parameter.numel())
        output[name] = vector[cursor : cursor + count].reshape(parameter.shape)
        cursor += count
    if cursor != int(vector.numel()):
        raise RuntimeError("flat vector length differs from parameter layout")
    return output


def output_jacobian_matrix(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Analytic per-example output Jacobians in named-parameter order."""
    with torch.no_grad():
        if isinstance(model, LinearRegressor):
            pieces = [inputs, torch.ones((len(inputs), 1), **_tensor_options(inputs))]
        elif isinstance(model, FunctionalUnitRegressor):
            preactivation = F.linear(
                inputs,
                parameters["atom_input_weight"],
                parameters["atom_input_bias"],
            )
            gate = (preactivation > 0).to(dtype=inputs.dtype)
            coefficient = (
                gate
                * parameters["atom_output_weight"].reshape(1, -1)
                * float(model.contribution_scale)
            )
            hidden = torch.relu(preactivation) * float(model.contribution_scale)
            by_name = {
                "base_weight": inputs,
                "base_bias": torch.ones((len(inputs), 1), **_tensor_options(inputs)),
                "atom_input_weight": torch.einsum("na,nd->nad", coefficient, inputs).reshape(len(inputs), -1),
                "atom_input_bias": coefficient,
                "atom_output_weight": hidden,
            }
            pieces = [by_name[name].reshape(len(inputs), -1) for name in parameters]
        else:
            raise TypeError(f"unsupported diagnostic model: {type(model).__name__}")
        result = torch.cat(pieces, dim=1)
    expected = sum(int(value.numel()) for value in parameters.values())
    if result.shape != (len(inputs), expected):
        raise RuntimeError(
            f"Jacobian shape {tuple(result.shape)} != {(len(inputs), expected)}"
        )
    return result


def _tensor_options(reference: torch.Tensor) -> dict[str, Any]:
    return {"dtype": reference.dtype, "device": reference.device}


def packet_strength(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mse_gradient: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], OrderedDict[str, torch.Tensor]]:
    """Return the exact half-squared-loss decomposition and strength metrics."""
    jacobian = output_jacobian_matrix(model, parameters, inputs).double()
    with torch.no_grad():
        prediction = A._call(model, parameters, inputs).double()
    target = targets.reshape(-1).double()
    n = int(len(target))
    if n < 2:
        raise ValueError("packet strength requires at least two examples")
    centered_j = jacobian - jacobian.mean(dim=0, keepdim=True)
    centered_y = target - target.mean()
    centered_f = prediction - prediction.mean()
    mean_component = (
        jacobian.mean(dim=0) * (prediction.mean() - target.mean())
        + centered_j.transpose(0, 1).matmul(centered_f) / float(n)
    )
    covariance = centered_j.transpose(0, 1).matmul(centered_y) / float(n)
    true_half = jacobian.transpose(0, 1).matmul(prediction - target) / float(n)
    autograd_half = _flatten(mse_gradient).double() / 2.0
    decomposition_error = float(torch.linalg.vector_norm(true_half - (mean_component - covariance)).item())
    autograd_error = float(torch.linalg.vector_norm(true_half - autograd_half).item())
    scale = max(float(torch.linalg.vector_norm(true_half).item()), 1.0e-12)
    s_j = float(centered_j.square().sum().item())
    s_y = float(centered_y.square().sum().item())
    permutation_mse = s_j * s_y / (float(n * n) * float(n - 1))
    covariance_norm = float(torch.linalg.vector_norm(covariance).item())
    mean_norm = float(torch.linalg.vector_norm(mean_component).item())
    true_norm = float(torch.linalg.vector_norm(true_half).item())
    permutation_sd = math.sqrt(max(permutation_mse, 0.0))
    denominator = max(true_norm * mean_norm, 1.0e-20)
    cosine = float(torch.dot(true_half, mean_component).item() / denominator)
    normalized_distance = float(
        torch.linalg.vector_norm(
            true_half / max(true_norm, 1.0e-20)
            - mean_component / max(mean_norm, 1.0e-20)
        ).item()
    )
    result = {
        "n": n,
        "S_J": s_j,
        "S_y": s_y,
        "covariance_norm": covariance_norm,
        "expected_permutation_gradient_norm": mean_norm,
        "true_half_gradient_norm": true_norm,
        "permutation_gradient_l2_sd": permutation_sd,
        "covariance_to_permutation_sd": covariance_norm / max(permutation_sd, 1.0e-20),
        "covariance_to_true_gradient": covariance_norm / max(true_norm, 1.0e-20),
        "jacobian_variation_per_example": s_j / float(n),
        "target_variation_per_example": s_y / float(n),
        "true_expected_permutation_cosine": cosine,
        "normalized_true_expected_distance": normalized_distance,
        "decomposition_absolute_error": decomposition_error,
        "decomposition_relative_error": decomposition_error / scale,
        "autograd_absolute_error": autograd_error,
        "autograd_relative_error": autograd_error / scale,
    }
    expected_mse_gradient = _unflatten_like(
        (2.0 * mean_component).to(dtype=inputs.dtype), parameters
    )
    return result, expected_mse_gradient


def _stack_directions(
    directions: Sequence[Mapping[str, torch.Tensor] | None],
    parameters: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, parameter in parameters.items():
        output[name] = torch.stack(
            [
                torch.zeros_like(parameter)
                if direction is None
                else direction[name].to(device=parameter.device, dtype=parameter.dtype)
                for direction in directions
            ],
            dim=0,
        )
    return output


def _states_from_directions(
    parameters: Mapping[str, torch.Tensor],
    directions: Sequence[Mapping[str, torch.Tensor] | None],
    *,
    radius: float,
    multiplier: float,
    mode: str,
    true_norm: float,
) -> OrderedDict[str, torch.Tensor]:
    stacked = _stack_directions(directions, parameters)
    replicas = len(directions)
    norms = torch.sqrt(
        sum(
            values.double().reshape(replicas, -1).square().sum(dim=1)
            for values in stacked.values()
        )
    ).to(dtype=next(iter(parameters.values())).dtype)
    if mode == "normalized_equal_radius":
        scales = torch.where(
            norms > 1.0e-20,
            -float(radius) * float(multiplier) / torch.clamp(norms, min=1.0e-20),
            torch.zeros_like(norms),
        )
    elif mode == "raw_calibrated":
        eta = float(radius) / max(float(true_norm), 1.0e-20)
        scales = torch.full_like(norms, -eta * float(multiplier))
        scales = torch.where(norms > 1.0e-20, scales, torch.zeros_like(scales))
    else:
        raise ValueError(f"unknown actuator mode: {mode}")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, parameter in parameters.items():
        shape = (replicas,) + (1,) * parameter.ndim
        output[name] = parameter.unsqueeze(0) + scales.reshape(shape) * stacked[name]
    return output


def _batched_predictions(
    model: torch.nn.Module,
    states: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    replicas = int(next(iter(states.values())).shape[0])
    with torch.no_grad():
        if isinstance(model, LinearRegressor):
            output = torch.einsum(
                "nd,pd->pn", inputs, states["linear.weight"].squeeze(1)
            ) + states["linear.bias"].reshape(replicas, 1)
        elif isinstance(model, FunctionalUnitRegressor):
            base = torch.einsum("nd,pd->pn", inputs, states["base_weight"])
            base = base + states["base_bias"].reshape(replicas, 1)
            preactivation = torch.einsum(
                "nd,pad->pna", inputs, states["atom_input_weight"]
            ) + states["atom_input_bias"].reshape(replicas, 1, -1)
            output = base + (
                torch.relu(preactivation)
                * states["atom_output_weight"].reshape(replicas, 1, -1)
                * float(model.contribution_scale)
            ).sum(dim=-1)
        else:
            raise TypeError(f"unsupported diagnostic model: {type(model).__name__}")
    return output


def _batched_gradients_at_states(
    model: torch.nn.Module,
    states: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    """Analytic MSE gradients for one target packet per functional state."""
    replicas, n = int(targets.shape[0]), int(targets.shape[1])
    factor = 2.0 / float(n)
    with torch.no_grad():
        prediction = _batched_predictions(model, states, inputs)
        residual = prediction - targets
        if isinstance(model, LinearRegressor):
            output = OrderedDict(
                [
                    ("linear.weight", (factor * torch.einsum("pn,nd->pd", residual, inputs)).unsqueeze(1)),
                    ("linear.bias", (factor * residual.sum(dim=1)).unsqueeze(1)),
                ]
            )
        elif isinstance(model, FunctionalUnitRegressor):
            preactivation = torch.einsum(
                "nd,pad->pna", inputs, states["atom_input_weight"]
            ) + states["atom_input_bias"].reshape(replicas, 1, -1)
            hidden = torch.relu(preactivation)
            gate = (preactivation > 0).to(dtype=inputs.dtype)
            coefficient = (
                residual.unsqueeze(-1)
                * gate
                * states["atom_output_weight"].reshape(replicas, 1, -1)
                * float(model.contribution_scale)
            )
            output = OrderedDict(
                [
                    ("base_weight", factor * torch.einsum("pn,nd->pd", residual, inputs)),
                    ("base_bias", factor * residual.sum(dim=1)),
                    ("atom_input_weight", factor * torch.einsum("pna,nd->pad", coefficient, inputs)),
                    ("atom_input_bias", factor * coefficient.sum(dim=1)),
                    (
                        "atom_output_weight",
                        factor
                        * float(model.contribution_scale)
                        * torch.einsum("pn,pna->pa", residual, hidden),
                    ),
                ]
            )
        else:
            raise TypeError(f"unsupported diagnostic model: {type(model).__name__}")
    if tuple(output) != tuple(states):
        raise RuntimeError("batched gradient layout differs from state layout")
    return output


def _repeat_states(
    parameters: Mapping[str, torch.Tensor], replicas: int
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.unsqueeze(0).expand((replicas,) + tuple(value.shape)).clone())
        for name, value in parameters.items()
    )


def _state_norms(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    replicas = int(next(iter(values.values())).shape[0])
    return torch.sqrt(
        sum(
            value.double().reshape(replicas, -1).square().sum(dim=1)
            for value in values.values()
        )
    )


def _replay_states(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    previous_inputs: torch.Tensor,
    previous_targets: torch.Tensor,
    *,
    radius: float,
    steps: Sequence[int],
) -> tuple[dict[int, OrderedDict[str, torch.Tensor]], float]:
    replicas = int(previous_targets.shape[0])
    states = _repeat_states(parameters, replicas)
    first = _batched_gradients_at_states(
        model, states, previous_inputs, previous_targets
    )
    first_true_norm = float(_state_norms(first)[0].item())
    eta = float(radius) / max(first_true_norm, 1.0e-20)
    requested = set(map(int, steps))
    snapshots: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for step in range(1, max(requested) + 1):
        gradients = first if step == 1 else _batched_gradients_at_states(
            model, states, previous_inputs, previous_targets
        )
        states = OrderedDict(
            (name, value - eta * gradients[name]) for name, value in states.items()
        )
        if step in requested:
            snapshots[step] = OrderedDict(
                (name, value.detach().clone()) for name, value in states.items()
            )
    return snapshots, eta


def _activation_crossing(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    states: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model, FunctionalUnitRegressor):
        return {
            "applicable": False,
            "fraction_crossing": None,
            "replica_fraction_with_any_crossing": None,
            "minimum_center_absolute_preactivation": None,
        }
    with torch.no_grad():
        center = F.linear(
            inputs,
            parameters["atom_input_weight"],
            parameters["atom_input_bias"],
        )
        replicas = int(states["atom_input_weight"].shape[0])
        endpoint = torch.einsum(
            "nd,pad->pna", inputs, states["atom_input_weight"]
        ) + states["atom_input_bias"].reshape(replicas, 1, -1)
        center_expanded = center.unsqueeze(0)
        crossing = (center_expanded == 0) | (endpoint == 0) | (
            torch.signbit(center_expanded) != torch.signbit(endpoint)
        )
        per_replica = crossing.reshape(replicas, -1).float().mean(dim=1)
    return {
        "applicable": True,
        "fraction_crossing": float(per_replica.mean().item()),
        "maximum_replica_fraction_crossing": float(per_replica.max().item()),
        "replica_fraction_with_any_crossing": float((per_replica > 0).float().mean().item()),
        "minimum_center_absolute_preactivation": float(center.abs().min().item()),
        "interpretation": "Exact endpoint test for a linear preactivation path on this packet; not a proof of uniform smoothness.",
    }


def _state_deltas(
    parameters: Mapping[str, torch.Tensor], states: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value - parameters[name].unsqueeze(0)) for name, value in states.items()
    )


def _taylor_diagnostic(
    center_prediction: torch.Tensor,
    branch_predictions: torch.Tensor,
    targets: torch.Tensor,
    current_gradient: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        center_risk = torch.mean((center_prediction - targets.reshape(-1)) ** 2)
        branch_risk = torch.mean(
            (branch_predictions - targets.reshape(1, -1)) ** 2, dim=1
        )
        actual = branch_risk - center_risk
        replicas = int(len(actual))
        linear = sum(
            (
                deltas[name].reshape(replicas, -1)
                * current_gradient[name].reshape(1, -1)
            ).sum(dim=1)
            for name in deltas
        )
        remainder = actual - linear
        delta_norm = _state_norms(deltas).to(dtype=actual.dtype)
        relative = remainder.abs() / torch.clamp(
            torch.maximum(actual.abs(), linear.abs()), min=1.0e-12
        )
        curvature = 2.0 * remainder.abs() / torch.clamp(delta_norm.square(), min=1.0e-20)
    return {
        "mean_absolute_taylor_remainder": float(remainder.abs().mean().item()),
        "maximum_absolute_taylor_remainder": float(remainder.abs().max().item()),
        "mean_relative_taylor_remainder": float(relative.mean().item()),
        "maximum_relative_taylor_remainder": float(relative.max().item()),
        "mean_empirical_directional_curvature": float(curvature.mean().item()),
        "maximum_empirical_directional_curvature": float(curvature.max().item()),
        "interpretation": "Finite-grid directional diagnostic only; it does not establish a uniform Lipschitz constant.",
    }


def _metric_tree(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    month: np.ndarray,
    scored: np.ndarray,
) -> dict[str, Any]:
    return {
        name: metric_by_month(value, target, month, scored)
        for name, value in predictions.items()
    }


def _packet_signature(packets: Sequence[Mapping[str, Any]]) -> str:
    array = np.asarray(
        [[len(packet["index"]), packet["n_acquisition_events"]] for packet in packets],
        dtype=np.int64,
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def run_diagnostic_unit(
    data: dict[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    model_type: str,
    optimizer_type: str,
    packet_size: int,
    n_permutations: int,
    permutation_salt: int,
    radius_multipliers: Sequence[float],
    multistep_counts: Sequence[int],
    device: str = "cpu",
) -> dict[str, Any]:
    """Replay one live learner and evaluate all discarded diagnostic branches."""
    if int(n_permutations) < 2:
        raise ValueError("at least two permutations are required")
    if sorted(set(map(int, multistep_counts))) != list(map(int, multistep_counts)):
        raise ValueError("multistep counts must be unique and increasing")
    if any(int(value) < 1 for value in multistep_counts):
        raise ValueError("multistep counts must be positive")
    multipliers = [float(value) for value in radius_multipliers]
    if any(not math.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError("radius multipliers must be finite and positive")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(int(config.get("torch_threads_per_worker", 1)))
    torch_device = torch.device(device)
    _seed_for_device(int(seed), torch_device)
    packets = packetize_stream(data, int(packet_size))
    model = _model(data["X"].shape[1], model_type, int(config["functional_units"]))
    model.to(torch_device).train()
    optimizer = _optimizer(model, optimizer_type, config)

    n = len(data["X"])
    replicas = int(n_permutations)
    predictions: dict[str, np.ndarray] = {K0: np.full(n, np.nan, dtype=np.float32)}
    for mode in ("normalized", "raw"):
        for multiplier in multipliers:
            key = f"{multiplier:g}"
            predictions[f"{mode}/r{key}/true"] = np.full(n, np.nan, dtype=np.float32)
            predictions[f"{mode}/r{key}/expected"] = np.full(n, np.nan, dtype=np.float32)
            for replica in range(replicas):
                predictions[f"{mode}/r{key}/perm_{replica:02d}"] = np.full(n, np.nan, dtype=np.float32)
    for steps in multistep_counts:
        predictions[f"replay_sgd/steps{int(steps)}/true"] = np.full(n, np.nan, dtype=np.float32)
        for replica in range(replicas):
            predictions[f"replay_sgd/steps{int(steps)}/perm_{replica:02d}"] = np.full(n, np.nan, dtype=np.float32)

    previous_true: Mapping[str, torch.Tensor] | None = None
    previous_expected: Mapping[str, torch.Tensor] | None = None
    previous_permutations: list[Mapping[str, torch.Tensor] | None] = [None] * replicas
    previous_inputs: torch.Tensor | None = None
    previous_target_replicas: torch.Tensor | None = None
    radii: deque[float] = deque(maxlen=int(config["branch_radius_history"]))
    packet_diagnostics: list[dict[str, Any]] = []
    isolation_checks = 0
    normalized_raw_true_max_difference = 0.0
    maximum_decomposition_relative_error = 0.0
    maximum_autograd_relative_error = 0.0
    locality_records = 0

    for packet_number, packet in enumerate(packets):
        index = np.asarray(packet["index"], dtype=np.int64)
        x = torch.as_tensor(packet["x"], dtype=torch.float32, device=torch_device)
        parameters = A._named_parameters(model)
        radius = float(np.median(radii)) if radii else 0.0
        snapshot = _live_snapshot_for_device(model, optimizer, torch_device)
        center_prediction = A._call(model, parameters, x).detach()
        predictions[K0][index] = center_prediction.cpu().numpy().astype(np.float32)

        directions = [previous_true, previous_expected, *previous_permutations]
        true_norm = A._dict_norm(previous_true)
        branch_cache: dict[str, tuple[OrderedDict[str, torch.Tensor], torch.Tensor]] = {}
        for mode, implementation in (
            ("normalized", "normalized_equal_radius"),
            ("raw", "raw_calibrated"),
        ):
            for multiplier in multipliers:
                key = f"{multiplier:g}"
                states = _states_from_directions(
                    parameters,
                    directions,
                    radius=radius,
                    multiplier=multiplier,
                    mode=implementation,
                    true_norm=true_norm,
                )
                outputs = _batched_predictions(model, states, x)
                branch_cache[f"{mode}/r{key}"] = (states, outputs)
                predictions[f"{mode}/r{key}/true"][index] = outputs[0].cpu().numpy()
                predictions[f"{mode}/r{key}/expected"][index] = outputs[1].cpu().numpy()
                for replica in range(replicas):
                    predictions[f"{mode}/r{key}/perm_{replica:02d}"][index] = (
                        outputs[2 + replica].cpu().numpy()
                    )
                if mode == "raw":
                    normalized = branch_cache[f"normalized/r{key}"][1][0]
                    normalized_raw_true_max_difference = max(
                        normalized_raw_true_max_difference,
                        float((outputs[0] - normalized).abs().max().item()),
                    )

        replay_cache: dict[int, tuple[OrderedDict[str, torch.Tensor], torch.Tensor]] = {}
        replay_eta = 0.0
        if previous_inputs is not None and previous_target_replicas is not None:
            states_by_step, replay_eta = _replay_states(
                model,
                parameters,
                previous_inputs,
                previous_target_replicas,
                radius=radius,
                steps=multistep_counts,
            )
            for steps, states in states_by_step.items():
                outputs = _batched_predictions(model, states, x)
                replay_cache[int(steps)] = (states, outputs)
                predictions[f"replay_sgd/steps{steps}/true"][index] = outputs[0].cpu().numpy()
                for replica in range(replicas):
                    predictions[f"replay_sgd/steps{steps}/perm_{replica:02d}"][index] = (
                        outputs[1 + replica].cpu().numpy()
                    )
        else:
            for steps in multistep_counts:
                predictions[f"replay_sgd/steps{int(steps)}/true"][index] = center_prediction.cpu().numpy()
                for replica in range(replicas):
                    predictions[f"replay_sgd/steps{int(steps)}/perm_{replica:02d}"][index] = center_prediction.cpu().numpy()

        # Current labels are first materialized only after every prospective
        # branch prediction has been computed.
        y = torch.as_tensor(packet["y"], dtype=torch.float32, device=torch_device)
        full_gradient, _ = A._gradient_dict(model, parameters, x, y)
        strength, expected_gradient = packet_strength(
            model, parameters, x, y, full_gradient
        )
        maximum_decomposition_relative_error = max(
            maximum_decomposition_relative_error,
            float(strength["decomposition_relative_error"]),
        )
        maximum_autograd_relative_error = max(
            maximum_autograd_relative_error,
            float(strength["autograd_relative_error"]),
        )
        indices = np.stack(
            [
                _permutation(
                    len(index),
                    initialization_seed=int(seed),
                    packet_number=packet_number,
                    replica=replica,
                    salt=int(permutation_salt),
                )
                for replica in range(replicas)
            ],
            axis=0,
        )
        permuted_targets = y[torch.as_tensor(indices, dtype=torch.long, device=torch_device)]
        permutation_gradients = _batched_mse_gradients(
            model, parameters, x, permuted_targets
        )
        half_permuted = torch.stack(
            [_flatten(value).double() / 2.0 for value in permutation_gradients]
        )
        expected_half = _flatten(expected_gradient).double() / 2.0
        squared_distance = (half_permuted - expected_half).square().sum(dim=1)
        strength.update(
            {
                "empirical_permutation_mean_distance_to_expectation": float(
                    torch.linalg.vector_norm(half_permuted.mean(dim=0) - expected_half).item()
                ),
                "empirical_permutation_mse_about_expectation": float(squared_distance.mean().item()),
                "empirical_to_theoretical_permutation_mse_ratio": float(
                    squared_distance.mean().item()
                    / max(float(strength["permutation_gradient_l2_sd"]) ** 2, 1.0e-20)
                ),
            }
        )

        locality: dict[str, Any] = {}
        for cache_key, (states, outputs) in branch_cache.items():
            locality[cache_key] = {
                "activation_segment": _activation_crossing(model, parameters, states, x),
                "taylor": _taylor_diagnostic(
                    center_prediction,
                    outputs,
                    y,
                    full_gradient,
                    _state_deltas(parameters, states),
                ),
            }
            locality_records += 1
        for steps, (states, outputs) in replay_cache.items():
            locality[f"replay_sgd/steps{steps}"] = {
                "activation_segment": _activation_crossing(model, parameters, states, x),
                "taylor": _taylor_diagnostic(
                    center_prediction,
                    outputs,
                    y,
                    full_gradient,
                    _state_deltas(parameters, states),
                ),
            }
            locality_records += 1

        if not _live_unchanged_for_device(snapshot, model, optimizer):
            raise RuntimeError("a reviewer diagnostic branch mutated the live trajectory")
        isolation_checks += 1
        packet_diagnostics.append(
            {
                "packet_number": int(packet_number),
                "n": int(len(index)),
                "n_acquisition_events": int(packet["n_acquisition_events"]),
                "source_packet_scored": bool(np.any(packet["scored"])),
                "used_by_next_packet": bool(packet_number + 1 < len(packets)),
                "destination_packet_scored": bool(
                    packet_number + 1 < len(packets)
                    and np.any(packets[packet_number + 1]["scored"])
                ),
                "base_radius": radius,
                "raw_calibrated_eta": radius / max(true_norm, 1.0e-20),
                "replay_calibrated_eta": replay_eta,
                "strength": strength,
                "locality": locality,
            }
        )

        optimizer.zero_grad(set_to_none=True)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for name, value in model.named_parameters():
            value.grad = full_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = {name: value.detach().clone() for name, value in model.named_parameters()}
        radii.append(A._difference_norm(before, after))
        previous_true = {name: value.detach().clone() for name, value in full_gradient.items()}
        previous_expected = {
            name: value.detach().clone() for name, value in expected_gradient.items()
        }
        previous_permutations = [
            {name: value.detach().clone() for name, value in gradient.items()}
            for gradient in permutation_gradients
        ]
        previous_inputs = x.detach().clone()
        previous_target_replicas = torch.cat(
            [y.reshape(1, -1), permuted_targets], dim=0
        ).detach().clone()

    scored = np.asarray(data["scored_mask"], dtype=bool)
    for name, values in predictions.items():
        if not np.isfinite(values[scored]).all():
            raise RuntimeError(f"{name}: missing or non-finite scored predictions")
    metrics = _metric_tree(
        predictions,
        np.asarray(data["y"], dtype=np.float32),
        np.asarray(data["month_id"]),
        scored,
    )
    diagnostics_finite = _finite_tree(packet_diagnostics)
    tolerance = float(config.get("diagnostic_relative_tolerance", 5.0e-5))
    audit = {
        "chronology_preserved": bool(data.get("chronology_preserved")),
        "all_rows_retained": bool(data.get("all_preprocessed_rows_retained")),
        "all_branches_evaluated_before_current_labels": True,
        "branches_state_isolated": isolation_checks == len(packets),
        "one_true_center_update_per_packet": True,
        "counter_based_independent_permutations": True,
        "packet_strength_complete": len(packet_diagnostics) == len(packets),
        "packet_strength_finite": diagnostics_finite,
        "exact_decomposition_within_tolerance": maximum_decomposition_relative_error <= tolerance,
        "analytic_jacobian_matches_autograd": maximum_autograd_relative_error <= tolerance,
        "normalized_true_matches_calibrated_raw_true": normalized_raw_true_max_difference <= 1.0e-5,
        "locality_grid_complete": locality_records >= len(packets) * 2 * len(multipliers),
        "scored_predictions_complete": True,
        "device_available": True,
    }
    audit["all_pass"] = bool(all(audit.values()))
    return {
        "schema_version": "fmmt.reviewer_diagnostics.unit.v1",
        "posthoc": True,
        "dataset": data["name"],
        "seed": int(seed),
        "model": model_type,
        "optimizer": optimizer_type,
        "packet_size": int(packet_size),
        "n_permutations": replicas,
        "radius_multipliers": multipliers,
        "multistep_counts": list(map(int, multistep_counts)),
        "device": str(device),
        "n_examples": int(n),
        "n_packets": int(len(packets)),
        "n_scored": int(scored.sum()),
        "packet_signature_sha256": _packet_signature(packets),
        "metrics": metrics,
        "packet_diagnostics": packet_diagnostics,
        "data_manifest": data.get("data_manifest"),
        "chronology_audit": data.get("chronology_audit"),
        "numerical_diagnostics": {
            "maximum_decomposition_relative_error": maximum_decomposition_relative_error,
            "maximum_autograd_relative_error": maximum_autograd_relative_error,
            "normalized_raw_true_max_prediction_difference": normalized_raw_true_max_difference,
        },
        "audit": audit,
        "interpretation_limits": [
            "Activation crossing is checked only for observed packet inputs and finite prespecified segments.",
            "Taylor remainders on a finite grid do not prove differentiability or uniform Lipschitz smoothness.",
            "All actuators are discarded counterfactual branches on one fixed live trajectory.",
            "Replay-SGD stores only the immediately preceding packet for this diagnostic and is not the original no-replay policy.",
        ],
        "resources": {
            "center_updates": int(len(packets)),
            "live_trajectory_count": 1,
            "permutations_per_packet": replicas,
            "prediction_rows": int(len(predictions)),
            "packet_diagnostic_rows": int(len(packet_diagnostics)),
        },
    }


def _metric_vector(metric: Mapping[str, Any], clusters: np.ndarray) -> np.ndarray:
    return np.asarray(
        [metric["per_month"][str(int(cluster))]["rmse"] for cluster in clusters],
        dtype=np.float64,
    )


def _circular_columns(
    rng: np.random.Generator, n_columns: int, block_length: int
) -> np.ndarray:
    length = min(max(int(block_length), 1), int(n_columns))
    blocks = int(math.ceil(n_columns / length))
    starts = rng.integers(0, n_columns, size=blocks)
    return np.concatenate(
        [(start + np.arange(length, dtype=np.int64)) % n_columns for start in starts]
    )[:n_columns]


def _nested_bootstrap(
    effects: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    draws: int,
    block_length: int,
) -> dict[str, Any]:
    effects = np.asarray(effects, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if effects.ndim != 3 or effects.shape != baseline.shape:
        raise ValueError("nested arrays must share [seed,permutation,cluster] shape")
    if not np.isfinite(effects).all() or not np.isfinite(baseline).all():
        raise ValueError("nested bootstrap arrays contain non-finite values")
    n_seed, n_permutation, n_cluster = effects.shape
    rng = np.random.default_rng(int(seed))
    absolute = np.empty(int(draws), dtype=np.float64)
    relative = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        seed_index = rng.integers(0, n_seed, size=n_seed)
        cluster_index = _circular_columns(rng, n_cluster, int(block_length))
        permutation_index = rng.integers(
            0, n_permutation, size=(n_seed, n_permutation)
        )
        sampled_effects = []
        sampled_baselines = []
        for row, selected_seed in enumerate(seed_index):
            sampled_effects.append(
                effects[selected_seed][np.ix_(permutation_index[row], cluster_index)]
            )
            sampled_baselines.append(
                baseline[selected_seed][np.ix_(permutation_index[row], cluster_index)]
            )
        absolute[draw] = float(np.mean(sampled_effects))
        relative[draw] = absolute[draw] / max(float(np.mean(sampled_baselines)), 1.0e-12)
    point = float(effects.mean())
    return {
        "improvement": point,
        "relative_improvement": point / max(float(baseline.mean()), 1.0e-12),
        "ci95": [float(value) for value in np.quantile(absolute, [0.025, 0.975])],
        "ci95_relative": [float(value) for value in np.quantile(relative, [0.025, 0.975])],
        "bootstrap": {
            "method": "nested_seed_circular_time_permutation",
            "draws": int(draws),
            "requested_block_length": int(block_length),
            "effective_block_length": min(int(block_length), int(n_cluster)),
            "initialization_seeds_resampled": True,
            "permutation_replicas_resampled_within_seed": True,
            "ordered_temporal_blocks": True,
        },
    }


def _common_clusters(units: Sequence[Mapping[str, Any]]) -> np.ndarray:
    shared: set[int] | None = None
    for unit in units:
        for metric in unit["metrics"].values():
            observed = {int(value) for value in metric["per_month"]}
            shared = observed if shared is None else shared & observed
    if not shared or len(shared) < 2:
        raise RuntimeError("fewer than two common temporal clusters")
    return np.asarray(sorted(shared), dtype=np.int64)


def _describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot describe empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def summarize_units(
    units: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate one dataset across seeds with permutation-aware inference."""
    if not units:
        raise ValueError("no diagnostic units")
    units = sorted(units, key=lambda value: int(value["seed"]))
    identity = {
        (
            unit["dataset"],
            unit["model"],
            unit["optimizer"],
            int(unit["packet_size"]),
            int(unit["n_permutations"]),
            unit["device"],
        )
        for unit in units
    }
    if len(identity) != 1:
        raise RuntimeError(f"mixed diagnostic identities: {sorted(identity)}")
    clusters = _common_clusters(units)
    permutations = int(units[0]["n_permutations"])
    settings: list[tuple[str, str]] = []
    for multiplier in units[0]["radius_multipliers"]:
        key = f"{float(multiplier):g}"
        settings.extend([("normalized", f"r{key}"), ("raw", f"r{key}")])
    settings.extend(
        ("replay_sgd", f"steps{int(steps)}")
        for steps in units[0]["multistep_counts"]
    )
    inference = config["inference"]
    actuator_results: dict[str, Any] = {}
    for setting_index, (mode, setting) in enumerate(settings):
        prefix = f"{mode}/{setting}"
        true = np.stack(
            [_metric_vector(unit["metrics"][f"{prefix}/true"], clusters) for unit in units]
        )
        permuted = np.stack(
            [
                np.stack(
                    [
                        _metric_vector(
                            unit["metrics"][f"{prefix}/perm_{replica:02d}"], clusters
                        )
                        for replica in range(permutations)
                    ]
                )
                for unit in units
            ]
        )
        true_expanded = np.broadcast_to(true[:, None, :], permuted.shape)
        pairing = permuted - true_expanded
        result = _nested_bootstrap(
            pairing,
            permuted,
            seed=int(inference["bootstrap_seed"]) + setting_index,
            draws=int(inference["bootstrap_draws"]),
            block_length=int(inference["primary_block_length"]),
        )
        expected_key = f"{prefix}/expected"
        result["macro_temporal_nrmse"] = {
            "true": float(true.mean()),
            "permutation_mean": float(permuted.mean()),
        }
        if expected_key in units[0]["metrics"]:
            expected = np.stack(
                [_metric_vector(unit["metrics"][expected_key], clusters) for unit in units]
            )
            result["macro_temporal_nrmse"]["exact_permutation_expectation_direction"] = float(expected.mean())
            result["finite_permutation_mean_minus_exact_expectation"] = float(permuted.mean() - expected.mean())
        actuator_results[prefix] = result

    primary = actuator_results["normalized/r1"]
    block_sensitivity: dict[str, Any] = {}
    true = np.stack(
        [_metric_vector(unit["metrics"]["normalized/r1/true"], clusters) for unit in units]
    )
    permuted = np.stack(
        [
            np.stack(
                [
                    _metric_vector(unit["metrics"][f"normalized/r1/perm_{replica:02d}"], clusters)
                    for replica in range(permutations)
                ]
            )
            for unit in units
        ]
    )
    true_expanded = np.broadcast_to(true[:, None, :], permuted.shape)
    for index, block_length in enumerate(inference["block_length_sensitivity"]):
        block_sensitivity[str(int(block_length))] = _nested_bootstrap(
            permuted - true_expanded,
            permuted,
            seed=int(inference["bootstrap_seed"]) + 1000 + index,
            draws=int(inference["bootstrap_draws"]),
            block_length=int(block_length),
        )

    strength_fields = (
        "S_J",
        "S_y",
        "covariance_norm",
        "permutation_gradient_l2_sd",
        "covariance_to_permutation_sd",
        "covariance_to_true_gradient",
        "normalized_true_expected_distance",
        "empirical_to_theoretical_permutation_mse_ratio",
    )
    strength_summary: dict[str, Any] = {}
    # A packet-t gradient powers the packet-(t+1) branch.  Strength is
    # therefore filtered by the destination packet's scoring status, not by
    # whether the source packet itself happened to be scored.
    scored_packets = [
        packet
        for unit in units
        for packet in unit["packet_diagnostics"]
        if packet["destination_packet_scored"]
    ]
    for field in strength_fields:
        strength_summary[field] = _describe(
            [float(packet["strength"][field]) for packet in scored_packets]
        )
    strength_summary["population"] = {
        "unit_of_analysis": "source packet gradient",
        "inclusion_rule": "source packet powers a scored destination packet t+1",
        "n_source_packets": len(scored_packets),
        "final_source_without_destination_excluded": True,
    }

    locality_entries = [
        entry
        for unit in units
        for packet in unit["packet_diagnostics"]
        # Locality belongs to branches evaluated *on this packet*, whereas
        # strength belongs to this packet's gradient used on the next packet.
        if packet["source_packet_scored"]
        for entry in packet["locality"].values()
    ]
    applicable = [
        entry["activation_segment"]
        for entry in locality_entries
        if entry["activation_segment"]["applicable"]
    ]
    locality_summary = {
        "finite_grid_only": True,
        "does_not_prove_smoothness": True,
        "n_packet_setting_records": len(locality_entries),
        "population": {
            "unit_of_analysis": "actuator endpoint on the current destination packet",
            "inclusion_rule": "current destination packet is scored",
        },
        "activation_fraction_crossing_center_to_endpoint": (
            _describe([float(value["fraction_crossing"]) for value in applicable])
            if applicable
            else None
        ),
        "fraction_of_records_with_any_observed_crossing": (
            float(np.mean([value["replica_fraction_with_any_crossing"] for value in applicable]))
            if applicable
            else None
        ),
        "multistep_geometry_limit": (
            "For replay_sgd, activation crossing is measured on the straight "
            "center-to-final-endpoint chord, not on every segment of the curved "
            "multi-step replay path. Taylor risk change remains endpoint-exact."
        ),
        "relative_taylor_remainder": _describe(
            [float(entry["taylor"]["mean_relative_taylor_remainder"]) for entry in locality_entries]
        ),
        "empirical_directional_curvature": _describe(
            [float(entry["taylor"]["mean_empirical_directional_curvature"]) for entry in locality_entries]
        ),
    }
    integrity = {
        "all_unit_audits_pass": all(unit["audit"]["all_pass"] for unit in units),
        "all_locked_block_lengths_present": set(block_sensitivity)
        == {str(int(value)) for value in inference["block_length_sensitivity"]},
        "packet_strength_rows_present": len(scored_packets) > 0,
        "locality_diagnostics_present": len(locality_entries) > 0,
        "primary_result_reused_exactly": primary == actuator_results["normalized/r1"],
    }
    integrity["all_pass"] = bool(all(integrity.values()))
    dataset, model, optimizer, packet_size, _, device = next(iter(identity))
    return {
        "schema_version": "fmmt.reviewer_diagnostics.summary.v1",
        "posthoc": True,
        "identity": {
            "dataset": dataset,
            "model": model,
            "optimizer": optimizer,
            "packet_size": packet_size,
            "device": device,
        },
        "n_initialization_seeds": len(units),
        "n_permutations_per_packet": permutations,
        "n_temporal_clusters": int(len(clusters)),
        "actuator_pairing_results": actuator_results,
        "primary_normalized_r1_pairing": primary,
        "circular_block_length_sensitivity": block_sensitivity,
        "packet_strength": strength_summary,
        "locality_diagnostics": locality_summary,
        "integrity": integrity,
        "interpretation": (
            "Scientific estimates are reported regardless of sign. PASS denotes only "
            "technical completeness and integrity, not support for a paper claim."
        ),
    }
