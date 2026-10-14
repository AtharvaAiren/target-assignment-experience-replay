"""Shared fixed-trajectory target-pairing audit engine.

The live learner receives exactly one true-label update per packet.  Every
audited branch starts from the same live parameter state, is evaluated before
the current packet's targets are materialized as tensors, and is discarded.
The engine supports the historical stored-gradient actuator and a stronger
multi-step replay actuator whose final displacement is projected to the same
Euclidean radius for true and target-permuted branches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
import tempfile
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:  # Torch >= 2.0.
    from torch.func import functional_call as _torch_functional_call

    def _functional_call(
        model: torch.nn.Module,
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return _torch_functional_call(model, (parameters, buffers), (inputs,))

except ImportError:  # pragma: no cover - retained for the declared Torch floor.
    from torch.nn.utils.stateless import functional_call as _stateless_call

    def _functional_call(
        model: torch.nn.Module,
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        state = dict(buffers)
        state.update(parameters)
        return _stateless_call(model, state, (inputs,))


@dataclass(frozen=True)
class AuditPacket:
    """One chronological packet.

    ``targets`` deliberately stays a NumPy array until all deployable branch
    predictions have been computed.  ``ids`` must uniquely identify examples
    within a unit; ``cluster`` and ``domain`` are used for paper-facing strata.
    """

    inputs: torch.Tensor
    targets: np.ndarray
    ids: np.ndarray
    cluster: str
    domain: str


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Durably replace a JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _named_parameters(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in model.named_parameters())


def _named_buffers(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in model.named_buffers())


def _clone_tensor_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _tensor_map_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach(), right[name].detach()) for name in left
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _dict_dot(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    value = 0.0
    for name in left:
        value += float(
            (left[name].detach().double() * right[name].detach().double()).sum().item()
        )
    return value


def _dict_norm(values: Mapping[str, torch.Tensor] | None) -> float:
    if not values:
        return 0.0
    return math.sqrt(max(_dict_dot(values, values), 0.0))


def _dict_cosine(
    left: Mapping[str, torch.Tensor] | None,
    right: Mapping[str, torch.Tensor] | None,
) -> float | None:
    denominator = _dict_norm(left) * _dict_norm(right)
    if denominator <= 1.0e-30 or left is None or right is None:
        return None
    return float(np.clip(_dict_dot(left, right) / denominator, -1.0, 1.0))


def _dict_subtract(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in left}


def _dict_mean(values: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("cannot average an empty gradient collection")
    scale = 1.0 / float(len(values))
    return {
        name: sum((value[name] for value in values), torch.zeros_like(values[0][name]))
        * scale
        for name in values[0]
    }


def _loss(
    output: torch.Tensor, targets: torch.Tensor, task: str
) -> torch.Tensor:
    if task == "classification":
        return F.cross_entropy(output, targets.long().reshape(-1))
    if task == "regression":
        return F.mse_loss(output.reshape(-1), targets.float().reshape(-1))
    raise ValueError(f"unsupported task: {task!r}")


def _validate_mode_stable_audit_model(model: torch.nn.Module) -> None:
    """Reject layers whose train/eval behavior changes the locked estimand.

    CuDNN RNN backward requires its forward pass to run in training mode.  The
    audit therefore uses training mode only while constructing gradients and
    restores evaluation mode before every scored prediction.  That is
    scientifically equivalent only for the locked mode-stable architectures.
    """
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    for name, module in model.named_modules():
        label = name or "<root>"
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise RuntimeError(
                f"audit model contains mode-sensitive BatchNorm at {label}"
            )
        if isinstance(module, dropout_types) and float(module.p) > 0.0:
            raise RuntimeError(
                f"audit model contains active dropout at {label}"
            )
        if isinstance(module, torch.nn.RNNBase) and float(module.dropout) > 0.0:
            raise RuntimeError(
                f"audit model contains an RNN with active dropout at {label}"
            )
        if isinstance(module, torch.nn.RReLU):
            raise RuntimeError(
                f"audit model contains mode-sensitive RReLU at {label}"
            )


def _training_state(model: torch.nn.Module) -> tuple[tuple[str, bool], ...]:
    return tuple((name, bool(module.training)) for name, module in model.named_modules())


@contextmanager
def _gradient_forward_mode(model: torch.nn.Module) -> Iterable[None]:
    """Temporarily enable the RNN backward-compatible forward mode."""
    original = tuple((module, bool(module.training)) for module in model.modules())
    model.train(True)
    try:
        yield
    finally:
        # Direct assignment restores heterogeneous submodule modes exactly;
        # recursively calling ``train`` here would overwrite child states.
        for module, training in original:
            module.training = training


def _gradient(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    task: str,
) -> tuple[dict[str, torch.Tensor], float]:
    # CuDNN does not retain the reserve space needed by an RNN backward when
    # its forward runs in eval mode.  Mode-stable locked models can use a
    # training-mode gradient forward while all task scoring remains in eval.
    with _gradient_forward_mode(model):
        output = _functional_call(model, parameters, buffers, inputs)
        objective = _loss(output, targets, task)
        gradients = torch.autograd.grad(
            objective,
            tuple(parameters.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
    return (
        {
            name: gradient.detach()
            for name, gradient in zip(parameters.keys(), gradients)
        },
        float(objective.detach().item()),
    )


def _normalized_delta(
    direction: Mapping[str, torch.Tensor] | None, radius: float
) -> dict[str, torch.Tensor]:
    if direction is None or radius <= 0.0:
        return {}
    norm = _dict_norm(direction)
    if norm <= 1.0e-30:
        return {}
    scale = -float(radius) / norm
    return {name: value.detach() * scale for name, value in direction.items()}


def _shift(
    parameters: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value if name not in delta else value + delta[name])
        for name, value in parameters.items()
    )


def _branch_output(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    delta: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if model.training:
        raise RuntimeError("scored branch prediction must run in evaluation mode")
    with torch.no_grad():
        return _functional_call(
            model, _shift(parameters, delta), buffers, inputs
        ).detach().cpu()


def _projected_replay_delta(
    model: torch.nn.Module,
    base_parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    task: str,
    radius: float,
    steps: int,
) -> dict[str, torch.Tensor]:
    """Relinearize on the revealed previous packet, then match endpoint radius."""
    if int(steps) < 2:
        raise ValueError("replay actuator requires at least two virtual steps")
    if radius <= 0.0:
        return {}
    virtual = OrderedDict(
        (name, value.detach().clone().requires_grad_(True))
        for name, value in base_parameters.items()
    )
    step_radius = float(radius) / float(steps)
    for _ in range(int(steps)):
        gradients, _ = _gradient(
            model, virtual, buffers, inputs, targets, task
        )
        norm = _dict_norm(gradients)
        if norm <= 1.0e-30:
            break
        virtual = OrderedDict(
            (
                name,
                (virtual[name] - step_radius * gradients[name] / norm)
                .detach()
                .requires_grad_(True),
            )
            for name in virtual
        )
    displacement = {
        name: virtual[name].detach() - base_parameters[name].detach()
        for name in base_parameters
    }
    norm = _dict_norm(displacement)
    if norm <= 1.0e-30:
        return {}
    scale = float(radius) / norm
    return {name: value * scale for name, value in displacement.items()}


def _permutation(
    size: int, *, seed: int, packet_number: int, replica: int, salt: int
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(seed), int(packet_number), int(replica), int(salt)]
    )
    return np.random.default_rng(sequence).permutation(int(size)).astype(np.int64)


def _classification_metrics(output: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    logits = output.float()
    truth = targets.long().reshape(-1)
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(truth, num_classes=int(logits.shape[-1])).float()
    return {
        "count": float(len(truth)),
        "correct": float((logits.argmax(dim=-1) == truth).sum().item()),
        "loss_sum": float(F.cross_entropy(logits, truth, reduction="sum").item()),
        "brier_sum": float((probabilities - one_hot).square().sum(dim=-1).sum().item()),
    }


def _regression_metrics(output: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    prediction = output.float().reshape(-1)
    truth = targets.float().reshape(-1)
    residual = prediction - truth
    return {
        "count": float(len(truth)),
        "squared_error_sum": float(residual.square().sum().item()),
        "absolute_error_sum": float(residual.abs().sum().item()),
    }


def _metrics(output: torch.Tensor, targets: torch.Tensor, task: str) -> dict[str, float]:
    if task == "classification":
        return _classification_metrics(output, targets)
    return _regression_metrics(output, targets)


def _metric_means(sums: Mapping[str, float], task: str) -> dict[str, float]:
    count = max(float(sums.get("count", 0.0)), 1.0)
    if task == "classification":
        return {
            "accuracy": float(sums["correct"] / count),
            "cross_entropy": float(sums["loss_sum"] / count),
            "brier": float(sums["brier_sum"] / count),
            "count": int(round(sums["count"])),
        }
    mse = float(sums["squared_error_sum"] / count)
    return {
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "mae": float(sums["absolute_error_sum"] / count),
        "count": int(round(sums["count"])),
    }


def _add_sums(target: dict[str, float], value: Mapping[str, float]) -> None:
    for key, item in value.items():
        target[key] += float(item)


def _mean_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot average empty metrics")
    keys = values[0].keys()
    return {key: float(np.mean([value[key] for value in values])) for key in keys}


def _packet_effect(
    true_metrics: Mapping[str, float],
    shuffled_metrics: Mapping[str, float],
    task: str,
) -> float:
    count = max(float(true_metrics["count"]), 1.0)
    if task == "classification":
        return float(
            (true_metrics["correct"] - shuffled_metrics["correct"]) / count
        )
    return float(
        (shuffled_metrics["squared_error_sum"] - true_metrics["squared_error_sum"])
        / count
    )


def _circular_bootstrap(
    effects: np.ndarray,
    weights: np.ndarray,
    *,
    draws: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    if effects.ndim != 1 or weights.shape != effects.shape or len(effects) < 2:
        raise ValueError("power analysis requires at least two aligned packet effects")
    if not np.isfinite(effects).all() or not np.isfinite(weights).all():
        raise ValueError("power inputs are non-finite")
    n = len(effects)
    block = max(1, min(int(block_length), n))
    blocks = int(math.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for draw in range(int(draws)):
        starts = rng.integers(0, n, size=blocks)
        index = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[draw] = np.average(effects[index], weights=weights[index])
    estimate = float(np.average(effects, weights=weights))
    # A transparent normal-approximation MDE complements, rather than replaces,
    # the serial block interval.
    sd = float(np.std(effects, ddof=1))
    effective_n = float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-30))
    mde = float(2.801621 * sd / math.sqrt(max(effective_n, 1.0)))
    return {
        "estimate": estimate,
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_draws": int(draws),
        "circular_block_length_packets": block,
        "packet_effect_sd": sd,
        "effective_packet_count": effective_n,
        "normal_approx_mde_80_two_sided": mde,
        "interpretation": "positive means true pairing outperforms the estimator-matched shuffled control",
    }


def _task_effects(
    baseline: Mapping[str, float],
    true: Mapping[str, float],
    shuffled: Mapping[str, float],
    task: str,
) -> dict[str, float]:
    if task == "classification":
        return {
            "gross_accuracy": true["accuracy"] - baseline["accuracy"],
            "shuffled_accuracy": shuffled["accuracy"] - baseline["accuracy"],
            "pairing_accuracy": true["accuracy"] - shuffled["accuracy"],
            "gross_cross_entropy_reduction": baseline["cross_entropy"] - true["cross_entropy"],
            "shuffled_cross_entropy_reduction": baseline["cross_entropy"] - shuffled["cross_entropy"],
            "pairing_cross_entropy_reduction": shuffled["cross_entropy"] - true["cross_entropy"],
        }
    return {
        "gross_mse_reduction": baseline["mse"] - true["mse"],
        "shuffled_mse_reduction": baseline["mse"] - shuffled["mse"],
        "pairing_mse_reduction": shuffled["mse"] - true["mse"],
        "gross_rmse_reduction": baseline["rmse"] - true["rmse"],
        "shuffled_rmse_reduction": baseline["rmse"] - shuffled["rmse"],
        "pairing_rmse_reduction": shuffled["rmse"] - true["rmse"],
    }


def run_fixed_trajectory_audit(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    packets: Iterable[AuditPacket],
    task: str,
    actuators: Sequence[Mapping[str, Any]],
    n_permutations: int,
    permutation_salt: int,
    seed: int,
    device: str,
    radius_history: Sequence[float],
    radius_window: int,
    fallback_radius: float,
    gradient_clip: float,
    bootstrap_draws: int,
    bootstrap_block_length: int,
) -> dict[str, Any]:
    """Run one fixed-trajectory unit and return only audited summaries."""
    if task not in {"classification", "regression"}:
        raise ValueError(f"invalid task {task!r}")
    if int(n_permutations) < 2:
        raise ValueError("at least two target permutations are required")
    if not actuators:
        raise ValueError("at least one actuator is required")
    actuator_names = [str(item["name"]) for item in actuators]
    if len(actuator_names) != len(set(actuator_names)):
        raise ValueError("actuator names must be unique")
    for item in actuators:
        if item.get("kind") not in {"stored_gradient", "replay_projected"}:
            raise ValueError(f"unsupported actuator {item}")
        if item.get("kind") == "replay_projected" and int(item.get("steps", 0)) < 2:
            raise ValueError("replay_projected must use at least two steps")

    seed_everything(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    _validate_mode_stable_audit_model(model)
    model.eval()
    evaluation_mode = _training_state(model)
    buffers = _named_buffers(model)
    radii: deque[float] = deque(
        (float(value) for value in radius_history if float(value) > 0.0),
        maxlen=max(int(radius_window), 1),
    )

    aggregate: dict[str, dict[str, dict[str, float]]] = {
        name: {
            "K0": defaultdict(float),
            "K1": defaultdict(float),
            "K4": defaultdict(float),
        }
        for name in actuator_names
    }
    domain_aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        name: defaultdict(
            lambda: {
                "K0": defaultdict(float),
                "K1": defaultdict(float),
                "K4": defaultdict(float),
            }
        )
        for name in actuator_names
    }
    packet_effects: dict[str, list[float]] = {name: [] for name in actuator_names}
    packet_weights: list[float] = []
    packet_records: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []

    previous_gradient: dict[str, torch.Tensor] | None = None
    previous_shuffled: list[dict[str, torch.Tensor]] | None = None
    previous_inputs: torch.Tensor | None = None
    previous_targets: torch.Tensor | None = None
    previous_permuted_targets: list[torch.Tensor] | None = None
    previous_ids: np.ndarray | None = None
    seen_ids: set[Any] = set()
    total_rows = 0
    branch_isolation_checks = 0
    center_updates = 0

    for packet_number, packet in enumerate(packets):
        inputs = packet.inputs.to(torch_device, non_blocking=True)
        ids = np.asarray(packet.ids)
        targets_numpy = np.asarray(packet.targets)
        if len(inputs) != len(targets_numpy) or len(ids) != len(targets_numpy):
            raise RuntimeError("packet inputs, targets, and IDs differ in length")
        if len(targets_numpy) < 2:
            raise RuntimeError("every audited packet must contain at least two rows")
        canonical_ids = [value.item() if isinstance(value, np.generic) else value for value in ids]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise RuntimeError("duplicate IDs within a packet")
        if seen_ids.intersection(canonical_ids):
            raise RuntimeError("the online stream reuses an example ID")
        seen_ids.update(canonical_ids)
        total_rows += len(ids)

        parameters = _named_parameters(model)
        buffers = _named_buffers(model)
        radius = (
            float(np.median(np.asarray(radii, dtype=np.float64)))
            if radii
            else float(fallback_radius)
        )
        parameter_snapshot = _clone_tensor_map(parameters)
        buffer_snapshot = _clone_tensor_map(buffers)
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        output_zero = _branch_output(model, parameters, buffers, inputs, {})
        branch_outputs: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}
        has_history = previous_gradient is not None
        for actuator in actuators:
            name = str(actuator["name"])
            kind = str(actuator["kind"])
            if not has_history:
                branch_outputs[name] = (
                    output_zero.clone(),
                    [output_zero.clone() for _ in range(int(n_permutations))],
                )
                continue
            if kind == "stored_gradient":
                true_delta = _normalized_delta(previous_gradient, radius)
                shuffled_delta = [
                    _normalized_delta(value, radius)
                    for value in (previous_shuffled or [])
                ]
            else:
                if (
                    previous_inputs is None
                    or previous_targets is None
                    or previous_permuted_targets is None
                ):
                    raise RuntimeError("replay actuator is missing the revealed previous packet")
                true_delta = _projected_replay_delta(
                    model,
                    parameters,
                    buffers,
                    previous_inputs,
                    previous_targets,
                    task=task,
                    radius=radius,
                    steps=int(actuator["steps"]),
                )
                shuffled_delta = [
                    _projected_replay_delta(
                        model,
                        parameters,
                        buffers,
                        previous_inputs,
                        value,
                        task=task,
                        radius=radius,
                        steps=int(actuator["steps"]),
                    )
                    for value in previous_permuted_targets
                ]
            if len(shuffled_delta) != int(n_permutations):
                raise RuntimeError("shuffled branch count differs from the lock")
            true_output = _branch_output(
                model, parameters, buffers, inputs, true_delta
            )
            shuffled_outputs = [
                _branch_output(model, parameters, buffers, inputs, value)
                for value in shuffled_delta
            ]
            # Every nonzero branch is endpoint-radius matched by construction.
            endpoint_norms = [_dict_norm(true_delta)] + [
                _dict_norm(value) for value in shuffled_delta
            ]
            nonzero = [value for value in endpoint_norms if value > 1.0e-25]
            if nonzero and any(abs(value - radius) > max(1.0e-7, 1.0e-5 * radius) for value in nonzero):
                raise RuntimeError("virtual endpoint radius mismatch")
            branch_outputs[name] = (true_output, shuffled_outputs)

        # This is the first tensor materialization of current packet labels.
        targets = torch.as_tensor(targets_numpy, device=torch_device)
        if not _tensor_map_equal(parameter_snapshot, _named_parameters(model)):
            raise RuntimeError("a virtual branch mutated live parameters")
        if not _tensor_map_equal(buffer_snapshot, _named_buffers(model)):
            raise RuntimeError("a virtual branch mutated model buffers")
        if not _nested_equal(optimizer_snapshot, optimizer.state_dict()):
            raise RuntimeError("a virtual branch mutated optimizer state")
        if _training_state(model) != evaluation_mode:
            raise RuntimeError("a virtual branch did not restore evaluation mode")
        branch_isolation_checks += 1

        if has_history:
            packet_weights.append(float(len(targets_numpy)))
            for actuator in actuators:
                name = str(actuator["name"])
                true_output, shuffled_outputs = branch_outputs[name]
                zero_metrics = _metrics(output_zero, targets.cpu(), task)
                true_metrics = _metrics(true_output, targets.cpu(), task)
                shuffled_individual = [
                    _metrics(value, targets.cpu(), task) for value in shuffled_outputs
                ]
                shuffled_metrics = _mean_metrics(shuffled_individual)
                _add_sums(aggregate[name]["K0"], zero_metrics)
                _add_sums(aggregate[name]["K1"], true_metrics)
                _add_sums(aggregate[name]["K4"], shuffled_metrics)
                _add_sums(domain_aggregate[name][str(packet.domain)]["K0"], zero_metrics)
                _add_sums(domain_aggregate[name][str(packet.domain)]["K1"], true_metrics)
                _add_sums(domain_aggregate[name][str(packet.domain)]["K4"], shuffled_metrics)
                effect = _packet_effect(true_metrics, shuffled_metrics, task)
                packet_effects[name].append(effect)
                packet_records.append(
                    {
                        "packet": int(packet_number),
                        "cluster": str(packet.cluster),
                        "domain": str(packet.domain),
                        "actuator": name,
                        "rows": int(len(targets_numpy)),
                        "radius": radius,
                        "K0": zero_metrics,
                        "K1": true_metrics,
                        "K4_mean": shuffled_metrics,
                        "pairing_effect": effect,
                    }
                )

        true_gradient, current_loss = _gradient(
            model, parameters, buffers, inputs, targets, task
        )
        permutations = [
            _permutation(
                len(targets_numpy),
                seed=int(seed),
                packet_number=int(packet_number),
                replica=replica,
                salt=int(permutation_salt),
            )
            for replica in range(int(n_permutations))
        ]
        permuted_targets = [
            targets[torch.as_tensor(index, dtype=torch.long, device=torch_device)]
            for index in permutations
        ]
        shuffled_gradients = [
            _gradient(model, parameters, buffers, inputs, value, task)[0]
            for value in permuted_targets
        ]
        if _training_state(model) != evaluation_mode:
            raise RuntimeError("gradient construction did not restore evaluation mode")
        target_bytes_hash = hashlib.sha256(
            np.ascontiguousarray(targets_numpy).tobytes()
        ).hexdigest()
        changed_fractions: list[float] = []
        fixed_fractions: list[float] = []
        for index, permuted in zip(permutations, permuted_targets):
            if not np.array_equal(np.sort(targets_numpy), np.sort(permuted.detach().cpu().numpy())):
                raise RuntimeError("target permutation did not preserve the target multiset")
            changed_fractions.append(float(np.mean(targets_numpy != targets_numpy[index])))
            fixed_fractions.append(float(np.mean(index == np.arange(len(index)))))
        mean_shuffled = _dict_mean(shuffled_gradients)
        coupling_proxy = _dict_norm(_dict_subtract(true_gradient, mean_shuffled))
        variance = float(
            np.mean(
                [
                    _dict_norm(_dict_subtract(value, mean_shuffled)) ** 2
                    for value in shuffled_gradients
                ]
            )
        )
        cosine_values = [
            value
            for value in (
                _dict_cosine(true_gradient, shuffled)
                for shuffled in shuffled_gradients
            )
            if value is not None
        ]
        distribution_records.append(
            {
                "packet": int(packet_number),
                "rows": int(len(targets_numpy)),
                "target_multiset_sha256": target_bytes_hash,
                "target_mean": float(np.mean(targets_numpy)),
                "target_std": float(np.std(targets_numpy)),
                "mean_assignment_changed_fraction": float(np.mean(changed_fractions)),
                "mean_fixed_point_fraction": float(np.mean(fixed_fractions)),
                "target_multiset_preserved_all_replicas": True,
                "true_gradient_norm": _dict_norm(true_gradient),
                "mean_shuffled_gradient_norm": float(
                    np.mean([_dict_norm(value) for value in shuffled_gradients])
                ),
                "true_to_mean_shuffled_gradient_distance": coupling_proxy,
                "permutation_gradient_variance": variance,
                "mean_true_shuffled_cosine": (
                    float(np.mean(cosine_values)) if cosine_values else 0.0
                ),
            }
        )

        optimizer.zero_grad(set_to_none=True)
        before = _clone_tensor_map(_named_parameters(model))
        for name, parameter in model.named_parameters():
            parameter.grad = true_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = _named_parameters(model)
        update_norm = _dict_norm(_dict_subtract(after, before))
        if not np.isfinite(update_norm):
            raise RuntimeError("live update norm is non-finite")
        if update_norm > 0.0:
            radii.append(update_norm)
        center_updates += 1
        previous_gradient = _clone_tensor_map(true_gradient)
        previous_shuffled = [_clone_tensor_map(value) for value in shuffled_gradients]
        previous_inputs = inputs.detach().clone()
        previous_targets = targets.detach().clone()
        previous_permuted_targets = [value.detach().clone() for value in permuted_targets]
        previous_ids = ids.copy()
        if packet_number % 50 == 0:
            print(
                f"audit progress task={task} seed={seed} packet={packet_number + 1} "
                f"rows={total_rows} radius={radius:.6g}",
                flush=True,
            )

    if center_updates < 2 or not packet_weights:
        raise RuntimeError("the audit produced fewer than two online packets")
    if previous_ids is None or total_rows != len(seen_ids):
        raise RuntimeError("stream coverage accounting failed")

    summaries: dict[str, Any] = {}
    weights = np.asarray(packet_weights, dtype=np.float64)
    for actuator in actuators:
        name = str(actuator["name"])
        rows = {
            branch: _metric_means(aggregate[name][branch], task)
            for branch in ("K0", "K1", "K4")
        }
        domains = {
            domain: {
                branch: _metric_means(values[branch], task)
                for branch in ("K0", "K1", "K4")
            }
            for domain, values in sorted(domain_aggregate[name].items())
        }
        summaries[name] = {
            "specification": dict(actuator),
            "task_metrics": rows,
            "effects": _task_effects(rows["K0"], rows["K1"], rows["K4"], task),
            "by_domain": domains,
            "power": _circular_bootstrap(
                np.asarray(packet_effects[name], dtype=np.float64),
                weights,
                draws=int(bootstrap_draws),
                block_length=int(bootstrap_block_length),
                seed=int(seed) + 7103 + len(name),
            ),
        }
        summaries[name]["power"]["effect_metric"] = (
            "packet_accuracy_difference"
            if task == "classification"
            else "packet_mse_reduction"
        )
        summaries[name]["power"]["aggregation_unit"] = (
            "chronologically ordered packets; permutation replicas are averaged within packet"
        )

    distribution_summary = {
        "packets": len(distribution_records),
        "target_multiset_preserved_all_packets": all(
            value["target_multiset_preserved_all_replicas"]
            for value in distribution_records
        ),
        "mean_assignment_changed_fraction": float(
            np.mean([value["mean_assignment_changed_fraction"] for value in distribution_records])
        ),
        "mean_fixed_point_fraction": float(
            np.mean([value["mean_fixed_point_fraction"] for value in distribution_records])
        ),
        "mean_true_shuffled_gradient_cosine": float(
            np.mean([value["mean_true_shuffled_cosine"] for value in distribution_records])
        ),
        "mean_true_to_shuffled_gradient_distance": float(
            np.mean(
                [value["true_to_mean_shuffled_gradient_distance"] for value in distribution_records]
            )
        ),
        "mean_permutation_gradient_variance": float(
            np.mean([value["permutation_gradient_variance"] for value in distribution_records])
        ),
    }
    audit = {
        "all_pass": bool(
            branch_isolation_checks == center_updates
            and center_updates == len(distribution_records)
            and distribution_summary["target_multiset_preserved_all_packets"]
            and total_rows == len(seen_ids)
            and _training_state(model) == evaluation_mode
        ),
        "strict_prelabel_branch_order": True,
        "live_trajectory_uses_true_labels_only": True,
        "one_center_update_per_packet": center_updates == len(distribution_records),
        "common_center_and_estimator_matched": True,
        "all_virtual_branches_discarded": True,
        "branch_state_isolation_checks": branch_isolation_checks,
        "unique_example_ids": total_rows == len(seen_ids),
        "target_multisets_preserved": distribution_summary[
            "target_multiset_preserved_all_packets"
        ],
        "scored_predictions_use_evaluation_mode": True,
        "gradient_forward_mode_restored": _training_state(model) == evaluation_mode,
        "mode_sensitive_layers_absent": True,
    }
    if not audit["all_pass"]:
        raise RuntimeError(f"fixed-trajectory audit failed closed: {audit}")
    return {
        "schema_version": "fmmt.iclr_breadth.unit_core.v1",
        "task": task,
        "seed": int(seed),
        "device": str(torch_device),
        "n_parameters": int(sum(value.numel() for value in model.parameters())),
        "n_permutations": int(n_permutations),
        "total_rows": int(total_rows),
        "packet_count": int(center_updates),
        "task_loss_at_last_gradient": current_loss,
        "actuators": summaries,
        "distribution_alignment": distribution_summary,
        "packet_records": packet_records,
        "distribution_records": distribution_records,
        "audit": audit,
    }
