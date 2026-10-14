"""Fixed-trajectory next-token gradient-reuse audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import validate_mode_stable_model

try:
    from torch.func import functional_call as _torch_functional_call

    def _functional_call(
        model: torch.nn.Module,
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return _torch_functional_call(model, (parameters, buffers), (inputs,))

except ImportError:  # pragma: no cover - retained for the repository Torch floor.
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
class NextTokenPacket:
    """One chronological packet with targets kept off-device until scoring."""

    input_ids: torch.Tensor
    targets: np.ndarray
    ids: tuple[str, ...]
    timestamps: tuple[str, ...]
    packet_number: int


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


def _parameters(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(model.named_parameters())


def _buffers(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(model.named_buffers())


def _clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _dot(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    total = 0.0
    for name in left:
        total += float(
            (left[name].detach().double() * right[name].detach().double()).sum().item()
        )
    return total


def _norm(values: Mapping[str, torch.Tensor] | None) -> float:
    return math.sqrt(max(_dot(values, values), 0.0)) if values else 0.0


def _subtract(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in left}


def _mean(values: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("cannot average an empty gradient collection")
    scale = 1.0 / len(values)
    return {
        name: sum(
            (value[name] for value in values), torch.zeros_like(values[0][name])
        )
        * scale
        for name in values[0]
    }


def _cosine(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float | None:
    denominator = _norm(left) * _norm(right)
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(_dot(left, right) / denominator, -1.0, 1.0))


def _gradient(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], float]:
    if model.training:
        raise RuntimeError("language-model gradients must use locked evaluation mode")
    logits = model(inputs)
    loss_logits = (
        logits.float()
        if logits.dtype in {torch.float16, torch.bfloat16}
        else logits
    )
    loss = F.cross_entropy(loss_logits, targets.long())
    parameters = _parameters(model)
    gradients = torch.autograd.grad(
        loss,
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
        float(loss.detach().item()),
    )


def _normalized_delta(
    direction: Mapping[str, torch.Tensor] | None, radius: float
) -> dict[str, torch.Tensor]:
    norm = _norm(direction)
    if direction is None or radius <= 0.0 or norm <= 1.0e-30:
        return {}
    scale = -float(radius) / norm
    return {name: value * scale for name, value in direction.items()}


def _branch_output(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    delta: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if model.training:
        raise RuntimeError("scored branches must use evaluation mode")
    with torch.no_grad():
        shifted = OrderedDict(
            (name, value if name not in delta else value + delta[name])
            for name, value in parameters.items()
        )
        return _functional_call(model, shifted, buffers, inputs).detach().float().cpu()


def permutation_indices(
    size: int, *, seed: int, packet_number: int, replica: int, salt: int
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(seed), int(packet_number), int(replica), int(salt)]
    )
    return np.random.default_rng(sequence).permutation(int(size)).astype(np.int64)


def next_token_metric_sums(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    logits = logits.float()
    truth = targets.long().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != len(truth):
        raise ValueError("logits/targets shape mismatch")
    top = min(5, int(logits.shape[-1]))
    top_ids = logits.topk(top, dim=-1).indices
    return {
        "count": float(len(truth)),
        "nll_sum": float(F.cross_entropy(logits, truth, reduction="sum").item()),
        "top1_correct": float((top_ids[:, 0] == truth).sum().item()),
        "top5_correct": float((top_ids == truth[:, None]).any(dim=1).sum().item()),
    }


def _mean_sums(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot average empty metric collection")
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in values[0]
    }


def _add_sums(target: dict[str, float], value: Mapping[str, float]) -> None:
    for key, item in value.items():
        target[key] += float(item)


def _metric_means(values: Mapping[str, float]) -> dict[str, float | int]:
    count = float(values["count"])
    if count <= 0:
        raise RuntimeError("no scored tokens")
    nll = float(values["nll_sum"] / count)
    return {
        "count": int(round(count)),
        "nll": nll,
        "perplexity": math.exp(nll),
        "top1_accuracy": float(values["top1_correct"] / count),
        "top5_accuracy": float(values["top5_correct"] / count),
    }


def _state_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "parameters": tuple(
            (name, int(value.data_ptr()), int(value._version))
            for name, value in model.named_parameters()
        ),
        "buffers": _clone_map(_buffers(model)),
        "training": tuple(
            (name, bool(module.training)) for name, module in model.named_modules()
        ),
    }


def _state_unchanged(model: torch.nn.Module, snapshot: Mapping[str, Any]) -> bool:
    parameters = tuple(
        (name, int(value.data_ptr()), int(value._version))
        for name, value in model.named_parameters()
    )
    buffers = _buffers(model)
    buffer_equal = buffers.keys() == snapshot["buffers"].keys() and all(
        torch.equal(value.detach(), snapshot["buffers"][name])
        for name, value in buffers.items()
    )
    training = tuple(
        (name, bool(module.training)) for name, module in model.named_modules()
    )
    return bool(
        parameters == snapshot["parameters"]
        and buffer_equal
        and training == snapshot["training"]
    )


def _structure_signature(value: Any) -> Any:
    """Cheaply detect in-place optimizer mutation without cloning moment tensors."""
    if torch.is_tensor(value):
        return (
            "tensor",
            int(value.data_ptr()),
            int(value._version),
            str(value.dtype),
            tuple(value.shape),
        )
    if isinstance(value, Mapping):
        return tuple(
            (repr(key), _structure_signature(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_structure_signature(item) for item in value)
    return (type(value).__name__, repr(value))


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _circular_bootstrap(
    effects: np.ndarray,
    weights: np.ndarray,
    *,
    draws: int,
    block_length: int,
    seed: int,
    metric: str,
    contrast: str,
    positive_means: str,
) -> dict[str, Any]:
    if len(effects) < 2 or effects.shape != weights.shape:
        raise ValueError("bootstrap needs two or more aligned packet effects")
    n = len(effects)
    block = max(1, min(int(block_length), n))
    blocks = int(math.ceil(n / block))
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        starts = rng.integers(0, n, size=blocks)
        index = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[draw] = np.average(effects[index], weights=weights[index])
    return {
        "estimate": float(np.average(effects, weights=weights)),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "metric": str(metric),
        "contrast": str(contrast),
        "bootstrap_draws": int(draws),
        "circular_block_length_packets": block,
        "aggregation_unit": "chronological packets; permutation replicas averaged within packet",
        "positive_means": str(positive_means),
    }


def _packet_metric_values(values: Mapping[str, float]) -> dict[str, float]:
    count = float(values["count"])
    if count <= 0.0:
        raise RuntimeError("packet has no scored tokens")
    return {
        "nll": float(values["nll_sum"] / count),
        "top1_accuracy": float(values["top1_correct"] / count),
        "top5_accuracy": float(values["top5_correct"] / count),
    }


def _contrast_value(
    preferred: Mapping[str, float],
    comparator: Mapping[str, float],
    metric: str,
) -> float:
    """Return an improvement, so positive always favors ``preferred``."""
    if metric == "nll":
        return float(comparator[metric] - preferred[metric])
    if metric in {"top1_accuracy", "top5_accuracy"}:
        return float(preferred[metric] - comparator[metric])
    raise ValueError(f"unsupported contrast metric: {metric}")


def _bootstrap_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base)}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def warmup_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    packets: Iterable[NextTokenPacket],
    *,
    device: str,
    epochs: int,
    gradient_clip: float,
    radius_window: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Chronological true-target warmup; return recent realized update radii."""
    model.to(torch.device(device)).eval()
    validate_mode_stable_model(model)
    values = list(packets)
    radii: deque[float] = deque(maxlen=max(1, int(radius_window)))
    losses: list[float] = []
    completed = 0
    total = int(epochs) * len(values)
    for epoch in range(int(epochs)):
        for packet in values:
            inputs = packet.input_ids.to(device)
            targets = torch.as_tensor(packet.targets, dtype=torch.long, device=device)
            gradients, loss = _gradient(model, inputs, targets)
            before = _clone_map(_parameters(model))
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                parameter.grad = gradients[name]
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update = _subtract(_parameters(model), before)
            radius = _norm(update)
            if radius > 0.0:
                radii.append(radius)
            losses.append(loss)
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "warmup",
                        "epoch": epoch,
                        "completed_packets": completed,
                        "total_packets": total,
                    }
                )
    return {
        "epochs": int(epochs),
        "packets": len(values),
        "mean_nll": float(np.mean(losses)),
        "recent_update_radii": list(radii),
        "true_targets_only": True,
    }


def run_next_token_audit(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    packets: Iterable[NextTokenPacket],
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
    fail_on_zero_norm_actuator_branch: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compare no reuse, true-gradient reuse, and permuted-target reuse."""
    if int(n_permutations) < 2:
        raise ValueError("at least two finite target permutations are required")
    if int(bootstrap_draws) < 20:
        raise ValueError("bootstrap_draws is too small")
    seed_everything(seed)
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    validate_mode_stable_model(model)
    packet_values = list(packets)
    radii: deque[float] = deque(
        (float(value) for value in radius_history if float(value) > 0.0),
        maxlen=max(1, int(radius_window)),
    )
    aggregate = {branch: defaultdict(float) for branch in ("K0", "K1", "K4")}
    prefix_sizes = tuple(
        sorted({value for value in (1, 2, int(n_permutations)) if value <= int(n_permutations)})
    )
    prefix_aggregate = {value: defaultdict(float) for value in prefix_sizes}
    previous_true: dict[str, torch.Tensor] | None = None
    previous_permuted: list[dict[str, torch.Tensor]] | None = None
    records: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    packet_branch_metrics: list[dict[str, dict[str, float]]] = []
    permutation_mc_variances: dict[str, list[float]] = {
        "nll": [],
        "top1_accuracy": [],
        "top5_accuracy": [],
    }
    permutation_mc_weights: list[float] = []
    seen_ids: set[str] = set()
    total_rows = 0
    center_updates = 0
    isolation_checks = 0
    actuator_branches_checked = 0
    zero_norm_actuator_branches = 0
    target_reveal_events: list[dict[str, Any]] = []
    last_timestamp: str | None = None

    for observed_packet, packet in enumerate(packet_values):
        if int(packet.packet_number) != observed_packet:
            raise RuntimeError("packet numbers are not contiguous chronological order")
        inputs = packet.input_ids.to(torch_device)
        if inputs.ndim != 2:
            raise RuntimeError("packet inputs must be a rank-two tensor")
        row_count = int(inputs.shape[0])
        if row_count < 2 or len(packet.ids) != row_count:
            raise RuntimeError("invalid next-token packet")
        if len(packet.timestamps) != row_count:
            raise RuntimeError("packet timestamp count differs from its inputs")
        if len(set(packet.ids)) != len(packet.ids) or seen_ids.intersection(packet.ids):
            raise RuntimeError("stream contains a duplicate example ID")
        if tuple(packet.timestamps) != tuple(sorted(packet.timestamps)):
            raise RuntimeError("packet is not chronological")
        if last_timestamp is not None and packet.timestamps[0] < last_timestamp:
            raise RuntimeError("packets are not globally chronological")
        last_timestamp = packet.timestamps[-1]
        seen_ids.update(packet.ids)
        total_rows += len(packet.ids)
        parameters = _parameters(model)
        buffers = _buffers(model)
        radius = (
            float(np.median(np.asarray(radii, dtype=np.float64)))
            if radii
            else float(fallback_radius)
        )
        state = _state_snapshot(model)
        optimizer_state = _structure_signature(optimizer.state_dict())
        branch_output_evaluations = 0

        # The packet target accessor has not been touched while deployable branch
        # logits are computed.  Tests exercise this ordering with a sentinel
        # accessor that raises on any early read.
        zero_output = _branch_output(model, parameters, buffers, inputs, {})
        branch_output_evaluations += 1
        has_history = previous_true is not None
        if has_history:
            true_delta = _normalized_delta(previous_true, radius)
            shuffled_deltas = [
                _normalized_delta(value, radius) for value in (previous_permuted or [])
            ]
            if len(shuffled_deltas) != int(n_permutations):
                raise RuntimeError("historical permutation count differs from lock")
            true_output = _branch_output(
                model, parameters, buffers, inputs, true_delta
            )
            branch_output_evaluations += 1
            shuffled_outputs = [
                _branch_output(model, parameters, buffers, inputs, delta)
                for delta in shuffled_deltas
            ]
            branch_output_evaluations += len(shuffled_outputs)
            endpoint_norms = [_norm(true_delta)] + [_norm(value) for value in shuffled_deltas]
            actuator_branches_checked += len(endpoint_norms)
            packet_zero_norms = sum(value <= 1.0e-25 for value in endpoint_norms)
            zero_norm_actuator_branches += packet_zero_norms
            if packet_zero_norms and fail_on_zero_norm_actuator_branch:
                raise RuntimeError(
                    "zero-norm actuator branch violates the locked full-run protocol"
                )
            if any(
                value > 1.0e-25
                and abs(value - radius) > max(1.0e-7, radius * 1.0e-5)
                for value in endpoint_norms
            ):
                raise RuntimeError("reuse branches are not endpoint-radius matched")
        else:
            true_output = zero_output.clone()
            shuffled_outputs = [zero_output.clone() for _ in range(int(n_permutations))]

        if not _state_unchanged(model, state):
            raise RuntimeError("a scored branch mutated the live model")
        if _structure_signature(optimizer.state_dict()) != optimizer_state:
            raise RuntimeError("a scored branch mutated the live optimizer")
        isolation_checks += 1
        expected_evaluations = 1 + (
            1 + int(n_permutations) if has_history else 0
        )
        branch_logits_complete = branch_output_evaluations == expected_evaluations

        # This is the first access to current predicted targets.  Record the
        # observed reveal event rather than trusting a hardcoded integrity flag.
        targets_numpy = np.asarray(packet.targets, dtype=np.int64)
        target_reveal_events.append(
            {
                "packet": observed_packet,
                "branch_logits_complete_before_reveal": branch_logits_complete,
                "branch_output_evaluations_before_reveal": branch_output_evaluations,
                "expected_branch_output_evaluations": expected_evaluations,
            }
        )
        if len(targets_numpy) != row_count:
            raise RuntimeError("packet input/target mismatch")
        targets = torch.as_tensor(targets_numpy, dtype=torch.long, device=torch_device)
        targets_cpu = targets.cpu()
        if targets_numpy.min() < 0 or targets_numpy.max() >= zero_output.shape[-1]:
            raise RuntimeError("next-token target is outside the model vocabulary")

        if has_history:
            zero_metrics = next_token_metric_sums(zero_output, targets_cpu)
            true_metrics = next_token_metric_sums(true_output, targets_cpu)
            shuffled_individual = [
                next_token_metric_sums(value, targets_cpu) for value in shuffled_outputs
            ]
            shuffled_metrics = _mean_sums(shuffled_individual)
            prefix_metrics = {
                size: _mean_sums(shuffled_individual[:size]) for size in prefix_sizes
            }
            for branch, values in (
                ("K0", zero_metrics),
                ("K1", true_metrics),
                ("K4", shuffled_metrics),
            ):
                _add_sums(aggregate[branch], values)
            for size, values in prefix_metrics.items():
                _add_sums(prefix_aggregate[size], values)
            count = float(true_metrics["count"])
            effect = float(
                (shuffled_metrics["nll_sum"] - true_metrics["nll_sum"]) / count
            )
            branch_values = {
                "K0": _packet_metric_values(zero_metrics),
                "K1": _packet_metric_values(true_metrics),
                "K4": _packet_metric_values(shuffled_metrics),
            }
            packet_branch_metrics.append(branch_values)
            individual_values = [
                _packet_metric_values(value) for value in shuffled_individual
            ]
            for metric in permutation_mc_variances:
                permutation_mc_variances[metric].append(
                    float(
                        np.var(
                            [value[metric] for value in individual_values],
                            ddof=1,
                        )
                    )
                )
            permutation_mc_weights.append(count)
            records.append(
                {
                    "packet": observed_packet,
                    "timestamp_start": packet.timestamps[0],
                    "timestamp_end": packet.timestamps[-1],
                    "rows": len(targets_numpy),
                    "radius": radius,
                    "K0": zero_metrics,
                    "K1": true_metrics,
                    "K4_mean": shuffled_metrics,
                    "permutation_prefix_means": {
                        str(size): values for size, values in prefix_metrics.items()
                    },
                    "pairing_nll_reduction": effect,
                }
            )

        true_gradient, current_nll = _gradient(model, inputs, targets)
        indices = [
            permutation_indices(
                len(targets_numpy),
                seed=int(seed),
                packet_number=observed_packet,
                replica=replica,
                salt=int(permutation_salt),
            )
            for replica in range(int(n_permutations))
        ]
        permuted_targets = [
            targets[torch.as_tensor(index, dtype=torch.long, device=torch_device)]
            for index in indices
        ]
        permuted_gradients = [
            _gradient(model, inputs, value)[0] for value in permuted_targets
        ]
        sorted_targets = np.sort(targets_numpy)
        changed: list[float] = []
        fixed: list[float] = []
        index_assignment_hashes: list[str] = []
        effective_target_hashes: list[str] = []
        for index, value in zip(indices, permuted_targets):
            observed = value.detach().cpu().numpy()
            if not np.array_equal(sorted_targets, np.sort(observed)):
                raise RuntimeError("target permutation changed the token multiset")
            changed.append(float(np.mean(targets_numpy != observed)))
            fixed.append(float(np.mean(index == np.arange(len(index)))))
            index_assignment_hashes.append(
                hashlib.sha256(np.ascontiguousarray(index).tobytes()).hexdigest()
            )
            effective_target_hashes.append(
                hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()
            )
        mean_permuted = _mean(permuted_gradients)
        variance = float(
            np.mean(
                [
                    _norm(_subtract(value, mean_permuted)) ** 2
                    for value in permuted_gradients
                ]
            )
        )
        cosines = [
            value
            for value in (_cosine(true_gradient, item) for item in permuted_gradients)
            if value is not None
        ]
        distribution.append(
            {
                "packet": observed_packet,
                "rows": len(targets_numpy),
                "target_multiset_sha256": hashlib.sha256(
                    np.ascontiguousarray(sorted_targets).tobytes()
                ).hexdigest(),
                "target_multiset_preserved_all_replicas": True,
                "mean_assignment_changed_fraction": float(np.mean(changed)),
                "mean_index_fixed_point_fraction": float(np.mean(fixed)),
                "unique_index_assignments": len(set(index_assignment_hashes)),
                "unique_effective_target_assignments": len(set(effective_target_hashes)),
                "true_gradient_norm": _norm(true_gradient),
                "true_to_mean_permuted_gradient_distance": _norm(
                    _subtract(true_gradient, mean_permuted)
                ),
                "finite_permutation_gradient_variance": variance,
                "mean_true_permuted_gradient_cosine": (
                    float(np.mean(cosines)) if cosines else None
                ),
            }
        )

        before = _clone_map(_parameters(model))
        raw_true_gradient_norm = _norm(true_gradient)
        optimizer.zero_grad(set_to_none=True)
        # Optimizer clipping mutates ``parameter.grad`` in place.  Give AdamW
        # private gradient tensors so ``true_gradient`` remains the exact raw
        # loss-gradient direction stored for the next prospective branch.
        for name, parameter in model.named_parameters():
            parameter.grad = true_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        clipped_live_gradient_norm = _norm(
            {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
        preserved_raw_true_gradient_norm = _norm(true_gradient)
        distribution[-1].update(
            {
                "live_true_gradient_norm_before_clip": raw_true_gradient_norm,
                "live_true_gradient_norm_after_clip": clipped_live_gradient_norm,
                "stored_true_gradient_norm_after_live_clip": (
                    preserved_raw_true_gradient_norm
                ),
                "raw_true_gradient_preserved_exactly": bool(
                    preserved_raw_true_gradient_norm == raw_true_gradient_norm
                ),
            }
        )
        if preserved_raw_true_gradient_norm != raw_true_gradient_norm:
            raise RuntimeError("live gradient clipping mutated the stored raw direction")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_norm = _norm(_subtract(_parameters(model), before))
        if not np.isfinite(update_norm):
            raise RuntimeError("live update norm is non-finite")
        if update_norm > 0.0:
            radii.append(update_norm)
        previous_true = true_gradient
        previous_permuted = permuted_gradients
        center_updates += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "online_audit",
                    "completed_packets": center_updates,
                    "total_packets": len(packet_values),
                    "last_packet": observed_packet,
                }
            )

    if center_updates < 3 or len(packet_branch_metrics) != center_updates - 1:
        raise RuntimeError("audit needs at least three chronological packets")
    metrics = {branch: _metric_means(aggregate[branch]) for branch in aggregate}
    prefix_metrics = {
        str(size): _metric_means(prefix_aggregate[size]) for size in prefix_sizes
    }
    contrast_specifications = {
        "K1_vs_K0": ("K1", "K0", "true-assignment reuse"),
        "K4_vs_K0": ("K4", "K0", "permuted-target reuse"),
        "K1_vs_K4": ("K1", "K4", "true-assignment reuse"),
    }
    weights = np.asarray(permutation_mc_weights, dtype=np.float64)
    inference: dict[str, dict[str, Any]] = {}
    for contrast, (preferred, comparator, preferred_label) in contrast_specifications.items():
        inference[contrast] = {}
        for metric in ("nll", "top1_accuracy", "top5_accuracy"):
            effects = np.asarray(
                [
                    _contrast_value(value[preferred], value[comparator], metric)
                    for value in packet_branch_metrics
                ],
                dtype=np.float64,
            )
            direction = (
                f"positive means {preferred_label} has lower NLL"
                if metric == "nll"
                else f"positive means {preferred_label} has higher {metric}"
            )
            inference[contrast][metric] = _circular_bootstrap(
                effects,
                weights,
                draws=int(bootstrap_draws),
                block_length=int(bootstrap_block_length),
                seed=_bootstrap_seed(int(seed) + 91421, f"{contrast}:{metric}"),
                metric=metric,
                contrast=contrast,
                positive_means=direction,
            )
    total_weight = float(weights.sum())
    permutation_mc_standard_errors = {
        metric: float(
            math.sqrt(
                np.sum(
                    weights**2
                    * np.asarray(variances, dtype=np.float64)
                    / float(n_permutations)
                )
            )
            / total_weight
        )
        for metric, variances in permutation_mc_variances.items()
    }
    # The terminal packet's generated gradients have no subsequent packet on
    # which they can be scored.  Keep its record for provenance, but exclude it
    # from summaries describing realized finite-permutation interventions.
    scored_distribution = distribution[:-1]
    if len(scored_distribution) != len(packet_branch_metrics):
        raise RuntimeError("scored permutation diagnostics are not packet aligned")
    for index, value in enumerate(distribution):
        value["used_by_subsequent_scored_branch"] = index < len(scored_distribution)
    valid_cosines = [
        value["mean_true_permuted_gradient_cosine"]
        for value in scored_distribution
        if value["mean_true_permuted_gradient_cosine"] is not None
    ]
    distribution_summary = {
        "generated_packets": len(distribution),
        "scored_source_packets": len(scored_distribution),
        "excluded_terminal_unscored_packets": len(distribution) - len(scored_distribution),
        "target_multiset_preserved_all_packets": all(
            value["target_multiset_preserved_all_replicas"]
            for value in scored_distribution
        ),
        "mean_assignment_changed_fraction": float(
            np.mean(
                [
                    value["mean_assignment_changed_fraction"]
                    for value in scored_distribution
                ]
            )
        ),
        "mean_index_fixed_point_fraction": float(
            np.mean(
                [
                    value["mean_index_fixed_point_fraction"]
                    for value in scored_distribution
                ]
            )
        ),
        "minimum_unique_index_assignments": int(
            min(value["unique_index_assignments"] for value in scored_distribution)
        ),
        "mean_unique_index_assignments": float(
            np.mean(
                [value["unique_index_assignments"] for value in scored_distribution]
            )
        ),
        "minimum_unique_effective_target_assignments": int(
            min(
                value["unique_effective_target_assignments"]
                for value in scored_distribution
            )
        ),
        "mean_unique_effective_target_assignments": float(
            np.mean(
                [
                    value["unique_effective_target_assignments"]
                    for value in scored_distribution
                ]
            )
        ),
        "mean_true_permuted_gradient_cosine": (
            float(np.mean(valid_cosines)) if valid_cosines else None
        ),
        "mean_true_to_permuted_gradient_distance": float(
            np.mean(
                [
                    value["true_to_mean_permuted_gradient_distance"]
                    for value in scored_distribution
                ]
            )
        ),
        "mean_finite_permutation_gradient_variance": float(
            np.mean(
                [
                    value["finite_permutation_gradient_variance"]
                    for value in scored_distribution
                ]
            )
        ),
    }
    reveal_order_pass = bool(
        len(target_reveal_events) == center_updates
        and all(
            value["branch_logits_complete_before_reveal"]
            for value in target_reveal_events
        )
    )
    audit = {
        "all_pass": bool(
            isolation_checks == center_updates
            and total_rows == len(seen_ids)
            and distribution_summary["target_multiset_preserved_all_packets"]
            and (
                not fail_on_zero_norm_actuator_branch
                or zero_norm_actuator_branches == 0
            )
            and not model.training
            and reveal_order_pass
        ),
        "strict_prelabel_branch_order": reveal_order_pass,
        "observed_target_reveal_events": len(target_reveal_events),
        "fixed_conditioning_tokens": True,
        "predicted_targets_only_are_permuted": True,
        "live_trajectory_uses_true_targets_only": True,
        "one_true_update_per_packet": True,
        "common_center_and_equal_endpoint_radius": True,
        "actuator_branches_checked": actuator_branches_checked,
        "zero_norm_actuator_branches": zero_norm_actuator_branches,
        "zero_norm_actuator_branch_fail_closed": bool(
            fail_on_zero_norm_actuator_branch
        ),
        "raw_true_loss_gradient_preserved_before_live_clipping": True,
        "virtual_branches_discarded": True,
        "branch_state_isolation_checks": isolation_checks,
        "unique_example_ids": total_rows == len(seen_ids),
        "mode_sensitive_layers_absent": True,
    }
    if not audit["all_pass"]:
        raise RuntimeError(f"next-token audit failed closed: {json.dumps(audit)}")
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.unit.v2",
        "trajectory_seed": int(seed),
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "permutation_replicates_nested_within_packet": True,
        "device": str(torch_device),
        "n_parameters": int(sum(value.numel() for value in model.parameters())),
        "n_permutations": int(n_permutations),
        "packet_count": center_updates,
        "scored_token_count": int(metrics["K0"]["count"]),
        "task_loss_at_last_live_gradient": current_nll,
        "task_metrics": metrics,
        "permutation_prefix_sensitivity": {
            "nested_prefix_sizes": list(prefix_sizes),
            "replicas_share_each_packet_and_live_trajectory": True,
            "task_metrics": prefix_metrics,
            "pairing_effects_vs_true_assignment": {
                str(size): {
                    "nll_reduction": float(
                        prefix_metrics[str(size)]["nll"] - metrics["K1"]["nll"]
                    ),
                    "top1_accuracy": float(
                        metrics["K1"]["top1_accuracy"]
                        - prefix_metrics[str(size)]["top1_accuracy"]
                    ),
                    "top5_accuracy": float(
                        metrics["K1"]["top5_accuracy"]
                        - prefix_metrics[str(size)]["top5_accuracy"]
                    ),
                }
                for size in prefix_sizes
            },
        },
        "effects": {
            "gross_nll_reduction": float(metrics["K0"]["nll"] - metrics["K1"]["nll"]),
            "permuted_nll_reduction": float(metrics["K0"]["nll"] - metrics["K4"]["nll"]),
            "pairing_nll_reduction": float(metrics["K4"]["nll"] - metrics["K1"]["nll"]),
            "gross_top1_accuracy": float(
                metrics["K1"]["top1_accuracy"] - metrics["K0"]["top1_accuracy"]
            ),
            "pairing_top1_accuracy": float(
                metrics["K1"]["top1_accuracy"] - metrics["K4"]["top1_accuracy"]
            ),
            "gross_top5_accuracy": float(
                metrics["K1"]["top5_accuracy"] - metrics["K0"]["top5_accuracy"]
            ),
            "permuted_top1_accuracy": float(
                metrics["K4"]["top1_accuracy"] - metrics["K0"]["top1_accuracy"]
            ),
            "permuted_top5_accuracy": float(
                metrics["K4"]["top5_accuracy"] - metrics["K0"]["top5_accuracy"]
            ),
            "pairing_top5_accuracy": float(
                metrics["K1"]["top5_accuracy"] - metrics["K4"]["top5_accuracy"]
            ),
        },
        "inference": {
            "paired_circular_block_intervals": inference,
            "perplexity_policy": (
                "perplexity point estimates are exp(aggregate NLL); uncertainty is "
                "reported on the additive NLL scale"
            ),
            "permutation_outcome_mc_standard_error": {
                "replicas_per_packet": int(n_permutations),
                "conditional_on_packets_and_live_trajectory": True,
                **permutation_mc_standard_errors,
            },
        },
        "actuator": {
            "name": "equal_radius_raw_gradient_actuator",
            "direction": "stored raw minibatch gradient",
            "true_direction_preserved_before_live_gradient_clipping": True,
            "radius_calibration": "median recent true-target AdamW update norm",
            "not_an_adamw_counterfactual": True,
        },
        "distribution_alignment": distribution_summary,
        "packet_records": records,
        "distribution_records": distribution,
        "final_live_model_sha256": _model_sha256(model),
        "audit": audit,
    }
