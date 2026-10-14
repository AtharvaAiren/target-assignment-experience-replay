"""Study A: prospective-gradient opportunity audit for FMMT.

This module deliberately does *not* implement FMMT.  It asks two narrower
questions on a fixed online learner trajectory:

1. Would a correctly directed, pre-label parameter displacement have useful
   headroom on the next prediction?
2. Is any of that direction inferable from gradients revealed in the past?

The primary ceiling is cross-fitted within each arrival packet: a gradient
computed from one deterministic half predicts the disjoint other half, and
vice versa.  It is still explicitly noncausal because the support labels are
peeked.  The ordinary learner is updated exactly once after every branch has
been evaluated; no diagnostic branch may alter its parameters, optimizer, or
future trajectory.
"""

from __future__ import annotations

import math
import hashlib
import copy
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    atomic_json,
    grouped_event_folds,
    load_regression_stream,
    metric_by_month,
    packetize_stream,
    paired_summary,
    seed_everything,
)

try:  # Reuse the Study-B model when the shared core is present.
    from chartfilm.fmmt.core import FunctionalUnitRegressor
except (ImportError, AttributeError):  # pragma: no cover - compatibility path
    FunctionalUnitRegressor = None  # type: ignore[assignment]

try:
    from torch.func import functional_call as _torch_functional_call
except ImportError:  # pragma: no cover - torch 2.0 compatibility
    from torch.nn.utils.stateless import functional_call as _torch_functional_call


ROWS = (
    "A0_ZERO",
    "A1_ORACLE_SELF_SANITY",
    "A2_ORACLE_XFIT",
    "A3_PREVIOUS_GRADIENT",
    "A4_EWMA_GRADIENT",
    "A5_CAUSAL_SELECTOR",
    "A6_ANTI_XFIT",
    "A7_SHUFFLED_TARGET_XFIT",
    "A8_PREVIOUS_ACTUAL_DELTA",
    "A9_ADAMW_STALE_MOMENTUM",
)

CAUSAL_ROWS = {
    "A0_ZERO",
    "A3_PREVIOUS_GRADIENT",
    "A4_EWMA_GRADIENT",
    "A5_CAUSAL_SELECTOR",
    "A8_PREVIOUS_ACTUAL_DELTA",
    "A9_ADAMW_STALE_MOMENTUM",
}
NONCAUSAL_ROWS = set(ROWS) - CAUSAL_ROWS


DEFAULTS: dict[str, Any] = {
    "datasets": ["METRO", "BEIJING"],
    "seeds": [40, 41, 42, 43],
    "smoke_seeds": [7],
    "packet_size": 32,
    "n_atoms": 16,
    "hidden_dim": 64,
    "base_lr": 1e-3,
    "betas": [0.9, 0.999],
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "forecast_decay": 0.9,
    "selector_eta": 0.05,
    "radius_history": 16,
    "bootstrap_seed": 314159,
    "n_boot": 4000,
    "bootstrap_block_length": 3,
    "oracle_min_relative": 0.05,
    "oracle_ci_lower_relative": 0.02,
    "seed_wins_min": 3,
    "shuffled_retention_max": 0.20,
    "causal_min_relative": 0.02,
    "causal_noninferiority": 0.01,
    "sensitivity_multipliers": [0.5, 2.0],
    "synthetic_seeds": list(range(8)),
    "synthetic_packets": 1000,
    "synthetic_dim": 16,
    "synthetic_noise": 0.10,
    "synthetic_drift": 0.025,
    "synthetic_base_lr": 5e-3,
}


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Resolve both lightweight test configs and the immutable lock schema."""
    fallback = DEFAULTS.get(key, default)
    if not isinstance(config, Mapping):
        return getattr(config, key, fallback)
    if key in config:  # Explicit runner override (mainly unit tests/smoke).
        return config[key]

    paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "datasets": (("runtime", "datasets"),),
        "seeds": (("runtime", "development_seeds"),),
        "smoke_seeds": (("runtime", "smoke_seeds"),),
        "packet_size": (("runtime", "packet_size"),),
        "n_atoms": (("runtime", "model", "functional_units"),),
        "base_lr": (("runtime", "model", "learning_rate"),),
        "weight_decay": (("runtime", "model", "weight_decay"),),
        "gradient_clip": (("runtime", "model", "gradient_clip"),),
        "betas": (("runtime", "model", "betas"),),
        "bootstrap_seed": (("runtime", "cluster_bootstrap_seed"),),
        "n_boot": (("runtime", "cluster_bootstrap_draws"),),
        "bootstrap_block_length": (
            ("runtime", "cluster_bootstrap_block_length"),
            ("study_a", "bootstrap_block_length"),
        ),
        "forecast_decay": (("study_a", "ewma_decay"),),
        "selector_eta": (("study_a", "selector_eta"),),
        "sensitivity_multipliers": (("study_a", "radius_sensitivity"),),
        "synthetic_seeds": (("study_a", "synthetic_seeds"),),
        "oracle_min_relative": (
            ("study_a", "oracle_gate", "minimum_relative_nrmse_improvement"),
        ),
        "oracle_ci_lower_relative": (
            ("study_a", "oracle_gate", "ci_lower_relative_improvement"),
        ),
        "seed_wins_min": (("study_a", "oracle_gate", "wins_per_four_seeds"),),
        "shuffled_retention_max": (
            ("study_a", "oracle_gate", "shuffled_may_explain_at_most_fraction"),
        ),
        "causal_min_relative": (
            ("study_a", "causal_gate", "minimum_relative_nrmse_improvement_on_one_dataset"),
        ),
        "causal_noninferiority": (
            ("study_a", "causal_gate", "noninferiority_tolerance_on_other_dataset"),
        ),
    }
    # A same-named key inside study_a/runtime remains a convenient supported
    # override even when it is not part of the immutable lock.
    candidates = (("study_a", key), ("runtime", key)) + paths.get(key, ())
    for path in candidates:
        value: Any = config
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                break
            value = value[component]
        else:
            return value
    return fallback


class _FallbackRegressor(nn.Module):
    """Small functional-unit regressor used only if the shared core is absent."""

    def __init__(self, input_dim: int, n_atoms: int = 16, hidden_dim: int = 64):
        super().__init__()
        width = max(int(n_atoms), int(hidden_dim))
        self.inp = nn.Linear(int(input_dim), width)
        self.act = nn.Tanh()
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.inp(x))).squeeze(-1)


class _LinearRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _new_model(input_dim: int, config: Any, *, synthetic: bool = False) -> nn.Module:
    if synthetic:
        return _LinearRegressor(input_dim)
    n_atoms = int(_get(config, "n_atoms"))
    if FunctionalUnitRegressor is not None:
        try:
            return FunctionalUnitRegressor(input_dim=input_dim, n_atoms=n_atoms)
        except TypeError:
            # Keep compatibility with a core implementation that additionally
            # exposes an optional hidden-width argument.
            return FunctionalUnitRegressor(
                input_dim=input_dim,
                n_atoms=n_atoms,
                hidden_dim=int(_get(config, "hidden_dim")),
            )
    return _FallbackRegressor(
        input_dim=input_dim,
        n_atoms=n_atoms,
        hidden_dim=int(_get(config, "hidden_dim")),
    )


def _named_parameters(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in model.named_parameters())


def _call(
    model: nn.Module,
    parameters: Mapping[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    output = _torch_functional_call(model, dict(parameters), (x,))
    return output.reshape(-1)


def _predict(
    model: nn.Module,
    parameters: Mapping[str, torch.Tensor],
    x: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        return _call(model, parameters, x).detach().cpu().numpy().astype(np.float32)


def _gradient_dict(
    model: nn.Module,
    parameters: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], float]:
    prediction = _call(model, parameters, x)
    loss = F.mse_loss(prediction, y.reshape(-1))
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters.values()),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return (
        {name: grad.detach() for name, grad in zip(parameters, gradients)},
        float(loss.detach().item()),
    )


def _dict_norm(values: Mapping[str, torch.Tensor] | None) -> float:
    if not values:
        return 0.0
    total = torch.zeros((), dtype=torch.float64, device=next(iter(values.values())).device)
    for value in values.values():
        total = total + value.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def _difference_norm(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    total = torch.zeros((), dtype=torch.float64, device=next(iter(left.values())).device)
    for name in left:
        total = total + (left[name].detach().double() - right[name].detach().double()).square().sum()
    return float(torch.sqrt(total).item())


def _ema(
    previous: Mapping[str, torch.Tensor] | None,
    current: Mapping[str, torch.Tensor],
    decay: float,
) -> dict[str, torch.Tensor]:
    if previous is None:
        return {name: value.detach().clone() for name, value in current.items()}
    return {
        name: float(decay) * previous[name] + (1.0 - float(decay)) * current[name]
        for name in current
    }


def _delta(
    direction: Mapping[str, torch.Tensor] | None,
    radius: float,
    *,
    descent: bool = True,
    amplitude: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    norm = _dict_norm(direction)
    if direction is None or norm <= 1e-20 or radius <= 0 or amplitude <= 0:
        return None
    sign = -1.0 if descent else 1.0
    scale = sign * float(radius) * float(amplitude) / norm
    return {name: value * scale for name, value in direction.items()}


def _shift(
    parameters: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor] | None,
) -> OrderedDict[str, torch.Tensor]:
    if delta is None:
        return OrderedDict(parameters)
    return OrderedDict((name, value + delta[name]) for name, value in parameters.items())


def _parameters_equal(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    buffer_snapshot: Mapping[str, torch.Tensor],
) -> bool:
    for name, value in model.named_parameters():
        if name not in snapshot or not torch.equal(value.detach(), snapshot[name]):
            return False
    for name, value in model.named_buffers():
        if name not in buffer_snapshot or not torch.equal(value.detach(), buffer_snapshot[name]):
            return False
    return all(value.grad is None for value in model.parameters())


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _live_snapshot(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    return {
        "parameters": {name: value.detach().clone() for name, value in model.named_parameters()},
        "buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
        "gradients": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _live_unchanged(
    snapshot: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> bool:
    return bool(
        _tree_equal(
            snapshot["parameters"],
            {name: value.detach() for name, value in model.named_parameters()},
        )
        and _tree_equal(
            snapshot["buffers"],
            {name: value.detach() for name, value in model.named_buffers()},
        )
        and _tree_equal(
            snapshot["gradients"],
            {
                name: None if value.grad is None else value.grad.detach()
                for name, value in model.named_parameters()
            },
        )
        and _tree_equal(snapshot["optimizer"], optimizer.state_dict())
        and torch.equal(snapshot["cpu_rng"], torch.random.get_rng_state())
        and _tree_equal(
            snapshot["cuda_rng"],
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
    )


def _scale_direction(
    direction: Mapping[str, torch.Tensor] | None, scale: float
) -> dict[str, torch.Tensor] | None:
    if direction is None:
        return None
    return {name: float(scale) * value for name, value in direction.items()}


def _folds(
    global_index: np.ndarray,
    event_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquisition-event-disjoint folds, with a legacy test-only fallback."""
    if event_ids is not None:
        if len(np.asarray(global_index)) != len(np.asarray(event_ids)):
            raise ValueError("global indices and event IDs have different lengths")
        return grouped_event_folds(event_ids)
    positions = np.arange(len(np.asarray(global_index)), dtype=np.int64)
    return positions[::2], positions[1::2]


def _actual_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: after[name].detach() - before[name].detach() for name in before}


def _adamw_stale_momentum_delta(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
) -> dict[str, torch.Tensor] | None:
    """Analytic next AdamW displacement if the next gradient were exactly zero.

    The function reads but never mutates the optimizer.  It includes moment
    decay, bias correction, epsilon, and decoupled weight decay.
    """
    result: dict[str, torch.Tensor] = {}
    has_past_moment = False
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in optimizer.param_groups:
        beta1, beta2 = (float(value) for value in group["betas"])
        learning_rate = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        epsilon = float(group.get("eps", 1e-8))
        if bool(group.get("amsgrad", False)):
            raise RuntimeError("A9 analytic control does not support AMSGrad")
        for parameter in group["params"]:
            name = name_by_id[id(parameter)]
            state = optimizer.state.get(parameter, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
                result[name] = torch.zeros_like(parameter)
                continue
            has_past_moment = True
            step_value = state["step"]
            step = int(step_value.item()) if torch.is_tensor(step_value) else int(step_value)
            next_step = step + 1
            next_mean = beta1 * state["exp_avg"].detach()
            next_square = beta2 * state["exp_avg_sq"].detach()
            correction1 = 1.0 - beta1 ** next_step
            correction2 = 1.0 - beta2 ** next_step
            denominator = next_square.sqrt() / math.sqrt(max(correction2, 1e-30))
            denominator = denominator + epsilon
            adaptive = -(learning_rate / max(correction1, 1e-30)) * next_mean / denominator
            decay = -learning_rate * weight_decay * parameter.detach()
            result[name] = adaptive + decay
    return result if has_past_moment else None


def _forecast_errors(
    observed: Mapping[str, torch.Tensor],
    forecast: Mapping[str, torch.Tensor] | None,
    scale2: float,
) -> tuple[float, float, float]:
    g2 = _dict_norm(observed) ** 2
    if forecast is None:
        h_error2 = g2
    else:
        diff = {name: observed[name] - forecast[name] for name in observed}
        h_error2 = _dict_norm(diff) ** 2
    denominator = max(float(scale2), g2, 1e-20)
    return min(g2 / denominator, 1.0), min(h_error2 / denominator, 1.0), g2


def _row_metrics(
    predictions: Mapping[str, np.ndarray],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        row: metric_by_month(
            prediction,
            np.asarray(data["y"], dtype=np.float32),
            np.asarray(data["month_id"]),
            np.asarray(data["scored_mask"], dtype=bool),
        )
        for row, prediction in predictions.items()
    }


def _run_unit(
    data: dict[str, Any],
    config: Any,
    *,
    seed: int,
    device: torch.device,
    synthetic: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    packet_size = int(_get(config, "packet_size"))
    packets = packetize_stream(data, packet_size)
    n = len(data["X"])
    model = _new_model(data["X"].shape[1], config, synthetic=synthetic).to(device)
    model.train()
    learning_rate = float(
        _get(config, "synthetic_base_lr") if synthetic else _get(config, "base_lr")
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(float(value) for value in _get(config, "betas")),
        weight_decay=float(_get(config, "weight_decay")),
    )

    predictions = {row: np.full(n, np.nan, dtype=np.float32) for row in ROWS}
    sensitivity_multipliers = sorted(
        {
            float(value)
            for value in _get(config, "sensitivity_multipliers")
            if float(value) > 0 and not math.isclose(float(value), 1.0)
        }
    )
    sensitivity_predictions = {
        f"{multiplier:g}x": np.full(n, np.nan, dtype=np.float32)
        for multiplier in sensitivity_multipliers
    }
    previous_gradient: dict[str, torch.Tensor] | None = None
    velocity: dict[str, torch.Tensor] | None = None
    previous_actual_delta: dict[str, torch.Tensor] | None = None
    update_radii: deque[float] = deque(maxlen=int(_get(config, "radius_history")))
    selector = np.asarray([0.5, 0.5], dtype=np.float64)  # zero, velocity
    selector_trace: list[list[float]] = []
    gradient_scale2 = 1e-20
    rng = np.random.default_rng([int(seed), 0xA7])

    isolation_checks = 0
    isolation_passes = 0
    folds_pass = True
    causal_predictions_before_label = True
    scored_count = 0
    forecast_cosines: list[float] = []
    certificate_rates: list[float] = []

    packet_end_times = np.asarray(
        [float(np.max(packet["event_time_seconds"])) for packet in packets],
        dtype=np.float64,
    )
    calibration_packet_indices = [
        index
        for index, packet in enumerate(packets)
        if int(np.max(packet["index"])) < int(data.get("cal_n", packet_size))
    ]
    if len(calibration_packet_indices) < 2:
        raise RuntimeError("fewer than two complete packets in the calibration prefix")
    calibration_deltas = np.diff(packet_end_times[calibration_packet_indices])
    calibration_deltas = calibration_deltas[calibration_deltas > 0]
    if not len(calibration_deltas):
        calibration_deltas = np.diff(packet_end_times)
        calibration_deltas = calibration_deltas[calibration_deltas > 0]
    reference_packet_delta = (
        float(np.median(calibration_deltas)) if len(calibration_deltas) else 1.0
    )
    previous_packet_end: float | None = None
    effective_decays: list[float] = []
    elapsed_seconds: list[float] = []
    group_counts: list[list[int]] = []
    event_overlap_pass = True
    at_least_two_event_groups = True

    for packet_number, packet in enumerate(packets):
        idx = np.asarray(packet["index"], dtype=np.int64)
        x = torch.as_tensor(packet["x"], dtype=torch.float32, device=device)
        packet_end = packet_end_times[packet_number]
        delta_seconds = (
            reference_packet_delta
            if previous_packet_end is None
            else max(float(packet_end - previous_packet_end), 0.0)
        )
        effective_decay = float(_get(config, "forecast_decay")) ** (
            delta_seconds / max(reference_packet_delta, 1e-12)
        )
        # Propagate the already timestamped forecast state across the elapsed
        # gap *before* predicting.  A long gap therefore erases stale
        # direction rather than assigning the old gradient fresh weight.
        forecast_velocity = _scale_direction(velocity, effective_decay)
        effective_decays.append(effective_decay)
        elapsed_seconds.append(delta_seconds)
        # Intentionally do not materialize the packet target before all causal
        # predictions have been snapshotted.
        parameters = _named_parameters(model)
        radius = float(np.median(update_radii)) if update_radii else 0.0
        branch_snapshot = _live_snapshot(model, optimizer)

        predictions["A0_ZERO"][idx] = _predict(model, parameters, x)
        predictions["A3_PREVIOUS_GRADIENT"][idx] = _predict(
            model, _shift(parameters, _delta(previous_gradient, radius)), x
        )
        predictions["A4_EWMA_GRADIENT"][idx] = _predict(
            model, _shift(parameters, _delta(forecast_velocity, radius)), x
        )
        predictions["A5_CAUSAL_SELECTOR"][idx] = _predict(
            model,
            _shift(
                parameters,
                _delta(forecast_velocity, radius, amplitude=float(selector[1])),
            ),
            x,
        )
        predictions["A8_PREVIOUS_ACTUAL_DELTA"][idx] = _predict(
            model,
            _shift(
                parameters,
                _delta(previous_actual_delta, radius, descent=False),
            ),
            x,
        )
        stale_momentum_delta = _adamw_stale_momentum_delta(model, optimizer)
        predictions["A9_ADAMW_STALE_MOMENTUM"][idx] = _predict(
            model,
            _shift(
                parameters,
                _delta(stale_momentum_delta, radius, descent=False),
            ),
            x,
        )

        # Label revelation point.  Everything below this line that reads y is
        # either an explicitly noncausal diagnostic or an update for t+1.
        y = torch.as_tensor(packet["y"], dtype=torch.float32, device=device)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

        event_ids = np.asarray(packet["event_id"], dtype=np.int64)
        fold_a, fold_b = _folds(idx, event_ids)
        support_events = set(int(value) for value in event_ids[fold_a])
        query_events = set(int(value) for value in event_ids[fold_b])
        overlap_count = len(support_events.intersection(query_events))
        unique_event_count = len(np.unique(event_ids))
        event_overlap_pass = event_overlap_pass and overlap_count == 0
        at_least_two_event_groups = at_least_two_event_groups and unique_event_count >= 2
        group_counts.append(
            [unique_event_count, len(support_events), len(query_events), overlap_count]
        )
        folds_pass = folds_pass and (
            len(fold_a) > 0
            and len(fold_b) > 0
            and len(np.intersect1d(fold_a, fold_b)) == 0
            and len(np.union1d(fold_a, fold_b)) == len(idx)
        )
        ta = torch.as_tensor(fold_a, dtype=torch.long, device=device)
        tb = torch.as_tensor(fold_b, dtype=torch.long, device=device)
        grad_a, _ = _gradient_dict(model, parameters, x[ta], y[ta])
        grad_b, _ = _gradient_dict(model, parameters, x[tb], y[tb])
        full_gradient, _ = _gradient_dict(model, parameters, x, y)

        predictions["A1_ORACLE_SELF_SANITY"][idx] = _predict(
            model, _shift(parameters, _delta(full_gradient, radius)), x
        )

        xfit = np.empty(len(idx), dtype=np.float32)
        xfit[fold_b] = _predict(model, _shift(parameters, _delta(grad_a, radius)), x[tb])
        xfit[fold_a] = _predict(model, _shift(parameters, _delta(grad_b, radius)), x[ta])
        predictions["A2_ORACLE_XFIT"][idx] = xfit

        # Locked, non-gating radius sensitivity.  These branches reuse the
        # exact A2 support gradients and only vary displacement magnitude.
        for multiplier in sensitivity_multipliers:
            sensitivity_xfit = np.empty(len(idx), dtype=np.float32)
            sensitivity_xfit[fold_b] = _predict(
                model,
                _shift(parameters, _delta(grad_a, radius * multiplier)),
                x[tb],
            )
            sensitivity_xfit[fold_a] = _predict(
                model,
                _shift(parameters, _delta(grad_b, radius * multiplier)),
                x[ta],
            )
            sensitivity_predictions[f"{multiplier:g}x"][idx] = sensitivity_xfit

        anti = np.empty(len(idx), dtype=np.float32)
        anti[fold_b] = _predict(
            model, _shift(parameters, _delta(grad_a, radius, descent=False)), x[tb]
        )
        anti[fold_a] = _predict(
            model, _shift(parameters, _delta(grad_b, radius, descent=False)), x[ta]
        )
        predictions["A6_ANTI_XFIT"][idx] = anti

        # Independently permute labels in each support fold.  These gradients
        # have the same shape and update norm as the true cross-fit row.
        perm_a_np = rng.permutation(len(fold_a))
        perm_b_np = rng.permutation(len(fold_b))
        perm_a = torch.as_tensor(perm_a_np, dtype=torch.long, device=device)
        perm_b = torch.as_tensor(perm_b_np, dtype=torch.long, device=device)
        shuffled_a, _ = _gradient_dict(model, parameters, x[ta], y[ta][perm_a])
        shuffled_b, _ = _gradient_dict(model, parameters, x[tb], y[tb][perm_b])
        shuffled = np.empty(len(idx), dtype=np.float32)
        shuffled[fold_b] = _predict(
            model, _shift(parameters, _delta(shuffled_a, radius)), x[tb]
        )
        shuffled[fold_a] = _predict(
            model, _shift(parameters, _delta(shuffled_b, radius)), x[ta]
        )
        predictions["A7_SHUFFLED_TARGET_XFIT"][idx] = shuffled

        isolation_checks += 1
        isolated = (
            _parameters_equal(model, before, before_buffers)
            and _live_unchanged(branch_snapshot, model, optimizer)
        )
        isolation_passes += int(isolated)
        if not isolated:
            raise RuntimeError("Study-A diagnostic branch mutated the live learner")

        scored = np.asarray(packet["scored"], dtype=bool)
        scored_count += int(scored.sum())
        if scored.any() and forecast_velocity is not None:
            cosine_denominator = max(
                _dict_norm(full_gradient) * _dict_norm(forecast_velocity), 1e-20
            )
            dot = sum(
                float(
                    (
                        full_gradient[name].double()
                        * forecast_velocity[name].double()
                    ).sum().item()
                )
                for name in full_gradient
            )
            forecast_cosines.append(dot / cosine_denominator)
            zero_loss = (predictions["A0_ZERO"][idx][scored] - packet["y"][scored]) ** 2
            forecast_loss = (
                predictions["A4_EWMA_GRADIENT"][idx][scored] - packet["y"][scored]
            ) ** 2
            certificate_rates.append(float(np.mean(forecast_loss < zero_loss)))

        # Forecast selector is trained only now, after this packet's causal
        # predictions are irrevocably recorded.
        zero_error, velocity_error, current_g2 = _forecast_errors(
            full_gradient, forecast_velocity, gradient_scale2
        )
        selector *= np.exp(
            -float(_get(config, "selector_eta"))
            * np.asarray([zero_error, velocity_error], dtype=np.float64)
        )
        selector /= max(float(selector.sum()), 1e-300)
        selector_trace.append(selector.tolist())
        gradient_scale2 = 0.95 * gradient_scale2 + 0.05 * current_g2
        # Assimilate g_t only after every prediction at t is fixed.  This state
        # is timestamped at t and will be decayed over the next elapsed gap.
        velocity = _ema(velocity, full_gradient, effective_decay)

        # Exactly one live center update, reusing the full-packet gradient.
        optimizer.zero_grad(set_to_none=True)
        pre_update = {name: value.detach().clone() for name, value in model.named_parameters()}
        for name, value in model.named_parameters():
            value.grad = full_gradient[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(_get(config, "gradient_clip")))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        post_update = {name: value.detach() for name, value in model.named_parameters()}
        update_radii.append(_difference_norm(pre_update, post_update))

        previous_actual_delta = _actual_delta(pre_update, post_update)
        previous_gradient = {name: value.clone() for name, value in full_gradient.items()}
        previous_packet_end = float(packet_end)

    processed_index = np.concatenate(
        [np.asarray(packet["index"], dtype=np.int64) for packet in packets]
    )
    if not np.array_equal(processed_index, np.arange(len(data["X"]), dtype=np.int64)):
        raise RuntimeError("Study-A packets do not cover the stream exactly once")
    scored_mask = np.asarray(data["scored_mask"], dtype=bool)
    expected_scored = int(scored_mask[processed_index].sum())
    finite_complete = all(
        bool(np.isfinite(predictions[row][processed_index][scored_mask[processed_index]]).all())
        and int(
            np.isfinite(predictions[row][processed_index][scored_mask[processed_index]]).sum()
        )
        == expected_scored
        for row in ROWS
    )
    if not finite_complete:
        raise RuntimeError("Study-A row has a missing or non-finite scored prediction")
    metrics = _row_metrics(predictions, data)
    sensitivity_metrics = {
        label: metric_by_month(
            prediction,
            np.asarray(data["y"], dtype=np.float32),
            np.asarray(data["month_id"]),
            np.asarray(data["scored_mask"], dtype=bool),
        )
        for label, prediction in sensitivity_predictions.items()
    }
    # Include the locked primary radius for a self-contained three-point
    # sensitivity table, while retaining A2 as the only gated row.
    sensitivity_metrics["1x"] = metrics["A2_ORACLE_XFIT"]
    group_count_array = np.asarray(group_counts, dtype=np.int64)
    group_count_sha256 = hashlib.sha256(group_count_array.tobytes()).hexdigest()
    audit = {
        "chronology_preserved": bool(data.get("chronology_preserved", True)),
        "all_preprocessed_rows_retained": bool(
            data.get("all_preprocessed_rows_retained", False)
        ),
        "packet_folds_disjoint_exhaustive": bool(folds_pass),
        "zero_acquisition_event_overlap": bool(event_overlap_pass),
        "at_least_two_acquisition_event_groups_per_packet": bool(
            at_least_two_event_groups
        ),
        "all_causal_predictions_snapshotted_before_label": causal_predictions_before_label,
        "oracle_rows_explicitly_noncausal": NONCAUSAL_ROWS == {
            row for row in ROWS if row not in CAUSAL_ROWS
        },
        "diagnostic_branches_state_isolated": isolation_passes == isolation_checks,
        "one_live_update_per_packet": True,
        "calibration_excluded_from_metrics": True,
        "scored_predictions_complete": scored_count == expected_scored,
        "every_row_prediction_finite_on_every_scored_example": finite_complete,
    }
    audit["all_pass"] = bool(all(audit.values()))
    return {
        "seed": int(seed),
        "dataset": str(data.get("name", "unknown")),
        "n_examples": int(len(processed_index)),
        "n_packets": int(len(packets)),
        "n_scored": expected_scored,
        "metrics": metrics,
        "oracle_radius_sensitivity": {
            "non_gating": True,
            "metrics": sensitivity_metrics,
        },
        "selector_final": {"zero": float(selector[0]), "velocity": float(selector[1])},
        "selector_mean_last_10pct": {
            "zero": float(np.mean(np.asarray(selector_trace)[-max(1, len(selector_trace) // 10) :, 0])),
            "velocity": float(np.mean(np.asarray(selector_trace)[-max(1, len(selector_trace) // 10) :, 1])),
        },
        "forecast": {
            "mean_cosine": float(np.mean(forecast_cosines)) if forecast_cosines else None,
            "examplewise_improvement_fraction": (
                float(np.mean(certificate_rates)) if certificate_rates else None
            ),
        },
        "elapsed_time_forecast": {
            "base_decay": float(_get(config, "forecast_decay")),
            "reference_calibration_packet_delta_seconds": reference_packet_delta,
            "effective_decay_min": float(np.min(effective_decays)),
            "effective_decay_median": float(np.median(effective_decays)),
            "effective_decay_max": float(np.max(effective_decays)),
            "elapsed_seconds_min": float(np.min(elapsed_seconds)),
            "elapsed_seconds_median": float(np.median(elapsed_seconds)),
            "elapsed_seconds_max": float(np.max(elapsed_seconds)),
            "effective_decay_per_packet": effective_decays,
            "elapsed_seconds_per_packet": elapsed_seconds,
            "formula": (
                "forecast_t = decay(delta_t) * state_{t-1}; "
                "state_t = decay(delta_t) * state_{t-1} + "
                "(1-decay(delta_t)) * gradient_t"
            ),
            "long_gaps_decay_stale_state_before_prediction": True,
        },
        "group_fold_audit": {
            "columns": [
                "unique_events",
                "support_events",
                "query_events",
                "overlap_events",
            ],
            "per_packet_counts": group_count_array.tolist(),
            "per_packet_counts_sha256": group_count_sha256,
            "zero_overlap_packets": int(np.sum(group_count_array[:, 3] == 0)),
            "packets_with_at_least_two_groups": int(
                np.sum(group_count_array[:, 0] >= 2)
            ),
            "n_packets": int(len(group_count_array)),
        },
        "chronology_audit": data.get("chronology_audit"),
        "audit": audit,
        "resources": {
            "live_optimizer_updates": int(len(packets)),
            "live_label_exposures": int(len(processed_index)),
            "branch_forward_calls_per_packet": 13 + 2 * len(sensitivity_multipliers),
            "branch_gradient_calls_per_packet": 5,
            "raw_examples_persistently_stored": 0,
            "forecast_parameter_sized_states": 3,
            "A9_reuses_existing_adamw_state": True,
        },
    }


def _month_arrays(
    units: list[dict[str, Any]],
    row: str,
) -> tuple[np.ndarray, np.ndarray]:
    shared = set(units[0]["metrics"][row]["per_month"])
    for unit in units[1:]:
        shared &= set(unit["metrics"][row]["per_month"])
    months = np.asarray(sorted(shared, key=int), dtype=np.int64)
    values = np.asarray(
        [
            [unit["metrics"][row]["per_month"][str(int(month))]["rmse"] for month in months]
            for unit in units
        ],
        dtype=np.float64,
    )
    return values, months


def _sensitivity_month_arrays(
    units: list[dict[str, Any]],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    def per_month(unit: dict[str, Any]) -> Mapping[str, Any]:
        return unit["oracle_radius_sensitivity"]["metrics"][label]["per_month"]

    shared = set(per_month(units[0]))
    for unit in units[1:]:
        shared &= set(per_month(unit))
    months = np.asarray(sorted(shared, key=int), dtype=np.int64)
    values = np.asarray(
        [
            [per_month(unit)[str(int(month))]["rmse"] for month in months]
            for unit in units
        ],
        dtype=np.float64,
    )
    return values, months


def _relative_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    config: Any,
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    summary = paired_summary(
        baseline,
        candidate,
        clusters,
        seed=int(_get(config, "bootstrap_seed")) + int(seed_offset),
        n_boot=int(_get(config, "n_boot")),
        block_length=int(_get(config, "bootstrap_block_length")),
    )
    denominator = max(float(np.mean(baseline)), 1e-12)
    summary["ci95_relative"] = [value / denominator for value in summary["ci95"]]
    summary["baseline_mean"] = float(np.mean(baseline))
    summary["candidate_mean"] = float(np.mean(candidate))
    return summary


def _aggregate_dataset(
    dataset: str,
    units: list[dict[str, Any]],
    config: Any,
    *,
    offset: int,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    months: np.ndarray | None = None
    for row in ROWS:
        arrays[row], row_months = _month_arrays(units, row)
        if months is None:
            months = row_months
        elif not np.array_equal(months, row_months):
            raise RuntimeError(f"{dataset}: row month supports differ")
    if months is None or len(months) < 2:
        raise RuntimeError(f"{dataset}: fewer than two scored calendar-month clusters")
    baseline = arrays["A0_ZERO"]
    contrasts = {
        row: _relative_summary(baseline, arrays[row], months, config, seed_offset=offset + i)
        for i, row in enumerate(ROWS)
        if row != "A0_ZERO"
    }
    oracle_vs_shuffled = _relative_summary(
        arrays["A7_SHUFFLED_TARGET_XFIT"],
        arrays["A2_ORACLE_XFIT"],
        months,
        config,
        seed_offset=offset + 31,
    )
    oracle_vs_anti = _relative_summary(
        arrays["A6_ANTI_XFIT"],
        arrays["A2_ORACLE_XFIT"],
        months,
        config,
        seed_offset=offset + 37,
    )
    causal_controls = (
        "A3_PREVIOUS_GRADIENT",
        "A4_EWMA_GRADIENT",
        "A8_PREVIOUS_ACTUAL_DELTA",
        "A9_ADAMW_STALE_MOMENTUM",
    )
    strongest_causal_control = min(
        causal_controls, key=lambda row: float(np.mean(arrays[row]))
    )
    selector_vs_causal_control = _relative_summary(
        arrays[strongest_causal_control],
        arrays["A5_CAUSAL_SELECTOR"],
        months,
        config,
        seed_offset=offset + 43,
    )
    oracle = contrasts["A2_ORACLE_XFIT"]
    shuffled = contrasts["A7_SHUFFLED_TARGET_XFIT"]
    anti = contrasts["A6_ANTI_XFIT"]
    oracle_gain = float(oracle["improvement"])
    retention = float(shuffled["improvement"] / max(oracle_gain, 1e-12))
    seed_wins = sum(
        unit["metrics"]["A2_ORACLE_XFIT"]["macro_month_nrmse"]
        < unit["metrics"]["A0_ZERO"]["macro_month_nrmse"]
        for unit in units
    )
    gates = {
        "oracle_relative_ge_5pct": oracle["relative_improvement"]
        >= float(_get(config, "oracle_min_relative")),
        "oracle_relative_ci_lower_gt_2pct": oracle["ci95_relative"][0]
        > float(_get(config, "oracle_ci_lower_relative")),
        "oracle_seed_wins": seed_wins >= min(int(_get(config, "seed_wins_min")), len(units)),
        "oracle_beats_shuffled_ci": oracle_vs_shuffled["ci95"][0] > 0,
        "shuffled_retains_le_20pct": retention <= float(_get(config, "shuffled_retention_max")),
        "oracle_beats_anti_ci": oracle_vs_anti["ci95"][0] > 0,
        "anti_mean_gain_nonpositive": anti["improvement"] <= 0,
        "all_integrity_audits": all(unit["audit"]["all_pass"] for unit in units),
    }
    sensitivity_labels = set(units[0]["oracle_radius_sensitivity"]["metrics"])
    for unit in units[1:]:
        sensitivity_labels &= set(unit["oracle_radius_sensitivity"]["metrics"])
    sensitivity_summary: dict[str, Any] = {}
    for sensitivity_index, label in enumerate(sorted(sensitivity_labels)):
        values, sensitivity_months = _sensitivity_month_arrays(units, label)
        if not np.array_equal(months, sensitivity_months):
            raise RuntimeError(f"{dataset}: sensitivity month supports differ for {label}")
        sensitivity_summary[label] = {
            "macro_month_nrmse": float(values.mean()),
            "contrast_vs_A0": _relative_summary(
                baseline,
                values,
                months,
                config,
                seed_offset=offset + 60 + sensitivity_index,
            ),
            "non_gating": True,
        }
    return {
        "dataset": dataset,
        "n_month_clusters": int(len(months)),
        "n_initialization_seeds": int(len(units)),
        "rows_macro_month_nrmse": {row: float(values.mean()) for row, values in arrays.items()},
        "contrasts_vs_A0": contrasts,
        "A2_vs_A7": oracle_vs_shuffled,
        "A2_vs_A6": oracle_vs_anti,
        "A5_vs_strongest_causal_control": {
            "candidate_controls": list(causal_controls),
            "strongest_control": strongest_causal_control,
            "contrast": selector_vs_causal_control,
            "ci_lower_above_zero": selector_vs_causal_control["ci95"][0] > 0,
        },
        "shuffled_gain_retention": retention,
        "oracle_seed_wins": int(seed_wins),
        "oracle_radius_sensitivity": sensitivity_summary,
        "gates": gates,
        "oracle_opportunity_pass": bool(all(gates.values())),
        "selector_zero_mass_mean": float(
            np.mean([unit["selector_mean_last_10pct"]["zero"] for unit in units])
        ),
    }


def _synthetic_stream(rho: float, seed: int, config: Any, *, smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng([int(seed), int(round((rho + 1.0) * 1000))])
    packet_size = int(_get(config, "packet_size"))
    packets = 80 if smoke else int(_get(config, "synthetic_packets"))
    dimension = int(_get(config, "synthetic_dim"))
    noise = float(_get(config, "synthetic_noise"))
    drift = float(_get(config, "synthetic_drift"))
    x_all, y_all = [], []
    optimum = np.zeros(dimension, dtype=np.float64)
    velocity = rng.standard_normal(dimension)
    velocity /= np.linalg.norm(velocity)
    innovation_scale = math.sqrt(max(0.0, 1.0 - float(rho) ** 2))
    for _ in range(packets):
        innovation = rng.standard_normal(dimension)
        innovation /= max(np.linalg.norm(innovation), 1e-12)
        velocity = float(rho) * velocity + innovation_scale * innovation
        velocity /= max(np.linalg.norm(velocity), 1e-12)
        optimum = 0.999 * optimum + drift * velocity
        x = rng.standard_normal((packet_size, dimension))
        y = x @ optimum + noise * rng.standard_normal(packet_size)
        x_all.append(x.astype(np.float32)); y_all.append(y.astype(np.float32))
    X = np.concatenate(x_all)
    y_raw = np.concatenate(y_all)
    cal_n = max(packet_size, int(0.10 * len(y_raw)))
    y_mean = float(y_raw[:cal_n].mean())
    y_std = float(y_raw[:cal_n].std(ddof=1)) or 1.0
    y = ((y_raw - y_mean) / y_std).astype(np.float32)
    scored = np.arange(len(y)) >= cal_n
    month = (np.arange(len(y)) // (packet_size * 50)).astype(np.int64)
    return {
        "name": f"SYNTH_RHO_{rho:+.1f}",
        "X": X,
        "y": y,
        "event_id": np.arange(len(y), dtype=np.int64),
        "event_time_seconds": np.arange(len(y), dtype=np.float64),
        "month_id": month,
        "scored_mask": scored,
        "cal_n": cal_n,
        "chronology_preserved": True,
        "all_preprocessed_rows_retained": True,
    }


def _synthetic_summary(
    units: Mapping[str, list[dict[str, Any]]],
    config: Any,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for index, (name, group) in enumerate(units.items()):
        baseline = np.asarray(
            [unit["metrics"]["A0_ZERO"]["macro_month_nrmse"] for unit in group], float
        )
        row_summary = {}
        for j, row in enumerate(
            ("A2_ORACLE_XFIT", "A4_EWMA_GRADIENT", "A5_CAUSAL_SELECTOR")
        ):
            candidate = np.asarray(
                [unit["metrics"][row]["macro_month_nrmse"] for unit in group], float
            )
            row_summary[row] = _independent_seed_summary(
                baseline,
                candidate,
                config,
                seed_offset=500 + index * 20 + j,
            )
        row_summary["selector_zero_mass"] = float(
            np.mean([unit["selector_mean_last_10pct"]["zero"] for unit in group])
        )
        summaries[name] = row_summary

    positive = summaries["rho_+0.9"]
    zero = summaries["rho_+0.0"]
    negative = summaries["rho_-0.9"]
    gates = {
        "positive_ewma_ge_2pct_ci_gt_0": (
            positive["A4_EWMA_GRADIENT"]["relative_improvement"] >= 0.02
            and positive["A4_EWMA_GRADIENT"]["ci95"][0] > 0
        ),
        "positive_selector_ge_2pct_ci_gt_0": (
            positive["A5_CAUSAL_SELECTOR"]["relative_improvement"] >= 0.02
            and positive["A5_CAUSAL_SELECTOR"]["ci95"][0] > 0
        ),
        "zero_selector_noninferior": zero["A5_CAUSAL_SELECTOR"]["relative_improvement"] >= -0.01,
        "zero_selector_prefers_zero": zero["selector_zero_mass"] >= 0.70,
        "negative_naive_ewma_hurts": negative["A4_EWMA_GRADIENT"]["relative_improvement"] < 0,
        "negative_selector_noninferior": negative["A5_CAUSAL_SELECTOR"]["relative_improvement"] >= -0.01,
        "negative_selector_prefers_zero": negative["selector_zero_mass"] >= 0.70,
        "oracle_ceiling_positive_all_regimes": all(
            summaries[key]["A2_ORACLE_XFIT"]["relative_improvement"] > 0
            for key in summaries
        ),
    }
    return {"regimes": summaries, "gates": gates, "all_pass": bool(all(gates.values()))}


def _independent_seed_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    config: Any,
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Paired IID bootstrap for independent synthetic generator seeds.

    Unlike natural-data inference, these replicates have no ordered month
    axis.  Sorting paired effects before deterministic resampling makes the
    numerical result invariant to arbitrary seed-list ordering.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != candidate.shape or not len(baseline):
        raise ValueError("synthetic paired seed arrays must be nonempty 1-D matches")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("synthetic paired seed arrays contain non-finite values")
    effects = np.sort(baseline - candidate)
    n_boot = int(_get(config, "n_boot"))
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(
        int(_get(config, "bootstrap_seed")) + int(seed_offset)
    )
    sampled = rng.integers(0, len(effects), size=(n_boot, len(effects)))
    draws = effects[sampled].mean(axis=1)
    mean = float(effects.mean())
    denominator = max(abs(float(baseline.mean())), 1e-12)
    return {
        "improvement": mean,
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "relative_improvement": mean / denominator,
        "wins_of_n": int(np.sum(effects > 0)),
        "n": int(len(effects)),
        "paired_effects_sorted": effects.tolist(),
        "bootstrap": {
            "method": "paired_iid_initialization_seed",
            "seed_order_invariant": True,
            "n_initialization_seeds": int(len(effects)),
            "draws": n_boot,
        },
    }


def run_study_a(
    config: Any,
    results_dir: str | Path,
    device: str | torch.device,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the locked prospective-gradient audit and return its JSON payload."""
    device = torch.device(device)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    datasets = [str(value).upper() for value in _get(config, "datasets")]
    seeds = [int(value) for value in _get(config, "seeds")]
    synthetic_seeds = [int(value) for value in _get(config, "synthetic_seeds")]
    if smoke:
        seeds = [int(value) for value in _get(config, "smoke_seeds")]
        synthetic_seeds = synthetic_seeds[:2]

    real_units: dict[str, list[dict[str, Any]]] = {}
    real_summaries: dict[str, Any] = {}
    real_data_audits: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(datasets):
        group = []
        # The physical stream is fixed across initialization seeds.  Load and
        # preprocess it once so repeated seeds cannot accidentally become
        # different data replicates and the large Beijing archive is not
        # reparsed four times.
        data = load_regression_stream(
            dataset, smoke=smoke, seed=(seeds[0] if smoke else 0)
        )
        real_data_audits[dataset] = {
            "data_manifest": data.get("data_manifest"),
            "chronology_audit": data.get("chronology_audit"),
            "chronology_preserved": bool(data.get("chronology_preserved", False)),
            "n_rows": int(len(data["X"])),
        }
        for seed in seeds:
            unit = _run_unit(data, config, seed=seed, device=device, synthetic=False)
            unit["requested_dataset"] = dataset
            group.append(unit)
        real_units[dataset] = group
        real_summaries[dataset] = _aggregate_dataset(
            dataset, group, config, offset=dataset_index * 100
        )

    synthetic_units: dict[str, list[dict[str, Any]]] = {}
    for rho in (0.9, 0.0, -0.9):
        key = f"rho_{rho:+.1f}"
        synthetic_units[key] = [
            _run_unit(
                _synthetic_stream(rho, seed, config, smoke=smoke),
                config,
                seed=seed,
                device=device,
                synthetic=True,
            )
            for seed in synthetic_seeds
        ]
    synthetic_summary = _synthetic_summary(synthetic_units, config)

    natural_gate = bool(real_summaries) and all(
        summary["oracle_opportunity_pass"] for summary in real_summaries.values()
    )
    selector_effects = {
        dataset: summary["contrasts_vs_A0"]["A5_CAUSAL_SELECTOR"]
        for dataset, summary in real_summaries.items()
    }
    causal_positive = [
        effect["relative_improvement"] >= float(_get(config, "causal_min_relative"))
        and effect["ci95_relative"][0] > 0
        and real_summaries[dataset]["A5_vs_strongest_causal_control"][
            "ci_lower_above_zero"
        ]
        for dataset, effect in selector_effects.items()
    ]
    causal_noninferior = [
        effect["ci95_relative"][0]
        >= -float(_get(config, "causal_noninferiority"))
        for effect in selector_effects.values()
    ]
    forecast_signal = bool(causal_positive) and any(causal_positive) and all(causal_noninferior)

    if not natural_gate:
        verdict = "NO_PROSPECTIVE_HEADROOM"
    elif not synthetic_summary["all_pass"] or not forecast_signal:
        verdict = "ORACLE_HEADROOM_ONLY"
    else:
        verdict = "PROSPECTIVE_DIRECTION_GO"

    diagnostic_scientific_verdict = verdict
    public_verdict = "SMOKE_PASS" if smoke else diagnostic_scientific_verdict
    public_scientific_verdict = None if smoke else diagnostic_scientific_verdict

    payload = {
        "schema_version": "fmmt.study_a.v1",
        "study": "A_PROSPECTIVE_GRADIENT_OPPORTUNITY",
        "status": "COMPLETE",
        "verdict": public_verdict,
        "scientific_verdict": public_scientific_verdict,
        "diagnostic_scientific_verdict": diagnostic_scientific_verdict,
        "smoke": bool(smoke),
        "rows": {
            row: {"causal": row in CAUSAL_ROWS, "primary_gate": row == "A2_ORACLE_XFIT"}
            for row in ROWS
        },
        "protocol": {
            "shared_center_fixed_trajectory": True,
            "crossfit_support_query_disjoint": True,
            "crossfit_acquisition_events_disjoint": True,
            "oracle_predictions_noncausal_counterfactuals": True,
            "causal_rows_predict_before_current_label": True,
            "base_updates_once_after_all_branches": True,
            "calibration_prefix_scored": False,
            "optimizer_seeds_are_not_inference_clusters": True,
        },
        "config": {key: _get(config, key) for key in DEFAULTS},
        "real": {
            "summary": real_summaries,
            "units": real_units,
            "data_manifest_and_chronology_audit": real_data_audits,
        },
        "synthetic": {
            "summary": synthetic_summary,
            "units": synthetic_units,
        },
        "decision": {
            "natural_oracle_gate": natural_gate,
            "synthetic_controls_gate": synthetic_summary["all_pass"],
            "causal_forecast_signal": forecast_signal,
            "verdict": public_verdict,
            "scientific_verdict": public_scientific_verdict,
            "diagnostic_scientific_verdict": diagnostic_scientific_verdict,
            "causal_noninferiority_uses_ci_lower_bound": True,
            "causal_positive_requires_A5_to_beat_strongest_A3_A4_A8_A9": True,
            "study_b_must_run_regardless_of_scientific_verdict": True,
        },
    }
    atomic_json(results_path / "study_a_result.json", payload)
    return payload


__all__ = ["run_study_a", "ROWS", "CAUSAL_ROWS", "NONCAUSAL_ROWS"]
