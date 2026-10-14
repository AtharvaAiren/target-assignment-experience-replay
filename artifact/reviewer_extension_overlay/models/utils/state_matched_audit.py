"""Pure utilities for read-only state-matched replay probes.

The probe evaluates normalized, one-step functional branches without updating
the live model, optimizer, buffer, module buffers, or random-number streams.
The paired and permuted directions are replay-only cross-entropy gradients;
the risk gradient is the mean cross-entropy gradient over the exact old-task
test union selected by the caller.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


EPSILON = 1.0e-12


def tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def tensor_multiset_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().reshape(len(value), -1).numpy())
    rows = sorted(hashlib.sha256(row.tobytes()).digest() for row in array)
    header = json.dumps(
        {"dtype": str(array.dtype), "row_shape": list(value.shape[1:])},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + b"".join(rows)).hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def deterministic_permutation(
    size: int, *, run_seed: int, boundary: int, replica: int
) -> torch.Tensor:
    size = int(size)
    if size < 2:
        raise RuntimeError("state-matched target permutation needs at least two rows")
    sequence = np.random.SeedSequence(
        [int(run_seed), 0x53544154, int(boundary), int(replica)]
    )
    permutation = np.random.default_rng(sequence).permutation(size).astype(np.int64)
    return torch.as_tensor(permutation, dtype=torch.long)


def _loader_generators(
    loaders: Sequence[Iterable[Any]],
) -> tuple[torch.Generator, ...]:
    """Return each distinct generator that loader iteration may advance.

    Mammoth gives every seeded DataLoader a private generator.  Restoring only
    global RNG state is therefore insufficient: constructing an iterator
    advances the loader generator used to seed its workers.  Samplers can own
    a second generator, so both locations are included and deduplicated.
    """

    generators: list[torch.Generator] = []
    seen: set[int] = set()
    for loader in loaders:
        if bool(getattr(loader, "persistent_workers", False)):
            raise RuntimeError(
                "state-matched probes forbid persistent DataLoader workers"
            )
        candidates = [
            loader,
            getattr(loader, "sampler", None),
            getattr(loader, "batch_sampler", None),
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
        ]
        for candidate in candidates:
            generator = getattr(candidate, "generator", None)
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                seen.add(id(generator))
                generators.append(generator)
    return tuple(generators)


def _gradient_snapshot(
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    )


def _gradient_sha256(gradients: Sequence[torch.Tensor | None]) -> str:
    payload = [
        None if gradient is None else tensor_sha256(gradient) for gradient in gradients
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _restore_gradients(
    model: torch.nn.Module, gradients: Sequence[torch.Tensor | None]
) -> None:
    parameters = tuple(model.parameters())
    if len(parameters) != len(gradients):
        raise RuntimeError("model parameter inventory changed during the probe")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.clone()


def _state_value(value: Any) -> Any:
    """Canonicalize optimizer/buffer state without serializing live objects."""

    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_sha256(value)}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "ndarray_dtype": str(array.dtype),
            "ndarray_shape": list(array.shape),
            "ndarray_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _state_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_state_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("live state contains a non-finite scalar")
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    encoded = json.dumps(
        _state_value(optimizer.state_dict()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def buffer_state_sha256(buffer: Any) -> str:
    """Fingerprint mutable replay contents, counters, and sampler metadata."""

    values: dict[str, Any] = {}
    for name, value in sorted(vars(buffer).items()):
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            values[name] = _state_value(value)
        elif (
            isinstance(value, (str, int, float, bool, list, tuple, dict))
            or value is None
        ):
            values[name] = _state_value(value)
        elif name == "sample_selection_fn" and hasattr(value, "__dict__"):
            values[name] = {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _state_value(vars(value)),
            }
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def preserve_live_state(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    buffer: Any,
    loaders: Sequence[Iterable[Any]],
):
    """Restore every live state source that a discarded probe can touch."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    modes = tuple((module, bool(module.training)) for module in model.modules())
    generators = _loader_generators(loaders)
    generator_states = tuple(generator.get_state().clone() for generator in generators)
    gradients = _gradient_snapshot(model)
    before = {
        "model": model_state_sha256(model),
        "optimizer": optimizer_state_sha256(optimizer),
        "buffer": buffer_state_sha256(buffer),
        "gradients": _gradient_sha256(gradients),
        "loader_generators": [tensor_sha256(state) for state in generator_states],
    }
    try:
        yield dict(before)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for generator, state in zip(generators, generator_states):
            generator.set_state(state)
        _restore_gradients(model, gradients)
        for module, training in modes:
            module.training = training
        after = {
            "model": model_state_sha256(model),
            "optimizer": optimizer_state_sha256(optimizer),
            "buffer": buffer_state_sha256(buffer),
            "gradients": _gradient_sha256(_gradient_snapshot(model)),
            "loader_generators": [
                tensor_sha256(generator.get_state()) for generator in generators
            ],
        }
        if after != before:
            changed = sorted(name for name in before if before[name] != after[name])
            raise RuntimeError(f"state-matched probe mutated live state: {changed}")


def trainable_parameters(
    model: torch.nn.Module,
) -> OrderedDict[str, torch.nn.Parameter]:
    parameters = OrderedDict(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    if not parameters:
        raise RuntimeError("state-matched probe found no trainable parameters")
    return parameters


def flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    pieces = [
        (
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
        for gradient, parameter in zip(gradients, parameters)
    ]
    if not pieces:
        raise RuntimeError("cannot flatten an empty gradient")
    return torch.cat(pieces)


def replay_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    parameters = trainable_parameters(model)
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    # Some pinned Mammoth backbones register the same module through several
    # attribute paths.  In particular, BaseMNISTMLP exposes its Linear layers
    # directly and again through ``_features`` and ``net``.  PyTorch's default
    # tied-weight swapping can leave such an aliased live module holding the
    # functional replacement after the call returns.  The canonical parameter
    # mapping already addresses the shared module object, so disabling the
    # tied-name expansion both preserves the intended branch and keeps the live
    # trajectory byte-identical.
    logits = torch.func.functional_call(
        model, (parameters, buffers), (inputs,), tie_weights=False
    )
    loss = torch.nn.functional.cross_entropy(logits, targets.long().reshape(-1))
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters.values()),
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    output = flatten_gradients(gradients, tuple(parameters.values()))
    if not torch.isfinite(output).all():
        raise RuntimeError("replay branch produced a non-finite gradient")
    return output


def normalize_gradient(value: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    denominator = torch.clamp(norm, min=float(epsilon))
    result = value / denominator
    if not torch.isfinite(result).all():
        raise RuntimeError("normalized replay direction is non-finite")
    return result


def branch_parameters(
    parameters: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    radius: float,
) -> OrderedDict[str, torch.Tensor]:
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("state-matched radius must be finite and positive")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for name, parameter in parameters.items():
        count = parameter.numel()
        delta = direction[offset : offset + count].reshape_as(parameter)
        output[name] = parameter - radius * delta
        offset += count
    if offset != int(direction.numel()):
        raise RuntimeError("branch direction has the wrong parameter dimension")
    return output


def _batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise RuntimeError("old-task loader did not return inputs and labels")
    return batch[0].to(device), batch[1].long().reshape(-1).to(device)


def evaluate_old_task_branches(
    model: torch.nn.Module,
    loaders: Sequence[Iterable[Any]],
    directions: Mapping[str, torch.Tensor],
    *,
    radius: float,
) -> dict[str, Any]:
    """Evaluate no-step and normalized replay-gradient branches exactly."""

    if not loaders:
        raise RuntimeError("state-matched probe requires at least one old-task loader")
    expected_names = {
        "paired",
        "permuted_0",
        "permuted_1",
        "permuted_2",
        "permuted_3",
    }
    if set(directions) != expected_names:
        raise RuntimeError("state-matched branch set differs from the lock")
    parameters = trainable_parameters(model)
    parameter_values = tuple(parameters.values())
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    normalized = {
        name: normalize_gradient(direction) for name, direction in directions.items()
    }
    branches = {
        name: branch_parameters(parameters, direction, radius)
        for name, direction in normalized.items()
    }
    totals = {
        "no_step": {"loss": 0.0, "correct": 0},
        **{name: {"loss": 0.0, "correct": 0} for name in sorted(branches)},
    }
    risk_gradient = [
        torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for parameter in parameter_values
    ]
    examples = 0
    device = next(iter(parameter_values)).device
    model.eval()
    for loader in loaders:
        for raw_batch in loader:
            inputs, targets = _batch(raw_batch, device)
            batch_size = int(targets.numel())
            if batch_size < 1:
                continue
            logits = torch.func.functional_call(
                model, (parameters, buffers), (inputs,), tie_weights=False
            )
            loss_sum = torch.nn.functional.cross_entropy(
                logits, targets, reduction="sum"
            )
            gradients = torch.autograd.grad(
                loss_sum,
                parameter_values,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            for accumulator, gradient in zip(risk_gradient, gradients):
                if gradient is not None:
                    accumulator.add_(gradient.detach())
            totals["no_step"]["loss"] += float(loss_sum.detach().item())
            totals["no_step"]["correct"] += int(
                (logits.detach().argmax(dim=1) == targets).sum().item()
            )
            with torch.no_grad():
                for name, branch in branches.items():
                    branch_logits = torch.func.functional_call(
                        model,
                        (branch, buffers),
                        (inputs,),
                        tie_weights=False,
                    )
                    totals[name]["loss"] += float(
                        torch.nn.functional.cross_entropy(
                            branch_logits, targets, reduction="sum"
                        ).item()
                    )
                    totals[name]["correct"] += int(
                        (branch_logits.argmax(dim=1) == targets).sum().item()
                    )
            examples += batch_size
    if examples < 1:
        raise RuntimeError("old-task union contains no examples")
    h = flatten_gradients(risk_gradient, parameter_values) / float(examples)
    if not torch.isfinite(h).all():
        raise RuntimeError("old-task risk gradient is non-finite")
    metrics = {
        name: {
            "cross_entropy": float(value["loss"]) / examples,
            "accuracy": 100.0 * float(value["correct"]) / examples,
        }
        for name, value in totals.items()
    }
    permutation_names = [f"permuted_{index}" for index in range(4)]
    paired_ce_value = (
        metrics["no_step"]["cross_entropy"] - metrics["paired"]["cross_entropy"]
    )
    permuted_ce_values = [
        metrics["no_step"]["cross_entropy"] - metrics[name]["cross_entropy"]
        for name in permutation_names
    ]
    paired_accuracy_value = (
        metrics["paired"]["accuracy"] - metrics["no_step"]["accuracy"]
    )
    permuted_accuracy_values = [
        metrics[name]["accuracy"] - metrics["no_step"]["accuracy"]
        for name in permutation_names
    ]
    mean_permuted_ce = sum(permuted_ce_values) / 4.0
    mean_permuted_accuracy = sum(permuted_accuracy_values) / 4.0
    mean_permuted_direction = torch.stack(
        [normalized[name] for name in permutation_names]
    ).mean(dim=0)
    direction_difference = normalized["paired"] - mean_permuted_direction
    first_order = float(radius) * float(
        torch.dot(h.double(), direction_difference.double()).item()
    )
    return {
        "examples": examples,
        "radius": float(radius),
        "branches": metrics,
        "cross_entropy": {
            "V_total": paired_ce_value,
            "V_perm": mean_permuted_ce,
            "V_pair": paired_ce_value - mean_permuted_ce,
            "identity_residual": (
                paired_ce_value - mean_permuted_ce + mean_permuted_ce - paired_ce_value
            ),
        },
        "accuracy": {
            "V_total": paired_accuracy_value,
            "V_perm": mean_permuted_accuracy,
            "V_pair": paired_accuracy_value - mean_permuted_accuracy,
            "identity_residual": (
                paired_accuracy_value
                - mean_permuted_accuracy
                + mean_permuted_accuracy
                - paired_accuracy_value
            ),
        },
        "first_order": {
            "risk_gradient_norm": float(torch.linalg.vector_norm(h).item()),
            "paired_direction_norm": float(
                torch.linalg.vector_norm(normalized["paired"]).item()
            ),
            "mean_permuted_direction_norm": float(
                torch.linalg.vector_norm(mean_permuted_direction).item()
            ),
            "radius_scaled_pairing_term": first_order,
        },
    }


@torch.no_grad()
def final_confusion_diagnostic(
    model: torch.nn.Module,
    old_loaders: Sequence[Iterable[Any]],
    newest_loader: Iterable[Any],
    *,
    setting: str,
) -> dict[str, Any]:
    """Return the prespecified final classifier/logit diagnostic."""

    if not old_loaders:
        raise RuntimeError("final confusion diagnostic has no old tasks")
    device = next(model.parameters()).device
    newest_labels: set[int] = set()
    for raw_batch in newest_loader:
        _, targets = _batch(raw_batch, device)
        newest_labels.update(int(value) for value in targets.detach().cpu().tolist())
    if not newest_labels:
        raise RuntimeError("newest task has no labels")

    total = 0
    correct = 0
    correct_logits = 0.0
    incorrect_logits = 0.0
    predictive_entropy = 0.0
    incorrect_prediction_counts: dict[int, int] = {}
    other_old = 0
    predicted_newest = 0
    predicted_other = 0
    other_old_logits = 0.0
    newest_logits = 0.0
    other_old_logit_terms = 0
    newest_logit_terms = 0
    observed_old_labels: set[int] = set()
    model.eval()
    # First inventory the old label support.  Caching the decoded inputs would
    # retain the full old-task union on the accelerator; the surrounding
    # live-state guard restores loader/sampler generators after both passes.
    for loader in old_loaders:
        for raw_batch in loader:
            _, targets = _batch(raw_batch, device)
            observed_old_labels.update(int(value) for value in targets.cpu().tolist())
    if not observed_old_labels:
        raise RuntimeError("old-task union has no labels")
    class_count: int | None = None
    for loader in old_loaders:
        for raw_batch in loader:
            inputs, targets = _batch(raw_batch, device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            rows = torch.arange(len(targets), device=device)
            correct_logits += float(logits[rows, targets].sum().item())
            total += int(len(targets))
            correct_mask = predictions == targets
            correct += int(correct_mask.sum().item())
            predictive_entropy += float(
                (
                    -(probabilities * torch.log(probabilities.clamp_min(1.0e-30))).sum(
                        dim=1
                    )
                )
                .sum()
                .item()
            )
            for predicted in predictions[~correct_mask].detach().cpu().tolist():
                key = int(predicted)
                incorrect_prediction_counts[key] = (
                    incorrect_prediction_counts.get(key, 0) + 1
                )
            batch_class_count = int(logits.shape[1])
            if class_count is None:
                class_count = batch_class_count
            elif class_count != batch_class_count:
                raise RuntimeError("classifier width changed during confusion audit")
            all_classes = set(range(batch_class_count))
            if setting == "class-il":
                other_old_classes = sorted(observed_old_labels)
                newest_classes = sorted(newest_labels)
                old_index = torch.as_tensor(other_old_classes, device=device)
                new_index = torch.as_tensor(newest_classes, device=device)
                old_mask = torch.zeros(
                    (len(targets), batch_class_count),
                    dtype=torch.bool,
                    device=device,
                )
                old_mask[:, old_index] = True
                old_mask[rows, targets] = False
                other_old_logits += float(logits[old_mask].sum().item())
                other_old_logit_terms += int(old_mask.sum().item())
                newest_logits += float(logits[:, new_index].sum().item())
                newest_logit_terms += len(newest_classes) * len(targets)
                for target, prediction in zip(
                    targets.detach().cpu().tolist(),
                    predictions.detach().cpu().tolist(),
                ):
                    if prediction == target:
                        continue
                    if int(prediction) in observed_old_labels:
                        other_old += 1
                    elif int(prediction) in newest_labels:
                        predicted_newest += 1
                    elif int(prediction) in all_classes:
                        predicted_other += 1
            incorrect_mask = torch.ones_like(logits, dtype=torch.bool)
            incorrect_mask[rows, targets] = False
            incorrect_logits += float(logits[incorrect_mask].sum().item())
    if total < 1:
        raise RuntimeError("final confusion diagnostic observed no examples")
    if class_count is None or class_count < 2:
        raise RuntimeError("final confusion diagnostic has an invalid classifier")
    # Every example contributes C-1 incorrect-class logit entries.
    incorrect_logit_terms = total * (class_count - 1)
    errors = sum(incorrect_prediction_counts.values())
    error_entropy = 0.0
    if errors:
        for count in incorrect_prediction_counts.values():
            probability = count / errors
            error_entropy -= probability * math.log(probability)
    common = {
        "examples": total,
        "accuracy": 100.0 * correct / total,
        "mean_correct_label_logit": correct_logits / total,
        "mean_incorrect_label_logit": incorrect_logits / incorrect_logit_terms,
        "mean_predictive_entropy": predictive_entropy / total,
        "incorrect_prediction_label_entropy": error_entropy,
    }
    if setting == "domain-il":
        return {"schema_version": "gradient_reuse.pmnist_confusion.v1", **common}
    if setting != "class-il":
        raise RuntimeError(f"unsupported confusion setting: {setting}")
    return {
        "schema_version": "gradient_reuse.cifar_confusion.v1",
        **common,
        "mean_correct_old_logit": common["mean_correct_label_logit"],
        "correct_old_class_fraction": correct / total,
        "other_old_class_fraction": other_old / total,
        "newest_task_class_fraction": predicted_newest / total,
        "other_unseen_class_fraction": predicted_other / total,
        "mean_other_old_logit": (
            other_old_logits / other_old_logit_terms if other_old_logit_terms else None
        ),
        "mean_newest_task_logit": (
            newest_logits / newest_logit_terms if newest_logit_terms else None
        ),
        "partition_residual": (correct + other_old + predicted_newest + predicted_other)
        / total
        - 1.0,
    }
