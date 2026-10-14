"""Final-example capture and task-relative metrics.

The module deliberately separates the two terms in

    -log p(y) = -log p(S_y) - log p(y | S_y),

where ``S_y`` is the five-class Split CIFAR-100 task containing ``y``.
Run seeds, rather than examples or tasks, remain the inferential units.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "gradient_reuse.task_relative_unit_metrics.v1"
RAW_SCHEMA = "gradient_reuse.task_relative_raw_logits.v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _logsumexp(value: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(value - maximum), axis=axis)
    )


def _finite_mean(value: np.ndarray) -> float:
    result = float(np.mean(value, dtype=np.float64))
    if not math.isfinite(result):
        raise RuntimeError("task-relative metric is non-finite")
    return result


def _confusion(
    true_task: np.ndarray, predicted_task: np.ndarray, task_count: int
) -> list[list[int]]:
    matrix = np.zeros((task_count, task_count), dtype=np.int64)
    np.add.at(matrix, (true_task, predicted_task), 1)
    return matrix.tolist()


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
    *,
    classes_per_task: int = 5,
    class_count: int = 100,
    input_stream_sha256: str,
) -> dict[str, Any]:
    """Compute exact example-level task/all-class diagnostics."""

    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    row_index = np.asarray(indexes, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape != (len(target), class_count):
        raise RuntimeError("task-relative logits have the wrong shape")
    if len(target) != len(row_index) or len(target) < 1:
        raise RuntimeError("task-relative labels/indexes are empty or misaligned")
    if len(np.unique(row_index)) != len(row_index):
        raise RuntimeError("task-relative held-out indexes are not unique")
    if not np.isfinite(scores).all():
        raise RuntimeError("task-relative logits contain non-finite values")
    if np.any((target < 0) | (target >= class_count)):
        raise RuntimeError("task-relative labels are out of range")
    if class_count % classes_per_task:
        raise RuntimeError("class count is not divisible by classes per task")
    if not (
        isinstance(input_stream_sha256, str)
        and len(input_stream_sha256) == 64
        and all(char in "0123456789abcdef" for char in input_stream_sha256)
    ):
        raise RuntimeError("invalid input-stream digest")

    task_count = class_count // classes_per_task
    true_task = target // classes_per_task
    local_target = target % classes_per_task
    task_logits = scores.reshape(len(target), task_count, classes_per_task)
    task_log_numerator = _logsumexp(task_logits, axis=2)
    global_log_denominator = _logsumexp(scores, axis=1)
    row = np.arange(len(target))
    correct_logits = scores[row, target]
    correct_task_log_numerator = task_log_numerator[row, true_task]

    all_class_ce = global_log_denominator - correct_logits
    task_mass_ce = global_log_denominator - correct_task_log_numerator
    within_task_ce = correct_task_log_numerator - correct_logits
    correct_task_log_probability_mass = (
        correct_task_log_numerator - global_log_denominator
    )
    # ``-expm1(log p(S_y))`` avoids cancellation when the correct task already
    # receives nearly all probability mass.  Computing the mass from log-sum-exp
    # terms also avoids materializing the full softmax.
    probability_mass_outside_correct_task = -np.expm1(
        correct_task_log_probability_mass
    )
    if np.any(
        (probability_mass_outside_correct_task < -1e-15)
        | (probability_mass_outside_correct_task > 1.0 + 1e-15)
    ):
        raise RuntimeError("correct-task probability mass is outside [0,1]")
    probability_mass_outside_correct_task = np.clip(
        probability_mass_outside_correct_task, 0.0, 1.0
    )
    residual = all_class_ce - task_mass_ce - within_task_ce
    max_abs_residual = float(np.max(np.abs(residual)))
    if max_abs_residual > 5e-12:
        raise RuntimeError("cross-entropy task decomposition lost numerical identity")

    all_prediction = np.argmax(scores, axis=1)
    class_prediction_task = all_prediction // classes_per_task
    task_mass_prediction = np.argmax(task_log_numerator, axis=1)
    within_logits = task_logits[row, true_task]
    within_prediction = (
        true_task * classes_per_task + np.argmax(within_logits, axis=1)
    )

    within_competitor = within_logits.copy()
    within_competitor[row, local_target] = -np.inf
    within_margin = correct_logits - np.max(within_competitor, axis=1)
    all_competitor = scores.copy()
    all_competitor[row, target] = -np.inf
    all_margin = correct_logits - np.max(all_competitor, axis=1)
    task_competitor = task_log_numerator.copy()
    task_competitor[row, true_task] = -np.inf
    task_mass_margin = correct_task_log_numerator - np.max(task_competitor, axis=1)

    per_task: list[dict[str, Any]] = []
    for task in range(task_count):
        mask = true_task == task
        if not mask.any():
            raise RuntimeError(f"held-out task {task} has no examples")
        per_task.append(
            {
                "task": task,
                "examples": int(mask.sum()),
                "all_class_accuracy_percent": 100.0 * _finite_mean(all_prediction[mask] == target[mask]),
                "task_aware_accuracy_percent": 100.0 * _finite_mean(within_prediction[mask] == target[mask]),
                "task_mass_accuracy_percent": 100.0 * _finite_mean(task_mass_prediction[mask] == task),
                "class_prediction_task_accuracy_percent": 100.0 * _finite_mean(class_prediction_task[mask] == task),
                "all_class_cross_entropy_nats": _finite_mean(all_class_ce[mask]),
                "task_mass_cross_entropy_nats": _finite_mean(task_mass_ce[mask]),
                "within_task_cross_entropy_nats": _finite_mean(within_task_ce[mask]),
                "probability_mass_outside_correct_task_percent": 100.0 * _finite_mean(
                    probability_mass_outside_correct_task[mask]
                ),
                "all_class_margin_logits": _finite_mean(all_margin[mask]),
                "within_task_margin_logits": _finite_mean(within_margin[mask]),
                "task_mass_margin_log_probability": _finite_mean(task_mass_margin[mask]),
            }
        )

    identity = np.column_stack((row_index, target)).astype(np.int64, copy=False)
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "examples": len(target),
        "class_count": class_count,
        "classes_per_task": classes_per_task,
        "task_count": task_count,
        "example_identity_sha256": array_sha256(identity),
        "indexes_sha256": array_sha256(row_index),
        "labels_sha256": array_sha256(target),
        "logits_sha256": array_sha256(np.asarray(logits)),
        "input_stream_sha256": input_stream_sha256,
        "cross_entropy_identity_max_abs_residual": max_abs_residual,
        "overall": {
            "all_class_accuracy_percent": 100.0 * _finite_mean(all_prediction == target),
            "task_aware_accuracy_percent": 100.0 * _finite_mean(within_prediction == target),
            "task_mass_accuracy_percent": 100.0 * _finite_mean(task_mass_prediction == true_task),
            "class_prediction_task_accuracy_percent": 100.0 * _finite_mean(class_prediction_task == true_task),
            "all_class_cross_entropy_nats": _finite_mean(all_class_ce),
            "task_mass_cross_entropy_nats": _finite_mean(task_mass_ce),
            "within_task_cross_entropy_nats": _finite_mean(within_task_ce),
            "probability_mass_outside_correct_task_percent": 100.0 * _finite_mean(
                probability_mass_outside_correct_task
            ),
            "all_class_margin_logits": _finite_mean(all_margin),
            "within_task_margin_logits": _finite_mean(within_margin),
            "task_mass_margin_log_probability": _finite_mean(task_mass_margin),
            "within_task_positive_margin_percent": 100.0 * _finite_mean(within_margin > 0.0),
        },
        "task_confusion_by_probability_mass_counts": _confusion(
            true_task, task_mass_prediction, task_count
        ),
        "task_confusion_by_top_class_counts": _confusion(
            true_task, class_prediction_task, task_count
        ),
        "per_task": per_task,
        "definitions": {
            "all_class_prediction": "argmax over all 100 logits",
            "task_aware_prediction": "argmax over the five logits belonging to the true task",
            "task_mass_prediction": "argmax over each task's summed softmax probability mass",
            "task_mass_cross_entropy": "-log sum_{c in true task} p(c|x)",
            "within_task_cross_entropy": "-log p(y|x,true task)",
            "probability_mass_outside_correct_task": "100 times the example mean of 1 - sum_{c in true task} p(c|x)",
            "decomposition": "all_class_cross_entropy = task_mass_cross_entropy + within_task_cross_entropy per example",
            "within_task_margin": "true-class logit minus largest other logit in the true task",
            "task_mass_margin": "log probability mass of true task minus largest competing-task log mass",
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe task-relative JSON destination: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe task-relative array destination: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_final_examples(
    model: Any,
    dataset: Any,
    *,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Evaluate one trained state on all 10,000 deterministic test examples."""

    import torch

    if len(dataset.test_loaders) != 20:
        raise RuntimeError("final task-relative capture requires all twenty test loaders")
    was_training = bool(model.training)
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indexes: list[np.ndarray] = []
    input_digest = hashlib.sha256()
    try:
        model.eval()
        with torch.inference_mode():
            for task, loader in enumerate(dataset.test_loaders):
                indexes = np.asarray(loader.dataset.indexes, dtype=np.int64).reshape(-1)
                cursor = 0
                for batch in loader:
                    inputs = batch[0].to(model.device)
                    labels = batch[1].to(model.device, dtype=torch.long).reshape(-1)
                    count = len(labels)
                    batch_indexes = indexes[cursor : cursor + count]
                    cursor += count
                    if len(batch_indexes) != count:
                        raise RuntimeError("test index stream and loader batch disagree")
                    if bool(((labels // 5) != task).any().item()):
                        raise RuntimeError("held-out label does not belong to its test-loader task")
                    outputs = model(inputs)
                    if not torch.is_tensor(outputs) or outputs.ndim != 2 or outputs.shape != (count, 100):
                        raise RuntimeError("model did not emit one hundred logits per example")
                    cpu_inputs = np.ascontiguousarray(inputs.detach().cpu().numpy())
                    input_digest.update(
                        json.dumps(
                            {"dtype": str(cpu_inputs.dtype), "shape": list(cpu_inputs.shape)},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\0"
                        + cpu_inputs.tobytes()
                    )
                    all_logits.append(np.ascontiguousarray(outputs.detach().cpu().float().numpy()))
                    all_labels.append(np.ascontiguousarray(labels.detach().cpu().numpy(), dtype=np.int64))
                    all_indexes.append(np.ascontiguousarray(batch_indexes, dtype=np.int64))
                if cursor != len(indexes):
                    raise RuntimeError("test loader did not consume its complete index stream")
    finally:
        model.train(was_training)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indexes = np.concatenate(all_indexes, axis=0)
    if len(labels) != 10_000:
        raise RuntimeError(
            f"terminal Split CIFAR-100 capture has {len(labels)} examples, expected 10000"
        )
    summary = compute_metrics(
        logits,
        labels,
        indexes,
        input_stream_sha256=input_digest.hexdigest(),
    )
    _atomic_npz(
        raw_path,
        schema_version=np.asarray(RAW_SCHEMA),
        logits=logits,
        labels=labels,
        indexes=indexes,
    )
    summary = {**summary, "raw_logits_file": raw_path.name}
    _atomic_json(summary_path, summary)
    return summary


def install_final_capture(output_directory: Path) -> None:
    """Patch only the terminal evaluation call; training is left unchanged."""

    from datasets.utils.continual_dataset import ContinualDataset

    raw_descriptor = ContinualDataset.__dict__.get("evaluate")
    if not isinstance(raw_descriptor, staticmethod):
        raise RuntimeError("pinned Mammoth ContinualDataset.evaluate descriptor changed")
    original = raw_descriptor.__func__
    completed = False
    destination = Path(output_directory).resolve()

    def evaluate(model: Any, dataset: Any, last: bool = False, return_loss: bool = False):
        nonlocal completed
        result = original(model, dataset, last=last, return_loss=return_loss)
        if not last and not return_loss and len(dataset.test_loaders) == 20:
            if completed:
                raise RuntimeError("terminal task-relative capture was requested more than once")
            summary = capture_final_examples(
                model,
                dataset,
                raw_path=destination / "FINAL_LOGITS.npz",
                summary_path=destination / "FINAL_TASK_RELATIVE_METRICS.json",
            )
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise RuntimeError("Mammoth final evaluation returned an unknown payload")
            expected_all = [float(value) for value in result[0]]
            expected_task = [float(value) for value in result[1]]
            observed_all = [row["all_class_accuracy_percent"] for row in summary["per_task"]]
            observed_task = [row["task_aware_accuracy_percent"] for row in summary["per_task"]]
            if len(expected_all) != 20 or len(expected_task) != 20 or not (
                np.allclose(expected_all, observed_all, rtol=0.0, atol=1e-12)
                and np.allclose(expected_task, observed_task, rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError("captured logits disagree with Mammoth's final accuracies")
            _atomic_json(
                destination / "MAMMOTH_ENDPOINT_CROSSCHECK.json",
                {
                    "schema_version": "gradient_reuse.task_relative_mammoth_crosscheck.v1",
                    "status": "PASS",
                    "all_class_per_task_accuracy_percent": expected_all,
                    "task_aware_per_task_accuracy_percent": expected_task,
                    "maximum_absolute_accuracy_difference": max(
                        [abs(a - b) for a, b in zip(expected_all, observed_all)]
                        + [abs(a - b) for a, b in zip(expected_task, observed_task)]
                    ),
                },
            )
            completed = True
        return result

    ContinualDataset.evaluate = staticmethod(evaluate)


__all__ = [
    "SCHEMA",
    "array_sha256",
    "canonical_sha256",
    "capture_final_examples",
    "compute_metrics",
    "install_final_capture",
]
