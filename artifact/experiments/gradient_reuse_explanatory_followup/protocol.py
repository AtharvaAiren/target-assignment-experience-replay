"""Locked identities and Mammoth log parsing for the explanatory follow-up."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "gradient_reuse.explanatory_followup_lock.v1"
STATUS = "OUTCOME_INFORMED_FOLLOWUP_LOCKED_BEFORE_EXPLANATORY_FOLLOWUP_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
SHARED_SEEDS = tuple(range(3601, 3607))
VARIANCE_SEEDS = tuple(range(3701, 3707))
SHARED_ARMS = ("no_replay", "paired", "within_domain", "across_domain")
VARIANCE_ARMS = ("k1", "k4", "k16", "kinfinity")
BENCHMARKS = ("split_cifar100", "permuted_mnist")
EXPECTED_UNITS = 72


@dataclass(frozen=True)
class StudyUnit:
    study: str
    benchmark: str
    seed: int
    arm: str

    @property
    def unit_id(self) -> str:
        return f"{self.study}__{self.benchmark}__{self.arm}__seed_{self.seed}"

    @property
    def mammoth_dataset(self) -> str:
        return {
            "split_cifar100": "seq-cifar100",
            "permuted_mnist": "perm-mnist",
        }[self.benchmark]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("explanatory-follow-up lock identity changed")
    if lock.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("Mammoth pin changed")
    runtime = lock.get("runtime", {})
    if not (
        tuple(runtime.get("shared_output_run_seeds", ())) == SHARED_SEEDS
        and tuple(runtime.get("variance_ladder_run_seeds", ())) == VARIANCE_SEEDS
        and runtime.get("device") == "cuda:0"
        and runtime.get("deterministic_algorithms") is True
        and int(runtime.get("maximum_concurrent_gpu_processes", 0)) == 2
        and int(runtime.get("audit_interval", 0)) == 500
        and runtime.get("scipy_version") == "1.18.0"
    ):
        raise RuntimeError("runtime contract changed")
    shared = lock.get("shared_output_structure", {})
    if not (
        shared.get("benchmark") == "permuted_mnist"
        and tuple(shared.get("arms", ())) == SHARED_ARMS
        and shared.get("mammoth_model") == "er_shared_output_structure"
        and shared.get("reference_model") == "er_shared_output_structure"
        and int(shared.get("buffer_size", 0)) == 500
        and int(shared.get("minibatch_size", 0)) == 32
        and shared.get("inference", {}).get("no_optional_stopping") is True
    ):
        raise RuntimeError("shared-output study contract changed")
    variance = lock.get("variance_ladder", {})
    if not (
        tuple(variance.get("benchmarks", ())) == BENCHMARKS
        and variance.get("k_values") == [1, 4, 16, "infinity"]
        and int(variance.get("lambda_basis_points", 0)) == 7500
        and variance.get("mammoth_model") == "er_variance_ladder"
        and int(variance.get("buffer_size", 0)) == 500
        and int(variance.get("minibatch_size", 0)) == 32
        and variance.get("inference", {}).get("no_optional_stopping") is True
    ):
        raise RuntimeError("variance-ladder study contract changed")
    benchmarks = lock.get("benchmarks", {})
    expected_benchmarks = {
        "split_cifar100": ("seq-cifar100", "class-il", "task-il", "20tasks", "resnet18", 20, 100, 32, 1, 0.03),
        "permuted_mnist": ("perm-mnist", "domain-il", "domain-il", None, "mnistmlp", 20, 10, 128, 1, 0.1),
    }
    if set(benchmarks) != set(expected_benchmarks):
        raise RuntimeError("benchmark set changed")
    for name, expected in expected_benchmarks.items():
        row = benchmarks[name]
        observed = (
            row.get("mammoth_dataset"), row.get("setting"),
            row.get("followup_primary_setting"), row.get("dataset_config"),
            row.get("backbone"), int(row.get("tasks", 0)),
            int(row.get("classes", 0)), int(row.get("batch_size", 0)),
            int(row.get("epochs", 0)), float(row.get("learning_rate", 0.0)),
        )
        if observed != expected:
            raise RuntimeError(f"benchmark contract changed: {name}")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("shared_output_units", 0)) == 24
        and int(execution.get("variance_ladder_units", 0)) == 48
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and execution.get("completed_units_immutable") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
    ):
        raise RuntimeError("execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("lock is not a JSON object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[StudyUnit]:
    validate_lock(lock)
    units = [
        StudyUnit("shared_output_structure", "permuted_mnist", seed, arm)
        for seed in SHARED_SEEDS for arm in SHARED_ARMS
    ]
    units.extend(
        StudyUnit("variance_ladder", benchmark, seed, arm)
        for benchmark in BENCHMARKS
        for seed in VARIANCE_SEEDS
        for arm in VARIANCE_ARMS
    )
    if len(units) != EXPECTED_UNITS or len({unit.unit_id for unit in units}) != EXPECTED_UNITS:
        raise RuntimeError("unit enumeration changed")
    return units


def parse_mammoth_log(path: Path) -> dict[str, Any]:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if not rows:
        raise RuntimeError(f"Mammoth log is empty: {path}")
    tree = ast.parse(rows[-1], mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key, item in zip(node.keys, node.values):
                if key is None:
                    raise ValueError("dictionary unpacking is forbidden")
                result[evaluate(key)] = evaluate(item)
            return result
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [evaluate(item) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else +value
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "device"
            and [item.arg for item in node.keywords] == ["type", "index"]
            and [evaluate(item.value) for item in node.keywords] == ["cuda", 0]
        ):
            return "cuda:0"
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            name = node.func.id if isinstance(node.func, ast.Name) else None
            attribute = (
                node.func.attr if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"} else None
            )
            if name in {"array", "tensor"} or attribute in {"float16", "float32", "float64", "int32", "int64"}:
                value = evaluate(node.args[0])
                if attribute and attribute.startswith("float"):
                    return float(value)
                if attribute and attribute.startswith("int"):
                    return int(value)
                return value
        raise ValueError(f"unsupported Mammoth log expression: {ast.dump(node)}")

    value = evaluate(tree)
    if not isinstance(value, dict):
        raise RuntimeError("Mammoth log row is not a mapping")
    return value


def metrics_from_mammoth_log(payload: Mapping[str, Any], tasks: int = 20) -> dict[str, Any]:
    rows = sorted({
        int(key.rsplit("task", 1)[1]) for key in payload
        if key.startswith("accuracy_") and "_task" in key
    })
    if rows != list(range(1, tasks + 1)):
        raise RuntimeError(f"Mammoth accuracy matrix must contain exactly {tasks} tasks")
    matrix: list[list[float]] = []
    for row in rows:
        values = []
        for column in range(1, row + 1):
            value = float(payload[f"accuracy_{column}_task{row}"])
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise RuntimeError("invalid Mammoth accuracy")
            values.append(value)
        reported = payload.get(f"accmean_task{row}")
        if reported is not None and abs(float(reported) - sum(values) / len(values)) > 1e-6:
            raise RuntimeError("Mammoth task mean disagrees")
        matrix.append(values)
    trajectory = [sum(row) / len(row) for row in matrix]
    backward = sum(matrix[-1][i] - matrix[i][i] for i in range(tasks - 1)) / (tasks - 1)
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": trajectory[-1],
        "average_incremental_accuracy": sum(trajectory) / len(trajectory),
        "backward_transfer": backward,
        "accuracy_matrix": matrix,
        "mean_accuracy_trajectory": trajectory,
        "evaluation_tasks": tasks,
    }


__all__ = [
    "BENCHMARKS", "EXPECTED_UNITS", "SHARED_ARMS", "SHARED_SEEDS",
    "VARIANCE_ARMS", "VARIANCE_SEEDS", "StudyUnit", "canonical_sha256",
    "enumerate_units", "load_lock", "metrics_from_mammoth_log",
    "parse_mammoth_log", "sha256_file", "strict_json", "validate_lock",
]
