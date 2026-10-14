"""Locked unit identities and Mammoth log parsing for task-structure swaps."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "gradient_reuse.task_structure_equal_severity_lock.v1"
STATUS = "OPTIONAL_OUTCOME_INFORMED_FOLLOWUP_LOCKED_BEFORE_TASK_STRUCTURE_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
RUN_SEEDS = tuple(range(3501, 3513))
ARMS = ("within_task", "across_task", "paired_probe")
EXPECTED_UNITS = 36
PREREQUISITE_CONTRACTS = {
    "assignment_strength": {
        "schema_version": "gradient_reuse.assignment_strength_queue_done.v1",
        "scientific_role": "outcome_informed_prospectively_locked_assignment_strength_followup",
        "expected_units": 432,
    },
    "mean_matched": {
        "schema_version": "gradient_reuse.mean_matched_queue_done.v1",
        "scientific_role": "prospectively_locked_mean_matched_companion",
        "expected_units": 96,
    },
}


@dataclass(frozen=True)
class StudyUnit:
    seed: int
    arm: str

    @property
    def unit_id(self) -> str:
        return f"split_cifar100__er__{self.arm}__seed_{self.seed}"

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
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_prerequisite_payload(
    value: Any, campaign: str
) -> dict[str, Any]:
    """Authenticate the terminal shape of an exact upstream campaign."""

    if campaign not in PREREQUISITE_CONTRACTS:
        raise RuntimeError(f"unknown task-structure prerequisite: {campaign}")
    contract = PREREQUISITE_CONTRACTS[campaign]
    raw_count = contract["expected_units"]
    if not isinstance(raw_count, int):
        raise RuntimeError("task-structure prerequisite unit contract is invalid")
    count = raw_count
    if not isinstance(value, dict) or not (
        value.get("schema_version") == contract["schema_version"]
        and value.get("scientific_role") == contract["scientific_role"]
        and value.get("status") == "PASS"
        and value.get("unit_count") == count
        and value.get("expected_units") == count
        and value.get("completed_units") == count
    ):
        raise RuntimeError(f"{campaign} prerequisite is not the locked terminal PASS")
    unit_ids = value.get("validated_unit_ids")
    if not (
        isinstance(unit_ids, list)
        and len(unit_ids) == count
        and len(set(unit_ids)) == count
        and all(isinstance(unit_id, str) and unit_id for unit_id in unit_ids)
        and value.get("validated_unit_ids_sha256") == canonical_sha256(unit_ids)
    ):
        raise RuntimeError(f"{campaign} prerequisite unit coverage is invalid")
    unit_done = value.get("unit_done_sha256")
    if not (
        isinstance(unit_done, Mapping)
        and set(unit_done) == set(unit_ids)
        and all(_is_sha256(digest) for digest in unit_done.values())
        and value.get("unit_done_bundle_sha256") == canonical_sha256(dict(unit_done))
    ):
        raise RuntimeError(f"{campaign} prerequisite unit seals are invalid")
    for name in (
        "lock_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "run_binding_sha256",
        "aggregate_sha256",
    ):
        if not _is_sha256(value.get(name)):
            raise RuntimeError(f"{campaign} prerequisite omitted {name}")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("task-structure lock identity changed")
    if lock.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("task-structure Mammoth pin changed")
    mapping = lock.get("upstream", {}).get("task_mapping_evidence", {})
    if mapping != {
        "datasets/seq_cifar100.py": "53c012c88ee8e7ee1dad5b175fd92a1b3fe94074c240fe1adb5251267150664d",
        "datasets/configs/seq-cifar100/20tasks.yaml": "813579db24a14c32787c1f2139cc09d264ff6523d2d540f90fe45c1513243a8e",
        "datasets/utils/continual_dataset.py": "65a12ec2c0c75428eb70e4dc6afb7b32a3a10eb9b3aacdc59328b9c034188975",
        "utils/buffer.py": "0a240bf1e46e11adc0175207c3139460a1800d38fd647a506be2804ae60a79da",
    }:
        raise RuntimeError("pinned task-mapping evidence changed")
    interpretation = lock.get("interpretation", {})
    required_true = (
        "current_stream_targets_always_correct",
        "probe_training_targets_always_correct",
        "within_and_across_arms_use_the_same_selected_replay_rows",
        "each_selected_quadruple_has_two_different-class_rows_from_each_of_two_distinct_historical_tasks",
        "all_four_selected_target_values_change_in_both_swap_arms",
        "target_multiset_is_preserved_exactly_per_replay_batch",
        "selection_never_uses_model_predictions_losses_gradients_or_scientific_outcomes",
        "forced_swaps_are_not_uniform_group_permutations",
        "grouped_covariance_expectation_identity_is_not_applied_to_forced_swap_realizations",
        "no_causal_mechanism_decomposition_claim",
    )
    if not all(interpretation.get(name) is True for name in required_true):
        raise RuntimeError("task-structure interpretation guard changed")
    runtime = lock.get("runtime", {})
    if not (
        tuple(runtime.get("run_seeds", ())) == RUN_SEEDS
        and tuple(runtime.get("arms", ())) == ARMS
        and runtime.get("device") == "cuda:0"
        and int(runtime.get("counter_seed_offset", -1)) == 200003
        and int(runtime.get("selection_seed_salt", -1)) == 3284386755
        and int(runtime.get("gradient_audit_interval", -1)) == 100
        and int(runtime.get("maximum_concurrent_gpu_processes", -1)) == 2
        and runtime.get("deterministic_algorithms") is True
    ):
        raise RuntimeError("task-structure runtime contract changed")
    benchmark = lock.get("benchmark", {})
    observed = (
        benchmark.get("name"),
        benchmark.get("mammoth_dataset"),
        benchmark.get("dataset_config"),
        benchmark.get("setting"),
        benchmark.get("followup_primary_setting"),
        benchmark.get("backbone"),
        int(benchmark.get("tasks", 0)),
        int(benchmark.get("classes_per_task", 0)),
        int(benchmark.get("batch_size", 0)),
        int(benchmark.get("epochs", 0)),
        float(benchmark.get("learning_rate", 0.0)),
    )
    if observed != (
        "split_cifar100",
        "seq-cifar100",
        "20tasks",
        "class-il",
        "task-il",
        "resnet18",
        20,
        5,
        32,
        1,
        0.03,
    ):
        raise RuntimeError("task-structure benchmark changed")
    if not (
        benchmark.get("class_permutation") is False
        and benchmark.get("custom_class_order") is None
        and benchmark.get("custom_task_order") is None
        and float(benchmark.get("noise_rate", -1.0)) == 0.0
        and benchmark.get("mandatory_secondary_settings") == ["class-il"]
    ):
        raise RuntimeError("task mapping or mandatory class-IL endpoint changed")
    method = lock.get("method", {})
    if not (
        method.get("mammoth_model") == "er_task_structure_swap"
        and int(method.get("buffer_size", 0)) == 500
        and int(method.get("minibatch_size", 0)) == 32
    ):
        raise RuntimeError("task-structure ER method changed")
    inference = lock.get("inference", {})
    if not (
        inference.get("primary_estimand")
        == "across_task minus within_task final task-IL average accuracy in percentage points"
        and inference.get("mandatory_secondary_estimand")
        == "across_task minus within_task final class-IL average accuracy"
        and inference.get("no_optional_stopping") is True
        and inference.get("all_planned_units_required") is True
    ):
        raise RuntimeError("task-structure inference contract changed")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("full_training_units", 0)) == 24
        and int(execution.get("paired_probe_units", 0)) == 12
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and execution.get("completed_units_immutable") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
    ):
        raise RuntimeError("task-structure execution contract changed")
    prerequisites = lock.get("queue_prerequisites", {})
    if prerequisites != {
        "assignment_strength_done": "results/assignment_strength/done.json",
        "mean_matched_done": "results/mean_matched/done.json",
        "policy": "launch wait-controller immediately, but start no GPU process until both declared DONE sentinels authenticate as PASS",
    }:
        raise RuntimeError("task-structure prerequisite paths or policy changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("task-structure lock is not an object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[StudyUnit]:
    validate_lock(lock)
    units = [StudyUnit(seed, arm) for seed in RUN_SEEDS for arm in ARMS]
    if len(units) != EXPECTED_UNITS or len({unit.unit_id for unit in units}) != EXPECTED_UNITS:
        raise RuntimeError("task-structure unit enumeration changed")
    return units


def parse_mammoth_log(path: Path) -> dict[str, Any]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Mammoth log is empty: {path}")
    tree = ast.parse(rows[-1], mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            result = {}
            for key, item_node in zip(node.keys, node.values):
                if key is None:
                    raise ValueError("dictionary unpacking is forbidden in Mammoth logs")
                result[evaluate(key)] = evaluate(item_node)
            return result
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [evaluate(value) for value in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            evaluated: Any = evaluate(node.operand)
            return -evaluated if isinstance(node.op, ast.USub) else +evaluated
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "device"
            and [keyword.arg for keyword in node.keywords] == ["type", "index"]
            and [evaluate(keyword.value) for keyword in node.keywords] == ["cuda", 0]
        ):
            return "cuda:0"
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            name = node.func.id if isinstance(node.func, ast.Name) else None
            attribute = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                else None
            )
            if name in {"array", "tensor"} or attribute in {
                "float16", "float32", "float64", "int32", "int64"
            }:
                evaluated = evaluate(node.args[0])
                if attribute and attribute.startswith("float"):
                    return float(evaluated)
                if attribute and attribute.startswith("int"):
                    return int(evaluated)
                return evaluated
        raise ValueError(f"unsupported expression in Mammoth log: {ast.dump(node)}")

    value = evaluate(tree)
    if not isinstance(value, dict):
        raise RuntimeError("Mammoth log row is not a mapping")
    return value


def metrics_from_mammoth_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(
        {int(key.rsplit("task", 1)[1]) for key in payload if key.startswith("accuracy_") and "_task" in key}
    )
    if rows != list(range(1, 21)):
        raise RuntimeError("Mammoth accuracy matrix must contain exactly 20 tasks")
    matrix: list[list[float]] = []
    for row in rows:
        values = []
        for column in range(1, row + 1):
            key = f"accuracy_{column}_task{row}"
            value = float(payload[key])
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise RuntimeError(f"invalid Mammoth accuracy: {key}={value}")
            values.append(value)
        reported = payload.get(f"accmean_task{row}")
        if reported is not None and abs(float(reported) - sum(values) / len(values)) > 1e-6:
            raise RuntimeError(f"Mammoth task mean disagrees at task {row}")
        matrix.append(values)
    trajectory = [sum(row) / len(row) for row in matrix]
    bwt = sum(matrix[-1][i] - matrix[i][i] for i in range(19)) / 19
    if "backward_transfer" in payload and abs(float(payload["backward_transfer"]) - bwt) > 1e-6:
        raise RuntimeError("Mammoth backward transfer disagrees")
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": trajectory[-1],
        "average_incremental_accuracy": sum(trajectory) / len(trajectory),
        "backward_transfer": bwt,
        "accuracy_matrix": matrix,
        "mean_accuracy_trajectory": trajectory,
        "evaluation_tasks": 20,
    }


__all__ = [
    "ARMS",
    "EXPECTED_UNITS",
    "PREREQUISITE_CONTRACTS",
    "RUN_SEEDS",
    "StudyUnit",
    "canonical_sha256",
    "enumerate_units",
    "load_lock",
    "metrics_from_mammoth_log",
    "parse_mammoth_log",
    "sha256_file",
    "strict_json",
    "validate_lock",
    "validate_prerequisite_payload",
]
