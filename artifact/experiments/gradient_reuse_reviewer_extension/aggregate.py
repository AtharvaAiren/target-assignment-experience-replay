"""Fail-closed aggregation for the locked ICLR reviewer extension.

The four target permutations are nested Monte Carlo draws inside a run seed;
they are never counted as independent replications.  The completed parent
campaign is immutable.  This module authenticates the 156 new continual-
learning units, the E2 reanalysis of the parent units, and the disjoint E7
transformer stream before emitting one compact reviewer-facing summary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from experiments.gradient_reuse_mechanisms.protocol import (
    canonical_sha256,
    enumerate_units as enumerate_parent_units,
    mean_and_ci,
    method_estimand_class,
    reference_baseline_method,
    sha256_file,
    strict_json,
)

from .overlay import BASE_BUNDLE_SHA256, extension_bundle_sha256
from .protocol import (
    DEFAULT_LOCK,
    EXPECTED_CL_UNITS,
    EXPECTED_PARENT_AGGREGATE_SHA256,
    ExtensionUnit,
    enumerate_units,
    load_lock,
    validate_parent_completion,
)
from .source import validate_source_manifest
from .secondary_endpoints import (
    compute_estimands as compute_parent_estimands,
    holm_assignment_family as compute_parent_holm,
    summarize_estimands as summarize_parent_estimands,
)
from .unit_runner import effective_args, validate_completed_unit


SCHEMA = "gradient_reuse.reviewer_extension_aggregate.v1"
DONE_SCHEMA = "gradient_reuse.reviewer_extension_aggregate_done.v1"
E2_SCHEMA = "gradient_reuse.reviewer_extension_e2.v1"
E7_SCHEMA = "gradient_reuse.reviewer_extension.transformer_stream2.v1"
PARENT_LOCK = Path(__file__).resolve().parents[2] / "configs/mechanisms_rtx_7day.json"
REPLICAS = (0, 1, 2, 3)
HEX = frozenset("0123456789abcdef")

DECOMPOSITION_EXPERIMENTS = (
    "E1_off_floor_cifar50",
    "E4_buffer2000",
    "E5_pmnist_seed_extension",
    "E8_miniimagenet10",
)
STATE_EXPERIMENT = "E3_E6_state_matched_bridge"

ENDPOINTS = (
    "primary_final_average_accuracy",
    "primary_old_task_only_accuracy",
    "primary_average_incremental_accuracy",
    "primary_backward_transfer",
    "task_identity_final_average_accuracy",
    "task_identity_average_incremental_accuracy",
    "task_identity_backward_transfer",
)

E5_METHODS = (
    "experience_replay",
    "gem",
    "agem",
    "mer",
    "pcgrad_replay",
)

COMMON_CONFUSION_FIELDS = (
    "accuracy",
    "mean_correct_label_logit",
    "mean_incorrect_label_logit",
    "mean_predictive_entropy",
    "incorrect_prediction_label_entropy",
)
CIFAR_CONFUSION_FIELDS = COMMON_CONFUSION_FIELDS + (
    "correct_old_class_fraction",
    "other_old_class_fraction",
    "newest_task_class_fraction",
    "other_unseen_class_fraction",
    "mean_other_old_logit",
    "mean_newest_task_logit",
)


class ReviewerExtensionAggregationError(RuntimeError):
    """Raised when the extension evidence cannot support an aggregate."""


def _regular_file(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ReviewerExtensionAggregationError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def _regular_directory(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ReviewerExtensionAggregationError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ReviewerExtensionAggregationError(f"invalid SHA-256 for {context}")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ReviewerExtensionAggregationError(f"{context} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReviewerExtensionAggregationError(f"{context} is not numeric") from error
    if not math.isfinite(number):
        raise ReviewerExtensionAggregationError(f"{context} is not finite")
    return number


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if not data:
        raise ReviewerExtensionAggregationError("cannot average an empty collection")
    return sum(data) / len(data)


def _mc_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ReviewerExtensionAggregationError("nested MC variance needs two replicas")
    center = _mean(values)
    sample = sum((float(value) - center) ** 2 for value in values) / (len(values) - 1)
    return sample / len(values)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ReviewerExtensionAggregationError("Pearson inputs must align")
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (float(a) - left_mean) * (float(b) - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((float(value) - left_mean) ** 2 for value in left)
    right_ss = sum((float(value) - right_mean) ** 2 for value in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _metric_payload(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewerExtensionAggregationError(f"{context} metrics are absent")
    if value.get("accuracy_unit") != "percentage_points":
        raise ReviewerExtensionAggregationError(f"{context} accuracy unit changed")
    if value.get("evaluation_tasks") != 20:
        raise ReviewerExtensionAggregationError(f"{context} lacks 20 tasks")
    for field in ("final_average_accuracy", "average_incremental_accuracy"):
        number = _finite(value.get(field), f"{context}.{field}")
        if not 0.0 <= number <= 100.0:
            raise ReviewerExtensionAggregationError(f"{context}.{field} is invalid")
    _finite(value.get("backward_transfer"), f"{context}.backward_transfer")
    if value.get("backward_transfer_available") is not True:
        raise ReviewerExtensionAggregationError(f"{context} BWT is unavailable")
    matrix = value.get("accuracy_matrix")
    if (
        value.get("accuracy_matrix_available") is not True
        or not isinstance(matrix, list)
        or len(matrix) != 20
        or any(
            not isinstance(row, list) or len(row) != index
            for index, row in enumerate(matrix, 1)
        )
    ):
        raise ReviewerExtensionAggregationError(f"{context} accuracy matrix changed")
    for row_index, row in enumerate(matrix, 1):
        for column_index, entry in enumerate(row, 1):
            number = _finite(
                entry, f"{context}.accuracy_matrix[{row_index},{column_index}]"
            )
            if not 0.0 <= number <= 100.0:
                raise ReviewerExtensionAggregationError(
                    f"{context} accuracy matrix is invalid"
                )
    trajectory = value.get("mean_accuracy_trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 20:
        raise ReviewerExtensionAggregationError(f"{context} trajectory changed")
    reconstructed_trajectory = [_mean(row) for row in matrix]
    if any(
        abs(_finite(observed, f"{context} trajectory") - expected) > 1.0e-6
        for observed, expected in zip(trajectory, reconstructed_trajectory)
    ):
        raise ReviewerExtensionAggregationError(f"{context} trajectory is inconsistent")
    reconstructed_final = _mean(matrix[-1])
    reconstructed_incremental = _mean(reconstructed_trajectory)
    reconstructed_bwt = _mean(
        matrix[-1][index] - matrix[index][index] for index in range(19)
    )
    for field, expected in (
        ("final_average_accuracy", reconstructed_final),
        ("average_incremental_accuracy", reconstructed_incremental),
        ("backward_transfer", reconstructed_bwt),
    ):
        if abs(_finite(value.get(field), f"{context}.{field}") - expected) > 1.0e-6:
            raise ReviewerExtensionAggregationError(
                f"{context}.{field} is inconsistent with its accuracy matrix"
            )
    return value


def scalar_endpoints(
    record: Mapping[str, Any], unit: ExtensionUnit
) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ReviewerExtensionAggregationError(
            f"unit metrics are absent: {unit.unit_id}"
        )
    expected_setting = "domain-il" if unit.benchmark == "permuted_mnist" else "class-il"
    if metrics.get("primary_setting") != expected_setting:
        raise ReviewerExtensionAggregationError(
            f"unit primary setting changed: {unit.unit_id}"
        )
    primary = _metric_payload(metrics.get("primary"), context=f"{unit.unit_id}/primary")
    task_il_raw = metrics.get("task_il")
    if expected_setting == "class-il":
        task_identity = _metric_payload(task_il_raw, context=f"{unit.unit_id}/task-il")
    else:
        if task_il_raw is not None:
            raise ReviewerExtensionAggregationError(
                f"domain-IL unit unexpectedly emitted task-IL: {unit.unit_id}"
            )
        task_identity = primary
    matrix = primary["accuracy_matrix"]
    computed_old = _mean(
        _finite(value, "old-task accuracy") for value in matrix[-1][:-1]
    )
    recorded_old = _finite(
        metrics.get("old_task_only_accuracy"),
        f"{unit.unit_id}/old_task_only_accuracy",
    )
    if abs(computed_old - recorded_old) > 1.0e-10:
        raise ReviewerExtensionAggregationError(
            f"old-task-only endpoint changed: {unit.unit_id}"
        )
    return {
        "primary_final_average_accuracy": _finite(
            primary["final_average_accuracy"], "primary final accuracy"
        ),
        "primary_old_task_only_accuracy": recorded_old,
        "primary_average_incremental_accuracy": _finite(
            primary["average_incremental_accuracy"], "primary incremental accuracy"
        ),
        "primary_backward_transfer": _finite(
            primary["backward_transfer"], "primary BWT"
        ),
        "task_identity_final_average_accuracy": _finite(
            task_identity["final_average_accuracy"], "task-identity final accuracy"
        ),
        "task_identity_average_incremental_accuracy": _finite(
            task_identity["average_incremental_accuracy"],
            "task-identity incremental accuracy",
        ),
        "task_identity_backward_transfer": _finite(
            task_identity["backward_transfer"], "task-identity BWT"
        ),
    }


def _expected_identity_static(
    unit: ExtensionUnit,
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_manifest: Path,
    source: Mapping[str, Any],
    parent_completion: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.reviewer_extension_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "parent_completion_sha256": sha256_file(parent_completion),
        "base_overlay_bundle_sha256": BASE_BUNDLE_SHA256,
        "extension_overlay_bundle_sha256": extension_bundle_sha256(),
        "effective_args_sha256": canonical_sha256(
            effective_args(unit, lock, device="cuda:0")
        ),
        "device": "cuda:0",
        "paper_eligible": True,
    }


def _validate_unit_record(
    record: Mapping[str, Any],
    unit: ExtensionUnit,
    expected_static: Mapping[str, Any],
    expected_effective_args: Mapping[str, Any],
) -> dict[str, float]:
    if set(record) != {
        "schema_version",
        "identity",
        "unit",
        "effective_args",
        "metrics",
        "pairing_diagnostics",
        "sampled_event_fingerprints",
        "state_matched_audit",
        "runtime_seconds",
        "attempt",
        "paper_eligible",
    }:
        raise ReviewerExtensionAggregationError(
            f"unit record fields changed: {unit.unit_id}"
        )
    if record.get("schema_version") != "gradient_reuse.reviewer_extension_unit.v1":
        raise ReviewerExtensionAggregationError(f"unit schema changed: {unit.unit_id}")
    if record.get("paper_eligible") is not True or record.get("unit") != unit.as_dict():
        raise ReviewerExtensionAggregationError(
            f"unit contract changed: {unit.unit_id}"
        )
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ReviewerExtensionAggregationError(f"unit identity absent: {unit.unit_id}")
    for key, expected in expected_static.items():
        if identity.get(key) != expected:
            raise ReviewerExtensionAggregationError(
                f"unit identity {key} changed: {unit.unit_id}"
            )
    expected_identity_keys = set(expected_static) | {
        "external_overlay_binding",
        "data_manifest_sha256",
        "data_bundle_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "python_sha256",
    }
    if set(identity) != expected_identity_keys:
        raise ReviewerExtensionAggregationError(
            f"unit identity keys changed: {unit.unit_id}"
        )
    if not isinstance(identity["external_overlay_binding"], Mapping):
        raise ReviewerExtensionAggregationError("external overlay binding is absent")
    for field in (
        "data_manifest_sha256",
        "data_bundle_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "python_sha256",
    ):
        _digest(identity[field], f"{unit.unit_id}/{field}")
    if record.get("effective_args") != dict(expected_effective_args):
        # This comparison catches a readable UNIT record that disagrees with
        # its own outcome-relevant effective-argument hash.
        raise ReviewerExtensionAggregationError(
            f"effective arguments changed: {unit.unit_id}"
        )
    attempt = record.get("attempt")
    if (
        not isinstance(attempt, str)
        or not attempt.startswith("attempt_")
        or not attempt[8:].isdigit()
        or attempt != f"attempt_{int(attempt[8:]):03d}"
    ):
        raise ReviewerExtensionAggregationError(f"unit attempt changed: {unit.unit_id}")
    runtime = _finite(record.get("runtime_seconds"), f"{unit.unit_id}/runtime")
    if runtime < 0.0:
        raise ReviewerExtensionAggregationError(f"negative runtime: {unit.unit_id}")
    pairing = record.get("pairing_diagnostics")
    fingerprints = record.get("sampled_event_fingerprints")
    if not isinstance(pairing, Mapping):
        raise ReviewerExtensionAggregationError(
            f"pairing diagnostics absent: {unit.unit_id}"
        )
    if not isinstance(fingerprints, list):
        raise ReviewerExtensionAggregationError(f"fingerprints absent: {unit.unit_id}")
    if method_estimand_class(unit.method) == "reference_baseline":
        if dict(pairing) != {"records": 0} or fingerprints:
            raise ReviewerExtensionAggregationError(
                f"reference unit emitted replay diagnostics: {unit.unit_id}"
            )
    elif not fingerprints or int(pairing.get("records", 0)) < 1:
        raise ReviewerExtensionAggregationError(
            f"replay diagnostics are empty: {unit.unit_id}"
        )
    state_audit = record.get("state_matched_audit")
    if unit.experiment == STATE_EXPERIMENT:
        if not isinstance(state_audit, Mapping):
            raise ReviewerExtensionAggregationError(
                f"state audit absent: {unit.unit_id}"
            )
    elif state_audit is not None:
        raise ReviewerExtensionAggregationError(
            f"non-state unit emitted a state audit: {unit.unit_id}"
        )
    return scalar_endpoints(record, unit)


def validate_unit_artifacts(
    unit_root: Path,
    identity: Mapping[str, Any],
    *,
    expected_record: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Authenticate a unit's byte inventory and selected completed attempt."""

    try:
        validated = validate_completed_unit(unit_root, identity)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewerExtensionAggregationError(
            f"unit DONE authentication failed: {unit_root.name}"
        ) from error
    if expected_record is not None and validated != expected_record:
        raise ReviewerExtensionAggregationError(
            f"UNIT reread changed: {unit_root.name}"
        )
    attempt_name = validated.get("attempt")
    if not isinstance(attempt_name, str):
        raise ReviewerExtensionAggregationError(
            f"unit attempt absent: {unit_root.name}"
        )
    attempt = unit_root / attempt_name / "RUN_MANIFEST.json"
    attempt_value = strict_json(attempt)
    if not isinstance(attempt_value, Mapping) or not (
        set(attempt_value)
        == {
            "schema_version",
            "status",
            "identity",
            "identity_sha256",
            "command",
            "attempt",
            "started_unix",
            "completed_unix",
            "returncode",
        }
        and attempt_value.get("schema_version")
        == "gradient_reuse.reviewer_extension_attempt.v1"
        and attempt_value.get("status") == "COMPLETED"
        and attempt_value.get("identity") == identity
        and attempt_value.get("identity_sha256") == canonical_sha256(identity)
        and attempt_value.get("attempt") == attempt_name
        and isinstance(attempt_value.get("command"), list)
        and bool(attempt_value["command"])
        and attempt_value.get("returncode") == 0
    ):
        raise ReviewerExtensionAggregationError(
            f"selected attempt is not an authenticated completion: {unit_root.name}"
        )
    started = _finite(attempt_value["started_unix"], "attempt start time")
    completed = _finite(attempt_value["completed_unix"], "attempt completion time")
    if started <= 0.0 or completed < started:
        raise ReviewerExtensionAggregationError(
            f"selected attempt timestamps are invalid: {unit_root.name}"
        )
    return validated


def collect_completed_units(
    out: Path,
    *,
    lock_path: Path,
    source_manifest: Path,
    parent_completion: Path,
) -> list[dict[str, Any]]:
    """Authenticate and return all 156 CL unit records in canonical order."""

    out = _regular_directory(Path(out), "reviewer-extension output root")
    raw_lock = Path(lock_path)
    raw_source = Path(source_manifest)
    raw_parent = Path(parent_completion)
    lock = load_lock(raw_lock)
    source = validate_source_manifest(raw_source, lock_path=raw_lock)
    validate_parent_completion(raw_parent)
    lock_path = _regular_file(raw_lock, "reviewer-extension lock")
    source_manifest = _regular_file(raw_source, "reviewer-extension source manifest")
    parent_completion = _regular_file(raw_parent, "parent completion marker")
    expected_units = enumerate_units(lock)
    if len(expected_units) != EXPECTED_CL_UNITS:
        raise ReviewerExtensionAggregationError("locked CL count changed")
    units_root = out / "units"
    if not units_root.is_dir() or units_root.is_symlink():
        raise ReviewerExtensionAggregationError(
            "extension unit root is absent or symbolic"
        )
    observed_directories = {
        path.name
        for path in units_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_ids = {unit.unit_id for unit in expected_units}
    if observed_directories != expected_ids:
        raise ReviewerExtensionAggregationError(
            "extension unit directory set changed: "
            f"missing={sorted(expected_ids - observed_directories)}, "
            f"unexpected={sorted(observed_directories - expected_ids)}"
        )
    for path in units_root.iterdir():
        if path.is_symlink():
            raise ReviewerExtensionAggregationError(f"symbolic unit-root entry: {path}")
        if path.is_file() and path.name.startswith(".") and path.name.endswith(".lock"):
            continue
        if not path.is_dir():
            raise ReviewerExtensionAggregationError(
                f"unexpected unit-root entry: {path}"
            )

    common_identity: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    for unit in expected_units:
        unit_root = units_root / unit.unit_id
        raw = strict_json(unit_root / "UNIT.json")
        if not isinstance(raw, Mapping):
            raise ReviewerExtensionAggregationError(
                f"UNIT is not an object: {unit.unit_id}"
            )
        identity = raw.get("identity")
        expected_static = _expected_identity_static(
            unit,
            lock=lock,
            lock_path=lock_path,
            source_manifest=source_manifest,
            source=source,
            parent_completion=parent_completion,
        )
        endpoints = _validate_unit_record(
            raw,
            unit,
            expected_static,
            effective_args(unit, lock, device="cuda:0"),
        )
        validate_unit_artifacts(unit_root, identity, expected_record=raw)
        variable_identity = {
            key: identity[key]
            for key in (
                "external_overlay_binding",
                "data_manifest_sha256",
                "data_bundle_sha256",
                "environment_manifest_sha256",
                "environment_bundle_sha256",
                "python_sha256",
            )
        }
        if common_identity is None:
            common_identity = variable_identity
        elif variable_identity != common_identity:
            raise ReviewerExtensionAggregationError(
                f"cross-unit runtime identity changed: {unit.unit_id}"
            )
        output.append({"unit": unit, "record": raw, "endpoints": endpoints})
    if len(output) != EXPECTED_CL_UNITS:
        raise ReviewerExtensionAggregationError("extension aggregation is incomplete")
    return output


def _indexed(
    records: Sequence[Mapping[str, Any]], experiment: str, benchmark: str
) -> tuple[
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, int, int], Mapping[str, Any]],
]:
    references: dict[tuple[str, int], Mapping[str, Any]] = {}
    paired: dict[tuple[str, int], Mapping[str, Any]] = {}
    permuted: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for wrapped in records:
        unit: ExtensionUnit = wrapped["unit"]
        if unit.experiment != experiment or unit.benchmark != benchmark:
            continue
        if method_estimand_class(unit.method) == "reference_baseline":
            key = (unit.method, unit.seed)
            if key in references:
                raise ReviewerExtensionAggregationError("duplicate extension reference")
            references[key] = wrapped
        elif unit.intervention == "paired":
            key = (unit.method, unit.seed)
            if key in paired:
                raise ReviewerExtensionAggregationError(
                    "duplicate extension paired arm"
                )
            paired[key] = wrapped
        elif unit.intervention == "permuted":
            key = (unit.method, unit.seed, int(unit.permutation_replica))
            if key in permuted:
                raise ReviewerExtensionAggregationError(
                    "duplicate extension permutation"
                )
            permuted[key] = wrapped
        else:
            raise ReviewerExtensionAggregationError("unknown extension intervention")
    return references, paired, permuted


def decomposition_seed_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    benchmark: str,
    methods: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    references, paired, permuted = _indexed(records, experiment, benchmark)
    required_references = {
        (reference_baseline_method(method), seed)
        for method in methods
        for seed in seeds
    }
    if not required_references.issubset(references):
        raise ReviewerExtensionAggregationError(
            f"reference seed coverage changed: {experiment}/{benchmark}"
        )
    output = []
    for method in methods:
        reference_method = reference_baseline_method(method)
        for seed in seeds:
            key = (method, seed)
            if key not in paired:
                raise ReviewerExtensionAggregationError(f"paired arm absent: {key}")
            replica_records = []
            for replica in REPLICAS:
                replica_key = (method, seed, replica)
                if replica_key not in permuted:
                    raise ReviewerExtensionAggregationError(
                        f"nested permutation absent: {replica_key}"
                    )
                replica_records.append(permuted[replica_key])
            effects: dict[str, Any] = {}
            for endpoint in ENDPOINTS:
                reference = float(
                    references[(reference_method, seed)]["endpoints"][endpoint]
                )
                paired_value = float(paired[key]["endpoints"][endpoint])
                replica_values = [
                    float(record["endpoints"][endpoint]) for record in replica_records
                ]
                permuted_mean = _mean(replica_values)
                total = paired_value - reference
                preserved = permuted_mean - reference
                pairing = paired_value - permuted_mean
                residual = total - preserved - pairing
                if abs(residual) > 1.0e-12:
                    raise ReviewerExtensionAggregationError("value identity failed")
                effects[endpoint] = {
                    "reference": reference,
                    "paired": paired_value,
                    "permuted_replicas": replica_values,
                    "permuted_mean": permuted_mean,
                    "permuted_mean_mc_variance": _mc_variance(replica_values),
                    "V_total": total,
                    "V_perm": preserved,
                    "V_pair": pairing,
                    "identity_residual": residual,
                }
            output.append(
                {
                    "experiment": experiment,
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "reference_method": reference_method,
                    "permutation_replicas": list(REPLICAS),
                    "effects": effects,
                }
            )
    return output


def summarize_decomposition(
    seed_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[
            (str(row["experiment"]), str(row["benchmark"]), str(row["method"]))
        ].append(row)
    output = []
    for (experiment, benchmark, method), group in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        if len(seeds) != len(set(seeds)):
            raise ReviewerExtensionAggregationError("duplicate independent run seed")
        reference_methods = {str(row["reference_method"]) for row in group}
        expected_reference = reference_baseline_method(method)
        if reference_methods != {expected_reference}:
            raise ReviewerExtensionAggregationError(
                "decomposition reference method changed"
            )
        endpoints = {}
        for endpoint in ENDPOINTS:
            nested = [
                float(row["effects"][endpoint]["permuted_mean_mc_variance"])
                for row in group
            ]
            endpoints[endpoint] = {
                "V_total": mean_and_ci(
                    (float(row["effects"][endpoint]["V_total"]) for row in group),
                    nested_mc_variances=(0.0 for _ in group),
                ),
                "V_perm": mean_and_ci(
                    (float(row["effects"][endpoint]["V_perm"]) for row in group),
                    nested_mc_variances=nested,
                ),
                "V_pair": mean_and_ci(
                    (float(row["effects"][endpoint]["V_pair"]) for row in group),
                    nested_mc_variances=nested,
                ),
            }
        output.append(
            {
                "experiment": experiment,
                "benchmark": benchmark,
                "method": method,
                "reference_method": expected_reference,
                "independent_unit": "run_seed",
                "seeds": seeds,
                "permutation_replicas_are_nested_monte_carlo": True,
                "endpoints": endpoints,
            }
        )
    return output


def path_contrast_seed_rows(
    records: Sequence[Mapping[str, Any]], *, benchmark: str
) -> list[dict[str, Any]]:
    references, paired, permuted = _indexed(records, STATE_EXPERIMENT, benchmark)
    if references:
        raise ReviewerExtensionAggregationError("E3 unexpectedly has a no-replay arm")
    output = []
    for seed in (3401, 3402, 3403):
        key = ("experience_replay", seed)
        if key not in paired:
            raise ReviewerExtensionAggregationError("E3 paired trajectory is absent")
        replicas = []
        for replica in REPLICAS:
            replica_key = ("experience_replay", seed, replica)
            if replica_key not in permuted:
                raise ReviewerExtensionAggregationError("E3 permutation is absent")
            replicas.append(permuted[replica_key])
        effects = {}
        for endpoint in ENDPOINTS:
            paired_value = float(paired[key]["endpoints"][endpoint])
            replica_values = [float(row["endpoints"][endpoint]) for row in replicas]
            effects[endpoint] = {
                "paired": paired_value,
                "permuted_replicas": replica_values,
                "permuted_mean": _mean(replica_values),
                "permuted_mean_mc_variance": _mc_variance(replica_values),
                "paired_minus_permuted": paired_value - _mean(replica_values),
            }
        output.append(
            {
                "experiment": STATE_EXPERIMENT,
                "benchmark": benchmark,
                "method": "experience_replay",
                "seed": seed,
                "contrast": "paired_minus_mean_permuted_divergent_trajectory",
                "not_a_shared_state_causal_contrast": True,
                "effects": effects,
            }
        )
    return output


def summarize_path_contrasts(
    seed_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[str(row["benchmark"])].append(row)
    output = []
    for benchmark, group in sorted(groups.items()):
        if {int(row["seed"]) for row in group} != {3401, 3402, 3403}:
            raise ReviewerExtensionAggregationError("E3 path seed coverage changed")
        endpoints = {}
        for endpoint in ENDPOINTS:
            nested = [
                float(row["effects"][endpoint]["permuted_mean_mc_variance"])
                for row in group
            ]
            endpoints[endpoint] = mean_and_ci(
                (
                    float(row["effects"][endpoint]["paired_minus_permuted"])
                    for row in group
                ),
                nested_mc_variances=nested,
            )
        output.append(
            {
                "benchmark": benchmark,
                "method": "experience_replay",
                "independent_unit": "run_seed",
                "contrast": "paired_minus_mean_permuted_divergent_trajectory",
                "not_a_shared_state_causal_contrast": True,
                "endpoints": endpoints,
            }
        )
    return output


def _state_boundary_row(
    raw: Mapping[str, Any], *, benchmark: str, seed: int
) -> dict[str, Any]:
    if set(raw) != {
        "schema_version",
        "model",
        "benchmark",
        "dataset",
        "intervention",
        "run_seed",
        "permutation_replica",
        "completed_tasks",
        "evaluated_old_tasks",
        "live_state_before_and_after_sha256",
        "sampled_indexes_sha256",
        "replay_inputs_sha256",
        "replay_targets_sha256",
        "replay_target_multiset_sha256",
        "permutation_sha256",
        "update_norm_window",
        "examples",
        "radius",
        "branches",
        "cross_entropy",
        "accuracy",
        "first_order",
    }:
        raise ReviewerExtensionAggregationError("state boundary fields changed")
    if raw.get("schema_version") != "gradient_reuse.state_matched_boundary.v1":
        raise ReviewerExtensionAggregationError("state boundary schema changed")
    completed_raw = raw.get("completed_tasks")
    if not isinstance(completed_raw, int) or isinstance(completed_raw, bool):
        raise ReviewerExtensionAggregationError("state boundary index is invalid")
    completed = completed_raw
    if completed not in range(2, 21) or raw.get("evaluated_old_tasks") != completed - 1:
        raise ReviewerExtensionAggregationError("state boundary task coverage changed")
    if raw.get("benchmark") != benchmark or raw.get("run_seed") != seed:
        raise ReviewerExtensionAggregationError("state boundary identity changed")
    expected_dataset = "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist"
    if not (
        raw.get("model") == "er_state_matched_audit"
        and raw.get("dataset") == expected_dataset
        and raw.get("intervention") == "paired"
        and raw.get("permutation_replica") == 0
        and raw.get("update_norm_window") == 16
    ):
        raise ReviewerExtensionAggregationError("state boundary run binding changed")
    for field in (
        "sampled_indexes_sha256",
        "replay_inputs_sha256",
        "replay_targets_sha256",
        "replay_target_multiset_sha256",
    ):
        _digest(raw.get(field), f"state boundary {field}")
    permutations = raw.get("permutation_sha256")
    if not isinstance(permutations, list) or len(permutations) != 4:
        raise ReviewerExtensionAggregationError("state permutation identity changed")
    for replica, digest in enumerate(permutations):
        _digest(digest, f"state permutation {replica}")
    live_state = raw.get("live_state_before_and_after_sha256")
    if not isinstance(live_state, Mapping) or set(live_state) != {
        "model",
        "optimizer",
        "buffer",
        "gradients",
        "loader_generators",
    }:
        raise ReviewerExtensionAggregationError("state preservation identity changed")
    for field in ("model", "optimizer", "buffer", "gradients"):
        _digest(live_state[field], f"state preservation {field}")
    loader_generators = live_state["loader_generators"]
    if not isinstance(loader_generators, list):
        raise ReviewerExtensionAggregationError("loader-generator identity changed")
    for index, digest in enumerate(loader_generators):
        _digest(digest, f"loader generator {index}")
    branches = raw.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {
        "no_step",
        "paired",
        "permuted_0",
        "permuted_1",
        "permuted_2",
        "permuted_3",
    }:
        raise ReviewerExtensionAggregationError("state branch inventory changed")
    cross_entropy = raw.get("cross_entropy")
    accuracy = raw.get("accuracy")
    first_order = raw.get("first_order")
    if not all(
        isinstance(value, Mapping) for value in (cross_entropy, accuracy, first_order)
    ):
        raise ReviewerExtensionAggregationError("state boundary endpoints are absent")
    if not (
        set(cross_entropy) == {"V_total", "V_perm", "V_pair", "identity_residual"}
        and set(accuracy) == {"V_total", "V_perm", "V_pair", "identity_residual"}
        and set(first_order)
        == {
            "risk_gradient_norm",
            "paired_direction_norm",
            "mean_permuted_direction_norm",
            "radius_scaled_pairing_term",
        }
    ):
        raise ReviewerExtensionAggregationError("state endpoint fields changed")
    for name, branch in branches.items():
        if not isinstance(branch, Mapping) or set(branch) != {
            "cross_entropy",
            "accuracy",
        }:
            raise ReviewerExtensionAggregationError(
                f"state branch fields changed: {name}"
            )
    ce_replicas = []
    accuracy_replicas = []
    permuted_ce_values = []
    permuted_accuracy_values = []
    paired_ce = _finite(branches["paired"].get("cross_entropy"), "paired CE")
    paired_accuracy = _finite(branches["paired"].get("accuracy"), "paired accuracy")
    no_step_ce = _finite(branches["no_step"].get("cross_entropy"), "no-step CE")
    no_step_accuracy = _finite(branches["no_step"].get("accuracy"), "no-step accuracy")
    if min(no_step_ce, paired_ce) < 0.0:
        raise ReviewerExtensionAggregationError("state cross-entropy is negative")
    if not all(0.0 <= value <= 100.0 for value in (no_step_accuracy, paired_accuracy)):
        raise ReviewerExtensionAggregationError("state accuracy is outside [0,100]")
    for replica in REPLICAS:
        branch = branches[f"permuted_{replica}"]
        permuted_ce = _finite(branch.get("cross_entropy"), "permuted CE")
        permuted_accuracy = _finite(branch.get("accuracy"), "permuted accuracy")
        if permuted_ce < 0.0 or not 0.0 <= permuted_accuracy <= 100.0:
            raise ReviewerExtensionAggregationError("state branch metric is invalid")
        permuted_ce_values.append(permuted_ce)
        permuted_accuracy_values.append(permuted_accuracy)
        ce_replicas.append(permuted_ce - paired_ce)
        accuracy_replicas.append(paired_accuracy - permuted_accuracy)
    ce_pair = _finite(cross_entropy.get("V_pair"), "state CE V_pair")
    accuracy_pair = _finite(accuracy.get("V_pair"), "state accuracy V_pair")
    if abs(ce_pair - _mean(ce_replicas)) > 1.0e-10:
        raise ReviewerExtensionAggregationError(
            "state CE replica reconstruction failed"
        )
    if abs(accuracy_pair - _mean(accuracy_replicas)) > 1.0e-10:
        raise ReviewerExtensionAggregationError(
            "state accuracy replica reconstruction failed"
        )
    for name, payload in (("cross_entropy", cross_entropy), ("accuracy", accuracy)):
        residual = _finite(payload.get("identity_residual"), f"state {name} residual")
        if abs(residual) > 1.0e-12:
            raise ReviewerExtensionAggregationError("state value identity failed")
        for field in ("V_total", "V_perm"):
            _finite(payload.get(field), f"state {name}.{field}")
    if abs(float(cross_entropy["V_total"]) - (no_step_ce - paired_ce)) > 1.0e-10:
        raise ReviewerExtensionAggregationError("state CE total reconstruction failed")
    if (
        abs(float(cross_entropy["V_perm"]) - (no_step_ce - _mean(permuted_ce_values)))
        > 1.0e-10
    ):
        raise ReviewerExtensionAggregationError(
            "state CE permuted reconstruction failed"
        )
    if abs(float(accuracy["V_total"]) - (paired_accuracy - no_step_accuracy)) > 1.0e-10:
        raise ReviewerExtensionAggregationError(
            "state accuracy total reconstruction failed"
        )
    if (
        abs(
            float(accuracy["V_perm"])
            - (_mean(permuted_accuracy_values) - no_step_accuracy)
        )
        > 1.0e-10
    ):
        raise ReviewerExtensionAggregationError(
            "state accuracy permuted reconstruction failed"
        )
    examples_raw = raw.get("examples")
    if not isinstance(examples_raw, int) or isinstance(examples_raw, bool):
        raise ReviewerExtensionAggregationError("state example count is invalid")
    examples = examples_raw
    radius = _finite(raw.get("radius"), "state radius")
    if examples < 1 or radius <= 0.0:
        raise ReviewerExtensionAggregationError("state probe size or radius is invalid")
    for field in (
        "risk_gradient_norm",
        "paired_direction_norm",
        "mean_permuted_direction_norm",
    ):
        if _finite(first_order.get(field), f"state {field}") < 0.0:
            raise ReviewerExtensionAggregationError("state gradient norm is negative")
    return {
        "benchmark": benchmark,
        "seed": seed,
        "completed_tasks": completed,
        "evaluated_old_tasks": completed - 1,
        "examples": examples,
        "radius": radius,
        "cross_entropy": {
            "V_total": float(cross_entropy["V_total"]),
            "V_perm": float(cross_entropy["V_perm"]),
            "V_pair": ce_pair,
            "replica_pairing_effects": ce_replicas,
            "permuted_mean_mc_variance": _mc_variance(ce_replicas),
        },
        "accuracy": {
            "V_total": float(accuracy["V_total"]),
            "V_perm": float(accuracy["V_perm"]),
            "V_pair": accuracy_pair,
            "replica_pairing_effects": accuracy_replicas,
            "permuted_mean_mc_variance": _mc_variance(accuracy_replicas),
        },
        "first_order_pairing_term": _finite(
            first_order.get("radius_scaled_pairing_term"), "first-order term"
        ),
    }


def state_matched_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boundary_rows = []
    paired_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for wrapped in records:
        unit: ExtensionUnit = wrapped["unit"]
        if unit.experiment != STATE_EXPERIMENT:
            continue
        audit = wrapped["record"].get("state_matched_audit")
        if not isinstance(audit, Mapping):
            raise ReviewerExtensionAggregationError("E3 state audit is absent")
        if set(audit) != {
            "record_count",
            "boundary_count",
            "boundary_records",
            "final_confusion",
            "raw_jsonl_sha256",
        }:
            raise ReviewerExtensionAggregationError("E3 state audit fields changed")
        _digest(audit.get("raw_jsonl_sha256"), "E3 state-audit JSONL")
        expected_record_count = 21 if unit.intervention == "paired" else 2
        if audit.get("record_count") != expected_record_count:
            raise ReviewerExtensionAggregationError("E3 state record count changed")
        if unit.intervention != "paired":
            if audit.get("boundary_count") != 0 or audit.get("boundary_records") != []:
                raise ReviewerExtensionAggregationError(
                    "permuted E3 trajectory contains a shared-state probe"
                )
            continue
        if audit.get("boundary_count") != 19:
            raise ReviewerExtensionAggregationError(
                "paired E3 state audit is incomplete"
            )
        raw_boundaries = audit.get("boundary_records")
        if not isinstance(raw_boundaries, list) or len(raw_boundaries) != 19:
            raise ReviewerExtensionAggregationError(
                "E3 boundary records are incomplete"
            )
        rows = [
            _state_boundary_row(row, benchmark=unit.benchmark, seed=unit.seed)
            for row in raw_boundaries
        ]
        if [row["completed_tasks"] for row in rows] != list(range(2, 21)):
            raise ReviewerExtensionAggregationError("E3 boundary order changed")
        boundary_rows.extend(rows)
        key = (unit.benchmark, unit.seed)
        if key in paired_records:
            raise ReviewerExtensionAggregationError("duplicate E3 paired state audit")
        paired_records[key] = wrapped
    if len(paired_records) != 6 or len(boundary_rows) != 114:
        raise ReviewerExtensionAggregationError("E3 shared-state coverage changed")

    seed_rows = []
    for benchmark in ("split_cifar100", "permuted_mnist"):
        for seed in (3401, 3402, 3403):
            rows = [
                row
                for row in boundary_rows
                if row["benchmark"] == benchmark and row["seed"] == seed
            ]
            if len(rows) != 19:
                raise ReviewerExtensionAggregationError(
                    "E3 seed boundary coverage changed"
                )
            ce_values = [float(row["cross_entropy"]["V_pair"]) for row in rows]
            accuracy_values = [float(row["accuracy"]["V_pair"]) for row in rows]
            first_order = [float(row["first_order_pairing_term"]) for row in rows]
            ce_replica_means = [
                _mean(
                    row["cross_entropy"]["replica_pairing_effects"][replica]
                    for row in rows
                )
                for replica in REPLICAS
            ]
            accuracy_replica_means = [
                _mean(
                    row["accuracy"]["replica_pairing_effects"][replica] for row in rows
                )
                for replica in REPLICAS
            ]
            seed_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "boundaries": 19,
                    "mean_cross_entropy_V_pair": _mean(ce_values),
                    "cross_entropy_nested_mc_variance": _mc_variance(ce_replica_means),
                    "mean_accuracy_V_pair": _mean(accuracy_values),
                    "accuracy_nested_mc_variance": _mc_variance(accuracy_replica_means),
                    "mean_first_order_pairing_term": _mean(first_order),
                    "first_order_vs_realized_cross_entropy_V_pair_pearson": _pearson(
                        first_order, ce_values
                    ),
                    "mean_absolute_first_order_error": _mean(
                        abs(predicted - realized)
                        for predicted, realized in zip(first_order, ce_values)
                    ),
                }
            )
    benchmark_summaries = []
    boundary_summaries = []
    for benchmark in ("split_cifar100", "permuted_mnist"):
        seeds = [row for row in seed_rows if row["benchmark"] == benchmark]
        benchmark_summaries.append(
            {
                "benchmark": benchmark,
                "independent_unit": "run_seed",
                "task_boundaries_repeated_within_seed": True,
                "mean_cross_entropy_V_pair": mean_and_ci(
                    (float(row["mean_cross_entropy_V_pair"]) for row in seeds),
                    nested_mc_variances=(
                        float(row["cross_entropy_nested_mc_variance"]) for row in seeds
                    ),
                ),
                "mean_accuracy_V_pair": mean_and_ci(
                    (float(row["mean_accuracy_V_pair"]) for row in seeds),
                    nested_mc_variances=(
                        float(row["accuracy_nested_mc_variance"]) for row in seeds
                    ),
                ),
                "mean_first_order_pairing_term": mean_and_ci(
                    float(row["mean_first_order_pairing_term"]) for row in seeds
                ),
                "first_order_vs_realized_cross_entropy_V_pair_pearson_by_seed": [
                    row["first_order_vs_realized_cross_entropy_V_pair_pearson"]
                    for row in seeds
                ],
            }
        )
        for completed in range(2, 21):
            rows = [
                row
                for row in boundary_rows
                if row["benchmark"] == benchmark and row["completed_tasks"] == completed
            ]
            if len(rows) != 3:
                raise ReviewerExtensionAggregationError(
                    "E3 boundary seed count changed"
                )
            boundary_summaries.append(
                {
                    "benchmark": benchmark,
                    "completed_tasks": completed,
                    "cross_entropy_V_pair": mean_and_ci(
                        (float(row["cross_entropy"]["V_pair"]) for row in rows),
                        nested_mc_variances=(
                            float(row["cross_entropy"]["permuted_mean_mc_variance"])
                            for row in rows
                        ),
                    ),
                    "accuracy_V_pair": mean_and_ci(
                        (float(row["accuracy"]["V_pair"]) for row in rows),
                        nested_mc_variances=(
                            float(row["accuracy"]["permuted_mean_mc_variance"])
                            for row in rows
                        ),
                    ),
                    "first_order_pairing_term": mean_and_ci(
                        float(row["first_order_pairing_term"]) for row in rows
                    ),
                }
            )
    return {
        "primary_endpoint": "old_task_union_cross_entropy_V_pair",
        "inference": "run_seed; task_boundaries_are_repeated_within_seed",
        "boundary_rows": boundary_rows,
        "seed_rows": seed_rows,
        "benchmark_summaries": benchmark_summaries,
        "boundary_summaries": boundary_summaries,
    }


def _descriptive(values: Sequence[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ReviewerExtensionAggregationError("empty descriptive effect")
    mean = _mean(data)
    standard_deviation = (
        0.0
        if len(data) == 1
        else math.sqrt(sum((value - mean) ** 2 for value in data) / (len(data) - 1))
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": min(data),
        "maximum": max(data),
        "seeds": len(data),
    }


def confusion_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for wrapped in records:
        unit: ExtensionUnit = wrapped["unit"]
        if unit.experiment != STATE_EXPERIMENT:
            continue
        audit = wrapped["record"].get("state_matched_audit")
        if not isinstance(audit, Mapping):
            raise ReviewerExtensionAggregationError("E6 audit absent")
        final = audit.get("final_confusion")
        if not isinstance(final, Mapping) or final.get("completed_tasks") != 20:
            raise ReviewerExtensionAggregationError("E6 final diagnostic absent")
        if set(final) != {
            "schema_version",
            "model",
            "benchmark",
            "dataset",
            "intervention",
            "run_seed",
            "permutation_replica",
            "completed_tasks",
            "live_state_before_and_after_sha256",
            "diagnostic",
        }:
            raise ReviewerExtensionAggregationError("E6 final record fields changed")
        expected_replica = (
            0 if unit.intervention == "paired" else int(unit.permutation_replica)
        )
        if not (
            final.get("schema_version") == "gradient_reuse.final_confusion_record.v1"
            and final.get("model") == "er_state_matched_audit"
            and final.get("benchmark") == unit.benchmark
            and final.get("dataset") == unit.mammoth_dataset
            and final.get("intervention") == unit.intervention
            and final.get("run_seed") == unit.seed
            and final.get("permutation_replica") == expected_replica
        ):
            raise ReviewerExtensionAggregationError("E6 run binding changed")
        live_state = final.get("live_state_before_and_after_sha256")
        if not isinstance(live_state, Mapping) or set(live_state) != {
            "model",
            "optimizer",
            "buffer",
            "gradients",
            "loader_generators",
        }:
            raise ReviewerExtensionAggregationError("E6 state preservation changed")
        for field in ("model", "optimizer", "buffer", "gradients"):
            _digest(live_state[field], f"E6 state preservation {field}")
        generators = live_state["loader_generators"]
        if not isinstance(generators, list):
            raise ReviewerExtensionAggregationError("E6 loader identity changed")
        for index, digest in enumerate(generators):
            _digest(digest, f"E6 loader generator {index}")
        diagnostic = final.get("diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise ReviewerExtensionAggregationError("E6 diagnostic malformed")
        expected_schema = (
            "gradient_reuse.cifar_confusion.v1"
            if unit.benchmark == "split_cifar100"
            else "gradient_reuse.pmnist_confusion.v1"
        )
        if diagnostic.get("schema_version") != expected_schema:
            raise ReviewerExtensionAggregationError("E6 diagnostic schema changed")
        expected_diagnostic_keys = {
            "schema_version",
            "examples",
            *COMMON_CONFUSION_FIELDS,
        }
        if unit.benchmark == "split_cifar100":
            expected_diagnostic_keys |= {
                "mean_correct_old_logit",
                "correct_old_class_fraction",
                "other_old_class_fraction",
                "newest_task_class_fraction",
                "other_unseen_class_fraction",
                "mean_other_old_logit",
                "mean_newest_task_logit",
                "partition_residual",
            }
        if set(diagnostic) != expected_diagnostic_keys:
            raise ReviewerExtensionAggregationError("E6 diagnostic fields changed")
        fields = (
            CIFAR_CONFUSION_FIELDS
            if unit.benchmark == "split_cifar100"
            else COMMON_CONFUSION_FIELDS
        )
        values = {
            field: _finite(diagnostic.get(field), f"E6/{field}") for field in fields
        }
        examples = int(diagnostic.get("examples", 0))
        if examples < 1 or not 0.0 <= values["accuracy"] <= 100.0:
            raise ReviewerExtensionAggregationError(
                "E6 sample count or accuracy changed"
            )
        if (
            values["mean_predictive_entropy"] < 0.0
            or values["incorrect_prediction_label_entropy"] < 0.0
        ):
            raise ReviewerExtensionAggregationError("E6 entropy is negative")
        if unit.benchmark == "split_cifar100" and any(
            not 0.0 <= values[field] <= 1.0
            for field in (
                "correct_old_class_fraction",
                "other_old_class_fraction",
                "newest_task_class_fraction",
                "other_unseen_class_fraction",
            )
        ):
            raise ReviewerExtensionAggregationError("E6 class fraction is invalid")
        if (
            unit.benchmark == "split_cifar100"
            and abs(
                _finite(diagnostic.get("partition_residual"), "E6 partition residual")
            )
            > 1.0e-12
        ):
            raise ReviewerExtensionAggregationError("E6 CIFAR partition failed")
        if unit.benchmark == "split_cifar100" and (
            abs(
                sum(
                    values[field]
                    for field in (
                        "correct_old_class_fraction",
                        "other_old_class_fraction",
                        "newest_task_class_fraction",
                        "other_unseen_class_fraction",
                    )
                )
                - 1.0
            )
            > 1.0e-12
            or abs(
                _finite(
                    diagnostic.get("mean_correct_old_logit"),
                    "E6 correct-old logit",
                )
                - values["mean_correct_label_logit"]
            )
            > 1.0e-12
        ):
            raise ReviewerExtensionAggregationError(
                "E6 CIFAR diagnostic reconstruction failed"
            )
        arm = (
            "paired"
            if unit.intervention == "paired"
            else f"permuted_{int(unit.permutation_replica)}"
        )
        if arm in grouped[(unit.benchmark, unit.seed)]:
            raise ReviewerExtensionAggregationError("duplicate E6 arm")
        grouped[(unit.benchmark, unit.seed)][arm] = values
    seed_rows = []
    for benchmark in ("split_cifar100", "permuted_mnist"):
        fields = (
            CIFAR_CONFUSION_FIELDS
            if benchmark == "split_cifar100"
            else COMMON_CONFUSION_FIELDS
        )
        for seed in (3401, 3402, 3403):
            arms = grouped.get((benchmark, seed), {})
            if set(arms) != {
                "paired",
                "permuted_0",
                "permuted_1",
                "permuted_2",
                "permuted_3",
            }:
                raise ReviewerExtensionAggregationError("E6 arm coverage changed")
            effects = {}
            for field in fields:
                replicas = [
                    float(arms[f"permuted_{replica}"][field]) for replica in REPLICAS
                ]
                effects[field] = {
                    "paired": float(arms["paired"][field]),
                    "permuted_replicas": replicas,
                    "permuted_mean": _mean(replicas),
                    "paired_minus_permuted": float(arms["paired"][field])
                    - _mean(replicas),
                    "permuted_mean_mc_variance": _mc_variance(replicas),
                }
            seed_rows.append({"benchmark": benchmark, "seed": seed, "effects": effects})
    summaries = []
    for benchmark in ("split_cifar100", "permuted_mnist"):
        group = [row for row in seed_rows if row["benchmark"] == benchmark]
        fields = (
            CIFAR_CONFUSION_FIELDS
            if benchmark == "split_cifar100"
            else COMMON_CONFUSION_FIELDS
        )
        summaries.append(
            {
                "benchmark": benchmark,
                "reporting": "exploratory_seed_level_effect_sizes_no_per_component_null_hypothesis_tests",
                "effects": {
                    field: _descriptive(
                        [
                            float(row["effects"][field]["paired_minus_permuted"])
                            for row in group
                        ]
                    )
                    for field in fields
                },
            }
        )
    return {"seed_rows": seed_rows, "summaries": summaries}


def validate_e2(path: Path, parent_completion: Path) -> dict[str, Any]:
    path = _regular_file(Path(path), "E2 result")
    value = strict_json(path)
    expected_top_keys = {
        "schema_version",
        "status",
        "scientific_result",
        "parent_completion_sha256",
        "parent_mechanism_aggregate_sha256",
        "raw_unit_count",
        "independent_unit",
        "permutation_replicas",
        "permutation_replicas_are_nested_monte_carlo",
        "old_task_only_definition",
        "raw_rows",
        "endpoints",
    }
    if not isinstance(value, dict) or not (
        set(value) == expected_top_keys
        and value.get("schema_version") == E2_SCHEMA
        and value.get("status") == "PASS"
        and value.get("scientific_result") is True
        and value.get("parent_completion_sha256") == sha256_file(parent_completion)
        and value.get("parent_mechanism_aggregate_sha256")
        == EXPECTED_PARENT_AGGREGATE_SHA256
        and value.get("raw_unit_count") == 288
        and value.get("independent_unit") == "run_seed"
        and value.get("permutation_replicas") == list(REPLICAS)
        and value.get("permutation_replicas_are_nested_monte_carlo") is True
        and value.get("old_task_only_definition")
        == "mean_final_accuracy_over_tasks_1_through_T_minus_1"
    ):
        raise ReviewerExtensionAggregationError("E2 output authentication failed")
    raw_rows = value.get("raw_rows")
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 288
        or len(
            {str(row.get("unit_id")) for row in raw_rows if isinstance(row, Mapping)}
        )
        != 288
    ):
        raise ReviewerExtensionAggregationError("E2 raw-unit inventory changed")
    parent_lock = strict_json(PARENT_LOCK.resolve(strict=True))
    expected_parent_units = {
        unit.unit_id: unit for unit in enumerate_parent_units(parent_lock)
    }
    if len(expected_parent_units) != 288:
        raise ReviewerExtensionAggregationError("parent lock unit count changed")
    observed_parent_units: set[str] = set()
    expected_raw_keys = {
        "unit_id",
        "benchmark",
        "method",
        "estimand_class",
        "intervention",
        "seed",
        "permutation_replica",
        "task_il_final_average_accuracy",
        "task_il_old_task_only_accuracy",
        "primary_old_task_only_accuracy",
        "task_il_source",
        "primary_log",
        "primary_log_sha256",
        "task_il_log",
        "task_il_log_sha256",
        "unit_done_sha256",
    }
    for row in raw_rows:
        if not isinstance(row, Mapping) or set(row) != expected_raw_keys:
            raise ReviewerExtensionAggregationError("E2 raw row is not an object")
        unit_id = str(row.get("unit_id"))
        unit = expected_parent_units.get(unit_id)
        if unit is None or unit_id in observed_parent_units:
            raise ReviewerExtensionAggregationError("E2 raw unit identity changed")
        observed_parent_units.add(unit_id)
        expected_task_source = (
            "domain_il_equals_task_identity_for_shared_labels"
            if unit.benchmark == "permuted_mnist"
            else "mammoth_task_il_log"
        )
        if not (
            row.get("benchmark") == unit.benchmark
            and row.get("method") == unit.method
            and row.get("estimand_class") == method_estimand_class(unit.method)
            and row.get("intervention") == unit.intervention
            and row.get("seed") == unit.seed
            and row.get("permutation_replica") == unit.permutation_replica
            and row.get("task_il_source") == expected_task_source
        ):
            raise ReviewerExtensionAggregationError("E2 raw unit binding changed")
        for endpoint_name in (
            "task_il_final_average_accuracy",
            "task_il_old_task_only_accuracy",
            "primary_old_task_only_accuracy",
        ):
            endpoint_value = _finite(row.get(endpoint_name), f"E2/{endpoint_name}")
            if not 0.0 <= endpoint_value <= 100.0:
                raise ReviewerExtensionAggregationError("E2 raw accuracy is invalid")
        for field in (
            "primary_log_sha256",
            "task_il_log_sha256",
            "unit_done_sha256",
        ):
            _digest(row.get(field), f"E2/{unit_id}/{field}")
        for field in ("primary_log", "task_il_log"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ReviewerExtensionAggregationError("E2 raw log path is absent")
    if observed_parent_units != set(expected_parent_units):
        raise ReviewerExtensionAggregationError("E2 parent unit coverage changed")
    endpoints = value.get("endpoints")
    expected_endpoints = {
        "task_il_final_average_accuracy",
        "task_il_old_task_only_accuracy",
        "primary_old_task_only_accuracy",
    }
    if not isinstance(endpoints, Mapping) or set(endpoints) != expected_endpoints:
        raise ReviewerExtensionAggregationError("E2 endpoint family changed")
    task_rows = endpoints["task_il_final_average_accuracy"].get("seed_rows")
    if not isinstance(task_rows, list) or len(task_rows) != 54:
        raise ReviewerExtensionAggregationError("E2 task-IL seed rows changed")
    replay_methods = {
        "experience_replay",
        "gem",
        "agem",
        "mer",
        "der",
        "pcgrad_replay",
    }
    expected_cells = {
        (benchmark, method, seed)
        for benchmark in (
            "split_cifar100",
            "split_miniimagenet",
            "permuted_mnist",
        )
        for method in replay_methods
        for seed in (3401, 3402, 3403)
    }
    for endpoint_name, endpoint in endpoints.items():
        if not isinstance(endpoint, Mapping):
            raise ReviewerExtensionAggregationError(f"E2 {endpoint_name} is malformed")
        rows = endpoint.get("seed_rows")
        summaries = endpoint.get("summaries")
        if not isinstance(rows, list) or len(rows) != 54:
            raise ReviewerExtensionAggregationError(f"E2 {endpoint_name} rows changed")
        if not isinstance(summaries, list) or len(summaries) != 18:
            raise ReviewerExtensionAggregationError(
                f"E2 {endpoint_name} summaries changed"
            )
        summary_cells = {
            (str(row.get("benchmark")), str(row.get("method")))
            for row in summaries
            if isinstance(row, Mapping)
        }
        if summary_cells != {
            (benchmark, method)
            for benchmark in (
                "split_cifar100",
                "split_miniimagenet",
                "permuted_mnist",
            )
            for method in replay_methods
        }:
            raise ReviewerExtensionAggregationError(
                f"E2 {endpoint_name} summary cells changed"
            )
        observed_cells = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ReviewerExtensionAggregationError("E2 seed row is not an object")
            key = (
                str(row.get("benchmark")),
                str(row.get("method")),
                int(row.get("seed", -1)),
            )
            if key in observed_cells:
                raise ReviewerExtensionAggregationError("duplicate E2 seed row")
            observed_cells.add(key)
            replicas = row.get("permuted_replicas")
            if not isinstance(replicas, list) or len(replicas) != 4:
                raise ReviewerExtensionAggregationError("E2 nested replicas changed")
            replica_values = [
                _finite(item, "E2 permutation replica") for item in replicas
            ]
            reconstructed_mean = _mean(replica_values)
            if (
                abs(
                    _finite(row.get("permuted_mean"), "E2 permutation mean")
                    - reconstructed_mean
                )
                > 1.0e-10
            ):
                raise ReviewerExtensionAggregationError("E2 permutation mean changed")
            observed_mc_variance = _finite(
                row.get("permuted_mean_mc_variance"), "E2 nested MC variance"
            )
            if (
                observed_mc_variance < 0.0
                or abs(observed_mc_variance - _mc_variance(replica_values)) > 1.0e-10
            ):
                raise ReviewerExtensionAggregationError("E2 nested MC variance changed")
            reference = _finite(row.get("reference"), "E2 reference")
            paired = _finite(row.get("paired"), "E2 paired")
            total = paired - reference
            preserved = reconstructed_mean - reference
            pairing = paired - reconstructed_mean
            if any(
                abs(_finite(row.get(field), f"E2 {field}") - expected) > 1.0e-10
                for field, expected in (
                    ("V_total", total),
                    ("V_perm", preserved),
                    ("V_pair", pairing),
                    ("identity_residual", 0.0),
                )
            ):
                raise ReviewerExtensionAggregationError("E2 value identity changed")
            expected_class = (
                "path_dependent_endogenous_targets"
                if key[1] == "der"
                else "assignment_only"
            )
            if row.get("estimand_class") != expected_class:
                raise ReviewerExtensionAggregationError("E2 estimand class changed")
        if observed_cells != expected_cells:
            raise ReviewerExtensionAggregationError(
                f"E2 {endpoint_name} cell set changed"
            )
    holm = endpoints["task_il_final_average_accuracy"].get(
        "holm_assignment_only_V_pair"
    )
    if not isinstance(holm, list) or len(holm) != 15:
        raise ReviewerExtensionAggregationError("E2 Holm family changed")
    expected_holm = {
        (benchmark, method)
        for benchmark in (
            "split_cifar100",
            "split_miniimagenet",
            "permuted_mnist",
        )
        for method in (
            "experience_replay",
            "gem",
            "agem",
            "mer",
            "pcgrad_replay",
        )
    }
    observed_holm = set()
    for row in holm:
        if not isinstance(row, Mapping):
            raise ReviewerExtensionAggregationError("E2 Holm row is malformed")
        observed_holm.add((str(row.get("benchmark")), str(row.get("method"))))
        for field in ("raw_two_sided_p", "holm_adjusted_p"):
            probability = _finite(row.get(field), f"E2 Holm {field}")
            if not 0.0 <= probability <= 1.0:
                raise ReviewerExtensionAggregationError("E2 Holm probability changed")
        if row.get("reject_at_0_05") is not (float(row["holm_adjusted_p"]) <= 0.05):
            raise ReviewerExtensionAggregationError("E2 Holm decision changed")
    if observed_holm != expected_holm:
        raise ReviewerExtensionAggregationError("E2 Holm cell set changed")
    # Require the compact endpoint tables to be the deterministic projection
    # of the authenticated 288 raw rows, rather than merely self-consistent
    # tables that happen to have the right shape.
    reconstructed_endpoints: dict[str, Any] = {}
    try:
        for endpoint_name in expected_endpoints:
            reconstructed_rows = compute_parent_estimands(raw_rows, endpoint_name)
            reconstructed_endpoints[endpoint_name] = {
                "seed_rows": reconstructed_rows,
                "summaries": summarize_parent_estimands(reconstructed_rows),
            }
        reconstructed_endpoints["task_il_final_average_accuracy"][
            "holm_assignment_only_V_pair"
        ] = compute_parent_holm(
            reconstructed_endpoints["task_il_final_average_accuracy"]["summaries"]
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ReviewerExtensionAggregationError(
            "E2 deterministic reconstruction failed"
        ) from error
    if dict(endpoints) != reconstructed_endpoints:
        raise ReviewerExtensionAggregationError(
            "E2 endpoint tables differ from the raw-unit reconstruction"
        )
    return value


def e5_five_seed_sensitivity(
    extension_rows: Sequence[Mapping[str, Any]], e2: Mapping[str, Any]
) -> dict[str, Any]:
    parent_rows = e2["endpoints"]["task_il_final_average_accuracy"]["seed_rows"]
    parent_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in parent_rows:
        if row.get("benchmark") == "permuted_mnist" and row.get("method") in E5_METHODS:
            key = (str(row["method"]), int(row["seed"]))
            if key in parent_index:
                raise ReviewerExtensionAggregationError("duplicate E5 parent seed row")
            parent_index[key] = row
    extension_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in extension_rows:
        if row.get("experiment") != "E5_pmnist_seed_extension":
            continue
        key = (str(row["method"]), int(row["seed"]))
        if key in extension_index:
            raise ReviewerExtensionAggregationError("duplicate E5 extension seed row")
        extension_index[key] = row
    expected_parent = {
        (method, seed) for method in E5_METHODS for seed in (3401, 3402, 3403)
    }
    expected_extension = {
        (method, seed) for method in E5_METHODS for seed in (3404, 3405)
    }
    if (
        set(parent_index) != expected_parent
        or set(extension_index) != expected_extension
    ):
        raise ReviewerExtensionAggregationError("E5 five-seed coverage changed")
    seed_rows = []
    for method in E5_METHODS:
        for seed in (3401, 3402, 3403):
            raw = parent_index[(method, seed)]
            expected_reference = reference_baseline_method(method)
            replicas = [
                _finite(value, "E5 parent replica")
                for value in raw.get("permuted_replicas", [])
            ]
            if (
                len(replicas) != 4
                or raw.get("estimand_class") != "assignment_only"
                or raw.get("reference_method") != expected_reference
            ):
                raise ReviewerExtensionAggregationError("E5 parent contrast changed")
            row = {
                "method": method,
                "seed": seed,
                "source": "authenticated_parent_E2",
                "reference_method": expected_reference,
                "reference": _finite(raw.get("reference"), "E5 parent reference"),
                "paired": _finite(raw.get("paired"), "E5 parent paired"),
                "permuted_replicas": replicas,
            }
            row["permuted_mean"] = _mean(replicas)
            row["permuted_mean_mc_variance"] = _mc_variance(replicas)
            row["V_total"] = row["paired"] - row["reference"]
            row["V_perm"] = row["permuted_mean"] - row["reference"]
            row["V_pair"] = row["paired"] - row["permuted_mean"]
            seed_rows.append(row)
        for seed in (3404, 3405):
            extension_row = extension_index[(method, seed)]
            expected_reference = reference_baseline_method(method)
            if extension_row.get("reference_method") != expected_reference:
                raise ReviewerExtensionAggregationError(
                    "E5 extension reference method changed"
                )
            raw = extension_row["effects"]["task_identity_final_average_accuracy"]
            replicas = [
                _finite(value, "E5 extension replica")
                for value in raw.get("permuted_replicas", [])
            ]
            if len(replicas) != 4:
                raise ReviewerExtensionAggregationError(
                    "E5 extension permutation coverage changed"
                )
            row = {
                "method": method,
                "seed": seed,
                "source": "reviewer_extension_E5",
                "reference_method": expected_reference,
                "reference": _finite(raw.get("reference"), "E5 extension reference"),
                "paired": _finite(raw.get("paired"), "E5 extension paired"),
                "permuted_replicas": replicas,
            }
            row["permuted_mean"] = _mean(replicas)
            row["permuted_mean_mc_variance"] = _mc_variance(replicas)
            row["V_total"] = row["paired"] - row["reference"]
            row["V_perm"] = row["permuted_mean"] - row["reference"]
            row["V_pair"] = row["paired"] - row["permuted_mean"]
            for field in (
                "permuted_mean",
                "permuted_mean_mc_variance",
                "V_total",
                "V_perm",
                "V_pair",
            ):
                if (
                    abs(_finite(raw.get(field), f"E5 extension {field}") - row[field])
                    > 1.0e-12
                ):
                    raise ReviewerExtensionAggregationError(
                        f"E5 extension {field} changed"
                    )
            if (
                abs(
                    _finite(
                        raw.get("identity_residual"), "E5 extension identity residual"
                    )
                )
                > 1.0e-12
            ):
                raise ReviewerExtensionAggregationError(
                    "E5 extension value identity changed"
                )
            seed_rows.append(row)
    summaries = []
    for method in E5_METHODS:
        group = [row for row in seed_rows if row["method"] == method]
        if {int(row["seed"]) for row in group} != {3401, 3402, 3403, 3404, 3405}:
            raise ReviewerExtensionAggregationError("E5 sensitivity seed set changed")
        expected_reference = reference_baseline_method(method)
        if {str(row["reference_method"]) for row in group} != {expected_reference}:
            raise ReviewerExtensionAggregationError(
                "E5 sensitivity reference method changed"
            )
        nested = [float(row["permuted_mean_mc_variance"]) for row in group]
        summaries.append(
            {
                "method": method,
                "reference_method": expected_reference,
                "seeds": [3401, 3402, 3403, 3404, 3405],
                "reporting": "post_hoc_five_seed_sensitivity_separate_from_frozen_three_seed_analysis",
                "V_total": mean_and_ci(
                    (float(row["V_total"]) for row in group),
                    nested_mc_variances=(0.0 for _ in group),
                ),
                "V_perm": mean_and_ci(
                    (float(row["V_perm"]) for row in group),
                    nested_mc_variances=nested,
                ),
                "V_pair": mean_and_ci(
                    (float(row["V_pair"]) for row in group),
                    nested_mc_variances=nested,
                ),
            }
        )
    return {"seed_rows": seed_rows, "summaries": summaries}


def validate_e7(
    path: Path,
    *,
    parent_completion: Path,
    source_manifest: Path,
    lock_path: Path,
    config_path: Path,
    asset_manifest: Path,
    parent_transformer_result: Path,
    correction_manifest: Path,
    correction_patch: Path,
) -> dict[str, Any]:
    path = _regular_file(Path(path), "E7 result")
    value = strict_json(path)
    # Rebuild and authenticate every live invocation binding rather than
    # trusting a self-consistent embedded bundle on resume/aggregation.
    from experiments.gradient_reuse_transformer.run import read_config

    from .transformer_stream2 import (
        _validate_parent_result,
        activated_correction_provenance,
        authenticate_correction_files,
        build_e7_input_bundle,
        validate_stream2_result,
    )

    config_path = _regular_file(config_path, "E7 Transformer config")
    asset_manifest = _regular_file(asset_manifest, "E7 asset manifest")
    parent_transformer_result = _regular_file(
        parent_transformer_result, "parent Transformer result"
    )
    correction_manifest = _regular_file(
        correction_manifest, "Transformer correction manifest"
    )
    correction_patch = _regular_file(correction_patch, "Transformer correction patch")
    input_bundle = build_e7_input_bundle(
        {
            "reviewer_extension_lock": lock_path,
            "reviewer_extension_source_manifest": source_manifest,
            "transformer_configuration": config_path,
            "transformer_asset_manifest": asset_manifest,
            "parent_completion": parent_completion,
            "parent_transformer_result": parent_transformer_result,
            "transformer_correction_manifest": correction_manifest,
            "transformer_correction_patch": correction_patch,
        }
    )
    config_sha256 = sha256_file(config_path)
    parent_result, parent_binding = _validate_parent_result(
        parent_completion,
        parent_transformer_result,
        config=read_config(config_path),
        config_sha256=config_sha256,
    )
    del parent_result
    correction = authenticate_correction_files(correction_manifest, correction_patch)
    correction_provenance = activated_correction_provenance(correction)

    try:
        validate_stream2_result(value, expected_input_bundle=input_bundle)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ReviewerExtensionAggregationError(
            "E7 native validation failed"
        ) from error
    if value.get("parent") != parent_binding:
        raise ReviewerExtensionAggregationError(
            "E7 parent result was not reauthenticated from live inputs"
        )
    if value.get("correction_provenance") != correction_provenance:
        raise ReviewerExtensionAggregationError(
            "E7 correction provenance differs from live correction bytes"
        )
    if not isinstance(value, dict) or not (
        value.get("schema_version") == E7_SCHEMA
        and value.get("status") == "PASS"
        and value.get("paper_evidence_eligible") is True
        and value.get("experiment") == "E7_second_pythia_stream"
        and value.get("independent_model_trajectories") == 1
        and value.get("seed_level_replication") is False
        and value.get("packet_blocks_are_not_corpus_replications") is True
        and value.get("parent_and_historical_streams_are_replications") is False
    ):
        raise ReviewerExtensionAggregationError("E7 output authentication failed")
    source = value.get("source_provenance")
    parent = value.get("parent")
    if not isinstance(source, Mapping) or not isinstance(parent, Mapping):
        raise ReviewerExtensionAggregationError("E7 invocation binding is absent")
    if not (
        source.get("source_manifest_file_sha256") == sha256_file(source_manifest)
        and source.get("lock_sha256") == sha256_file(lock_path)
        and parent.get("completion_file_sha256") == sha256_file(parent_completion)
    ):
        raise ReviewerExtensionAggregationError(
            "E7 result belongs to different aggregate inputs"
        )
    stream = value.get("stream")
    if not isinstance(stream, Mapping) or not (
        stream.get("warmup_examples") == 4096
        and stream.get("online_examples") == 8192
        and stream.get("required_examples") == 12288
        and stream.get("selection")
        == (
            "latest_required_eligible_articles_strictly_before_parent_cutoff_"
            "after_parent_id_and_text_exclusion"
        )
        and stream.get("historical_cutoff_utc") == "2021-01-01T00:00:00+00:00"
        and stream.get("comparison_reporting")
        == "separate_historical_stream_not_a_replication"
        and stream.get("example_id_overlap_count") == 0
        and stream.get("article_text_overlap_count") == 0
        and stream.get("timestamp_overlap_count") == 0
        and stream.get("strict_temporal_separation") is True
        and stream.get("parent_stream_rebuilt_from_exact_asset") is True
        and stream.get("parent_stream_manifest_matches_completed_result") is True
        and stream.get("post_parent_candidate_upper_bound") == 291
    ):
        raise ReviewerExtensionAggregationError(
            "E7 disjoint historical-stream contract changed"
        )
    _digest(stream.get("stream_sha256"), "E7 stream")
    _digest(stream.get("parent_stream_sha256"), "E7 parent stream")
    unit = value.get("unit")
    scored_token_count = (
        unit.get("scored_token_count") if isinstance(unit, Mapping) else None
    )
    if not isinstance(unit, Mapping) or not (
        unit.get("schema_version") == "fmmt.gradient_reuse_transformer.unit.v2"
        and unit.get("trajectory_seed") == 300
        and unit.get("permutation_replicates_nested_within_packet") is True
        and unit.get("seed_level_replication") is False
        and unit.get("packet_count") == 512
        and unit.get("warmup", {}).get("packets") == 256
        and unit.get("warmup", {}).get("epochs") == 1
        and unit.get("warmup", {}).get("true_targets_only") is True
        and isinstance(scored_token_count, int)
        and not isinstance(scored_token_count, bool)
        and scored_token_count == 8176
    ):
        raise ReviewerExtensionAggregationError("E7 transformer unit changed")
    metrics = unit.get("task_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"K0", "K1", "K4"}:
        raise ReviewerExtensionAggregationError("E7 task metrics changed")
    for branch, payload in metrics.items():
        if not isinstance(payload, Mapping):
            raise ReviewerExtensionAggregationError(f"E7 {branch} metrics absent")
        for field in ("count", "nll", "perplexity", "top1_accuracy", "top5_accuracy"):
            _finite(payload.get(field), f"E7/{branch}/{field}")
        if _finite(payload["count"], f"E7/{branch}/count") != float(scored_token_count):
            raise ReviewerExtensionAggregationError(f"E7 {branch} count changed")
        if (
            float(payload["nll"]) < 0.0
            or float(payload["perplexity"]) <= 0.0
            or not 0.0 <= float(payload["top1_accuracy"]) <= 1.0
            or not 0.0 <= float(payload["top5_accuracy"]) <= 1.0
            or float(payload["top5_accuracy"]) < float(payload["top1_accuracy"])
            or not math.isclose(
                float(payload["perplexity"]),
                math.exp(float(payload["nll"])),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ReviewerExtensionAggregationError(f"E7 {branch} metric is invalid")
    effects = unit.get("effects")
    if not isinstance(effects, Mapping):
        raise ReviewerExtensionAggregationError("E7 effects are absent")
    expected_nll_effects = {
        "gross_nll_reduction": float(metrics["K0"]["nll"])
        - float(metrics["K1"]["nll"]),
        "permuted_nll_reduction": float(metrics["K0"]["nll"])
        - float(metrics["K4"]["nll"]),
        "pairing_nll_reduction": float(metrics["K4"]["nll"])
        - float(metrics["K1"]["nll"]),
    }
    expected_accuracy_effects = {
        "gross_top1_accuracy": float(metrics["K1"]["top1_accuracy"])
        - float(metrics["K0"]["top1_accuracy"]),
        "permuted_top1_accuracy": float(metrics["K4"]["top1_accuracy"])
        - float(metrics["K0"]["top1_accuracy"]),
        "pairing_top1_accuracy": float(metrics["K1"]["top1_accuracy"])
        - float(metrics["K4"]["top1_accuracy"]),
        "gross_top5_accuracy": float(metrics["K1"]["top5_accuracy"])
        - float(metrics["K0"]["top5_accuracy"]),
        "permuted_top5_accuracy": float(metrics["K4"]["top5_accuracy"])
        - float(metrics["K0"]["top5_accuracy"]),
        "pairing_top5_accuracy": float(metrics["K1"]["top5_accuracy"])
        - float(metrics["K4"]["top5_accuracy"]),
    }
    expected_effects = {**expected_nll_effects, **expected_accuracy_effects}
    if set(effects) != set(expected_effects):
        raise ReviewerExtensionAggregationError("E7 effect family changed")
    for field, expected in expected_effects.items():
        if abs(_finite(effects.get(field), f"E7 {field}") - expected) > 1.0e-12:
            raise ReviewerExtensionAggregationError(f"E7 {field} changed")
    return value


def _e7_summary(value: Mapping[str, Any], path: Path) -> dict[str, Any]:
    unit = value["unit"]
    return {
        "result_sha256": sha256_file(path),
        "stream_sha256": value["stream"]["stream_sha256"],
        "parent_stream_sha256": value["stream"]["parent_stream_sha256"],
        "historical_cutoff_utc": value["stream"]["historical_cutoff_utc"],
        "comparison_reporting": value["stream"]["comparison_reporting"],
        "strict_temporal_separation": value["stream"]["strict_temporal_separation"],
        "example_id_overlap_count": value["stream"]["example_id_overlap_count"],
        "article_text_overlap_count": value["stream"]["article_text_overlap_count"],
        "timestamp_overlap_count": value["stream"]["timestamp_overlap_count"],
        "scored_token_count": unit["scored_token_count"],
        "trajectory_seed": unit.get("trajectory_seed"),
        "task_metrics": unit["task_metrics"],
        "effects": unit["effects"],
        "seed_level_replication": False,
        "packet_blocks_are_not_corpus_replications": True,
        "parent_and_historical_streams_are_replications": False,
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    e2: Mapping[str, Any],
    e2_path: Path,
    e7: Mapping[str, Any],
    e7_path: Path,
    lock_path: Path,
    source_manifest: Path,
    parent_completion: Path,
) -> dict[str, Any]:
    lock = load_lock(lock_path)
    decomposition_rows = []
    for experiment in DECOMPOSITION_EXPERIMENTS:
        spec = lock["experiments"][experiment]
        benchmarks = (
            spec["benchmark"]
            if isinstance(spec["benchmark"], list)
            else [spec["benchmark"]]
        )
        methods = [
            method
            for method in spec["methods"]
            if method_estimand_class(method) != "reference_baseline"
        ]
        for benchmark in benchmarks:
            decomposition_rows.extend(
                decomposition_seed_rows(
                    records,
                    experiment=experiment,
                    benchmark=benchmark,
                    methods=methods,
                    seeds=[int(seed) for seed in spec["seeds"]],
                )
            )
    # 1 E1 + 2 E4 + 5 E5 + 1 E8 cells, each at its locked seed count.
    if len(decomposition_rows) != 22:
        raise ReviewerExtensionAggregationError(
            "extension decomposition row count changed"
        )
    path_rows = [
        *path_contrast_seed_rows(records, benchmark="split_cifar100"),
        *path_contrast_seed_rows(records, benchmark="permuted_mnist"),
    ]
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "profile_id": "iclr_reviewer_extension_v1",
        "unit_count": len(records),
        "unit_ids_sha256": canonical_sha256(
            [wrapped["unit"].unit_id for wrapped in records]
        ),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "parent_completion_sha256": sha256_file(parent_completion),
        "independent_unit": "run_seed",
        "permutation_replicas": list(REPLICAS),
        "permutation_replicas_are_nested_monte_carlo": True,
        "decomposition": {
            "definition": "V_total=V_perm+V_pair",
            "seed_rows": decomposition_rows,
            "summaries": summarize_decomposition(decomposition_rows),
        },
        "state_matched_bridge": {
            "path_level": {
                "seed_rows": path_rows,
                "summaries": summarize_path_contrasts(path_rows),
            },
            "shared_state": state_matched_summary(records),
        },
        "final_confusion": confusion_summary(records),
        "e2_existing_run_reanalysis": {
            "sha256": sha256_file(e2_path),
            "raw_unit_count": e2["raw_unit_count"],
            "endpoints": e2["endpoints"],
        },
        "e5_five_seed_sensitivity": e5_five_seed_sensitivity(decomposition_rows, e2),
        "e7_second_pythia_stream": _e7_summary(e7, e7_path),
    }


def run(
    *,
    out: Path,
    lock_path: Path,
    source_manifest: Path,
    parent_completion: Path,
    secondary_output: Path,
    transformer_output: Path,
    transformer_config: Path,
    asset_manifest: Path,
    parent_transformer_result: Path,
    transformer_correction_manifest: Path,
    transformer_correction_patch: Path,
) -> Path:
    out = _regular_directory(Path(out), "reviewer-extension output root")
    raw_lock = Path(lock_path)
    raw_source = Path(source_manifest)
    raw_parent = Path(parent_completion)
    load_lock(raw_lock)
    validate_parent_completion(raw_parent)
    validate_source_manifest(raw_source, lock_path=raw_lock)
    lock_path = _regular_file(raw_lock, "reviewer-extension lock")
    source_manifest = _regular_file(raw_source, "reviewer-extension source manifest")
    parent_completion = _regular_file(raw_parent, "parent completion marker")
    secondary_output = _regular_file(Path(secondary_output), "E2 result")
    transformer_output = _regular_file(Path(transformer_output), "E7 result")
    e2 = validate_e2(secondary_output, parent_completion)
    e7 = validate_e7(
        transformer_output,
        parent_completion=parent_completion,
        source_manifest=source_manifest,
        lock_path=lock_path,
        config_path=transformer_config,
        asset_manifest=asset_manifest,
        parent_transformer_result=parent_transformer_result,
        correction_manifest=transformer_correction_manifest,
        correction_patch=transformer_correction_patch,
    )
    records = collect_completed_units(
        out,
        lock_path=lock_path,
        source_manifest=source_manifest,
        parent_completion=parent_completion,
    )
    summary = build_summary(
        records,
        e2=e2,
        e2_path=secondary_output,
        e7=e7,
        e7_path=transformer_output,
        lock_path=lock_path,
        source_manifest=source_manifest,
        parent_completion=parent_completion,
    )
    aggregate = out / "aggregate"
    result = aggregate / "SUMMARY.json"
    done_path = aggregate / "DONE.json"
    if done_path.exists() or done_path.is_symlink():
        if not (
            result.is_file()
            and not result.is_symlink()
            and done_path.is_file()
            and not done_path.is_symlink()
        ):
            raise ReviewerExtensionAggregationError(
                "aggregate DONE exists without a safe SUMMARY"
            )
        if strict_json(result) != summary:
            raise ReviewerExtensionAggregationError(
                "existing aggregate summary changed"
            )
        done = strict_json(done_path)
        expected_inventory = {"SUMMARY.json": sha256_file(result)}
        if not isinstance(done, Mapping) or not (
            done.get("schema_version") == DONE_SCHEMA
            and done.get("status") == "PASS"
            and done.get("unit_count") == EXPECTED_CL_UNITS
            and done.get("output_inventory") == expected_inventory
            and done.get("output_bundle_sha256") == canonical_sha256(expected_inventory)
        ):
            raise ReviewerExtensionAggregationError("existing aggregate DONE changed")
        return result
    if result.exists() or result.is_symlink():
        if result.is_symlink() or not result.is_file():
            raise ReviewerExtensionAggregationError(
                "partial aggregate SUMMARY is not a regular file"
            )
        if strict_json(result) != summary:
            raise ReviewerExtensionAggregationError(
                "partial aggregate summary differs from fresh reconstruction"
            )
    else:
        if aggregate.exists() and any(aggregate.iterdir()):
            raise ReviewerExtensionAggregationError("aggregate directory is not empty")
        atomic_json(result, summary)
    inventory = {"SUMMARY.json": sha256_file(result)}
    atomic_json(
        done_path,
        {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "scientific_result": True,
            "profile_id": "iclr_reviewer_extension_v1",
            "unit_count": EXPECTED_CL_UNITS,
            "identity_sha256": canonical_sha256(
                {
                    "lock_sha256": sha256_file(lock_path),
                    "source_manifest_sha256": sha256_file(source_manifest),
                    "parent_completion_sha256": sha256_file(parent_completion),
                    "e2_sha256": sha256_file(secondary_output),
                    "e7_sha256": sha256_file(transformer_output),
                }
            ),
            "output_inventory": inventory,
            "output_bundle_sha256": canonical_sha256(inventory),
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK), type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--parent-completion", required=True, type=Path)
    parser.add_argument("--secondary-output", required=True, type=Path)
    parser.add_argument("--transformer-output", required=True, type=Path)
    parser.add_argument("--transformer-config", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--parent-transformer-result", required=True, type=Path)
    public_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--transformer-correction-manifest",
        type=Path,
        default=public_root / "provenance/transformer_correction_portable.json",
    )
    parser.add_argument(
        "--transformer-correction-patch",
        type=Path,
        default=(
            public_root
            / "experiments/gradient_reuse_reviewer_extension/"
            "transformer_correction.py"
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        out=args.out,
        lock_path=args.lock,
        source_manifest=args.source_manifest,
        parent_completion=args.parent_completion,
        secondary_output=args.secondary_output,
        transformer_output=args.transformer_output,
        transformer_config=args.transformer_config,
        asset_manifest=args.asset_manifest,
        parent_transformer_result=args.parent_transformer_result,
        transformer_correction_manifest=args.transformer_correction_manifest,
        transformer_correction_patch=args.transformer_correction_patch,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
