#!/usr/bin/env python3
"""Authenticate and combine the frozen 3-seed result with seeds 3404--3405.

This is deliberately a stand-alone, fail-closed aggregator.  It does not
rewrite the original three-seed result.  It reports the two newly completed
seeds separately and labels the pooled five-seed analysis as post hoc.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "gradient_reuse.primary_seed_completion_aggregate.v1"
DESIGN_SCHEMA = "gradient_reuse.primary_seed_completion_lock.v1"
HISTORICAL_FULL_LOCK_SHA256 = (
    "be5247f6f94ba5835be7b87377fda106487306b2215ba30e3955377b4a402012"
)
PUBLIC_FULL_LOCK_SHA256 = (
    "4d38847ce83ae36850c78573707095568fa8c31cbd9b5b93feb9be3514c61cbc"
)
PARENT_SUMMARY_SHA256 = "06015597a934d8dba380b8e2e5b278eaf220c96e7c4288444cc0145bd84b7e86"
PARENT_E2_SHA256 = "14d6cc2b32422b993986e86d37a1068d77b26b53cd4ee0a19efaa1e2946e5f4b"
E5_UNIT_DONE_BUNDLE_SHA256 = "d0b5b77defe49badff91db0861ee04cef642634c4a52ec013ce39e61cacc1c6e"
PARENT_OVERLAY_BUNDLE_SHA256 = "15586d828e1388653328c4e1a971ff3e5839411da7c1cbca3a9b59a82dfbf8d6"
FROZEN_SEEDS = (3401, 3402, 3403)
COMPLETION_SEEDS = (3404, 3405)
ALL_SEEDS = FROZEN_SEEDS + COMPLETION_SEEDS
REPLICAS = (0, 1, 2, 3)
BENCHMARKS = ("permuted_mnist", "split_cifar100", "split_miniimagenet")
IMAGE_BENCHMARKS = ("split_cifar100", "split_miniimagenet")
METHODS = ("experience_replay", "gem", "agem", "mer", "pcgrad_replay")
REFERENCES = ("naive_sgd", "mer_current_only")
IDENTITY_TOLERANCE = 1.0e-10
T95_DF4 = 2.7764451051977987


class CompletionAggregationError(RuntimeError):
    """Raised when evidence is missing, mixed, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise CompletionAggregationError(f"duplicate JSON key {key!r}: {path}")
            output[key] = value
        return output

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except CompletionAggregationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise CompletionAggregationError(f"invalid strict JSON: {path}") from error


def regular_file(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CompletionAggregationError(f"{role} is absent or symbolic: {path}")
    return path.resolve(strict=True)


def regular_directory(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CompletionAggregationError(f"{role} is absent or symbolic: {path}")
    return path.resolve(strict=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.is_symlink():
        raise CompletionAggregationError(f"aggregate output is symbolic: {path}")
    if path.is_file():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise CompletionAggregationError(f"refusing to overwrite different aggregate output: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    if path.exists() or path.is_symlink():
        temporary.unlink(missing_ok=True)
        raise CompletionAggregationError(f"refusing to overwrite aggregate output: {path}")
    temporary.replace(path)


def artifact_inventory(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CompletionAggregationError(f"artifact tree contains a symlink: {path}")
        if not path.is_file() or (path.parent == root and path.name == "DONE.json"):
            continue
        output[path.relative_to(root).as_posix()] = sha256_file(path)
    return output


def validate_inventory_done(root: Path, expected_schema: str) -> dict[str, Any]:
    done = strict_json(regular_file(root / "DONE.json", "DONE marker"))
    if not isinstance(done, dict) or not (
        done.get("schema_version") == expected_schema and done.get("status") == "PASS"
    ):
        raise CompletionAggregationError(f"DONE contract changed: {root}")
    observed = artifact_inventory(root)
    if done.get("artifact_inventory") != observed:
        raise CompletionAggregationError(f"artifact inventory mismatch: {root}")
    if done.get("artifact_count") != len(observed):
        raise CompletionAggregationError(f"artifact count mismatch: {root}")
    if done.get("artifact_bundle_sha256") != canonical_sha256(observed):
        raise CompletionAggregationError(f"artifact bundle mismatch: {root}")
    return done


def load_design(path: Path) -> dict[str, Any]:
    path = regular_file(path, "design lock")
    design = strict_json(path)
    if not isinstance(design, dict) or design.get("schema_version") != DESIGN_SCHEMA:
        raise CompletionAggregationError("primary seed-completion design-lock schema changed")
    if design.get("status") != "POST_HOC_EXECUTION_PLAN_LOCKED_BEFORE_NEW_IMAGE_OUTCOMES":
        raise CompletionAggregationError("primary seed-completion design lock has wrong status")
    if design["new_execution"].get("expected_new_units") != 108:
        raise CompletionAggregationError("design lock no longer specifies 108 new units")
    return design


def import_protocol(project: Path) -> tuple[Any, Any, Any, Any, Any]:
    import sys

    project_text = str(project)
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    from experiments.gradient_reuse_mechanisms.protocol import (  # type: ignore
        MechanismUnit,
        compute_mechanism_estimands,
        enumerate_units,
        load_and_validate_lock,
        metrics_from_mammoth_log,
        parse_mammoth_log,
    )
    from experiments.gradient_reuse_mechanisms.run import effective_mammoth_args  # type: ignore

    return (
        MechanismUnit,
        compute_mechanism_estimands,
        enumerate_units,
        load_and_validate_lock,
        (metrics_from_mammoth_log, parse_mammoth_log, effective_mammoth_args),
    )


def expected_units(lock: Mapping[str, Any], enumerate_units: Any) -> tuple[list[Any], list[Any]]:
    selected = [
        unit
        for unit in enumerate_units(lock)
        if unit.seed in COMPLETION_SEEDS
        and unit.method in METHODS + REFERENCES
    ]
    new = [unit for unit in selected if unit.benchmark in IMAGE_BENCHMARKS]
    e5 = [unit for unit in selected if unit.benchmark == "permuted_mnist"]
    if len(new) != 108 or len(e5) != 54:
        raise CompletionAggregationError(
            f"unexpected selected unit counts: new={len(new)}, E5={len(e5)}"
        )
    return new, e5


def validate_metric(metrics: Any, tasks: int, role: str) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise CompletionAggregationError(f"{role} metrics are not an object")
    if metrics.get("accuracy_unit") != "percentage_points":
        raise CompletionAggregationError(f"{role} accuracy unit changed")
    if metrics.get("evaluation_tasks") != tasks:
        raise CompletionAggregationError(f"{role} task coverage changed")
    matrix = metrics.get("accuracy_matrix")
    if not isinstance(matrix, list) or len(matrix) != tasks:
        raise CompletionAggregationError(f"{role} lacks a full accuracy matrix")
    for index, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != index:
            raise CompletionAggregationError(f"{role} accuracy matrix is malformed")
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 100 for value in row):
            raise CompletionAggregationError(f"{role} accuracy matrix has invalid values")
    final = sum(map(float, matrix[-1])) / tasks
    if abs(final - float(metrics.get("final_average_accuracy", math.nan))) > 1e-8:
        raise CompletionAggregationError(f"{role} final accuracy is inconsistent")
    return metrics


def validate_new_units(
    completion_root: Path,
    project: Path,
    lock: Mapping[str, Any],
    expected: Sequence[Any],
    helpers: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics_from_log, parse_log, effective_args = helpers
    campaigns = regular_directory(completion_root / "unit_campaigns", "unit campaigns root")
    expected_by_id = {unit.unit_id: unit for unit in expected}
    observed_directories = {
        path.name for path in campaigns.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if observed_directories != set(expected_by_id):
        missing = sorted(set(expected_by_id) - observed_directories)
        extra = sorted(observed_directories - set(expected_by_id))
        raise CompletionAggregationError(
            f"new-unit coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(path.is_symlink() for path in campaigns.iterdir()):
        raise CompletionAggregationError("unit campaigns root contains a symlink")

    primary_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    identity_counter: Counter[str] = Counter()
    unit_bindings: dict[str, Any] = {}
    project_lock = regular_file(project / "configs/mechanisms.json", "original mechanism lock")
    project_source = regular_file(
        project / "provenance/source/mechanisms.json", "mechanism source manifest"
    )
    if sha256_file(project_lock) != PUBLIC_FULL_LOCK_SHA256:
        raise CompletionAggregationError("public mechanism lock bytes changed")
    from experiments.gradient_reuse_mechanisms.protocol import (  # type: ignore
        validate_source_manifest,
    )

    validate_source_manifest(project_source, repo=project, lock_path=project_lock)

    for unit_id in sorted(expected_by_id):
        unit = expected_by_id[unit_id]
        campaign = campaigns / unit_id
        outer_done = validate_inventory_done(campaign, "gradient_reuse.mechanism_done.v2")
        if not (
            outer_done.get("paper_eligible") is False
            and outer_done.get("complete_protocol") is False
            and outer_done.get("completed_unit_ids") == [unit_id]
        ):
            raise CompletionAggregationError(f"selected-unit campaign contract changed: {unit_id}")
        if sha256_file(campaign / "LOCK.json") != HISTORICAL_FULL_LOCK_SHA256:
            raise CompletionAggregationError(f"unit uses another scientific lock: {unit_id}")
        if sha256_file(campaign / "SOURCE_MANIFEST.json") != sha256_file(project_source):
            raise CompletionAggregationError(f"unit uses another source manifest: {unit_id}")

        unit_root = regular_directory(campaign / "units" / unit_id, "mechanism unit root")
        inner_done = validate_inventory_done(
            unit_root, "gradient_reuse.mechanism_unit_done.v2"
        )
        record_path = regular_file(unit_root / "UNIT.json", "mechanism UNIT record")
        record = strict_json(record_path)
        if not isinstance(record, dict) or record.get("schema_version") != "gradient_reuse.mechanism_unit.v2":
            raise CompletionAggregationError(f"new UNIT schema changed: {unit_id}")
        expected_fields = {
            "benchmark": unit.benchmark,
            "method": unit.method,
            "mammoth_model": unit.mammoth_model,
            "intervention": unit.intervention,
            "seed": unit.seed,
            "permutation_replica": unit.permutation_replica,
        }
        if any(record.get(name) != value for name, value in expected_fields.items()):
            raise CompletionAggregationError(f"new UNIT identity changed: {unit_id}")
        if record.get("paper_eligible") is not True:
            raise CompletionAggregationError(f"new unit is not paper eligible: {unit_id}")
        expected_args = effective_args(unit, lock, device="cuda:0", smoke=False)
        if record.get("effective_args") != expected_args:
            raise CompletionAggregationError(f"effective Mammoth arguments changed: {unit_id}")
        identity = record.get("identity")
        if (
            not isinstance(identity, dict)
            or identity.get("lock_sha256") != HISTORICAL_FULL_LOCK_SHA256
        ):
            raise CompletionAggregationError(f"new unit identity is not bound to full lock: {unit_id}")
        if identity.get("paper_eligible") is not True or identity.get("smoke") is not False:
            raise CompletionAggregationError(f"new unit identity is not scientific: {unit_id}")
        if identity.get("overlay_bundle_sha256") != PARENT_OVERLAY_BUNDLE_SHA256:
            raise CompletionAggregationError(
                f"new unit does not use the parent 19-file overlay identity: {unit_id}"
            )
        if inner_done.get("identity_sha256") != canonical_sha256(identity):
            raise CompletionAggregationError(f"new unit identity digest mismatch: {unit_id}")
        shared_identity = {
            key: value
            for key, value in identity.items()
            if key not in {"unit", "estimand_class", "effective_args_sha256"}
        }
        identity_counter[canonical_sha256(shared_identity)] += 1

        tasks = int(lock["benchmarks"][unit.benchmark]["tasks"])
        validate_metric(record.get("metrics"), tasks, f"{unit_id}/primary")
        primary_records.append(record)
        attempt = str(record.get("attempt", ""))
        if not attempt.startswith("attempt_") or "/" in attempt:
            raise CompletionAggregationError(f"new unit attempt name is unsafe: {unit_id}")
        task_log = (
            unit_root
            / attempt
            / "mammoth_results"
            / "task-il"
            / unit.mammoth_dataset
            / unit.mammoth_model
            / "logs.pyd"
        )
        regular_file(task_log, f"{unit_id} task-IL log")
        task_metrics = metrics_from_log(parse_log(task_log))
        validate_metric(task_metrics, tasks, f"{unit_id}/task-il")
        task_record = dict(record)
        task_record["metrics"] = task_metrics
        task_records.append(task_record)
        unit_bindings[unit_id] = {
            "campaign_done_sha256": sha256_file(campaign / "DONE.json"),
            "unit_done_sha256": sha256_file(unit_root / "DONE.json"),
            "unit_sha256": sha256_file(record_path),
            "artifact_bundle_sha256": inner_done["artifact_bundle_sha256"],
        }

    if len(identity_counter) != 1 or next(iter(identity_counter.values())) != 108:
        raise CompletionAggregationError("new units mix execution identities")
    return primary_records, task_records, unit_bindings


def validate_e5_units(
    e5_root: Path,
    lock: Mapping[str, Any],
    expected: Sequence[Any],
    effective_args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    units_root = e5_root / "units" if (e5_root / "units").is_dir() else e5_root
    units_root = regular_directory(units_root, "E5 units root")
    expected_by_name = {
        f"E5_pmnist_seed_extension__{unit.unit_id}": unit for unit in expected
    }
    observed = {
        path.name
        for path in units_root.glob("E5_pmnist_seed_extension__*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed != set(expected_by_name):
        raise CompletionAggregationError(
            f"E5 coverage mismatch: observed={len(observed)}, expected={len(expected_by_name)}"
        )

    pair_inventory: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    for directory_name in sorted(expected_by_name):
        unit = expected_by_name[directory_name]
        root = units_root / directory_name
        done = validate_inventory_done(root, "gradient_reuse.reviewer_extension_unit_done.v1")
        unit_path = regular_file(root / "UNIT.json", "E5 UNIT record")
        payload = strict_json(unit_path)
        if not isinstance(payload, dict) or payload.get("schema_version") != "gradient_reuse.reviewer_extension_unit.v1":
            raise CompletionAggregationError(f"E5 UNIT schema changed: {directory_name}")
        metadata = payload.get("identity", {}).get("unit")
        if not isinstance(metadata, dict) or metadata.get("experiment") != "E5_pmnist_seed_extension":
            raise CompletionAggregationError(f"E5 experiment identity changed: {directory_name}")
        expected_metadata = {
            "benchmark": unit.benchmark,
            "method": unit.method,
            "mammoth_dataset": unit.mammoth_dataset,
            "mammoth_model": unit.mammoth_model,
            "intervention": unit.intervention,
            "seed": unit.seed,
            "permutation_replica": unit.permutation_replica,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise CompletionAggregationError(f"E5 unit identity changed: {directory_name}")
        if payload.get("paper_eligible") is not True:
            raise CompletionAggregationError(f"E5 unit is not paper eligible: {directory_name}")
        expected_args = dict(effective_args(unit, lock, device="cuda:0", smoke=False))
        expected_args.pop("runtime_patch_domain_il_fullacc", None)
        if payload.get("effective_args") != expected_args:
            raise CompletionAggregationError(f"E5 effective arguments differ: {directory_name}")
        metrics_wrapper = payload.get("metrics")
        if not isinstance(metrics_wrapper, dict) or metrics_wrapper.get("primary_setting") != "domain-il":
            raise CompletionAggregationError(f"E5 metric setting changed: {directory_name}")
        metrics = validate_metric(metrics_wrapper.get("primary"), 20, f"{directory_name}/domain-il")
        record = {
            "benchmark": unit.benchmark,
            "method": unit.method,
            "estimand_class": (
                "reference_baseline" if unit.method in REFERENCES else "assignment_only"
            ),
            "intervention": unit.intervention,
            "seed": unit.seed,
            "permutation_replica": unit.permutation_replica,
            "metrics": metrics,
        }
        records.append(record)
        pair_inventory[f"{directory_name}/UNIT.json"] = sha256_file(unit_path)
        pair_inventory[f"{directory_name}/DONE.json"] = sha256_file(root / "DONE.json")
        bindings[unit.unit_id] = {
            "e5_directory": directory_name,
            "unit_sha256": sha256_file(unit_path),
            "done_sha256": sha256_file(root / "DONE.json"),
            "artifact_bundle_sha256": done["artifact_bundle_sha256"],
        }
    if canonical_sha256(pair_inventory) != E5_UNIT_DONE_BUNDLE_SHA256:
        raise CompletionAggregationError("E5 UNIT/DONE byte inventory differs from transfer")
    return records, [dict(record) for record in records], bindings


def normalize_primary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "benchmark": str(row["benchmark"]),
        "method": str(row["method"]),
        "seed": int(row["seed"]),
        "reference_method": str(row["reference_method"]),
        "estimand_class": str(row["estimand_class"]),
        "reference": float(row["reference_accuracy"]),
        "paired": float(row["paired_accuracy"]),
        "permuted_replicas": [float(v) for v in row["permuted_replica_accuracy"]],
        "permuted_mean": float(row["permuted_mean_accuracy"]),
        "permuted_mean_mc_variance": float(row["permuted_mean_accuracy_mc_variance"]),
        "V_total": float(row["accuracy_V_total"]),
        "V_perm": float(row["accuracy_V_perm"]),
        "V_pair": float(row["accuracy_V_pair"]),
        "identity_residual": float(row["accuracy_identity_residual"]),
        "accuracy_unit": "percentage_points",
    }
    validate_estimand_row(output)
    return output


def normalize_e2_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "benchmark": str(row["benchmark"]),
        "method": str(row["method"]),
        "seed": int(row["seed"]),
        "reference_method": str(row["reference_method"]),
        "estimand_class": str(row["estimand_class"]),
        "reference": float(row["reference"]),
        "paired": float(row["paired"]),
        "permuted_replicas": [float(v) for v in row["permuted_replicas"]],
        "permuted_mean": float(row["permuted_mean"]),
        "permuted_mean_mc_variance": float(row["permuted_mean_mc_variance"]),
        "V_total": float(row["V_total"]),
        "V_perm": float(row["V_perm"]),
        "V_pair": float(row["V_pair"]),
        "identity_residual": float(row["identity_residual"]),
        "accuracy_unit": "percentage_points",
    }
    validate_estimand_row(output)
    return output


def validate_estimand_row(row: Mapping[str, Any]) -> None:
    if row.get("benchmark") not in BENCHMARKS or row.get("method") not in METHODS:
        raise CompletionAggregationError("estimand row is outside the assignment-only grid")
    if list(range(4)) != list(REPLICAS) or len(row.get("permuted_replicas", [])) != 4:
        raise CompletionAggregationError("estimand row lacks four nested permutations")
    values = [
        float(row[name])
        for name in ("reference", "paired", "permuted_mean", "V_total", "V_perm", "V_pair")
    ]
    if any(not math.isfinite(value) for value in values):
        raise CompletionAggregationError("estimand row contains a non-finite value")
    residual = float(row["V_total"]) - float(row["V_perm"]) - float(row["V_pair"])
    if abs(residual) > IDENTITY_TOLERANCE or abs(float(row["identity_residual"])) > IDENTITY_TOLERANCE:
        raise CompletionAggregationError(f"estimand identity failed: residual={residual}")


def validate_parent_inputs(parent_summary: Path, parent_e2: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent_summary = regular_file(parent_summary, "frozen three-seed summary")
    parent_e2 = regular_file(parent_e2, "frozen three-seed E2 reanalysis")
    if sha256_file(parent_summary) != PARENT_SUMMARY_SHA256:
        raise CompletionAggregationError("frozen three-seed summary hash mismatch")
    if sha256_file(parent_e2) != PARENT_E2_SHA256:
        raise CompletionAggregationError("frozen E2 hash mismatch")
    summary = strict_json(parent_summary)
    e2 = strict_json(parent_e2)
    if not isinstance(summary, dict) or not (
        summary.get("schema_version") == "gradient_reuse.mechanism_aggregate_summary.v3"
        and summary.get("paper_eligible") is True
        and summary.get("unit_count") == 288
        and summary.get("permutation_replicas_are_nested") is True
    ):
        raise CompletionAggregationError("frozen three-seed summary contract changed")
    if not isinstance(e2, dict) or not (
        e2.get("schema_version") == "gradient_reuse.reviewer_extension_e2.v1"
        and e2.get("status") == "PASS"
        and e2.get("raw_unit_count") == 288
        and e2.get("permutation_replicas") == list(REPLICAS)
        and e2.get("permutation_replicas_are_nested_monte_carlo") is True
    ):
        raise CompletionAggregationError("frozen E2 contract changed")
    primary = [
        normalize_primary_row(row)
        for row in summary.get("seed_rows", [])
        if row.get("estimand_class") == "assignment_only"
    ]
    task_rows = e2.get("endpoints", {}).get("task_il_final_average_accuracy", {}).get("seed_rows", [])
    secondary = [
        normalize_e2_row(row)
        for row in task_rows
        if row.get("estimand_class") == "assignment_only"
    ]
    if len(primary) != 45 or len(secondary) != 45:
        raise CompletionAggregationError(
            f"frozen assignment-only row count changed: primary={len(primary)}, secondary={len(secondary)}"
        )
    if {row["seed"] for row in primary + secondary} != set(FROZEN_SEEDS):
        raise CompletionAggregationError("frozen rows use another seed set")
    return primary, secondary


def exact_df4_two_sided_p(t_statistic: float) -> float:
    """Exact two-sided Student-t p value for four degrees of freedom."""

    t_value = abs(float(t_statistic))
    if math.isnan(t_value):
        raise ValueError("t statistic is NaN")
    if math.isinf(t_value):
        return 0.0
    u = t_value / math.sqrt(t_value * t_value + 4.0)
    return max(0.0, min(1.0, 1.0 - 1.5 * u + 0.5 * u**3))


def pooled_statistic(values: Iterable[float], nested: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    q = [float(value) for value in nested]
    if len(data) != 5 or len(q) != 5:
        raise CompletionAggregationError("pooled inference requires exactly five aligned seeds")
    mean = sum(data) / 5.0
    sample_variance = sum((value - mean) ** 2 for value in data) / 4.0
    mean_nested = sum(q) / 5.0
    if mean_nested < 0 or not math.isfinite(mean_nested):
        raise CompletionAggregationError("nested Monte Carlo variance is invalid")
    standard_error = math.sqrt(max(sample_variance, mean_nested) / 5.0)
    if standard_error == 0:
        t_statistic = math.copysign(math.inf, mean) if mean != 0 else 0.0
    else:
        t_statistic = mean / standard_error
    return {
        "mean": mean,
        "sample_variance_across_run_seeds": sample_variance,
        "nested_permutation_mean_variance": mean_nested,
        "standard_error": standard_error,
        "degrees_of_freedom": 4,
        "t_critical_95": T95_DF4,
        "ci95_low": mean - T95_DF4 * standard_error,
        "ci95_high": mean + T95_DF4 * standard_error,
        "t_statistic": t_statistic,
        "p_two_sided": exact_df4_two_sided_p(t_statistic),
        "interval": "two_sided_student_t_df4_with_nested_permutation_mc",
        "independent_seeds": 5,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in p_values):
        raise ValueError("Holm inputs must be finite probabilities")
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def summarize_endpoint(rows: Sequence[Mapping[str, Any]], endpoint: str) -> dict[str, Any]:
    keys = {(benchmark, method) for benchmark in BENCHMARKS for method in METHODS}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_estimand_row(row)
        groups[(str(row["benchmark"]), str(row["method"]))].append(row)
    if set(groups) != keys:
        raise CompletionAggregationError(f"{endpoint} does not cover exactly 15 cells")
    summaries: list[dict[str, Any]] = []
    for benchmark, method in sorted(keys):
        group = sorted(groups[(benchmark, method)], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in group] != list(ALL_SEEDS):
            raise CompletionAggregationError(f"{endpoint}/{benchmark}/{method} seed coverage changed")
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "reference_method": group[0]["reference_method"],
            "estimand_class": "assignment_only",
        }
        for estimand in ("V_total", "V_perm", "V_pair"):
            nested = [
                0.0 if estimand == "V_total" else float(row["permuted_mean_mc_variance"])
                for row in group
            ]
            summary[estimand] = pooled_statistic(
                [float(row[estimand]) for row in group], nested
            )
        summary["max_abs_identity_residual"] = max(
            abs(float(row["identity_residual"])) for row in group
        )
        summaries.append(summary)
    holm_families: dict[str, Any] = {}
    for estimand in ("V_total", "V_perm", "V_pair"):
        adjusted = holm_adjust(
            [summary[estimand]["p_two_sided"] for summary in summaries]
        )
        for summary, p_adjusted in zip(summaries, adjusted):
            effect = summary[estimand]
            effect["holm_adjusted_p_two_sided"] = p_adjusted
            effect["holm_positive_supported_0.05"] = bool(
                effect["mean"] > 0 and p_adjusted <= 0.05
            )
            effect["holm_negative_supported_0.05"] = bool(
                effect["mean"] < 0 and p_adjusted <= 0.05
            )
        holm_families[estimand] = {
            "estimand": estimand,
            "tests": 15,
            "method": "Holm step-down, two-sided df=4 t tests",
            "positive_supported": sum(
                summary[estimand]["holm_positive_supported_0.05"]
                for summary in summaries
            ),
            "negative_supported": sum(
                summary[estimand]["holm_negative_supported_0.05"]
                for summary in summaries
            ),
        }
    return {
        "endpoint": endpoint,
        "seed_rows": list(rows),
        "summaries": summaries,
        "holm_families": holm_families,
    }


def held_out_report(primary: Sequence[Mapping[str, Any]], secondary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def endpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["benchmark"]), str(row["method"]))].append(row)
        output: list[dict[str, Any]] = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda row: int(row["seed"]))
            if [row["seed"] for row in group] != list(COMPLETION_SEEDS):
                raise CompletionAggregationError(f"held-out seed coverage changed: {key}")
            output.append(
                {
                    "benchmark": key[0],
                    "method": key[1],
                    "seed_values": [
                        {name: row[name] for name in ("seed", "V_total", "V_perm", "V_pair")}
                        for row in group
                    ],
                    "mean_V_total": sum(float(row["V_total"]) for row in group) / 2.0,
                    "mean_V_perm": sum(float(row["V_perm"]) for row in group) / 2.0,
                    "mean_V_pair": sum(float(row["V_pair"]) for row in group) / 2.0,
                    "positive_V_pair_seeds": sum(float(row["V_pair"]) > 0 for row in group),
                }
            )
        if len(output) != 15:
            raise CompletionAggregationError("held-out report does not cover 15 cells")
        return output

    return {
        "schema_version": "gradient_reuse.primary_seed_completion_held_out.v1",
        "status": "PASS",
        "scientific_result": True,
        "analysis_role": "descriptive held-out directional replication; no n=2 significance claims",
        "independent_seeds": list(COMPLETION_SEEDS),
        "permutation_replicas_are_nested_monte_carlo": True,
        "endpoints": {
            "class_or_domain_il_final_average_accuracy": endpoint_rows(primary),
            "task_or_domain_il_final_average_accuracy": endpoint_rows(secondary),
        },
    }


def verify_row_coverage(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], role: str) -> None:
    expected = {
        (benchmark, method, seed)
        for benchmark in BENCHMARKS
        for method in METHODS
        for seed in seeds
    }
    observed = [(str(row["benchmark"]), str(row["method"]), int(row["seed"])) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise CompletionAggregationError(f"{role} row coverage is not exact")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project", required=True)
    result.add_argument("--completion-root", required=True)
    result.add_argument("--parent-summary", required=True)
    result.add_argument("--parent-e2", required=True)
    result.add_argument("--e5-root", required=True)
    result.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    project = regular_directory(Path(arguments.project), "source project")
    design_path = project / "configs/primary_seed_completion.json"
    design = load_design(design_path)
    completion_root = regular_directory(Path(arguments.completion_root), "completion root")
    output = Path(arguments.output)
    if output != completion_root / "PRIMARY_SEED_COMPLETION_AGGREGATE.json":
        raise CompletionAggregationError("aggregate output must be inside the bound completion root")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise CompletionAggregationError(f"aggregate output path is unsafe: {output}")

    (
        _MechanismUnit,
        compute_estimands,
        enumerate_units,
        load_lock,
        helpers,
    ) = import_protocol(project)
    lock_path = regular_file(project / "configs/mechanisms.json", "full mechanism lock")
    if sha256_file(lock_path) != PUBLIC_FULL_LOCK_SHA256:
        raise CompletionAggregationError("public full mechanisms lock hash mismatch")
    lock = load_lock(lock_path)
    new_expected, e5_expected = expected_units(lock, enumerate_units)

    parent_primary, parent_secondary = validate_parent_inputs(
        Path(arguments.parent_summary), Path(arguments.parent_e2)
    )
    new_primary_records, new_task_records, new_bindings = validate_new_units(
        completion_root, project, lock, new_expected, helpers
    )
    e5_primary_records, e5_task_records, e5_bindings = validate_e5_units(
        Path(arguments.e5_root), lock, e5_expected, helpers[2]
    )

    new_primary_rows = [normalize_primary_row(row) for row in compute_estimands(new_primary_records)]
    new_secondary_rows = [normalize_primary_row(row) for row in compute_estimands(new_task_records)]
    e5_primary_rows = [normalize_primary_row(row) for row in compute_estimands(e5_primary_records)]
    e5_secondary_rows = [normalize_primary_row(row) for row in compute_estimands(e5_task_records)]
    added_primary = sorted(
        new_primary_rows + e5_primary_rows,
        key=lambda row: (row["benchmark"], row["method"], row["seed"]),
    )
    added_secondary = sorted(
        new_secondary_rows + e5_secondary_rows,
        key=lambda row: (row["benchmark"], row["method"], row["seed"]),
    )
    verify_row_coverage(added_primary, COMPLETION_SEEDS, "held-out primary")
    verify_row_coverage(added_secondary, COMPLETION_SEEDS, "held-out secondary")

    primary_rows = sorted(
        parent_primary + added_primary,
        key=lambda row: (row["benchmark"], row["method"], row["seed"]),
    )
    secondary_rows = sorted(
        parent_secondary + added_secondary,
        key=lambda row: (row["benchmark"], row["method"], row["seed"]),
    )
    verify_row_coverage(primary_rows, ALL_SEEDS, "pooled primary")
    verify_row_coverage(secondary_rows, ALL_SEEDS, "pooled secondary")

    held_out = held_out_report(added_primary, added_secondary)
    pooled = {
        "schema_version": "gradient_reuse.primary_seed_completion_five_seed_summary.v1",
        "status": "PASS",
        "scientific_result": True,
        "analysis_role": "post-hoc completion of the seed set in the original pre-outcome lock",
        "frozen_primary_analysis_preserved": True,
        "independent_seeds": list(ALL_SEEDS),
        "degrees_of_freedom": 4,
        "permutation_replicas": list(REPLICAS),
        "permutation_replicas_are_nested_monte_carlo": True,
        "endpoints": {
            "class_or_domain_il_final_average_accuracy": summarize_endpoint(
                primary_rows, "class_or_domain_il_final_average_accuracy"
            ),
            "task_or_domain_il_final_average_accuracy": summarize_endpoint(
                secondary_rows, "task_or_domain_il_final_average_accuracy"
            ),
        },
    }
    held_path = completion_root / "HELD_OUT_REPLICATION.json"
    pooled_path = completion_root / "FIVE_SEED_PRIMARY_SUMMARY.json"
    atomic_json(held_path, held_out)
    atomic_json(pooled_path, pooled)
    aggregate = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "analysis_role": {
            "frozen_three_seed": "prespecified primary, unchanged",
            "held_out_two_seed": "descriptive directional replication",
            "pooled_five_seed": "post-hoc completion of originally specified seed set",
        },
        "design_lock_sha256": sha256_file(design_path),
        "original_lock_sha256": HISTORICAL_FULL_LOCK_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_e2_sha256": PARENT_E2_SHA256,
        "new_unit_count": 108,
        "reused_e5_unit_count": 54,
        "pooled_seed_rows_per_endpoint": 75,
        "pooled_cells_per_endpoint": 15,
        "new_unit_bindings": new_bindings,
        "reused_e5_bindings": e5_bindings,
        "new_unit_binding_sha256": canonical_sha256(new_bindings),
        "reused_e5_binding_sha256": canonical_sha256(e5_bindings),
        "outputs": {
            "FIVE_SEED_PRIMARY_SUMMARY.json": sha256_file(pooled_path),
            "HELD_OUT_REPLICATION.json": sha256_file(held_path),
        },
        "outcome_independent_gates": {
            gate: "PASS" for gate in design["outcome_independent_gates"]
        },
    }
    atomic_json(output, aggregate)
    print(output)
    print(f"sha256={sha256_file(output)}")
    print("PRIMARY FIVE-SEED COMPLETION AGGREGATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
