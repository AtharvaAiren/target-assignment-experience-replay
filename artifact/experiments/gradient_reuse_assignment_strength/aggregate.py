"""Fail-closed aggregation for the assignment-strength dose--response follow-up.

The scientific replicate is a fresh run seed.  The four assignment draws at
each nonzero strength are nested Monte Carlo draws, not independent runs.  All
432 locked trajectories must be sealed before this module will publish an
aggregate.  The resulting intervals use a conservative nested-Monte-Carlo
variance floor and Student-t calibration over the twelve independent seeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats

from .integrity import (
    atomic_json,
    executable_entry,
    git_output,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from .protocol import (
    BENCHMARKS,
    EXPECTED_UNITS,
    LAMBDA_BASIS_POINTS,
    PERMUTATION_REPLICAS,
    RUN_SEEDS,
    SweepUnit,
    canonical_sha256,
    enumerate_units,
    load_lock,
    sha256_file,
    strict_json,
)
from .source import validate_source_manifest
from .unit_runner import effective_args, validate_completed_unit


SCHEMA = "gradient_reuse.assignment_strength_aggregate.v1"
AGGREGATE_NAME = "ASSIGNMENT_STRENGTH_AGGREGATE.json"
DF = len(RUN_SEEDS) - 1
ALPHA = 0.05
HEX = frozenset("0123456789abcdef")
DIAGNOSTIC_RATES = (
    "realized_lambda",
    "selected_fraction",
    "index_displaced_fraction",
    "target_value_changed_fraction",
)
SCALAR_ENDPOINTS = (
    "followup_primary_final_average_accuracy",
    "benchmark_default_final_average_accuracy",
    "followup_primary_average_incremental_accuracy",
    "benchmark_default_average_incremental_accuracy",
    "followup_primary_backward_transfer",
    "benchmark_default_backward_transfer",
)


class AssignmentStrengthAggregationError(RuntimeError):
    """Raised when the locked evidence cannot support an aggregate."""


def _fail(message: str) -> None:
    raise AssignmentStrengthAggregationError(message)


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        _fail(f"{context} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AssignmentStrengthAggregationError(
            f"{context} is not numeric"
        ) from error
    if not math.isfinite(number):
        _fail(f"{context} is not finite")
    return number


def _digest(value: Any, context: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    ):
        _fail(f"invalid SHA-256 for {context}")
    return value


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if not data:
        _fail("cannot average an empty collection")
    return sum(data) / len(data)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        _fail("sample variance requires at least two values")
    center = _mean(values)
    return sum((float(value) - center) ** 2 for value in values) / (
        len(values) - 1
    )


def _nested_variance_of_mean(values: Sequence[float]) -> float:
    """Return the estimated variance of a four-draw Monte Carlo mean."""

    if len(values) != len(PERMUTATION_REPLICAS):
        _fail("nested Monte Carlo variance requires exactly four replicas")
    return _sample_variance(values) / len(values)


def student_t_summary(
    values: Iterable[float],
    *,
    nested_mc_variances: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Summarize twelve seed values with the locked conservative variance floor."""

    data = [float(value) for value in values]
    if len(data) != len(RUN_SEEDS) or any(not math.isfinite(value) for value in data):
        _fail("Student-t summary requires twelve finite independent-seed values")
    center = _mean(data)
    seed_variance = _sample_variance(data)
    if nested_mc_variances is None:
        nested = [0.0] * len(data)
    else:
        nested = [float(value) for value in nested_mc_variances]
        if len(nested) != len(data) or any(
            not math.isfinite(value) or value < 0.0 for value in nested
        ):
            _fail("nested Monte Carlo variances must align and be nonnegative")
    mean_nested = _mean(nested)
    variance_used = max(seed_variance, mean_nested)
    standard_error = math.sqrt(variance_used / len(data))
    critical = float(stats.t.ppf(1.0 - ALPHA / 2.0, DF))
    if standard_error == 0.0:
        # JSON deliberately has no infinity: the p-value still records the
        # exact degenerate decision, while ``None`` signals an undefined
        # finite t ratio.
        statistic = None
        p_value = 1.0 if center == 0.0 else 0.0
    else:
        statistic = center / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), DF))
    return {
        "mean": center,
        "independent_seed_values": data,
        "independent_seed_count": len(data),
        "degrees_of_freedom": DF,
        "seed_sample_variance": seed_variance,
        "mean_nested_mc_variance": mean_nested,
        "variance_floor_rule": "max(seed_sample_variance,mean_nested_mc_variance)",
        "variance_used": variance_used,
        "standard_error": standard_error,
        "critical_value_0_975": critical,
        "ci95": [center - critical * standard_error, center + critical * standard_error],
        "t_statistic": statistic,
        "raw_two_sided_p": p_value,
        "interval": "two_sided_student_t_over_12_run_seeds_with_nested_mc_variance_floor",
    }


def holm_adjust(
    hypotheses: Sequence[tuple[str, float]],
) -> dict[str, dict[str, Any]]:
    """Return deterministic Holm-adjusted p-values for one declared family."""

    if not hypotheses or len({key for key, _ in hypotheses}) != len(hypotheses):
        _fail("Holm family must be nonempty and uniquely keyed")
    ordered: list[tuple[float, str]] = []
    for key, raw in hypotheses:
        value = _finite(raw, f"Holm p-value {key}")
        if not 0.0 <= value <= 1.0:
            _fail(f"Holm p-value is outside [0,1]: {key}")
        ordered.append((value, key))
    ordered.sort(key=lambda row: (row[0], row[1]))
    adjusted: dict[str, dict[str, Any]] = {}
    running = 0.0
    total = len(ordered)
    for rank, (raw, key) in enumerate(ordered, 1):
        running = max(running, min(1.0, (total - rank + 1) * raw))
        adjusted[key] = {
            "family_size": total,
            "holm_rank": rank,
            "raw_two_sided_p": raw,
            "holm_adjusted_p": running,
            "reject_at_0_05": running <= ALPHA,
        }
    return adjusted


def _metric_payload(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} metrics are absent")
    if value.get("accuracy_unit") != "percentage_points":
        _fail(f"{context} accuracy unit changed")
    if int(value.get("evaluation_tasks", 0)) != 20:
        _fail(f"{context} does not contain 20 evaluation tasks")
    matrix = value.get("accuracy_matrix")
    trajectory = value.get("mean_accuracy_trajectory")
    if not isinstance(matrix, list) or len(matrix) != 20:
        _fail(f"{context} accuracy matrix changed")
    if not isinstance(trajectory, list) or len(trajectory) != 20:
        _fail(f"{context} accuracy trajectory changed")
    normalized_matrix: list[list[float]] = []
    for row_number, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != row_number:
            _fail(f"{context} accuracy matrix is not triangular")
        values = [_finite(entry, f"{context} accuracy") for entry in row]
        if any(not 0.0 <= entry <= 100.0 for entry in values):
            _fail(f"{context} accuracy lies outside [0,100]")
        normalized_matrix.append(values)
    expected_trajectory = [_mean(row) for row in normalized_matrix]
    observed_trajectory = [
        _finite(entry, f"{context} trajectory") for entry in trajectory
    ]
    if any(
        abs(observed - expected) > 1.0e-8
        for observed, expected in zip(observed_trajectory, expected_trajectory)
    ):
        _fail(f"{context} trajectory is inconsistent with its matrix")
    expected_final = expected_trajectory[-1]
    expected_incremental = _mean(expected_trajectory)
    expected_bwt = _mean(
        normalized_matrix[-1][index] - normalized_matrix[index][index]
        for index in range(19)
    )
    for field, expected in (
        ("final_average_accuracy", expected_final),
        ("average_incremental_accuracy", expected_incremental),
        ("backward_transfer", expected_bwt),
    ):
        if abs(_finite(value.get(field), f"{context}.{field}") - expected) > 1.0e-8:
            _fail(f"{context}.{field} is inconsistent with its matrix")
    return value


def scalar_endpoints(record: Mapping[str, Any], unit: SweepUnit) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "primary_setting",
        "primary",
        "task_il",
        "effective_argument_evidence",
    }:
        _fail(f"unit metrics are absent: {unit.unit_id}")
    expected_setting = "class-il" if unit.benchmark == "split_cifar100" else "domain-il"
    if metrics.get("primary_setting") != expected_setting:
        _fail(f"primary setting changed: {unit.unit_id}")
    benchmark_default = _metric_payload(
        metrics.get("primary"), f"{unit.unit_id}/primary"
    )
    task_raw = metrics.get("task_il")
    if unit.benchmark == "split_cifar100":
        followup_primary = _metric_payload(task_raw, f"{unit.unit_id}/task-il")
    else:
        if task_raw is not None:
            _fail(f"domain-IL unit unexpectedly emitted task-IL metrics: {unit.unit_id}")
        followup_primary = benchmark_default
    return {
        "followup_primary_final_average_accuracy": _finite(
            followup_primary["final_average_accuracy"],
            "follow-up primary final accuracy",
        ),
        "benchmark_default_final_average_accuracy": _finite(
            benchmark_default["final_average_accuracy"],
            "benchmark-default final accuracy",
        ),
        "followup_primary_average_incremental_accuracy": _finite(
            followup_primary["average_incremental_accuracy"],
            "follow-up primary incremental accuracy",
        ),
        "benchmark_default_average_incremental_accuracy": _finite(
            benchmark_default["average_incremental_accuracy"],
            "benchmark-default incremental accuracy",
        ),
        "followup_primary_backward_transfer": _finite(
            followup_primary["backward_transfer"], "follow-up primary BWT"
        ),
        "benchmark_default_backward_transfer": _finite(
            benchmark_default["backward_transfer"], "benchmark-default BWT"
        ),
        "followup_primary_mean_accuracy_trajectory": [
            _finite(value, "follow-up primary trajectory")
            for value in followup_primary["mean_accuracy_trajectory"]
        ],
        "benchmark_default_mean_accuracy_trajectory": [
            _finite(value, "benchmark-default trajectory")
            for value in benchmark_default["mean_accuracy_trajectory"]
        ],
    }


def _validate_effective_argument_evidence(
    record: Mapping[str, Any], unit: SweepUnit
) -> None:
    """Validate the sealed evidence for arguments Mammoth actually recorded."""

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        _fail(f"unit metrics are absent: {unit.unit_id}")
    evidence = metrics.get("effective_argument_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {"primary", "task_il"}:
        _fail(f"effective-argument evidence changed: {unit.unit_id}")
    primary = evidence.get("primary")
    if not isinstance(primary, Mapping) or set(primary) != {"sha256", "effective_args"}:
        _fail(f"primary effective-argument evidence changed: {unit.unit_id}")
    _digest(primary.get("sha256"), f"{unit.unit_id}/primary Mammoth log")
    observed = primary.get("effective_args")
    intended = record.get("effective_args")
    if not isinstance(observed, Mapping) or not isinstance(intended, Mapping):
        _fail(f"effective arguments are absent: {unit.unit_id}")
    expected = dict(intended)
    for flag in ("enable_other_metrics", "non_verbose"):
        expected[flag] = bool(expected[flag])
    expected.update(
        {
            "base_path": observed.get("base_path"),
            "results_path": observed.get("results_path"),
            "debug_mode": False,
        }
    )
    if unit.method == "experience_replay":
        expected["assignment_audit_path"] = observed.get("assignment_audit_path")
    if dict(observed) != expected:
        _fail(f"logged and intended effective arguments disagree: {unit.unit_id}")
    for field in ("base_path", "results_path"):
        value = observed.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            _fail(f"logged {field} is not absolute: {unit.unit_id}")
    if unit.method == "experience_replay":
        audit_path = observed.get("assignment_audit_path")
        if not (
            isinstance(audit_path, str)
            and Path(audit_path).is_absolute()
            and Path(audit_path).name == "assignment_audit.jsonl"
        ):
            _fail(f"logged assignment-audit path changed: {unit.unit_id}")
    task_il = evidence.get("task_il")
    if unit.benchmark == "split_cifar100":
        if not isinstance(task_il, Mapping) or set(task_il) != {
            "sha256",
            "effective_args",
        }:
            _fail(f"task-IL effective-argument evidence changed: {unit.unit_id}")
        _digest(task_il.get("sha256"), f"{unit.unit_id}/task-IL Mammoth log")
        if task_il.get("effective_args") != observed:
            _fail(f"class-IL and task-IL effective arguments disagree: {unit.unit_id}")
    elif task_il is not None:
        _fail(f"domain-IL unit has task-IL argument evidence: {unit.unit_id}")


def _unit_from_record(record: Mapping[str, Any]) -> SweepUnit:
    raw = record.get("unit")
    if not isinstance(raw, Mapping) or set(raw) != {
        "benchmark",
        "method",
        "mammoth_dataset",
        "mammoth_model",
        "seed",
        "lambda_bp",
        "permutation_replica",
    }:
        _fail("UNIT has an invalid unit identity payload")
    try:
        unit = SweepUnit(
            benchmark=str(raw["benchmark"]),
            method=str(raw["method"]),
            mammoth_dataset=str(raw["mammoth_dataset"]),
            mammoth_model=str(raw["mammoth_model"]),
            seed=int(raw["seed"]),
            lambda_bp=None if raw["lambda_bp"] is None else int(raw["lambda_bp"]),
            permutation_replica=(
                None
                if raw["permutation_replica"] is None
                else int(raw["permutation_replica"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AssignmentStrengthAggregationError("invalid UNIT identity values") from error
    if unit.as_dict() != dict(raw):
        _fail(f"UNIT identity uses noncanonical scalar types: {unit.unit_id}")
    return unit


def _validate_logical_record(
    record: Mapping[str, Any], unit: SweepUnit, lock: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "identity",
        "unit",
        "effective_args",
        "metrics",
        "assignment_diagnostics",
        "runtime_seconds",
        "attempt",
        "paper_eligible",
    }
    if set(record) != expected_keys:
        _fail(f"UNIT schema changed: {unit.unit_id}")
    if not (
        record.get("schema_version") == "gradient_reuse.assignment_strength_unit.v1"
        and record.get("paper_eligible") is True
        and record.get("unit") == unit.as_dict()
    ):
        _fail(f"UNIT metadata changed: {unit.unit_id}")
    if record.get("effective_args") != effective_args(unit, lock, "cuda:0"):
        _fail(f"UNIT effective arguments changed: {unit.unit_id}")
    runtime = _finite(record.get("runtime_seconds"), f"{unit.unit_id} runtime")
    if runtime < 0.0:
        _fail(f"UNIT runtime is negative: {unit.unit_id}")
    attempt = record.get("attempt")
    if not (
        isinstance(attempt, str)
        and len(attempt) == 11
        and attempt.startswith("attempt_")
        and attempt[8:].isdigit()
    ):
        _fail(f"UNIT attempt is invalid: {unit.unit_id}")
    _validate_effective_argument_evidence(record, unit)
    scalar_endpoints(record, unit)
    diagnostics = record.get("assignment_diagnostics")
    if not isinstance(diagnostics, Mapping):
        _fail(f"assignment diagnostics are absent: {unit.unit_id}")
    if unit.method == "reference":
        if dict(diagnostics) != {"record_count": 0}:
            _fail(f"reference emitted assignment diagnostics: {unit.unit_id}")
        return
    required = {
        "record_count",
        "intervention_records",
        "gradient_diagnostic_records",
        "minimum_replay_batch_rows",
        "maximum_replay_batch_rows",
        "means",
        "minima",
        "maxima",
        "sampled_event_fingerprints",
        "audit_sha256",
    }
    if set(diagnostics) != required:
        _fail(f"assignment diagnostic schema changed: {unit.unit_id}")
    if min(
        int(diagnostics.get("record_count", 0)),
        int(diagnostics.get("intervention_records", 0)),
        int(diagnostics.get("gradient_diagnostic_records", 0)),
    ) < 1:
        _fail(f"assignment diagnostics are empty: {unit.unit_id}")
    minimum_rows = int(diagnostics.get("minimum_replay_batch_rows", 0))
    maximum_rows = int(diagnostics.get("maximum_replay_batch_rows", 0))
    if minimum_rows < 1 or maximum_rows < minimum_rows:
        _fail(f"assignment replay-batch row range changed: {unit.unit_id}")
    _digest(diagnostics.get("audit_sha256"), f"{unit.unit_id} audit")
    for section_name in ("means", "minima", "maxima"):
        section = diagnostics.get(section_name)
        if not isinstance(section, Mapping) or set(section) != set(DIAGNOSTIC_RATES):
            _fail(f"assignment diagnostic rates changed: {unit.unit_id}/{section_name}")
        for name in DIAGNOSTIC_RATES:
            value = _finite(section[name], f"{unit.unit_id}/{section_name}/{name}")
            if not 0.0 <= value <= 1.0:
                _fail(f"assignment diagnostic rate is invalid: {unit.unit_id}/{name}")
            if not (
                _finite(diagnostics["minima"][name], "diagnostic minimum")
                <= _finite(diagnostics["means"][name], "diagnostic mean")
                <= _finite(diagnostics["maxima"][name], "diagnostic maximum")
            ):
                _fail(f"assignment diagnostic extrema are inconsistent: {unit.unit_id}/{name}")
    fingerprints = diagnostics.get("sampled_event_fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints:
        _fail(f"sampled-event fingerprints are absent: {unit.unit_id}")
    for row in fingerprints:
        if not isinstance(row, Mapping) or set(row) != {
            "observation",
            "event",
            "channel",
            "sampled_indexes_sha256",
            "input_sha256",
            "original_target_sha256",
        }:
            _fail(f"sampled-event fingerprint schema changed: {unit.unit_id}")
        for name in ("sampled_indexes_sha256", "input_sha256", "original_target_sha256"):
            _digest(row.get(name), f"{unit.unit_id}/{name}")


def _index_records(
    records: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> tuple[
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, int, int, int | None], Mapping[str, Any]],
]:
    expected_units = enumerate_units(lock)
    expected = {unit.unit_id: unit for unit in expected_units}
    if len(records) != EXPECTED_UNITS:
        _fail(f"aggregate requires {EXPECTED_UNITS} UNIT records, found {len(records)}")
    observed: dict[str, tuple[SweepUnit, Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            _fail("UNIT record is not an object")
        unit = _unit_from_record(record)
        if unit.unit_id in observed:
            _fail(f"duplicate UNIT record: {unit.unit_id}")
        if unit.unit_id not in expected or unit != expected[unit.unit_id]:
            _fail(f"unexpected UNIT identity: {unit.unit_id}")
        _validate_logical_record(record, unit, lock)
        observed[unit.unit_id] = (unit, record)
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        _fail(f"locked unit coverage is incomplete; missing={missing[:3]}")
    references: dict[tuple[str, int], Mapping[str, Any]] = {}
    replay: dict[tuple[str, int, int, int | None], Mapping[str, Any]] = {}
    for unit_id in (unit.unit_id for unit in expected_units):
        unit, record = observed[unit_id]
        if unit.method == "reference":
            key = (unit.benchmark, unit.seed)
            if key in references:
                _fail(f"duplicate reference: {key}")
            references[key] = record
        elif unit.method == "experience_replay" and unit.lambda_bp is not None:
            key = (
                unit.benchmark,
                unit.seed,
                unit.lambda_bp,
                unit.permutation_replica,
            )
            if key in replay:
                _fail(f"duplicate replay arm: {key}")
            replay[key] = record
        else:
            _fail(f"unsupported locked arm: {unit.unit_id}")
    return references, replay


def _check_sampled_event_comparability(
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]]
) -> dict[str, Any]:
    """Authenticate that assignment is the only replay-batch change at audits."""

    groups = []
    for benchmark in BENCHMARKS:
        for seed in RUN_SEEDS:
            base = replay[(benchmark, seed, 0, None)]["assignment_diagnostics"][
                "sampled_event_fingerprints"
            ]
            compared = 0
            for strength in LAMBDA_BASIS_POINTS[1:]:
                for replica in PERMUTATION_REPLICAS:
                    candidate = replay[(benchmark, seed, strength, replica)][
                        "assignment_diagnostics"
                    ]["sampled_event_fingerprints"]
                    if candidate != base:
                        _fail(
                            "sampled replay input/index/original-target fingerprints differ "
                            f"across strengths: {benchmark}/seed={seed}/lambda={strength}/replica={replica}"
                        )
                    compared += 1
            groups.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "sampled_events": len(base),
                    "arms_compared_to_lambda_0": compared,
                    "status": "PASS",
                }
            )
    return {
        "status": "PASS",
        "interpretation": "sampled replay inputs, indexes, and original targets match; only target assignment varies at audited events",
        "groups": groups,
    }


def _replica_records(
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]],
    benchmark: str,
    seed: int,
    strength: int,
) -> list[Mapping[str, Any]]:
    if strength == 0:
        return [replay[(benchmark, seed, strength, None)]]
    return [
        replay[(benchmark, seed, strength, replica)]
        for replica in PERMUTATION_REPLICAS
    ]


def _endpoint_value(record: Mapping[str, Any], endpoint: str) -> float:
    unit = _unit_from_record(record)
    values = scalar_endpoints(record, unit)
    return float(values[endpoint])


def _trajectory_value(
    record: Mapping[str, Any], endpoint: str, task_index: int
) -> float:
    unit = _unit_from_record(record)
    values = scalar_endpoints(record, unit)
    trajectory = values[endpoint]
    if not isinstance(trajectory, list) or len(trajectory) != 20:
        _fail(f"trajectory endpoint changed: {unit.unit_id}/{endpoint}")
    return float(trajectory[task_index])


def _seed_strength_values(
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]],
    benchmark: str,
    endpoint: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    output: dict[int, dict[int, dict[str, Any]]] = {}
    for seed in RUN_SEEDS:
        strengths: dict[int, dict[str, Any]] = {}
        for strength in LAMBDA_BASIS_POINTS:
            values = [
                _endpoint_value(record, endpoint)
                for record in _replica_records(replay, benchmark, seed, strength)
            ]
            strengths[strength] = {
                "replica_values": values,
                "mean": _mean(values),
                "nested_mc_variance_of_mean": (
                    0.0 if strength == 0 else _nested_variance_of_mean(values)
                ),
            }
        output[seed] = strengths
    return output


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        _fail("OLS slope inputs do not align")
    x_mean = _mean(x)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        _fail("OLS strength grid has zero variance")
    y_mean = _mean(y)
    return sum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y)
    ) / denominator


def _slope_summary(seed_strengths: Mapping[int, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    x = [strength / 10000.0 for strength in LAMBDA_BASIS_POINTS]
    seed_rows = []
    slopes = []
    nested = []
    for seed in RUN_SEEDS:
        shared_zero = float(seed_strengths[seed][0]["mean"])
        replica_slopes = []
        for replica_index, replica in enumerate(PERMUTATION_REPLICAS):
            y = [shared_zero]
            for strength in LAMBDA_BASIS_POINTS[1:]:
                values = seed_strengths[seed][strength]["replica_values"]
                if len(values) != 4:
                    _fail("nonzero strength lacks four nested values")
                y.append(float(values[replica_index]))
            replica_slopes.append(_ols_slope(x, y))
        seed_slope = _mean(replica_slopes)
        seed_nested = _nested_variance_of_mean(replica_slopes)
        slopes.append(seed_slope)
        nested.append(seed_nested)
        seed_rows.append(
            {
                "seed": seed,
                "replica_slopes": replica_slopes,
                "mean_slope": seed_slope,
                "nested_mc_variance_of_mean_slope": seed_nested,
            }
        )
    return {
        "x_axis": "requested_lambda",
        "units": "percentage_points_per_unit_requested_lambda",
        "seed_rows": seed_rows,
        "inference": student_t_summary(slopes, nested_mc_variances=nested),
        "inference_role": "descriptive_unadjusted_unless_promoted_by_parent_family",
        "negative_supported_interpretation": (
            "graded negative trend, not proof of monotonicity"
        ),
    }


def _audit_estimands(
    reference_values: Sequence[float],
    seed_strengths: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return the binary endpoint estimands embedded in the graded sweep."""

    rows = []
    total_values = []
    perm_values = []
    pair_values = []
    full_nested = []
    for seed, reference in zip(RUN_SEEDS, reference_values):
        identity = float(seed_strengths[seed][0]["mean"])
        full = float(seed_strengths[seed][10000]["mean"])
        nested = float(
            seed_strengths[seed][10000]["nested_mc_variance_of_mean"]
        )
        total = identity - float(reference)
        perm = full - float(reference)
        pair = identity - full
        residual = total - (perm + pair)
        if abs(residual) > 1.0e-12:
            _fail("assignment-strength endpoint decomposition lost numerical closure")
        total_values.append(total)
        perm_values.append(perm)
        pair_values.append(pair)
        full_nested.append(nested)
        rows.append(
            {
                "seed": seed,
                "reference": float(reference),
                "identity_replay": identity,
                "full_reassignment_replay_mean": full,
                "v_total": total,
                "v_perm": perm,
                "v_pair": pair,
                "decomposition_residual": residual,
                "full_reassignment_nested_mc_variance_of_mean": nested,
            }
        )
    return {
        "inference_role": "descriptive_unadjusted",
        "definitions": {
            "v_total": "M(lambda=0)-M(no_replay_reference)",
            "v_perm": "M(lambda=1)-M(no_replay_reference)",
            "v_pair": "M(lambda=0)-M(lambda=1)",
            "identity": "v_total=v_perm+v_pair",
        },
        "seed_rows": rows,
        "v_total": student_t_summary(total_values),
        "v_perm": student_t_summary(
            perm_values, nested_mc_variances=full_nested
        ),
        "v_pair": student_t_summary(
            pair_values, nested_mc_variances=full_nested
        ),
        "maximum_absolute_seed_decomposition_residual": max(
            abs(float(row["decomposition_residual"])) for row in rows
        ),
    }


def _endpoint_summary(
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]],
    benchmark: str,
    endpoint: str,
) -> dict[str, Any]:
    seed_strengths = _seed_strength_values(replay, benchmark, endpoint)
    reference_values = [
        _endpoint_value(references[(benchmark, seed)], endpoint) for seed in RUN_SEEDS
    ]
    reference_summary = student_t_summary(reference_values)
    curve = []
    usefulness = []
    for strength in LAMBDA_BASIS_POINTS:
        values = [float(seed_strengths[seed][strength]["mean"]) for seed in RUN_SEEDS]
        nested = [
            float(seed_strengths[seed][strength]["nested_mc_variance_of_mean"])
            for seed in RUN_SEEDS
        ]
        effects = [value - reference for value, reference in zip(values, reference_values)]
        point = student_t_summary(values, nested_mc_variances=nested)
        effect = student_t_summary(effects, nested_mc_variances=nested)
        curve.append(
            {
                "lambda_bp": strength,
                "requested_lambda": strength / 10000.0,
                "replicas_per_seed": 1 if strength == 0 else 4,
                "accuracy": point,
                "inference_role": "descriptive_unadjusted",
            }
        )
        usefulness.append(
            {
                "lambda_bp": strength,
                "requested_lambda": strength / 10000.0,
                "contrast": "experience_replay_minus_same_seed_no_replay_reference",
                "effect": effect,
                "inference_role": "descriptive_unadjusted_unless_promoted_by_parent_family",
            }
        )
    adjacent = []
    for previous, current in zip(LAMBDA_BASIS_POINTS, LAMBDA_BASIS_POINTS[1:]):
        seed_differences = []
        seed_nested = []
        seed_rows = []
        for seed in RUN_SEEDS:
            previous_values = seed_strengths[seed][previous]["replica_values"]
            current_values = seed_strengths[seed][current]["replica_values"]
            if previous == 0:
                left = [float(previous_values[0])] * 4
            else:
                left = [float(value) for value in previous_values]
            right = [float(value) for value in current_values]
            differences = [new - old for new, old in zip(right, left)]
            seed_mean = _mean(differences)
            nested_variance = _nested_variance_of_mean(differences)
            seed_differences.append(seed_mean)
            seed_nested.append(nested_variance)
            seed_rows.append(
                {
                    "seed": seed,
                    "replica_differences": differences,
                    "mean_difference": seed_mean,
                    "nested_mc_variance_of_mean_difference": nested_variance,
                }
            )
        adjacent.append(
            {
                "from_lambda_bp": previous,
                "to_lambda_bp": current,
                "contrast": "higher_strength_minus_lower_strength",
                "seed_rows": seed_rows,
                "inference": student_t_summary(
                    seed_differences, nested_mc_variances=seed_nested
                ),
                "inference_role": "descriptive_unadjusted",
            }
        )
    return {
        "reference": reference_summary,
        "reference_inference_role": "descriptive_unadjusted",
        "strength_curve": curve,
        "usefulness_vs_reference": usefulness,
        "audit_estimands": _audit_estimands(reference_values, seed_strengths),
        "slope": _slope_summary(seed_strengths),
        "adjacent_differences": adjacent,
        "monotonicity_point_estimate_rule": {
            "rule": "all four pooled adjacent higher-minus-lower point estimates are nonpositive",
            "all_adjacent_means_nonpositive": all(
                float(row["inference"]["mean"]) <= 0.0 for row in adjacent
            ),
            "not_an_inferential_monotonicity_test": True,
        },
    }


def _trajectory_curve(
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]],
    benchmark: str,
    endpoint: str,
) -> dict[str, Any]:
    reference = []
    for task_index in range(20):
        values = [
            _trajectory_value(references[(benchmark, seed)], endpoint, task_index)
            for seed in RUN_SEEDS
        ]
        reference.append(student_t_summary(values))
    curves = []
    for strength in LAMBDA_BASIS_POINTS:
        tasks = []
        for task_index in range(20):
            seed_values = []
            nested = []
            for seed in RUN_SEEDS:
                values = [
                    _trajectory_value(record, endpoint, task_index)
                    for record in _replica_records(
                        replay, benchmark, seed, strength
                    )
                ]
                seed_values.append(_mean(values))
                nested.append(
                    0.0 if strength == 0 else _nested_variance_of_mean(values)
                )
            tasks.append(
                student_t_summary(seed_values, nested_mc_variances=nested)
            )
        curves.append(
            {
                "lambda_bp": strength,
                "requested_lambda": strength / 10000.0,
                "tasks": tasks,
            }
        )
    return {
        "task_numbers": list(range(1, 21)),
        "reference": reference,
        "strength_curves": curves,
    }


def _diagnostic_summary(
    replay: Mapping[tuple[str, int, int, int | None], Mapping[str, Any]],
    benchmark: str,
) -> list[dict[str, Any]]:
    rows = []
    for strength in LAMBDA_BASIS_POINTS:
        rates: dict[str, Any] = {}
        audited_row_minima: list[int] = []
        audited_row_maxima: list[int] = []
        for name in DIAGNOSTIC_RATES:
            seed_values = []
            nested = []
            all_unit_means = []
            all_unit_minima = []
            all_unit_maxima = []
            for seed in RUN_SEEDS:
                records = _replica_records(replay, benchmark, seed, strength)
                if name == DIAGNOSTIC_RATES[0]:
                    audited_row_minima.extend(
                        int(
                            record["assignment_diagnostics"][
                                "minimum_replay_batch_rows"
                            ]
                        )
                        for record in records
                    )
                    audited_row_maxima.extend(
                        int(
                            record["assignment_diagnostics"][
                                "maximum_replay_batch_rows"
                            ]
                        )
                        for record in records
                    )
                values = [
                    _finite(
                        record["assignment_diagnostics"]["means"][name],
                        f"{benchmark}/{strength}/{name}",
                    )
                    for record in records
                ]
                seed_values.append(_mean(values))
                nested.append(
                    0.0 if strength == 0 else _nested_variance_of_mean(values)
                )
                all_unit_means.extend(values)
                all_unit_minima.extend(
                    _finite(
                        record["assignment_diagnostics"]["minima"][name],
                        "diagnostic minimum",
                    )
                    for record in records
                )
                all_unit_maxima.extend(
                    _finite(
                        record["assignment_diagnostics"]["maxima"][name],
                        "diagnostic maximum",
                    )
                    for record in records
                )
            rates[name] = {
                "unit_mean_summary": student_t_summary(
                    seed_values, nested_mc_variances=nested
                ),
                "minimum_audited_event_rate": min(all_unit_minima),
                "maximum_audited_event_rate": max(all_unit_maxima),
                "minimum_unit_mean": min(all_unit_means),
                "maximum_unit_mean": max(all_unit_means),
            }
        rows.append(
            {
                "lambda_bp": strength,
                "requested_lambda": strength / 10000.0,
                "minimum_audited_replay_batch_rows": min(audited_row_minima),
                "maximum_audited_replay_batch_rows": max(audited_row_maxima),
                "rates": rates,
            }
        )
    return rows


def summarize_complete_records(
    records: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate logical coverage and compute the prespecified scientific summary."""

    references, replay = _index_records(records, lock)
    comparability = _check_sampled_event_comparability(replay)
    benchmark_summaries = []
    slope_hypotheses: list[tuple[str, float]] = []
    usefulness_hypotheses: list[tuple[str, float]] = []
    for benchmark in BENCHMARKS:
        endpoints = {
            endpoint: _endpoint_summary(
                references, replay, benchmark, endpoint
            )
            for endpoint in SCALAR_ENDPOINTS
        }
        primary = endpoints["followup_primary_final_average_accuracy"]
        slope_key = f"{benchmark}/followup_primary_final_accuracy_slope"
        slope_hypotheses.append(
            (slope_key, float(primary["slope"]["inference"]["raw_two_sided_p"]))
        )
        for row in primary["usefulness_vs_reference"]:
            key = f"{benchmark}/lambda_{int(row['lambda_bp']):05d}bp"
            usefulness_hypotheses.append(
                (key, float(row["effect"]["raw_two_sided_p"]))
            )
        benchmark_summaries.append(
            {
                "benchmark": benchmark,
                "followup_primary_setting": (
                    "task-il" if benchmark == "split_cifar100" else "domain-il"
                ),
                "benchmark_default_setting": (
                    "class-il" if benchmark == "split_cifar100" else "domain-il"
                ),
                "accuracy_unit": "percentage_points",
                "endpoints": endpoints,
                "accuracy_trajectories": {
                    "followup_primary": _trajectory_curve(
                        references,
                        replay,
                        benchmark,
                        "followup_primary_mean_accuracy_trajectory",
                    ),
                    "benchmark_default": _trajectory_curve(
                        references,
                        replay,
                        benchmark,
                        "benchmark_default_mean_accuracy_trajectory",
                    ),
                },
                "effective_corruption": _diagnostic_summary(replay, benchmark),
            }
        )
    slope_holm = holm_adjust(slope_hypotheses)
    usefulness_holm = holm_adjust(usefulness_hypotheses)
    for benchmark_summary in benchmark_summaries:
        benchmark = str(benchmark_summary["benchmark"])
        primary = benchmark_summary["endpoints"][
            "followup_primary_final_average_accuracy"
        ]
        slope_key = f"{benchmark}/followup_primary_final_accuracy_slope"
        slope_adjustment = slope_holm[slope_key]
        slope_mean = float(primary["slope"]["inference"]["mean"])
        slope_rejected = bool(slope_adjustment["reject_at_0_05"])
        primary["slope"]["holm_primary_family"] = slope_adjustment
        primary["slope"]["inference_role"] = "confirmatory_primary_holm_adjusted"
        primary["slope"]["directional_decision"] = {
            "alpha": ALPHA,
            "rule": "a direction is supported only when the two-sided Holm-adjusted test rejects and the estimated mean has that sign",
            "supported_negative_slope": slope_rejected and slope_mean < 0.0,
            "supported_positive_slope": slope_rejected and slope_mean > 0.0,
            "no_supported_direction": (not slope_rejected) or slope_mean == 0.0,
        }
        for row in primary["usefulness_vs_reference"]:
            key = f"{benchmark}/lambda_{int(row['lambda_bp']):05d}bp"
            usefulness_adjustment = usefulness_holm[key]
            usefulness_mean = float(row["effect"]["mean"])
            usefulness_rejected = bool(usefulness_adjustment["reject_at_0_05"])
            row["holm_secondary_usefulness_family"] = usefulness_adjustment
            row["inference_role"] = "prespecified_secondary_holm_adjusted"
            row["directional_decision"] = {
                "alpha": ALPHA,
                "rule": "useful or harmful reuse is supported only when the two-sided Holm-adjusted test rejects and the estimated ER-minus-reference mean has the corresponding sign",
                "supported_positive_reuse": (
                    usefulness_rejected and usefulness_mean > 0.0
                ),
                "supported_negative_reuse": (
                    usefulness_rejected and usefulness_mean < 0.0
                ),
                "no_supported_direction": (
                    (not usefulness_rejected) or usefulness_mean == 0.0
                ),
            }
    return {
        "schema_version": "gradient_reuse.assignment_strength_scientific_summary.v1",
        "status": "PASS",
        "scientific_result": True,
        "analysis_role": "outcome-informed_but_prospectively_locked_fresh_seed_followup",
        "not_a_replacement_for_original_primary": True,
        "independent_unit": "run_seed",
        "independent_seeds": list(RUN_SEEDS),
        "nested_monte_carlo_replicas": list(PERMUTATION_REPLICAS),
        "requested_lambda_basis_points": list(LAMBDA_BASIS_POINTS),
        "unit_count": EXPECTED_UNITS,
        "primary_inference": {
            "statistic": "within-replica OLS slope over requested lambda followed by within-seed replica averaging",
            "interval": "two-sided Student-t over 12 run seeds with nested Monte Carlo variance floor",
            "family": "two benchmark-specific follow-up-primary-final-accuracy slopes",
            "multiplicity": "Holm",
            "directional_decision_requires_estimate_sign": True,
            "holm_results": slope_holm,
        },
        "secondary_usefulness_inference": {
            "contrast": "experience replay minus same-seed no-replay reference",
            "family": "ten benchmark-by-strength follow-up-primary-final-accuracy contrasts",
            "multiplicity": "Holm",
            "directional_decision_requires_estimate_sign": True,
            "holm_results": usefulness_holm,
        },
        "comparability": comparability,
        "benchmark_summaries": benchmark_summaries,
    }


def _expected_unit_identity(
    unit: SweepUnit,
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    binding_path: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.assignment_strength_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": binding["mammoth_commit"],
        "data_manifest_sha256": binding["data_manifest_sha256"],
        "data_bundle_sha256": binding["data_bundle_sha256"],
        "environment_manifest_sha256": binding["environment_manifest_sha256"],
        "environment_bundle_sha256": binding["environment_bundle_sha256"],
        "python_sha256": binding["python_sha256"],
        "binding_sha256": sha256_file(binding_path),
        "effective_args_sha256": canonical_sha256(
            effective_args(unit, lock, str(binding["device"]))
        ),
        "device": binding["device"],
        "paper_eligible": True,
    }


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "project",
        "lock_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "mammoth_root",
        "mammoth_commit",
        "data_root",
        "data_manifest_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "device",
        "python_entry",
        "python_sha256",
        "data_bundle_sha256",
        "expected_units",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        _fail("assignment-strength run binding schema changed")
    if not (
        binding.get("schema_version")
        == "gradient_reuse.assignment_strength_run_binding.v1"
        and binding.get("status") == "PASS"
        and binding.get("project") == str(project)
        and binding.get("lock_sha256") == sha256_file(lock_path)
        and binding.get("source_manifest_sha256") == sha256_file(source_manifest_path)
        and binding.get("source_bundle_sha256") == source.get("source_bundle_sha256")
        and binding.get("mammoth_commit")
        == "e75a491c69fd729edeb01431afb753d9157d9a81"
        and binding.get("device") == "cuda:0"
        and int(binding.get("expected_units", 0)) == EXPECTED_UNITS
    ):
        _fail("assignment-strength run binding identity changed")
    for name in (
        "data_manifest_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "data_bundle_sha256",
        "python_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "lock_sha256",
    ):
        _digest(binding.get(name), f"binding/{name}")


def _validate_units_root_topology(
    units_root: Path, expected_names: set[str]
) -> None:
    """Admit exactly one unit directory and one runner lock per unit."""

    expected_lock_names = {f".{name}.lock" for name in expected_names}
    observed_names: set[str] = set()
    observed_lock_names: set[str] = set()
    unsafe: list[Path] = []
    for path in units_root.iterdir():
        if path.is_symlink():
            unsafe.append(path)
        elif path.is_dir() and path.name in expected_names:
            observed_names.add(path.name)
        elif path.is_file() and path.name in expected_lock_names:
            observed_lock_names.add(path.name)
        else:
            unsafe.append(path)
    if (
        unsafe
        or observed_names != expected_names
        or observed_lock_names != expected_lock_names
    ):
        _fail(
            "unit-root topology changed; "
            f"missing_units={len(expected_names - observed_names)}, "
            f"missing_locks={len(expected_lock_names - observed_lock_names)}, "
            f"unexpected={sorted(path.name for path in unsafe)[:3]}"
        )


def aggregate_campaign(
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    binding_path: Path,
    out: Path,
    output: Path | None = None,
    mammoth_root: Path | None = None,
    data_root: Path | None = None,
    data_manifest_path: Path | None = None,
    environment_manifest_path: Path | None = None,
) -> Path:
    """Authenticate all sealed units and publish only the scientific aggregate."""

    project = real_directory(project, "assignment-strength project")
    lock_path = real_file(lock_path, "assignment-strength lock")
    source_manifest_path = real_file(
        source_manifest_path, "assignment-strength source manifest"
    )
    binding_path = real_file(binding_path, "assignment-strength run binding")
    raw_out = Path(out).expanduser()
    if raw_out.is_symlink() or not raw_out.is_dir():
        _fail(f"assignment-strength output root is absent or symbolic: {raw_out}")
    out = raw_out.resolve(strict=True)
    aggregate_path = out / AGGREGATE_NAME
    if output is not None:
        raw_output = Path(output).expanduser()
        if raw_output.is_symlink() or raw_output.resolve(strict=False) != aggregate_path:
            _fail(
                f"aggregate output must be the canonical campaign path: {aggregate_path}"
            )
    lock = load_lock(lock_path)
    source = validate_source_manifest(source_manifest_path, project)
    raw_binding = strict_json(binding_path)
    if not isinstance(raw_binding, Mapping):
        _fail("assignment-strength binding is not an object")
    binding = dict(raw_binding)
    _validate_binding(
        binding,
        project=project,
        lock_path=lock_path,
        source_manifest_path=source_manifest_path,
        source=source,
    )
    if mammoth_root is not None:
        mammoth = validate_pristine_mammoth(mammoth_root)
        if binding.get("mammoth_root") != str(mammoth):
            _fail("Mammoth root differs from the run binding")
        if binding.get("mammoth_commit") != git_output(mammoth, "rev-parse", "HEAD"):
            _fail("Mammoth commit differs from the run binding")
    if data_root is not None:
        data = real_directory(data_root, "assignment-strength data root")
        if binding.get("data_root") != str(data):
            _fail("data root differs from the run binding")
    if data_manifest_path is not None:
        data_manifest = real_file(
            data_manifest_path, "assignment-strength data manifest"
        )
        if binding.get("data_manifest_sha256") != sha256_file(data_manifest):
            _fail("data manifest differs from the run binding")
        if data_root is None:
            _fail("data-root is required when authenticating the data manifest")
        data_value = validate_data_manifest(data_manifest, data)
        if binding.get("data_bundle_sha256") != data_value.get("data_bundle_sha256"):
            _fail("data bundle differs from the run binding")
    if environment_manifest_path is not None:
        environment_manifest = real_file(
            environment_manifest_path, "assignment-strength environment manifest"
        )
        if binding.get("environment_manifest_sha256") != sha256_file(
            environment_manifest
        ):
            _fail("environment manifest differs from the run binding")
        if mammoth_root is None:
            _fail("mammoth-root is required when authenticating the environment")
        environment_value = validate_environment_manifest(
            environment_manifest, mammoth, str(binding["device"])
        )
        if binding.get("environment_bundle_sha256") != environment_value.get(
            "environment_bundle_sha256"
        ):
            _fail("environment bundle differs from the run binding")
    python = executable_entry(
        Path(str(binding["python_entry"])), "assignment-strength Python"
    )
    if sha256_file(python) != binding.get("python_sha256"):
        _fail("Python bytes differ from the run binding")
    aggregate_exists = aggregate_path.exists() or aggregate_path.is_symlink()
    if aggregate_exists:
        # Do not trust a prior aggregate as a shortcut.  Reconstruct the
        # complete summary from all sealed units below, then require byte-level
        # semantic equality before preserving it.
        real_file(aggregate_path, "assignment-strength aggregate")
    units_root = real_directory(out / "units", "assignment-strength units")
    units = enumerate_units(lock)
    expected_names = {unit.unit_id for unit in units}
    _validate_units_root_topology(units_root, expected_names)
    records = []
    unit_done_sha256: dict[str, str] = {}
    for unit in units:
        identity = _expected_unit_identity(
            unit,
            lock=lock,
            lock_path=lock_path,
            source_manifest_path=source_manifest_path,
            source=source,
            binding_path=binding_path,
            binding=binding,
        )
        unit_root = units_root / unit.unit_id
        try:
            record = validate_completed_unit(unit_root, identity)
        except RuntimeError as error:
            raise AssignmentStrengthAggregationError(str(error)) from error
        _validate_logical_record(record, unit, lock)
        records.append(record)
        unit_done_sha256[unit.unit_id] = sha256_file(unit_root / "DONE.json")
    summary = summarize_complete_records(records, lock)
    aggregate = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "paper_eligible": True,
        "complete_protocol": True,
        "unit_count": len(records),
        "independent_seed_count": len(RUN_SEEDS),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "binding_sha256": sha256_file(binding_path),
        "unit_done_sha256": unit_done_sha256,
        "unit_done_bundle_sha256": canonical_sha256(unit_done_sha256),
        "summary": summary,
    }
    if aggregate_exists:
        if strict_json(aggregate_path) != aggregate:
            _fail("existing assignment-strength aggregate differs from reconstruction")
    else:
        atomic_json(aggregate_path, aggregate)
    return aggregate_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = aggregate_campaign(
        project=arguments.project,
        lock_path=arguments.lock,
        source_manifest_path=arguments.source_manifest,
        mammoth_root=arguments.mammoth_root,
        data_root=arguments.data_root,
        data_manifest_path=arguments.data_manifest,
        environment_manifest_path=arguments.environment_manifest,
        binding_path=arguments.binding,
        out=arguments.out,
        output=arguments.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
