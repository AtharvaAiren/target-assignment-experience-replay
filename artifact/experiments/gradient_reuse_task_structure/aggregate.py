"""Deterministic paired aggregation for the task-structure follow-up."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats

from .integrity import atomic_json, real_directory, real_file
from .protocol import (
    EXPECTED_UNITS,
    RUN_SEEDS,
    StudyUnit,
    canonical_sha256,
    enumerate_units,
    load_lock,
    sha256_file,
    strict_json,
)
from .unit_runner import effective_args


SCHEMA = "gradient_reuse.task_structure_aggregate.v1"
AGGREGATE_NAME = "TASK_STRUCTURE_AGGREGATE.json"


def student_t_summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if len(data) != len(RUN_SEEDS) or any(not math.isfinite(value) for value in data):
        raise RuntimeError("paired Student-t summary requires twelve finite seed values")
    center = sum(data) / len(data)
    variance = sum((value - center) ** 2 for value in data) / (len(data) - 1)
    standard_error = math.sqrt(variance / len(data))
    degrees = len(data) - 1
    critical = float(stats.t.ppf(0.975, degrees))
    if standard_error == 0.0:
        statistic = None
        p_value = 1.0 if center == 0.0 else 0.0
    else:
        statistic = center / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), degrees))
    return {
        "mean": center,
        "independent_seed_values": data,
        "independent_seed_count": len(data),
        "degrees_of_freedom": degrees,
        "sample_variance": variance,
        "standard_error": standard_error,
        "critical_value_0_975": critical,
        "ci95": [center - critical * standard_error, center + critical * standard_error],
        "t_statistic": statistic,
        "raw_two_sided_p": p_value,
        "reject_at_0_05": p_value <= 0.05,
        "interval": "two_sided_paired_student_t_over_12_run_seeds",
    }


def _unit_from_record(record: Mapping[str, Any]) -> StudyUnit:
    raw = record.get("unit")
    if not isinstance(raw, Mapping) or set(raw) != {"seed", "arm"}:
        raise RuntimeError("task-structure UNIT has invalid identity")
    unit = StudyUnit(seed=int(raw["seed"]), arm=str(raw["arm"]))
    if unit.as_dict() != dict(raw):
        raise RuntimeError("task-structure UNIT identity is noncanonical")
    return unit


def _metric(record: Mapping[str, Any], setting: str, field: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("task-structure UNIT has no metrics")
    payload = metrics.get(setting)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"task-structure UNIT has no {setting} endpoint")
    value = float(payload.get(field, float("nan")))
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite task-structure endpoint: {setting}/{field}")
    return value


def _trajectory(record: Mapping[str, Any], setting: str) -> list[float]:
    metrics = record["metrics"][setting]
    value = [float(item) for item in metrics["mean_accuracy_trajectory"]]
    if len(value) != 20 or any(not math.isfinite(item) for item in value):
        raise RuntimeError("task-structure trajectory is invalid")
    return value


def _validate_record(
    record: Mapping[str, Any], expected: StudyUnit, lock: Mapping[str, Any]
) -> None:
    if not (
        record.get("schema_version") == "gradient_reuse.task_structure_unit.v1"
        and record.get("paper_eligible") is True
        and _unit_from_record(record) == expected
    ):
        raise RuntimeError(f"task-structure UNIT metadata changed: {expected.unit_id}")
    if record.get("effective_args") != effective_args(expected, lock, "cuda:0"):
        raise RuntimeError(f"task-structure effective arguments changed: {expected.unit_id}")
    runtime = float(record.get("runtime_seconds", float("nan")))
    if not math.isfinite(runtime) or runtime < 0.0:
        raise RuntimeError(f"task-structure runtime is invalid: {expected.unit_id}")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "benchmark_default_setting",
        "followup_primary_setting",
        "class_il",
        "task_il",
        "effective_argument_evidence",
    }:
        raise RuntimeError(f"task-structure metrics schema changed: {expected.unit_id}")
    if not (
        metrics.get("benchmark_default_setting") == "class-il"
        and metrics.get("followup_primary_setting") == "task-il"
    ):
        raise RuntimeError(f"task-structure metric settings changed: {expected.unit_id}")
    diagnostics = record.get("task_structure_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError(f"task-structure diagnostics absent: {expected.unit_id}")
    if int(diagnostics.get("exposure_count", 0)) < 1:
        raise RuntimeError(f"task-structure exposure is empty: {expected.unit_id}")
    if int(diagnostics.get("eligible_exposure_count", 0)) < 1:
        raise RuntimeError(f"no eligible task-structure exposure: {expected.unit_id}")
    expected_gradients = expected.arm == "paired_probe"
    if (int(diagnostics.get("common_state_gradient_count", 0)) > 0) != expected_gradients:
        raise RuntimeError(f"task-structure gradient arm changed: {expected.unit_id}")
    for setting in ("task_il", "class_il"):
        for field in (
            "final_average_accuracy", "average_incremental_accuracy", "backward_transfer"
        ):
            _metric(record, setting, field)
        _trajectory(record, setting)


def _paired_endpoint(
    records: Mapping[tuple[int, str], Mapping[str, Any]],
    setting: str,
    field: str,
) -> dict[str, Any]:
    rows = []
    differences = []
    within_probe = []
    across_probe = []
    for seed in RUN_SEEDS:
        within = _metric(records[(seed, "within_task")], setting, field)
        across = _metric(records[(seed, "across_task")], setting, field)
        probe = _metric(records[(seed, "paired_probe")], setting, field)
        difference = across - within
        rows.append(
            {
                "seed": seed,
                "within_task": within,
                "across_task": across,
                "paired_probe": probe,
                "across_minus_within": difference,
                "within_minus_probe": within - probe,
                "across_minus_probe": across - probe,
            }
        )
        differences.append(difference)
        within_probe.append(within - probe)
        across_probe.append(across - probe)
    return {
        "setting": setting.replace("_", "-"),
        "endpoint": field,
        "units": "percentage_points",
        "seed_rows": rows,
        "across_minus_within": student_t_summary(differences),
        "within_minus_paired_probe_descriptive": student_t_summary(within_probe),
        "across_minus_paired_probe_descriptive": student_t_summary(across_probe),
    }


def _trajectory_summary(
    records: Mapping[tuple[int, str], Mapping[str, Any]], setting: str
) -> list[dict[str, Any]]:
    output = []
    for task in range(20):
        differences = [
            _trajectory(records[(seed, "across_task")], setting)[task]
            - _trajectory(records[(seed, "within_task")], setting)[task]
            for seed in RUN_SEEDS
        ]
        output.append({"after_task": task + 1, "across_minus_within": student_t_summary(differences)})
    return output


def _comparability(
    records: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    groups = []
    for seed in RUN_SEEDS:
        fingerprints = {
            arm: records[(seed, arm)]["task_structure_diagnostics"]["sampled_plan_fingerprints"]
            for arm in ("within_task", "across_task", "paired_probe")
        }
        if not (
            fingerprints["within_task"]
            == fingerprints["across_task"]
            == fingerprints["paired_probe"]
        ):
            raise RuntimeError(f"paired replay batches/plans differ across arms for seed {seed}")
        counts = {
            arm: {
                name: int(records[(seed, arm)]["task_structure_diagnostics"][name])
                for name in (
                    "exposure_count", "eligible_exposure_count", "ineligible_exposure_count",
                    "selected_quadruples", "selected_rows",
                )
            }
            for arm in ("within_task", "across_task", "paired_probe")
        }
        if not (counts["within_task"] == counts["across_task"] == counts["paired_probe"]):
            raise RuntimeError(f"task-structure exposure counts differ across arms for seed {seed}")
        groups.append(
            {
                "seed": seed,
                "status": "PASS",
                **counts["within_task"],
                "fingerprint_sha256": canonical_sha256(fingerprints["within_task"]),
            }
        )
    return {
        "status": "PASS",
        "same_replay_inputs_targets_tasks_and_selected_quadruples_across_all_three_arms": True,
        "groups": groups,
    }


def _common_state_summary(
    records: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    names = (
        "true_norm", "within_norm", "across_norm", "true_within_distance",
        "true_across_distance", "across_within_distance", "true_within_dot",
        "true_across_dot", "across_within_dot", "true_within_cosine",
        "true_across_cosine", "across_within_cosine",
    )
    seed_rows: list[dict[str, Any]] = []
    for seed in RUN_SEEDS:
        rows = records[(seed, "paired_probe")]["task_structure_diagnostics"]["common_state_gradients"]
        if not rows:
            raise RuntimeError(f"probe seed {seed} has no common-state gradients")
        seed_rows.append(
            {
                "seed": seed,
                "diagnostic_events": len(rows),
                "event_means": {
                    name: sum(float(row[name]) for row in rows) / len(rows) for name in names
                },
            }
        )
    return {
        "inference_role": "descriptive_only",
        "gradient_definition": "raw_full_replay_mean_cross_entropy_parameter_gradient_without_direction_normalization",
        "common_parameter_state": True,
        "seed_rows": seed_rows,
        "across_seed_summaries": {
            name: student_t_summary([row["event_means"][name] for row in seed_rows])
            for name in names
        },
    }


def aggregate(out: Path, lock_path: Path) -> dict[str, Any]:
    out = real_directory(out, "task-structure output")
    lock = load_lock(real_file(lock_path, "task-structure lock"))
    expected = {unit.unit_id: unit for unit in enumerate_units(lock)}
    records: dict[tuple[int, str], Mapping[str, Any]] = {}
    done_hashes: dict[str, str] = {}
    for unit_id, unit in expected.items():
        root = out / "units" / unit_id
        done_path = real_file(root / "DONE.json", f"{unit_id} DONE")
        record = strict_json(real_file(root / "UNIT.json", f"{unit_id} UNIT"))
        if not isinstance(record, Mapping):
            raise RuntimeError(f"task-structure UNIT is not an object: {unit_id}")
        _validate_record(record, unit, lock)
        records[(unit.seed, unit.arm)] = record
        done_hashes[unit_id] = sha256_file(done_path)
    if len(records) != EXPECTED_UNITS:
        raise RuntimeError("task-structure aggregate lacks the complete 36-unit panel")

    endpoints = {}
    for setting in ("task_il", "class_il"):
        endpoints[setting] = {
            field: _paired_endpoint(records, setting, field)
            for field in (
                "final_average_accuracy", "average_incremental_accuracy", "backward_transfer"
            )
        }
    primary = endpoints["task_il"]["final_average_accuracy"]["across_minus_within"]
    secondary = endpoints["class_il"]["final_average_accuracy"]["across_minus_within"]
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "complete_protocol": True,
        "paper_eligible": True,
        "unit_count": EXPECTED_UNITS,
        "independent_seed_count": len(RUN_SEEDS),
        "primary_estimand": {
            "definition": "across_task minus within_task final task-IL average accuracy",
            "result": primary,
        },
        "mandatory_class_il_secondary": {
            "definition": "across_task minus within_task final class-IL average accuracy",
            "result": secondary,
        },
        "all_endpoints": endpoints,
        "mean_accuracy_trajectories": {
            setting: _trajectory_summary(records, setting)
            for setting in ("task_il", "class_il")
        },
        "paired_exposure_authentication": _comparability(records),
        "common_state_gradient_diagnostics": _common_state_summary(records),
        "forced_swap_theory_guard": {
            "uniform_group_permutation_expectation_identity_applied": False,
            "reason": "the intervention conditions on unequal labels, forces derangements, and greedily matches task pairs; its finite set of swaps is not the uniform group-permutation law",
        },
        "unit_done_sha256": dict(sorted(done_hashes.items())),
        "unit_done_bundle_sha256": canonical_sha256(dict(sorted(done_hashes.items()))),
        "lock_sha256": sha256_file(lock_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--lock", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    value = aggregate(arguments.out, arguments.lock)
    atomic_json(arguments.output, value)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
