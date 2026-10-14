"""Statistical and fail-closed tests for the assignment-strength aggregate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from experiments.gradient_reuse_assignment_strength.aggregate import (
    AssignmentStrengthAggregationError,
    _validate_units_root_topology,
    holm_adjust,
    student_t_summary,
    summarize_complete_records,
)
from experiments.gradient_reuse_assignment_strength.unit_runner import effective_args
from experiments.gradient_reuse_assignment_strength.protocol import (
    LAMBDA_BASIS_POINTS,
    RUN_SEEDS,
    SweepUnit,
    enumerate_units,
    load_lock,
)


REPO = Path(__file__).resolve().parents[1]
LOCK = load_lock(REPO / "configs" / "assignment_strength_sweep.json")


def _metrics(value: float) -> dict[str, Any]:
    matrix = [[float(value)] * row for row in range(1, 21)]
    trajectory = [float(value)] * 20
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": float(value),
        "average_incremental_accuracy": float(value),
        "backward_transfer": 0.0,
        "accuracy_matrix": matrix,
        "mean_accuracy_trajectory": trajectory,
        "evaluation_tasks": 20,
    }


def _fingerprints(benchmark: str, seed: int) -> list[dict[str, Any]]:
    prefix = (sum(map(ord, benchmark)) + seed) % 16
    return [
        {
            "observation": 100,
            "event": 0,
            "channel": "replay_labels",
            "sampled_indexes_sha256": f"{prefix:x}" * 64,
            "input_sha256": f"{(prefix + 1) % 16:x}" * 64,
            "original_target_sha256": f"{(prefix + 2) % 16:x}" * 64,
        }
    ]


def _diagnostics(unit: SweepUnit) -> dict[str, Any]:
    if unit.method == "reference":
        return {"record_count": 0}
    strength = float(unit.lambda_bp) / 10000.0
    rates = {
        "realized_lambda": strength,
        "selected_fraction": strength,
        "index_displaced_fraction": 0.9 * strength,
        "target_value_changed_fraction": 0.8 * strength,
    }
    return {
        "record_count": 5,
        "intervention_records": 2,
        "gradient_diagnostic_records": 2,
        "minimum_replay_batch_rows": 32,
        "maximum_replay_batch_rows": 32,
        "means": rates,
        "minima": rates,
        "maxima": rates,
        "sampled_event_fingerprints": _fingerprints(unit.benchmark, unit.seed),
        "audit_sha256": "a" * 64,
    }


def _primary_value(unit: SweepUnit) -> float:
    seed_offset = (unit.seed - RUN_SEEDS[0]) * 0.02
    benchmark_offset = 30.0 if unit.benchmark == "split_cifar100" else 50.0
    if unit.method == "reference":
        return benchmark_offset - 3.0 + seed_offset
    strength = float(unit.lambda_bp) / 10000.0
    slope = -8.0 if unit.benchmark == "split_cifar100" else -20.0
    replica_noise = 0.0
    if unit.permutation_replica is not None:
        replica_noise = (unit.permutation_replica - 1.5) * 0.2 * strength
    return benchmark_offset + seed_offset + slope * strength + replica_noise


def _record(unit: SweepUnit) -> dict[str, Any]:
    primary = _primary_value(unit)
    intended = effective_args(unit, LOCK, "cuda:0")
    logged = dict(intended)
    logged["enable_other_metrics"] = True
    logged["non_verbose"] = True
    logged.update(
        {
            "base_path": "/data",
            "results_path": f"/results/{unit.unit_id}",
            "debug_mode": False,
        }
    )
    if unit.method == "experience_replay":
        logged["assignment_audit_path"] = (
            f"/results/{unit.unit_id}/assignment_audit.jsonl"
        )
    evidence = {
        "primary": {"sha256": "c" * 64, "effective_args": logged},
        "task_il": (
            {"sha256": "d" * 64, "effective_args": logged}
            if unit.benchmark == "split_cifar100"
            else None
        ),
    }
    if unit.benchmark == "split_cifar100":
        # The prospectively locked follow-up diagnostic endpoint is task-IL.
        # Class-IL remains mandatory and deliberately has a different response.
        class_value = primary - 5.0 + 2.0 * (float(unit.lambda_bp or 0) / 10000.0)
        metrics = {
            "primary_setting": "class-il",
            "primary": _metrics(class_value),
            "task_il": _metrics(primary),
            "effective_argument_evidence": evidence,
        }
    else:
        metrics = {
            "primary_setting": "domain-il",
            "primary": _metrics(primary),
            "task_il": None,
            "effective_argument_evidence": evidence,
        }
    return {
        "schema_version": "gradient_reuse.assignment_strength_unit.v1",
        "identity": {"test": unit.unit_id},
        "unit": unit.as_dict(),
        "effective_args": intended,
        "metrics": metrics,
        "assignment_diagnostics": _diagnostics(unit),
        "runtime_seconds": 1.0,
        "attempt": "attempt_001",
        "paper_eligible": True,
    }


def _records() -> list[dict[str, Any]]:
    return [_record(unit) for unit in enumerate_units(LOCK)]


def test_student_t_summary_uses_nested_mc_variance_as_a_floor() -> None:
    summary = student_t_summary(
        [1.0] * len(RUN_SEEDS),
        nested_mc_variances=[4.0] * len(RUN_SEEDS),
    )
    assert summary["seed_sample_variance"] == 0.0
    assert summary["mean_nested_mc_variance"] == 4.0
    assert summary["variance_used"] == 4.0
    assert summary["standard_error"] == pytest.approx(math.sqrt(4.0 / 12.0))
    assert summary["degrees_of_freedom"] == 11


def test_holm_adjustment_is_step_down_and_monotone() -> None:
    adjusted = holm_adjust((("a", 0.01), ("b", 0.02), ("c", 0.50)))
    assert adjusted["a"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert adjusted["b"]["holm_adjusted_p"] == pytest.approx(0.04)
    assert adjusted["c"]["holm_adjusted_p"] == pytest.approx(0.50)
    assert adjusted["a"]["reject_at_0_05"] is True
    assert adjusted["b"]["reject_at_0_05"] is True
    assert adjusted["c"]["reject_at_0_05"] is False


def test_complete_design_uses_seed_level_slopes_and_reports_both_endpoints() -> None:
    summary = summarize_complete_records(_records(), LOCK)
    assert summary["unit_count"] == 432
    assert summary["independent_seeds"] == list(RUN_SEEDS)
    assert len(summary["primary_inference"]["holm_results"]) == 2
    assert len(summary["secondary_usefulness_inference"]["holm_results"]) == 10
    by_benchmark = {
        row["benchmark"]: row for row in summary["benchmark_summaries"]
    }

    cifar = by_benchmark["split_cifar100"]
    cifar_followup = cifar["endpoints"][
        "followup_primary_final_average_accuracy"
    ]
    cifar_class = cifar["endpoints"][
        "benchmark_default_final_average_accuracy"
    ]
    assert cifar["followup_primary_setting"] == "task-il"
    assert cifar["benchmark_default_setting"] == "class-il"
    assert cifar_followup["slope"]["inference"]["mean"] == pytest.approx(-8.0)
    assert cifar_class["slope"]["inference"]["mean"] == pytest.approx(-6.0)
    assert cifar_class["slope"]["inference_role"] == (
        "descriptive_unadjusted_unless_promoted_by_parent_family"
    )
    assert (
        cifar_followup["monotonicity_point_estimate_rule"]
        ["all_adjacent_means_nonpositive"]
        is True
    )
    assert cifar_followup["slope"]["holm_primary_family"][
        "reject_at_0_05"
    ] is True
    assert cifar_followup["slope"]["directional_decision"] == {
        "alpha": 0.05,
        "rule": "a direction is supported only when the two-sided Holm-adjusted test rejects and the estimated mean has that sign",
        "supported_negative_slope": True,
        "supported_positive_slope": False,
        "no_supported_direction": False,
    }
    assert cifar_followup["audit_estimands"]["v_total"]["mean"] == pytest.approx(
        3.0
    )
    assert cifar_followup["audit_estimands"]["v_perm"]["mean"] == pytest.approx(
        -5.0
    )
    assert cifar_followup["audit_estimands"]["v_pair"]["mean"] == pytest.approx(
        8.0
    )
    assert cifar_followup["audit_estimands"]["inference_role"] == (
        "descriptive_unadjusted"
    )
    first_usefulness = cifar_followup["usefulness_vs_reference"][0]
    assert (
        first_usefulness["directional_decision"]["supported_positive_reuse"] is True
    )
    assert (
        first_usefulness["directional_decision"]["supported_negative_reuse"] is False
    )
    last_usefulness = cifar_followup["usefulness_vs_reference"][-1]
    assert last_usefulness["directional_decision"]["supported_positive_reuse"] is False
    assert last_usefulness["directional_decision"]["supported_negative_reuse"] is True
    assert (
        len(cifar["accuracy_trajectories"]["followup_primary"]["strength_curves"])
        == 5
    )
    assert len(cifar["effective_corruption"]) == len(LAMBDA_BASIS_POINTS)

    pmnist = by_benchmark["permuted_mnist"]
    assert pmnist["followup_primary_setting"] == "domain-il"
    assert pmnist["endpoints"]["followup_primary_final_average_accuracy"]["slope"][
        "inference"
    ]["mean"] == pytest.approx(-20.0)
    assert summary["comparability"]["status"] == "PASS"


def test_missing_unit_fails_closed_before_inference() -> None:
    with pytest.raises(
        AssignmentStrengthAggregationError, match="requires 432 UNIT records"
    ):
        summarize_complete_records(_records()[:-1], LOCK)


def test_nested_draw_is_not_accepted_as_an_independent_unit() -> None:
    records = _records()
    records[-1] = dict(records[-2])
    with pytest.raises(AssignmentStrengthAggregationError, match="duplicate UNIT"):
        summarize_complete_records(records, LOCK)


def test_logged_effective_argument_drift_fails_closed() -> None:
    records = _records()
    evidence = records[0]["metrics"]["effective_argument_evidence"]["primary"]
    evidence["effective_args"]["lr"] = 999.0
    with pytest.raises(
        AssignmentStrengthAggregationError,
        match="logged and intended effective arguments disagree",
    ):
        summarize_complete_records(records, LOCK)


def test_units_root_admits_only_exact_runner_locks(tmp_path: Path) -> None:
    expected = {"unit_a", "unit_b"}
    for name in expected:
        (tmp_path / name).mkdir()
        (tmp_path / f".{name}.lock").touch()
    _validate_units_root_topology(tmp_path, expected)

    (tmp_path / "unrelated.txt").touch()
    with pytest.raises(
        AssignmentStrengthAggregationError, match="unit-root topology changed"
    ):
        _validate_units_root_topology(tmp_path, expected)
