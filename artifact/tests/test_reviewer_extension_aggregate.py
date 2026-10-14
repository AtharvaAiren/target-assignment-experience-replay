"""Focused tests for the fail-closed reviewer-extension aggregate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.gradient_reuse_reviewer_extension import aggregate as aggregate_module
from experiments.gradient_reuse_mechanisms.protocol import canonical_sha256, sha256_file
from experiments.gradient_reuse_mechanisms.protocol import (
    enumerate_units as enumerate_parent_units,
    method_estimand_class,
    reference_baseline_method,
    strict_json,
)
from experiments.gradient_reuse_reviewer_extension.aggregate import (
    E2_SCHEMA,
    PARENT_LOCK,
    ReviewerExtensionAggregationError,
    confusion_summary,
    decomposition_seed_rows,
    e5_five_seed_sensitivity,
    path_contrast_seed_rows,
    scalar_endpoints,
    state_matched_summary,
    summarize_decomposition,
    summarize_path_contrasts,
    validate_e2,
    validate_unit_artifacts,
)
from experiments.gradient_reuse_reviewer_extension.protocol import (
    ExtensionUnit,
    enumerate_units,
    load_lock,
)
from experiments.gradient_reuse_reviewer_extension.unit_runner import (
    artifact_inventory,
    unit_lock,
)
from experiments.gradient_reuse_reviewer_extension.secondary_endpoints import (
    compute_estimands,
    holm_assignment_family,
    summarize_estimands,
)


REPO = Path(__file__).resolve().parents[1]
LOCK = load_lock(REPO / "configs" / "reviewer_extension.json")


def _arm_value(unit: ExtensionUnit) -> float:
    center = 40.0 + (unit.seed - 3400) / 10.0
    if unit.intervention == "none":
        return center
    if unit.intervention == "paired":
        return center + 5.0
    return center + 1.0 + float(unit.permutation_replica)


def _endpoints(unit: ExtensionUnit) -> dict[str, float]:
    value = _arm_value(unit)
    return {
        "primary_final_average_accuracy": value,
        "primary_old_task_only_accuracy": value - 1.0,
        "primary_average_incremental_accuracy": value + 1.0,
        "primary_backward_transfer": value - 50.0,
        "task_identity_final_average_accuracy": value + 0.5,
        "task_identity_average_incremental_accuracy": value + 1.5,
        "task_identity_backward_transfer": value - 49.5,
    }


def _confusion(unit: ExtensionUnit) -> dict[str, Any]:
    delta = (
        1.0 if unit.intervention == "paired" else 0.1 * int(unit.permutation_replica)
    )
    common = {
        "examples": 1000,
        "accuracy": 40.0 + delta,
        "mean_correct_label_logit": 2.0 + delta,
        "mean_incorrect_label_logit": -1.0 + delta,
        "mean_predictive_entropy": 1.5 - delta / 10.0,
        "incorrect_prediction_label_entropy": 2.5 - delta / 10.0,
    }
    if unit.benchmark == "permuted_mnist":
        diagnostic = {
            "schema_version": "gradient_reuse.pmnist_confusion.v1",
            **common,
        }
    else:
        diagnostic = {
            "schema_version": "gradient_reuse.cifar_confusion.v1",
            **common,
            "mean_correct_old_logit": common["mean_correct_label_logit"],
            "correct_old_class_fraction": 0.4 + delta / 100.0,
            "other_old_class_fraction": 0.3 - delta / 100.0,
            "newest_task_class_fraction": 0.2,
            "other_unseen_class_fraction": 0.1,
            "mean_other_old_logit": -0.5 + delta,
            "mean_newest_task_logit": -0.25 + delta,
            "partition_residual": 0.0,
        }
    return {
        "schema_version": "gradient_reuse.final_confusion_record.v1",
        "model": "er_state_matched_audit",
        "benchmark": unit.benchmark,
        "dataset": unit.mammoth_dataset,
        "intervention": unit.intervention,
        "run_seed": unit.seed,
        "permutation_replica": (
            0 if unit.intervention == "paired" else int(unit.permutation_replica)
        ),
        "completed_tasks": 20,
        "live_state_before_and_after_sha256": {
            "model": "9" * 64,
            "optimizer": "a" * 64,
            "buffer": "b" * 64,
            "gradients": "c" * 64,
            "loader_generators": ["d" * 64],
        },
        "diagnostic": diagnostic,
    }


def _boundary(completed: int, seed: int, benchmark: str) -> dict[str, Any]:
    no_step_ce = 2.0 + completed / 100.0
    paired_reduction = completed / 100.0 + (seed - 3400) / 1000.0
    permutation_reductions = [
        completed * factor / 1000.0 for factor in (2.0, 3.0, 4.0, 5.0)
    ]
    paired_ce = no_step_ce - paired_reduction
    permuted_ce = [no_step_ce - reduction for reduction in permutation_reductions]
    no_step_accuracy = 40.0
    paired_gain = completed / 10.0
    permutation_gains = [completed * factor / 100.0 for factor in (1.0, 2.0, 3.0, 4.0)]
    paired_accuracy = no_step_accuracy + paired_gain
    permuted_accuracy = [no_step_accuracy + gain for gain in permutation_gains]
    ce_perm = sum(permutation_reductions) / 4.0
    accuracy_perm = sum(permutation_gains) / 4.0
    return {
        "schema_version": "gradient_reuse.state_matched_boundary.v1",
        "benchmark": benchmark,
        "run_seed": seed,
        "model": "er_state_matched_audit",
        "dataset": "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist",
        "intervention": "paired",
        "permutation_replica": 0,
        "update_norm_window": 16,
        "completed_tasks": completed,
        "evaluated_old_tasks": completed - 1,
        "sampled_indexes_sha256": "1" * 64,
        "replay_inputs_sha256": "2" * 64,
        "replay_targets_sha256": "3" * 64,
        "replay_target_multiset_sha256": "4" * 64,
        "permutation_sha256": [str(replica + 5) * 64 for replica in range(4)],
        "live_state_before_and_after_sha256": {
            "model": "9" * 64,
            "optimizer": "a" * 64,
            "buffer": "b" * 64,
            "gradients": "c" * 64,
            "loader_generators": ["d" * 64],
        },
        "examples": 100 * (completed - 1),
        "radius": 0.01 + completed / 10000.0,
        "branches": {
            "no_step": {
                "cross_entropy": no_step_ce,
                "accuracy": no_step_accuracy,
            },
            "paired": {
                "cross_entropy": paired_ce,
                "accuracy": paired_accuracy,
            },
            **{
                f"permuted_{replica}": {
                    "cross_entropy": permuted_ce[replica],
                    "accuracy": permuted_accuracy[replica],
                }
                for replica in range(4)
            },
        },
        "cross_entropy": {
            "V_total": paired_reduction,
            "V_perm": ce_perm,
            "V_pair": paired_reduction - ce_perm,
            "identity_residual": 0.0,
        },
        "accuracy": {
            "V_total": paired_gain,
            "V_perm": accuracy_perm,
            "V_pair": paired_gain - accuracy_perm,
            "identity_residual": 0.0,
        },
        "first_order": {
            "risk_gradient_norm": 1.0,
            "paired_direction_norm": 1.0,
            "mean_permuted_direction_norm": 0.5,
            "radius_scaled_pairing_term": 0.9 * (paired_reduction - ce_perm),
        },
    }


def _wrapped_records() -> list[dict[str, Any]]:
    output = []
    for unit in enumerate_units(LOCK):
        record: dict[str, Any] = {"state_matched_audit": None}
        if unit.experiment == "E3_E6_state_matched_bridge":
            boundaries = (
                [
                    _boundary(completed, unit.seed, unit.benchmark)
                    for completed in range(2, 21)
                ]
                if unit.intervention == "paired"
                else []
            )
            record["state_matched_audit"] = {
                "record_count": 21 if unit.intervention == "paired" else 2,
                "boundary_count": len(boundaries),
                "boundary_records": boundaries,
                "final_confusion": _confusion(unit),
                "raw_jsonl_sha256": "e" * 64,
            }
        output.append({"unit": unit, "record": record, "endpoints": _endpoints(unit)})
    return output


def test_locked_decompositions_use_seeds_not_nested_replicas() -> None:
    records = _wrapped_records()
    rows = []
    selections = (
        (
            "E1_off_floor_cifar50",
            "split_cifar100",
            ("experience_replay",),
            (3401, 3402, 3403),
        ),
        ("E4_buffer2000", "split_cifar100", ("experience_replay",), (3401, 3402, 3403)),
        ("E4_buffer2000", "permuted_mnist", ("experience_replay",), (3401, 3402, 3403)),
        (
            "E5_pmnist_seed_extension",
            "permuted_mnist",
            ("experience_replay", "gem", "agem", "mer", "pcgrad_replay"),
            (3404, 3405),
        ),
        (
            "E8_miniimagenet10",
            "split_miniimagenet",
            ("experience_replay",),
            (3401, 3402, 3403),
        ),
    )
    for experiment, benchmark, methods, seeds in selections:
        rows.extend(
            decomposition_seed_rows(
                records,
                experiment=experiment,
                benchmark=benchmark,
                methods=methods,
                seeds=seeds,
            )
        )
    assert len(records) == 156
    assert len(rows) == 22
    summaries = summarize_decomposition(rows)
    assert len(summaries) == 9
    e5 = next(
        row
        for row in summaries
        if row["experiment"] == "E5_pmnist_seed_extension" and row["method"] == "agem"
    )
    assert e5["seeds"] == [3404, 3405]
    assert e5["endpoints"]["primary_final_average_accuracy"]["V_pair"]["seeds"] == 2
    assert all(
        abs(endpoint["identity_residual"]) <= 1.0e-12
        for row in rows
        for endpoint in row["effects"].values()
    )


def test_missing_nested_replica_fails_closed() -> None:
    records = _wrapped_records()
    records = [
        row
        for row in records
        if row["unit"].unit_id
        != "E1_off_floor_cifar50__split_cifar100__experience_replay__permuted__seed_3401__replica_3"
    ]
    with pytest.raises(
        ReviewerExtensionAggregationError, match="nested permutation absent"
    ):
        decomposition_seed_rows(
            records,
            experiment="E1_off_floor_cifar50",
            benchmark="split_cifar100",
            methods=("experience_replay",),
            seeds=(3401, 3402, 3403),
        )


def _metric(value: float) -> dict[str, Any]:
    matrix = [[value for _ in range(row)] for row in range(1, 21)]
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": value,
        "average_incremental_accuracy": value,
        "backward_transfer": 0.0,
        "backward_transfer_available": True,
        "accuracy_matrix": matrix,
        "accuracy_matrix_available": True,
        "mean_accuracy_trajectory": [value] * 20,
        "evaluation_tasks": 20,
    }


def test_scalar_endpoints_are_reconstructed_from_accuracy_matrices() -> None:
    unit = next(
        unit
        for unit in enumerate_units(LOCK)
        if unit.experiment == "E1_off_floor_cifar50"
        and unit.method == "naive_sgd"
        and unit.seed == 3401
    )
    record = {
        "metrics": {
            "primary_setting": "class-il",
            "primary": _metric(40.0),
            "task_il": _metric(60.0),
            "old_task_only_accuracy": 40.0,
        }
    }
    endpoints = scalar_endpoints(record, unit)
    assert endpoints["primary_final_average_accuracy"] == 40.0
    assert endpoints["task_identity_final_average_accuracy"] == 60.0

    record["metrics"]["primary"]["backward_transfer"] = 1.0
    with pytest.raises(ReviewerExtensionAggregationError, match="inconsistent"):
        scalar_endpoints(record, unit)


def test_state_matched_and_confusion_summaries_preserve_inference_level() -> None:
    records = _wrapped_records()
    state = state_matched_summary(records)
    assert len(state["boundary_rows"]) == 114
    assert len(state["seed_rows"]) == 6
    assert len(state["benchmark_summaries"]) == 2
    assert len(state["boundary_summaries"]) == 38
    assert all(
        row["first_order_vs_realized_cross_entropy_V_pair_pearson"]
        == pytest.approx(1.0)
        for row in state["seed_rows"]
    )
    assert all(
        row["task_boundaries_repeated_within_seed"] is True
        for row in state["benchmark_summaries"]
    )

    confusion = confusion_summary(records)
    assert len(confusion["seed_rows"]) == 6
    assert len(confusion["summaries"]) == 2
    assert all(
        "no_per_component_null_hypothesis_tests" in row["reporting"]
        and "p_value" not in json.dumps(row)
        and "ci95" not in json.dumps(row)
        for row in confusion["summaries"]
    )

    path_rows = [
        *path_contrast_seed_rows(records, benchmark="split_cifar100"),
        *path_contrast_seed_rows(records, benchmark="permuted_mnist"),
    ]
    path = summarize_path_contrasts(path_rows)
    assert len(path_rows) == 6
    assert len(path) == 2
    assert all(row["not_a_shared_state_causal_contrast"] is True for row in path)


def test_five_seed_sensitivity_combines_three_parent_and_two_new_seeds() -> None:
    records = _wrapped_records()
    extension = decomposition_seed_rows(
        records,
        experiment="E5_pmnist_seed_extension",
        benchmark="permuted_mnist",
        methods=("experience_replay", "gem", "agem", "mer", "pcgrad_replay"),
        seeds=(3404, 3405),
    )
    parent_rows = []
    for method in ("experience_replay", "gem", "agem", "mer", "pcgrad_replay"):
        for seed in (3401, 3402, 3403):
            replicas = [41.0, 42.0, 43.0, 44.0]
            parent_rows.append(
                {
                    "benchmark": "permuted_mnist",
                    "method": method,
                    "estimand_class": "assignment_only",
                    "seed": seed,
                    "reference_method": reference_baseline_method(method),
                    "reference": 40.0,
                    "paired": 45.0,
                    "permuted_replicas": replicas,
                }
            )
    e2 = {"endpoints": {"task_il_final_average_accuracy": {"seed_rows": parent_rows}}}
    sensitivity = e5_five_seed_sensitivity(extension, e2)
    assert len(sensitivity["seed_rows"]) == 25
    assert len(sensitivity["summaries"]) == 5
    assert (
        next(row for row in sensitivity["summaries"] if row["method"] == "mer")[
            "reference_method"
        ]
        == "mer_current_only"
    )
    assert all(row["V_pair"]["seeds"] == 5 for row in sensitivity["summaries"])
    assert all(
        "post_hoc_five_seed_sensitivity" in row["reporting"]
        for row in sensitivity["summaries"]
    )


def _e2_payload(parent_completion: Path) -> dict[str, Any]:
    parent_units = enumerate_parent_units(strict_json(PARENT_LOCK))
    method_offsets = {
        method: index / 10.0
        for index, method in enumerate(sorted({unit.method for unit in parent_units}))
    }
    raw_rows = []
    for unit in parent_units:
        value = 20.0 + (unit.seed - 3401) / 10.0 + method_offsets[unit.method]
        if unit.intervention == "paired":
            value += 5.0
        elif unit.intervention == "permuted":
            value += 1.0 + float(unit.permutation_replica)
        task_source = (
            "domain_il_equals_task_identity_for_shared_labels"
            if unit.benchmark == "permuted_mnist"
            else "mammoth_task_il_log"
        )
        raw_rows.append(
            {
                "unit_id": unit.unit_id,
                "benchmark": unit.benchmark,
                "method": unit.method,
                "estimand_class": method_estimand_class(unit.method),
                "intervention": unit.intervention,
                "seed": unit.seed,
                "permutation_replica": unit.permutation_replica,
                "task_il_final_average_accuracy": value,
                "task_il_old_task_only_accuracy": value - 0.5,
                "primary_old_task_only_accuracy": value - 1.0,
                "task_il_source": task_source,
                "primary_log": f"/sealed/{unit.unit_id}/primary/logs.pyd",
                "primary_log_sha256": "1" * 64,
                "task_il_log": f"/sealed/{unit.unit_id}/task/logs.pyd",
                "task_il_log_sha256": "2" * 64,
                "unit_done_sha256": "3" * 64,
            }
        )
    endpoints = {}
    for endpoint_name in (
        "task_il_final_average_accuracy",
        "task_il_old_task_only_accuracy",
        "primary_old_task_only_accuracy",
    ):
        seed_rows = compute_estimands(raw_rows, endpoint_name)
        endpoints[endpoint_name] = {
            "seed_rows": seed_rows,
            "summaries": summarize_estimands(seed_rows),
        }
    endpoints["task_il_final_average_accuracy"]["holm_assignment_only_V_pair"] = (
        holm_assignment_family(endpoints["task_il_final_average_accuracy"]["summaries"])
    )
    return {
        "schema_version": E2_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "parent_completion_sha256": sha256_file(parent_completion),
        "parent_mechanism_aggregate_sha256": (
            "14316b3c1b5ae2bac509aa4e75c3fd255020f0e97f8072b0247260625c08e1f3"
        ),
        "raw_unit_count": 288,
        "independent_unit": "run_seed",
        "permutation_replicas": [0, 1, 2, 3],
        "permutation_replicas_are_nested_monte_carlo": True,
        "old_task_only_definition": (
            "mean_final_accuracy_over_tasks_1_through_T_minus_1"
        ),
        "raw_rows": raw_rows,
        "endpoints": endpoints,
    }


def test_e2_tables_are_reconstructed_from_exact_parent_unit_inventory(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "parent.json"
    completion.write_text("immutable parent\n", encoding="utf-8")
    payload = _e2_payload(completion)
    output = tmp_path / "E2.json"
    _write_json(output, payload)
    assert validate_e2(output, completion) == payload

    payload["raw_rows"][0]["task_il_final_average_accuracy"] += 0.25
    tampered = tmp_path / "E2_tampered.json"
    _write_json(tampered, payload)
    with pytest.raises(
        ReviewerExtensionAggregationError,
        match="differ from the raw-unit reconstruction",
    ):
        validate_e2(tampered, completion)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_unit_done_inventory_and_selected_attempt_are_authenticated(
    tmp_path: Path,
) -> None:
    unit_root = tmp_path / "unit"
    attempt = unit_root / "attempt_001"
    identity = {"schema_version": "test.identity.v1", "value": 1}
    record = {
        "identity": identity,
        "attempt": "attempt_001",
    }
    _write_json(unit_root / "UNIT.json", record)
    _write_json(
        attempt / "RUN_MANIFEST.json",
        {
            "schema_version": "gradient_reuse.reviewer_extension_attempt.v1",
            "status": "COMPLETED",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "command": ["python", "runner.py"],
            "attempt": "attempt_001",
            "started_unix": 10.0,
            "completed_unix": 20.0,
            "returncode": 0,
        },
    )
    (attempt / "stdout.log").write_text("ok\n", encoding="utf-8")
    inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
    _write_json(
        unit_root / "DONE.json",
        {
            "schema_version": "gradient_reuse.reviewer_extension_unit_done.v1",
            "status": "PASS",
            "identity_sha256": canonical_sha256(identity),
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )
    assert (
        validate_unit_artifacts(unit_root, identity, expected_record=record) == record
    )

    (attempt / "stdout.log").write_text("tampered\n", encoding="utf-8")
    assert sha256_file(attempt / "stdout.log") != inventory["attempt_001/stdout.log"]
    with pytest.raises(ReviewerExtensionAggregationError, match="DONE authentication"):
        validate_unit_artifacts(unit_root, identity, expected_record=record)


def test_unit_lock_rejects_symbolic_and_hardlinked_paths(tmp_path: Path) -> None:
    unit_root = tmp_path / "units" / "unit_a"
    unit_root.parent.mkdir()
    lock_path = unit_root.parent / ".unit_a.lock"
    target = tmp_path / "preserved.lock"
    target.write_text("preserve\n", encoding="utf-8")

    lock_path.symlink_to(target)
    with pytest.raises(RuntimeError, match="unsafe|safely"):
        with unit_lock(unit_root):
            pytest.fail("symbolic lock must not be acquired")
    lock_path.unlink()

    lock_path.hardlink_to(target)
    with pytest.raises(RuntimeError, match="singly linked"):
        with unit_lock(unit_root):
            pytest.fail("hardlinked lock must not be acquired")
    assert target.read_text(encoding="utf-8") == "preserve\n"


def test_matching_summary_only_is_resumed_by_publishing_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
    aggregate = out / "aggregate"
    aggregate.mkdir(parents=True)
    summary = {"schema_version": "test.summary.v1", "value": 7}
    _write_json(aggregate / "SUMMARY.json", summary)
    inputs = {}
    for name in (
        "lock",
        "source",
        "parent_completion",
        "secondary",
        "transformer",
        "config",
        "assets",
        "parent_transformer",
        "correction_manifest",
        "correction_patch",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path

    monkeypatch.setattr(aggregate_module, "load_lock", lambda _: {})
    monkeypatch.setattr(aggregate_module, "validate_parent_completion", lambda _: {})
    monkeypatch.setattr(
        aggregate_module, "validate_source_manifest", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(aggregate_module, "validate_e2", lambda *args: {})
    monkeypatch.setattr(aggregate_module, "validate_e7", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        aggregate_module, "collect_completed_units", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        aggregate_module, "build_summary", lambda *args, **kwargs: summary
    )

    result = aggregate_module.run(
        out=out,
        lock_path=inputs["lock"],
        source_manifest=inputs["source"],
        parent_completion=inputs["parent_completion"],
        secondary_output=inputs["secondary"],
        transformer_output=inputs["transformer"],
        transformer_config=inputs["config"],
        asset_manifest=inputs["assets"],
        parent_transformer_result=inputs["parent_transformer"],
        transformer_correction_manifest=inputs["correction_manifest"],
        transformer_correction_patch=inputs["correction_patch"],
    )
    assert result == aggregate / "SUMMARY.json"
    done = strict_json(aggregate / "DONE.json")
    assert done["status"] == "PASS"
    assert done["output_inventory"] == {
        "SUMMARY.json": sha256_file(aggregate / "SUMMARY.json")
    }
