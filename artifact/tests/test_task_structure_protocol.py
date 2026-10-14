"""Tests for the locked task-structure protocol and scheduler."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_task_structure.protocol import (
    ARMS,
    EXPECTED_UNITS,
    RUN_SEEDS,
    enumerate_units,
    load_lock,
    metrics_from_mammoth_log,
    canonical_sha256,
    validate_lock,
    validate_prerequisite_payload,
)
from experiments.gradient_reuse_task_structure.queue import scheduler_key
from experiments.gradient_reuse_task_structure.unit_runner import build_command


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/task_structure_equal_severity.json"


def test_lock_enumerates_24_training_and_12_probe_units() -> None:
    lock = load_lock(LOCK)
    units = enumerate_units(lock)
    assert len(units) == EXPECTED_UNITS == 36
    assert len({unit.unit_id for unit in units}) == 36
    assert sum(unit.arm == "paired_probe" for unit in units) == 12
    assert sum(unit.arm in {"within_task", "across_task"} for unit in units) == 24
    for seed in RUN_SEEDS:
        assert {unit.arm for unit in units if unit.seed == seed} == set(ARMS)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["runtime"].__setitem__("run_seeds", list(range(12))),
        lambda value: value["benchmark"].__setitem__("classes_per_task", 10),
        lambda value: value["benchmark"].__setitem__("class_permutation", True),
        lambda value: value["inference"].__setitem__("no_optional_stopping", False),
        lambda value: value["interpretation"].__setitem__(
            "forced_swaps_are_not_uniform_group_permutations", False
        ),
        lambda value: value["queue_prerequisites"].__setitem__(
            "mean_matched_done", "/tmp/not-the-locked-parent.json"
        ),
    ],
)
def test_lock_rejects_scientifically_material_changes(mutation) -> None:
    value = json.loads(LOCK.read_text())
    mutation(value)
    with pytest.raises(RuntimeError):
        validate_lock(value)


def test_scheduler_starts_within_and_across_together_before_probe() -> None:
    units = sorted(enumerate_units(load_lock(LOCK)), key=scheduler_key)
    assert [(unit.seed, unit.arm) for unit in units[:3]] == [
        (3501, "within_task"),
        (3501, "across_task"),
        (3501, "paired_probe"),
    ]


def test_mammoth_command_has_new_namespace_and_unique_flags() -> None:
    lock = load_lock(LOCK)
    unit = enumerate_units(lock)[0]
    command = build_command(
        unit,
        lock,
        python=Path("/python"),
        mammoth_root=Path("/mammoth"),
        data_root=Path("/data"),
        results=Path("/results"),
        audit=Path("/audit.jsonl"),
        device="cuda:0",
    )
    assert command[3] == "experiments.gradient_reuse_task_structure.mammoth_bootstrap"
    flags = command[4:][0::2]
    assert len(flags) == len(set(flags))
    assert "--task_structure_arm" in flags
    assert "--task_structure_audit_path" in flags


def test_metric_reconstruction_requires_full_20_task_matrix() -> None:
    payload = {}
    for row in range(1, 21):
        values = []
        for column in range(1, row + 1):
            value = float(50 + column - row)
            payload[f"accuracy_{column}_task{row}"] = value
            values.append(value)
        payload[f"accmean_task{row}"] = sum(values) / len(values)
    payload["backward_transfer"] = sum(
        payload[f"accuracy_{i + 1}_task20"] - payload[f"accuracy_{i + 1}_task{i + 1}"]
        for i in range(19)
    ) / 19
    metrics = metrics_from_mammoth_log(payload)
    assert metrics["evaluation_tasks"] == 20
    bad = copy.deepcopy(payload)
    del bad["accuracy_1_task20"]
    with pytest.raises((KeyError, RuntimeError)):
        metrics_from_mammoth_log(bad)


@pytest.mark.parametrize(
    ("campaign", "schema", "role", "count"),
    [
        (
            "assignment_strength",
            "gradient_reuse.assignment_strength_queue_done.v1",
            "outcome_informed_prospectively_locked_assignment_strength_followup",
            432,
        ),
        (
            "mean_matched",
            "gradient_reuse.mean_matched_queue_done.v1",
            "prospectively_locked_mean_matched_companion",
            96,
        ),
    ],
)
def test_prerequisite_contract_rejects_generic_pass(
    campaign: str, schema: str, role: str, count: int
) -> None:
    unit_ids = [f"unit-{index}" for index in range(count)]
    unit_done = {unit_id: "a" * 64 for unit_id in unit_ids}
    payload = {
        "schema_version": schema,
        "scientific_role": role,
        "status": "PASS",
        "unit_count": count,
        "expected_units": count,
        "completed_units": count,
        "validated_unit_ids": unit_ids,
        "validated_unit_ids_sha256": canonical_sha256(unit_ids),
        "unit_done_sha256": unit_done,
        "unit_done_bundle_sha256": canonical_sha256(unit_done),
        "lock_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "source_bundle_sha256": "d" * 64,
        "run_binding_sha256": "e" * 64,
        "aggregate_sha256": "f" * 64,
    }
    assert validate_prerequisite_payload(payload, campaign) == payload
    with pytest.raises(RuntimeError):
        validate_prerequisite_payload({"status": "PASS"}, campaign)
    payload["completed_units"] -= 1
    with pytest.raises(RuntimeError):
        validate_prerequisite_payload(payload, campaign)
