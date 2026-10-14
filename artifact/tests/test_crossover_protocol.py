from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_crossover.protocol import (
    EXPECTED_UNITS_PER_CAMPAIGN,
    REPLICAS,
    SEEDS,
    STRENGTHS_BP,
    enumerate_units,
    load_lock,
    selected_rows,
    validate_lock,
    validate_runtime_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/crossover_threshold_followup.json"


def test_complete_new_inventory_has_no_controls_or_old_endpoints() -> None:
    lock = load_lock(LOCK)
    all_ids: set[str] = set()
    for campaign in ("pmnist", "cifar"):
        units = enumerate_units(lock, campaign)
        assert len(units) == EXPECTED_UNITS_PER_CAMPAIGN
        assert {unit.seed for unit in units} == set(SEEDS[campaign])
        assert {unit.lambda_bp for unit in units} == set(STRENGTHS_BP)
        assert {unit.assignment_replica for unit in units} == set(REPLICAS)
        assert all(unit.arm == "er_reassigned" and unit.is_replay for unit in units)
        assert not any(unit.lambda_bp in {0, 2500, 5000, 7500, 10000} for unit in units)
        identifiers = {unit.unit_id for unit in units}
        assert len(identifiers) == EXPECTED_UNITS_PER_CAMPAIGN
        assert all_ids.isdisjoint(identifiers)
        all_ids.update(identifiers)
    assert len(all_ids) == 672


def test_dense_grid_is_exactly_the_missing_n32_supports() -> None:
    supports = [selected_rows(value, 32) for value in STRENGTHS_BP]
    assert supports == list(range(25, 32))
    assert len(set(supports)) == len(supports)
    old_75_support = 1 + (2 * (7500 * 31) + 10000) // 20000
    full_support = 1 + (2 * (10000 * 31) + 10000) // 20000
    assert old_75_support == 24
    assert full_support == 32


def test_lock_mutations_fail_closed() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    mutations = []
    changed = copy.deepcopy(lock)
    changed["runtime"]["requested_strengths_bp"][0] = 7800
    mutations.append(changed)
    changed = copy.deepcopy(lock)
    changed["execution"]["new_units_total"] = 671
    mutations.append(changed)
    changed = copy.deepcopy(lock)
    changed["interpretation"]["new_strengths_selected_after_reading_original_outcomes"] = False
    mutations.append(changed)
    changed = copy.deepcopy(lock)
    changed["inference"]["bootstrap_replicates"] = 1000
    mutations.append(changed)
    for value in mutations:
        with pytest.raises(RuntimeError):
            validate_lock(value)


def test_selected_rows_rejects_unlocked_search_points() -> None:
    for value in (7501, 8000, 9999):
        with pytest.raises(RuntimeError):
            selected_rows(value, 32)


def test_external_runtime_inputs_are_prospectively_pinned() -> None:
    lock = load_lock(LOCK)
    runtime = lock["runtime"]
    provenance = {
        "data_manifest_sha256": runtime["expected_data_manifest_sha256"],
        "environment_manifest_sha256": runtime[
            "expected_environment_manifest_sha256"
        ],
        "python_sha256": runtime["expected_python_sha256"],
    }
    validate_runtime_provenance(lock, provenance)
    changed = dict(provenance)
    changed["python_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runtime provenance changed"):
        validate_runtime_provenance(lock, changed)
