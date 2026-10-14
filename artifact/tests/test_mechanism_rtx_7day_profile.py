"""Focused checks for the prespecified seven-day RTX campaign profile."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_mechanisms.aggregate import (
    deterministic_shard_assignments,
)
from experiments.gradient_reuse_mechanisms.protocol import (
    FULL_PROTOCOL_UNIT_COUNT,
    FULL_RUN_SEEDS,
    PERMUTATION_REPLICAS,
    PUBLIC_RTX_7DAY_LOCK_SHA256,
    RTX_7DAY_PROFILE_ID,
    RTX_7DAY_PROTOCOL_UNIT_COUNT,
    RTX_7DAY_RUN_SEEDS,
    canonical_sha256,
    enumerate_units,
    expected_paper_unit_count,
    load_and_validate_lock,
    mechanism_profile_id,
    mechanism_source_experiment,
    sha256_file,
    validate_rtx_queue_completion,
    validate_source_manifest as validate_mechanism_source_manifest,
    validate_lock,
)
from experiments.source_provenance import build_source_manifest, get_source_spec


REPO = Path(__file__).resolve().parents[1]
FULL_LOCK = REPO / "configs" / "mechanisms.json"
RTX_LOCK = REPO / "configs" / "mechanisms_rtx_7day.json"


@pytest.fixture(scope="module")
def full_lock() -> dict:
    return load_and_validate_lock(FULL_LOCK)


@pytest.fixture(scope="module")
def rtx_lock() -> dict:
    return load_and_validate_lock(RTX_LOCK)


def test_original_full_lock_remains_the_480_unit_protocol(full_lock: dict) -> None:
    units = enumerate_units(full_lock)
    assert tuple(full_lock["runtime"]["seeds"]) == FULL_RUN_SEEDS
    assert len(units) == FULL_PROTOCOL_UNIT_COUNT == 480
    assert expected_paper_unit_count(full_lock) == 480
    assert mechanism_profile_id(full_lock) == "full_v1"
    assert mechanism_source_experiment(full_lock) == "mechanisms"


def test_rtx_profile_changes_only_the_independent_seed_count(
    full_lock: dict, rtx_lock: dict
) -> None:
    reconstructed = deepcopy(rtx_lock)
    amendment = reconstructed.pop("compute_budget_amendment")
    reconstructed["runtime"]["seeds"] = list(FULL_RUN_SEEDS)
    assert reconstructed == full_lock
    assert amendment["profile_id"] == RTX_7DAY_PROFILE_ID
    assert tuple(rtx_lock["runtime"]["seeds"]) == RTX_7DAY_RUN_SEEDS
    assert tuple(rtx_lock["runtime"]["permutation_replicas"]) == (
        PERMUTATION_REPLICAS
    )
    assert expected_paper_unit_count(rtx_lock) == RTX_7DAY_PROTOCOL_UNIT_COUNT
    assert mechanism_profile_id(rtx_lock) == RTX_7DAY_PROFILE_ID
    assert mechanism_source_experiment(rtx_lock) == "mechanisms_rtx_7day"


def test_rtx_profile_enumerates_all_288_prespecified_units(rtx_lock: dict) -> None:
    units = enumerate_units(rtx_lock)
    assert len(units) == RTX_7DAY_PROTOCOL_UNIT_COUNT == 288
    assert len({unit.unit_id for unit in units}) == 288
    assert {unit.benchmark for unit in units} == {
        "split_cifar100",
        "split_miniimagenet",
        "permuted_mnist",
    }
    assert {unit.seed for unit in units} == set(RTX_7DAY_RUN_SEEDS)

    benchmark_counts = Counter(unit.benchmark for unit in units)
    assert set(benchmark_counts.values()) == {96}
    method_counts = Counter(unit.method for unit in units)
    assert method_counts["naive_sgd"] == 9
    assert method_counts["mer_current_only"] == 9
    for method in (
        "experience_replay",
        "gem",
        "agem",
        "mer",
        "der",
        "pcgrad_replay",
    ):
        assert method_counts[method] == 45
    intervention_counts = Counter(unit.intervention for unit in units)
    assert intervention_counts == {"none": 18, "paired": 54, "permuted": 216}
    assert {
        unit.permutation_replica
        for unit in units
        if unit.intervention == "permuted"
    } == set(PERMUTATION_REPLICAS)


def test_four_shards_are_deterministic_complete_and_balanced(rtx_lock: dict) -> None:
    first = deterministic_shard_assignments(rtx_lock, 4)
    second = deterministic_shard_assignments(rtx_lock, 4)
    assert first == second
    assert set(first) == {0, 1, 2, 3}
    assert [len(first[index]) for index in range(4)] == [72, 72, 72, 72]
    flattened = [unit_id for index in range(4) for unit_id in first[index]]
    assert len(flattened) == len(set(flattened)) == 288


@pytest.mark.parametrize("mutation", ("amendment", "seed", "hyperparameter"))
def test_rtx_profile_fails_closed_for_unprespecified_changes(
    rtx_lock: dict, mutation: str
) -> None:
    changed = deepcopy(rtx_lock)
    if mutation == "amendment":
        changed["compute_budget_amendment"]["maximum_wall_time_days"] = 8
        match = "amendment differs"
    elif mutation == "seed":
        changed["runtime"]["seeds"].append(3404)
        match = "run seeds differ"
    elif mutation == "hyperparameter":
        changed["methods"]["gem"]["hyperparameters"]["split_cifar100"]["lr"] = 0.04
        match = "fields beyond runtime.seeds"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(RuntimeError, match=match):
        validate_lock(changed)


def test_rtx_public_source_closure_excludes_deployment_launcher() -> None:
    full = get_source_spec("mechanisms")
    rtx = get_source_spec("mechanisms_rtx_7day")
    assert full.configuration_path == "configs/mechanisms.json"
    assert rtx.configuration_path == "configs/mechanisms_rtx_7day.json"
    assert "configs/mechanisms.json" in rtx.files
    assert "scripts/launch_iclr_upgrade_campaign_rtx_7day.sh" not in rtx.files
    assert "tests/test_mechanism_rtx_7day_profile.py" in rtx.files


def test_rtx_source_manifest_authenticates_both_profile_and_parent(
    tmp_path: Path,
) -> None:
    manifest = build_source_manifest("mechanisms_rtx_7day", repo=REPO)
    path = tmp_path / "mechanisms_rtx_7day.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle, observed = validate_mechanism_source_manifest(
        path, repo=REPO, lock_path=RTX_LOCK
    )
    declared = {row["path"]: row["sha256"] for row in manifest["files"]}
    assert bundle == manifest["source_bundle_sha256"]
    assert observed["configs/mechanisms.json"] == declared[
        "configs/mechanisms.json"
    ]
    assert "configs/mechanisms_rtx_7day.json" in observed


def test_committed_rtx_source_manifest_maps_to_sanitized_public_checkout() -> None:
    committed = REPO / "provenance" / "source" / "mechanisms_rtx_7day.json"
    bundle, observed = validate_mechanism_source_manifest(
        committed, repo=REPO, lock_path=RTX_LOCK
    )
    payload = json.loads(committed.read_text(encoding="utf-8"))
    assert bundle == payload["source_bundle_sha256"]
    assert payload["configuration_sha256"] != PUBLIC_RTX_7DAY_LOCK_SHA256
    assert (
        observed["configs/mechanisms_rtx_7day.json"]
        == PUBLIC_RTX_7DAY_LOCK_SHA256
    )


def _queue_completion(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    beyond_root = tmp_path / "beyond"
    beyond_root.mkdir()
    beyond_plan = {
        "status": "PASS",
        "manifest_sha256": "d" * 64,
        "selection_profile": {
            "profile_file_sha256": "e" * 64,
            "profile_lock_manifest_sha256": "f" * 64,
        },
    }
    beyond_plan_path = beyond_root / "PROFILE_PLAN.json"
    beyond_plan_path.write_text(
        json.dumps(beyond_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = {
        "external_campaign": beyond_root / "RTX_7DAY_PROFILE_DONE.json",
        "controller_synthetic": tmp_path / "controller" / "RESULT.json",
        "controller_beijing": tmp_path / "controller_natural" / "RESULT.json",
        "mechanism_audit": tmp_path / "mechanisms" / "aggregate" / "DONE.json",
        "transformer_audit": tmp_path / "transformer" / "attempt_1" / "RESULT.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["external_campaign"].write_text("external\n", encoding="utf-8")
    paths["controller_synthetic"].write_text("controller\n", encoding="utf-8")
    paths["controller_beijing"].write_text("natural\n", encoding="utf-8")
    paths["transformer_audit"].write_text("transformer\n", encoding="utf-8")
    paths["mechanism_audit"].write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.mechanism_aggregate_done.v3",
                "status": "PASS",
                "paper_eligible": True,
                "complete_protocol": True,
                "profile_id": "rtx_7day_v1",
                "protocol_profile_id": "rtx_7day_v1",
                "complete_selected_profile": True,
                "complete_original_protocol": False,
                "unit_count": 288,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {name: path.resolve() for name, path in paths.items()}
    profile = tmp_path / "CAMPAIGN_PROFILE.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.iclr_rtx_7day_profile.v1",
                "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
                "profile_id": "rtx_7day_v1",
                "mechanism_expected_units": 288,
                "mechanism_shard_count": 4,
                "mechanism_parallel_shards": 1,
                "transformer_attempt_limit": 3,
                "mechanism_lock_sha256": "1" * 64,
                "mechanism_source_manifest_sha256": "2" * 64,
                "mechanism_source_bundle_sha256": "3" * 64,
                "beyond_profile_plan_sha256": sha256_file(beyond_plan_path),
                "beyond_profile_plan_manifest_sha256": "d" * 64,
                "beyond_profile_file_sha256": "e" * 64,
                "beyond_profile_lock_manifest_sha256": "f" * 64,
                "controller_config_sha256": "4" * 64,
                "controller_source_manifest_sha256": "5" * 64,
                "controller_source_bundle_sha256": "6" * 64,
                "controller_natural_config_sha256": "7" * 64,
                "controller_natural_source_manifest_sha256": "8" * 64,
                "controller_natural_source_bundle_sha256": "9" * 64,
                "transformer_config_sha256": "a" * 64,
                "transformer_source_manifest_sha256": "b" * 64,
                "transformer_source_bundle_sha256": "c" * 64,
                "beyond_results_root": str(beyond_root.resolve()),
                "beyond_tta_cache_root": "/cache/beyond",
                "beyond_play3_store_root": "/cache/play3",
                "transformer_asset_root": "/assets/transformer",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["campaign_profile"] = profile.resolve()
    payload = {
        "schema_version": "gradient_reuse.iclr_rtx_7day_queue_done.v1",
        "status": "PASS",
        "profile_id": "rtx_7day_v1",
        "mechanism_unit_count": 288,
        "transformer_attempt_index": 1,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    completion = tmp_path / "RTX_7DAY_PROFILE_DONE.json"
    completion.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion, artifacts


def test_queue_completion_authenticates_exact_artifacts(tmp_path: Path) -> None:
    completion, artifacts = _queue_completion(tmp_path)
    observed = validate_rtx_queue_completion(
        completion, expected_profile_path=artifacts["campaign_profile"]
    )
    assert observed["mechanism_unit_count"] == 288


def test_queue_completion_rejects_artifact_tampering(tmp_path: Path) -> None:
    completion, artifacts = _queue_completion(tmp_path)
    artifacts["mechanism_audit"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact digest mismatch"):
        validate_rtx_queue_completion(
            completion, expected_profile_path=artifacts["campaign_profile"]
        )


def test_queue_completion_rejects_stale_external_profile_plan(tmp_path: Path) -> None:
    completion, artifacts = _queue_completion(tmp_path)
    beyond_plan = artifacts["external_campaign"].parent / "PROFILE_PLAN.json"
    beyond_plan.write_text("stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="profile plan digest mismatch"):
        validate_rtx_queue_completion(
            completion, expected_profile_path=artifacts["campaign_profile"]
        )


def test_queue_completion_rejects_reduced_mechanism_claiming_original(
    tmp_path: Path,
) -> None:
    completion, artifacts = _queue_completion(tmp_path)
    mechanism = json.loads(artifacts["mechanism_audit"].read_text())
    mechanism["complete_original_protocol"] = True
    artifacts["mechanism_audit"].write_text(
        json.dumps(mechanism, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload = json.loads(completion.read_text())
    payload["artifacts"]["mechanism_audit"]["sha256"] = sha256_file(
        artifacts["mechanism_audit"]
    )
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    completion.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not the selected profile"):
        validate_rtx_queue_completion(
            completion, expected_profile_path=artifacts["campaign_profile"]
        )


def test_queue_completion_binds_the_selected_transformer_attempt(
    tmp_path: Path,
) -> None:
    completion, artifacts = _queue_completion(tmp_path)
    payload = json.loads(completion.read_text())
    payload["transformer_attempt_index"] = 2
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    completion.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="artifact path changed: transformer_audit"):
        validate_rtx_queue_completion(
            completion, expected_profile_path=artifacts["campaign_profile"]
        )
