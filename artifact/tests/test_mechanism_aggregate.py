"""Focused tests for fail-closed mechanism-shard aggregation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from experiments.gradient_reuse_mechanisms.aggregate import (
    AggregationIdentity,
    EXPECTED_PAPER_UNITS,
    MechanismAggregationError,
    aggregate_shards,
    deterministic_shard_assignments,
    expected_unit_identity,
    load_aggregation_identity,
    _validate_cross_arm,
    _validate_unit,
)
from experiments.gradient_reuse_mechanisms.protocol import (
    DIAGNOSTIC_FIELDS,
    MechanismUnit,
    canonical_sha256,
    enumerate_units,
    load_and_validate_lock,
    sha256_file,
)
from experiments.gradient_reuse_mechanisms.run import (
    SHARD_SELECTION_SCHEME,
    effective_mammoth_args,
)
from experiments.source_provenance import build_source_manifest


REPO = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO / "configs" / "mechanisms.json"


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    return load_and_validate_lock(LOCK_PATH)


@pytest.fixture(scope="module")
def identity() -> AggregationIdentity:
    return AggregationIdentity(
        lock_sha256=sha256_file(LOCK_PATH),
        source_manifest_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        data_manifest_sha256="3" * 64,
        data_bundle_sha256="4" * 64,
        environment_manifest_sha256="5" * 64,
        environment_bundle_sha256="6" * 64,
        overlay_manifest_sha256="7" * 64,
        overlay_bundle_sha256="8" * 64,
        upstream_commit="e75a491c69fd729edeb01431afb753d9157d9a81",
        device="cuda:0",
    )


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _value(unit: MechanismUnit) -> float:
    if unit.intervention == "none":
        return 50.0
    if unit.intervention == "paired":
        return 55.0
    return 51.0 + float(unit.permutation_replica)


def _metrics(unit: MechanismUnit, tasks: int) -> dict[str, Any]:
    value = _value(unit)
    matrix = [[value for _ in range(row)] for row in range(1, tasks + 1)]
    trajectory = [value for _ in range(tasks)]
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": value,
        "average_incremental_accuracy": value,
        "backward_transfer": 0.0,
        "backward_transfer_available": True,
        "accuracy_matrix": matrix,
        "accuracy_matrix_available": True,
        "mean_accuracy_trajectory": trajectory,
        "evaluation_tasks": tasks,
    }


def _fingerprints(unit: MechanismUnit) -> list[dict[str, Any]]:
    if unit.intervention == "none":
        return []
    # This intentionally excludes intervention/replica, matching the scientific
    # cross-arm input/index/original-target fingerprint.
    original_target = "c" * 64
    target_multiset = "d" * 64
    if (
        unit.method == "der"
        and unit.intervention == "permuted"
        and unit.permutation_replica == 1
    ):
        original_target = "e" * 64
        target_multiset = "f" * 64
    return [
        {
            "observation": 100,
            "event": 0,
            "channel": "replay",
            "rows": 32,
            "method": unit.mammoth_model,
            "benchmark": unit.benchmark,
            "dataset": unit.mammoth_dataset,
            "run_seed": unit.seed,
            "input_sha256": "a" * 64,
            "sampled_indexes_sha256": "b" * 64,
            "original_target_sha256": original_target,
            "target_multiset_sha256": target_multiset,
        }
    ]


def _diagnostics(unit: MechanismUnit) -> dict[str, Any]:
    if unit.intervention == "none":
        return {"records": 0}
    value = 4.0 if unit.intervention == "paired" else (
        1.0 + float(unit.permutation_replica)
    )
    projection_method = unit.method in {"gem", "agem", "pcgrad_replay"}
    payload = {
        "records": 1,
        "mean_pairing_component_norm": value,
        "mean_finite_permutation_variance": value / 10.0,
        "mean_true_permuted_cosine": value / 20.0,
        "mean_fixed_point_fraction": (
            1.0 if unit.intervention == "paired" else 0.25
        ),
        "projection_activation_rate": value / 100.0 if projection_method else None,
        "mean_pre_projection_constraint": -value if projection_method else None,
        "mean_post_projection_constraint": value if projection_method else None,
    }
    assert set(DIAGNOSTIC_FIELDS).issubset(payload)
    return payload


def _inventory(unit_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(unit_root)): sha256_file(path)
        for path in sorted(unit_root.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }


def _write_unit(
    unit_root: Path,
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
) -> None:
    attempt_name = "attempt_001"
    attempt = unit_root / attempt_name
    expected_identity = expected_unit_identity(unit, lock, identity)
    effective = effective_mammoth_args(
        unit, lock, device=identity.device, smoke=False
    )
    record = {
        "schema_version": "gradient_reuse.mechanism_unit.v2",
        "identity": expected_identity,
        "benchmark": unit.benchmark,
        "method": unit.method,
        "mammoth_model": unit.mammoth_model,
        "estimand_class": lock["methods"][unit.method]["estimand_class"],
        "intervention": unit.intervention,
        "seed": unit.seed,
        "permutation_replica": unit.permutation_replica,
        "metrics": _metrics(unit, int(lock["benchmarks"][unit.benchmark]["tasks"])),
        "diagnostics": _diagnostics(unit),
        "sampled_event_fingerprints": _fingerprints(unit),
        "effective_args": effective,
        "runtime_seconds": 1.0,
        "paper_eligible": True,
        "attempt": attempt_name,
    }
    _json(unit_root / "UNIT.json", record)
    _json(
        attempt / "RUN_MANIFEST.json",
        {
            "schema_version": "gradient_reuse.mechanism_attempt.v1",
            "status": "COMPLETED",
            "identity": expected_identity,
            "identity_sha256": canonical_sha256(expected_identity),
            "attempt": attempt_name,
            "command": ["python", "main.py"],
            "started_unix": 1.0,
            "completed_unix": 2.0,
            "returncode": 0,
        },
    )
    (attempt / "stdout.log").write_text("ok\n", encoding="utf-8")
    result = attempt / "mammoth_results" / "setting" / "data" / "model" / "logs.pyd"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("{}\n", encoding="utf-8")
    if unit.intervention != "none":
        (attempt / "pairing_audit.jsonl").write_text("{}\n", encoding="utf-8")
    inventory = _inventory(unit_root)
    _json(
        unit_root / "DONE.json",
        {
            "schema_version": "gradient_reuse.mechanism_unit_done.v2",
            "status": "PASS",
            "identity_sha256": canonical_sha256(expected_identity),
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )


def _invocation(
    selected: Sequence[MechanismUnit],
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
    *,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.mechanism_invocation.v2",
        **identity.as_dict(),
        "smoke": False,
        "paper_eligible": False,
        "device": identity.device,
        "selection_scheme": SHARD_SELECTION_SCHEME,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selected_unit_ids": [unit.unit_id for unit in selected],
        "effective_args_sha256": {
            unit.unit_id: canonical_sha256(
                effective_mammoth_args(
                    unit, lock, device=identity.device, smoke=False
                )
            )
            for unit in selected
        },
    }


def _finish_shard(
    shard: Path,
    selected: Sequence[MechanismUnit],
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
    *,
    shard_count: int,
    shard_index: int,
) -> None:
    completed_unit_ids = [unit.unit_id for unit in selected]
    completed_unit_done_sha256 = {
        unit_id: sha256_file(shard / "units" / unit_id / "DONE.json")
        for unit_id in completed_unit_ids
    }
    _json(
        shard / "RUN_MANIFEST.json",
        {
            "schema_version": "gradient_reuse.mechanism_campaign.v2",
            "status": "COMPLETED",
            "started_unix": 1.0,
            "run_attempt": 1,
            "completed_unix": 2.0,
            "completed_units": len(selected),
            "completed_unit_ids": completed_unit_ids,
            "completed_unit_done_sha256": completed_unit_done_sha256,
            "complete_protocol": False,
            "invocation": _invocation(
                selected,
                lock,
                identity,
                shard_count=shard_count,
                shard_index=shard_index,
            ),
        },
    )
    inventory = {
        str(path.relative_to(shard)): sha256_file(path)
        for path in sorted(shard.rglob("*"))
        if path.is_file() and path != shard / "DONE.json"
    }
    _json(
        shard / "DONE.json",
        {
            "schema_version": "gradient_reuse.mechanism_done.v2",
            "status": "PASS",
            "paper_eligible": False,
            "complete_protocol": False,
            "completed_unit_ids": completed_unit_ids,
            "completed_unit_done_sha256": completed_unit_done_sha256,
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )


def _write_shard(
    root: Path,
    selected: Sequence[MechanismUnit],
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
    *,
    shard_count: int,
    shard_index: int,
) -> Path:
    (root / "units").mkdir(parents=True)
    for unit in selected:
        _write_unit(root / "units" / unit.unit_id, unit, lock, identity)
    _finish_shard(
        root,
        selected,
        lock,
        identity,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    return root


def _refresh_shard_done(shard: Path) -> None:
    done = json.loads((shard / "DONE.json").read_text())
    inventory = {
        str(path.relative_to(shard)): sha256_file(path)
        for path in sorted(shard.rglob("*"))
        if path.is_file() and path != shard / "DONE.json"
    }
    done["artifact_count"] = len(inventory)
    done["artifact_inventory"] = inventory
    done["artifact_bundle_sha256"] = canonical_sha256(inventory)
    _json(shard / "DONE.json", done)


def _refresh_unit_done(unit_root: Path) -> None:
    done = json.loads((unit_root / "DONE.json").read_text())
    inventory = _inventory(unit_root)
    done["artifact_count"] = len(inventory)
    done["artifact_inventory"] = inventory
    done["artifact_bundle_sha256"] = canonical_sha256(inventory)
    _json(unit_root / "DONE.json", done)


@pytest.fixture(scope="module")
def complete_aggregate(
    tmp_path_factory: pytest.TempPathFactory,
    lock: dict[str, Any],
    identity: AggregationIdentity,
) -> Path:
    root = tmp_path_factory.mktemp("mechanism_complete")
    units = enumerate_units(lock)
    shards = []
    for shard_index in range(3):
        selected = [
            unit for index, unit in enumerate(units) if index % 3 == shard_index
        ]
        shards.append(
            _write_shard(
                root / f"shard_{shard_index}",
                selected,
                lock,
                identity,
                shard_count=3,
                shard_index=shard_index,
            )
        )
    return aggregate_shards(
        shards, output=root / "aggregate", lock=lock, identity=identity
    )


def test_modulo_partition_is_deterministic_and_complete(lock: dict[str, Any]) -> None:
    first = deterministic_shard_assignments(lock, 3)
    second = deterministic_shard_assignments(lock, 3)
    assert first == second
    assert [len(first[index]) for index in range(3)] == [160, 160, 160]
    flattened = [unit_id for values in first.values() for unit_id in values]
    assert len(flattened) == EXPECTED_PAPER_UNITS
    assert len(flattened) == len(set(flattened))


def test_config_and_unit_estimand_classes_are_exact(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    changed = deepcopy(lock)
    changed["methods"]["der"]["estimand_class"] = "assignment_only"
    with pytest.raises(RuntimeError, match="estimand class changed"):
        enumerate_units(changed)

    der = next(unit for unit in enumerate_units(lock) if unit.method == "der")
    unit_root = tmp_path / der.unit_id
    _write_unit(unit_root, der, lock, identity)
    record = json.loads((unit_root / "UNIT.json").read_text())
    record["estimand_class"] = "assignment_only"
    _json(unit_root / "UNIT.json", record)
    _refresh_unit_done(unit_root)
    with pytest.raises(MechanismAggregationError, match="metadata mismatch estimand_class"):
        _validate_unit(unit_root, der, lock, identity)


def test_complete_aggregate_is_authenticated_and_recomputes_estimands(
    complete_aggregate: Path,
) -> None:
    done = json.loads((complete_aggregate / "DONE.json").read_text())
    manifest = json.loads((complete_aggregate / "RUN_MANIFEST.json").read_text())
    summary = json.loads((complete_aggregate / "SUMMARY.json").read_text())
    assert done["schema_version"] == "gradient_reuse.mechanism_aggregate_done.v3"
    assert manifest["schema_version"] == (
        "gradient_reuse.mechanism_aggregate_campaign.v3"
    )
    assert summary["schema_version"] == (
        "gradient_reuse.mechanism_aggregate_summary.v3"
    )
    for payload in (done, manifest, summary):
        assert payload["profile_id"] == "full_v1"
        assert payload["protocol_profile_id"] == "full_v1"
        assert payload["complete_selected_profile"] is True
        assert payload["complete_original_protocol"] is True
    assert done["paper_eligible"] is True
    assert done["unit_count"] == EXPECTED_PAPER_UNITS
    assert manifest["unit_count"] == EXPECTED_PAPER_UNITS
    assert len(manifest["unit_artifacts"]) == EXPECTED_PAPER_UNITS
    assert summary["permutation_replica_ids"] == [0, 1, 2, 3]
    assert len(summary["method_summaries"]) == 18
    assert len(summary["diagnostic_summaries"]) == 18
    assert summary["primary_inference"] == "within_method_paired_vs_permuted"
    assert summary["primary_endpoint"] == "final_average_accuracy_across_observed_tasks"
    assert summary["primary_estimand"] == "V_pair"
    assert summary["cross_method_accuracy_ranking_supported"] is False
    assert summary["estimand_classes"]["der"] == (
        "path_dependent_endogenous_targets"
    )
    assert summary["methods_pooled_across_estimand_classes"] is False
    assert all(
        row["estimand_class"] == summary["estimand_classes"][row["method"]]
        for row in summary["method_summaries"]
    )
    comparability = summary["cross_arm_comparability"]
    assert len(comparability["assignment_only_groups"]) == 75
    assert len(comparability["path_dependent_endogenous_target_groups"]) == 15
    assert comparability["audit_scope"] == "sampled_events_only"
    assert comparability["sampled_endogenous_target_divergence_count"] == 15
    row = summary["method_summaries"][0]
    assert row["V_total"]["mean"] == pytest.approx(5.0)
    assert row["V_perm"]["mean"] == pytest.approx(2.5)
    assert row["V_pair"]["mean"] == pytest.approx(2.5)
    assert row["V_perm"]["interval"] == (
        "two_sided_student_t_with_nested_permutation_mc"
    )
    diagnostic = summary["diagnostic_summaries"][0]["diagnostics"][
        "mean_pairing_component_norm"
    ]
    assert diagnostic["paired_minus_permuted"]["mean"] == pytest.approx(1.5)
    assert diagnostic["paired_minus_permuted"]["interval"] == (
        "two_sided_student_t_with_nested_permutation_mc"
    )
    for relative, digest in done["outputs"].items():
        assert sha256_file(complete_aggregate / relative) == digest


def test_aggregate_requires_a_fresh_output(
    complete_aggregate: Path,
    lock: dict[str, Any],
    identity: AggregationIdentity,
) -> None:
    with pytest.raises(MechanismAggregationError, match="must be fresh"):
        aggregate_shards(
            [complete_aggregate.parent / "shard_0"],
            output=complete_aggregate,
            lock=lock,
            identity=identity,
        )


def test_missing_unit_coverage_is_rejected(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    unit = enumerate_units(lock)[0]
    shard = _write_shard(
        tmp_path / "shard",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    with pytest.raises(MechanismAggregationError, match="each canonical shard"):
        aggregate_shards([shard], output=tmp_path / "out", lock=lock, identity=identity)


def test_duplicate_units_across_shards_are_rejected(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    unit = enumerate_units(lock)[0]
    first = _write_shard(
        tmp_path / "one",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    second = _write_shard(
        tmp_path / "two",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    with pytest.raises(MechanismAggregationError, match="duplicate unit"):
        aggregate_shards(
            [first, second], output=tmp_path / "out", lock=lock, identity=identity
        )


def test_smoke_or_mixed_identity_is_rejected(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    unit = enumerate_units(lock)[0]
    smoke = _write_shard(
        tmp_path / "smoke",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    run = json.loads((smoke / "RUN_MANIFEST.json").read_text())
    run["invocation"]["smoke"] = True
    _json(smoke / "RUN_MANIFEST.json", run)
    _refresh_shard_done(smoke)
    with pytest.raises(MechanismAggregationError, match="mixed shard identity smoke"):
        aggregate_shards([smoke], output=tmp_path / "smoke_out", lock=lock, identity=identity)

    mixed = _write_shard(
        tmp_path / "mixed",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    run = json.loads((mixed / "RUN_MANIFEST.json").read_text())
    run["invocation"]["data_bundle_sha256"] = "f" * 64
    _json(mixed / "RUN_MANIFEST.json", run)
    _refresh_shard_done(mixed)
    with pytest.raises(MechanismAggregationError, match="mixed shard identity data_bundle"):
        aggregate_shards([mixed], output=tmp_path / "mixed_out", lock=lock, identity=identity)


def test_unit_artifact_tampering_is_rejected(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    unit = enumerate_units(lock)[0]
    shard = _write_shard(
        tmp_path / "shard",
        [unit],
        lock,
        identity,
        shard_count=EXPECTED_PAPER_UNITS,
        shard_index=0,
    )
    stdout = shard / "units" / unit.unit_id / "attempt_001" / "stdout.log"
    stdout.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MechanismAggregationError, match="artifact inventory mismatch"):
        aggregate_shards([shard], output=tmp_path / "out", lock=lock, identity=identity)


def test_paper_permuted_mnist_unit_requires_full_matrix_and_bwt(
    tmp_path: Path, lock: dict[str, Any], identity: AggregationIdentity
) -> None:
    unit = next(
        value
        for value in enumerate_units(lock)
        if value.benchmark == "permuted_mnist" and value.method == "naive_sgd"
    )
    unit_root = tmp_path / unit.unit_id
    _write_unit(unit_root, unit, lock, identity)
    record = json.loads((unit_root / "UNIT.json").read_text())
    record["metrics"]["accuracy_matrix"] = None
    record["metrics"]["accuracy_matrix_available"] = False
    record["metrics"]["backward_transfer"] = None
    record["metrics"]["backward_transfer_available"] = False
    _json(unit_root / "UNIT.json", record)
    _refresh_unit_done(unit_root)
    with pytest.raises(MechanismAggregationError, match="lacks a full accuracy matrix"):
        _validate_unit(unit_root, unit, lock, identity)


def test_external_identity_manifests_are_validated_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _json(source, build_source_manifest("mechanisms", repo=REPO))
    files = {"dataset.bin": "a" * 64}
    data = tmp_path / "data.json"
    _json(
        data,
        {
            "schema_version": "gradient_reuse.mechanism_data_manifest.v1",
            "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
            "files": files,
            "data_bundle_sha256": canonical_sha256(files),
            "benchmarks": {},
        },
    )
    observed = {"device": "cuda:0", "python": "test"}
    environment = tmp_path / "environment.json"
    _json(
        environment,
        {
            "schema_version": "gradient_reuse.mechanism_environment.v1",
            "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
            "observed": observed,
            "environment_bundle_sha256": canonical_sha256(observed),
        },
    )
    overlay_files = {"models/a.py": "b" * 64}
    overlay = tmp_path / "overlay.json"
    _json(
        overlay,
        {
            "schema_version": "gradient_reuse.mammoth_overlay.v1",
            "upstream_commit": "e75a491c69fd729edeb01431afb753d9157d9a81",
            "files": overlay_files,
            "overlay_bundle_sha256": canonical_sha256(overlay_files),
        },
    )
    lock, identity = load_aggregation_identity(
        lock_path=LOCK_PATH,
        source_manifest=source,
        data_manifest=data,
        environment_manifest=environment,
        overlay_manifest=overlay,
        repo=REPO,
    )
    assert len(enumerate_units(lock)) == EXPECTED_PAPER_UNITS
    assert identity.data_bundle_sha256 == canonical_sha256(files)

    payload = json.loads(data.read_text())
    payload["data_bundle_sha256"] = "0" * 64
    _json(data, payload)
    with pytest.raises(MechanismAggregationError, match="data manifest bundle"):
        load_aggregation_identity(
            lock_path=LOCK_PATH,
            source_manifest=source,
            data_manifest=data,
            environment_manifest=environment,
            overlay_manifest=overlay,
            repo=REPO,
        )
