from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory,
    atomic_json,
)
from experiments.gradient_reuse_crossover.protocol import enumerate_units, load_lock
from experiments.gradient_reuse_crossover.unit_runner import (
    ATTEMPT_SCHEMA,
    DONE_SCHEMA,
    STREAM_SCHEMA,
    UNIT_SCHEMA,
    _final_accuracy,
    _validate_assignment_geometry,
    allocate_attempt,
    effective_args,
    stream_identity,
    validate_completed_unit,
)
from experiments.gradient_reuse_crossover.protocol import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/crossover_threshold_followup.json"


def test_runner_schemas_are_crossover_scoped() -> None:
    assert all(
        value.startswith("gradient_reuse.crossover_")
        for value in (UNIT_SCHEMA, DONE_SCHEMA, ATTEMPT_SCHEMA, STREAM_SCHEMA)
    )


@pytest.mark.parametrize("campaign", ["pmnist", "cifar"])
def test_effective_arguments_preserve_locked_assignment_geometry(campaign: str) -> None:
    lock = load_lock(LOCK_PATH)
    unit = enumerate_units(lock, campaign)[-1]
    arguments = effective_args(unit, lock)
    assert arguments["model"] == "er_assignment_strength"
    assert arguments["assignment_lambda_bp"] == 9677
    assert arguments["assignment_replica"] == 3
    assert arguments["minibatch_size"] == 32
    assert arguments["assignment_run_seed"] == unit.seed
    if campaign == "cifar":
        assert not {
            "permute_classes",
            "custom_class_order",
            "custom_task_order",
        } & set(arguments)


def test_streams_bind_parent_pmnist_bank_and_original_cifar_order() -> None:
    lock = load_lock(LOCK_PATH)
    pmnist = enumerate_units(lock, "pmnist")[0]
    cifar = enumerate_units(lock, "cifar")[0]
    pmnist_stream = stream_identity(lock, pmnist, ROOT)
    cifar_stream = stream_identity(lock, cifar, ROOT)
    assert pmnist_stream["parent_lock_sha256"] == lock["benchmarks"]["pmnist"][
        "task_bank_lock_sha256"
    ]
    assert pmnist_stream["task_bank_sha256"]
    assert cifar_stream["task_order"] == list(range(20))
    assert cifar_stream["class_order"] == list(range(100))
    assert cifar_stream["permute_classes"] is False
    assert cifar_stream["custom_class_order"] is None
    assert cifar_stream["custom_task_order"] is None


def test_assignment_geometry_requires_exact_locked_support() -> None:
    lock = load_lock(LOCK_PATH)
    unit = enumerate_units(lock, "cifar")[3 * 4]
    diagnostics = {
        "intervention_records": 2,
        "gradient_diagnostic_records": 2,
        "minimum_replay_batch_rows": 32,
        "maximum_replay_batch_rows": 32,
        "means": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
        },
        "minima": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
        },
        "maxima": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
        },
    }
    _validate_assignment_geometry(unit, diagnostics)
    changed = copy.deepcopy(diagnostics)
    changed["maximum_replay_batch_rows"] = 31
    with pytest.raises(RuntimeError, match="32-row"):
        _validate_assignment_geometry(unit, changed)


def test_endpoint_normalization_keeps_both_cifar_views() -> None:
    pmnist = {
        "domain_il": {"metrics": {"final_average_accuracy": 78.5}}
    }
    cifar = {
        "primary": {"final_average_accuracy": 12.25},
        "task_il": {"final_average_accuracy": 48.75},
    }
    assert _final_accuracy(pmnist, "pmnist") == {"domain_il": 78.5}
    assert _final_accuracy(cifar, "cifar") == {
        "task_il": 48.75,
        "class_il": 12.25,
    }


def test_attempt_allocator_preserves_failures_and_rejects_unknown_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unit"
    assert allocate_attempt(root).name == "attempt_001"
    assert allocate_attempt(root).name == "attempt_002"
    (root / "unexpected.txt").write_text("unsafe", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        allocate_attempt(root)


def test_completed_cifar_unit_returns_authenticated_seal_digests(
    tmp_path: Path,
) -> None:
    lock = load_lock(LOCK_PATH)
    unit = enumerate_units(lock, "cifar")[0]
    root = tmp_path / unit.unit_id
    attempt = root / "attempt_001"
    results = attempt / "mammoth_results"
    for setting in ("class-il", "task-il"):
        log = (
            results
            / setting
            / unit.mammoth_dataset
            / unit.mammoth_model
            / "logs.pyd"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "{'permute_classes': False, 'custom_class_order': None, "
            "'custom_task_order': None, 'noise_rate': 0.0}\n",
            encoding="utf-8",
        )
    arguments = effective_args(unit, lock)
    diagnostics = {
        "intervention_records": 1,
        "gradient_diagnostic_records": 1,
        "minimum_replay_batch_rows": 32,
        "maximum_replay_batch_rows": 32,
        "means": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
            "target_value_changed_fraction": 0.5,
        },
        "minima": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
        },
        "maxima": {
            "realized_lambda": unit.induced_strength,
            "selected_fraction": unit.selected_rows / 32,
        },
    }
    metrics = {
        "primary": {"final_average_accuracy": 12.0},
        "task_il": {"final_average_accuracy": 48.0},
    }
    provenance = {
        "lock_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "source_bundle_sha256": "c" * 64,
        "mammoth_commit": lock["upstream"]["commit"],
        "data_manifest_sha256": "d" * 64,
        "data_bundle_sha256": "e" * 64,
        "environment_manifest_sha256": "f" * 64,
        "environment_bundle_sha256": "0" * 64,
        "python_sha256": "1" * 64,
    }
    stream = stream_identity(lock, unit, ROOT)
    atomic_json(
        attempt / "RUN_MANIFEST.json",
        {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "COMPLETED",
            "campaign": "cifar",
            "unit": unit.as_dict(),
            "lock_content_sha256": canonical_sha256(lock),
            "stream_identity": stream,
            "effective_args_sha256": canonical_sha256(arguments),
            **provenance,
            "command": ["python", "-m", "runner"],
            "attempt": "attempt_001",
            "started_unix": 0.0,
            "completed_unix": 1.0,
            "returncode": 0,
        },
    )
    atomic_json(
        root / "UNIT.json",
        {
            "schema_version": UNIT_SCHEMA,
            "status": "PASS",
            "campaign": "cifar",
            "unit": unit.as_dict(),
            **provenance,
            "lock_content_sha256": canonical_sha256(lock),
            "stream_identity": stream,
            "task_bank_sha256": None,
            "order_sha256": stream["order_sha256"],
            "effective_args": arguments,
            "effective_args_sha256": canonical_sha256(arguments),
            "metrics": metrics,
            "endpoint_values": {"task_il": 48.0, "class_il": 12.0},
            "assignment_diagnostics": diagnostics,
            "bootstrap_audit_sha256": None,
            "attempt": "attempt_001",
            "runtime_seconds": 1.0,
        },
    )
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    atomic_json(
        root / "DONE.json",
        {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "unit_id": unit.unit_id,
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )
    sealed = validate_completed_unit(
        root,
        unit,
        campaign="cifar",
        lock=lock,
        expected_provenance=provenance,
        project=ROOT,
    )
    assert set(sealed["seal_digests"]) == {
        "unit_json_sha256",
        "unit_done_sha256",
        "artifact_bundle_sha256",
    }
    (root / "UNIT.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        validate_completed_unit(
            root,
            unit,
            campaign="cifar",
            lock=lock,
            expected_provenance=provenance,
            project=ROOT,
        )
