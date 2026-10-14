from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from experiments.gradient_reuse_corrected_reference_order import (
    unit_runner as corrected_unit_runner,
)
from experiments.gradient_reuse_corrected_reference_order.cifar_protocol import (
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    validate_lock as validate_cifar_lock,
)
from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory,
    atomic_json,
)
from experiments.gradient_reuse_assignment_strength.protocol import parse_mammoth_log
from experiments.gradient_reuse_corrected_reference_order.bank import canonical_sha256
from experiments.gradient_reuse_corrected_reference_order.protocol import (
    enumerate_units as enumerate_pmnist_units,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
    validate_lock as validate_pmnist_lock,
)
from experiments.gradient_reuse_corrected_reference_order.queue import queue_plan
from experiments.gradient_reuse_corrected_reference_order.source import (
    SOURCE_FILES,
    build_source_manifest,
)
from experiments.gradient_reuse_corrected_reference_order.unit_runner import (
    DONE_SCHEMA,
    UNIT_SCHEMA,
    build_command,
    effective_args,
    validate_completed_unit,
)


ROOT = Path(__file__).resolve().parents[1]
PMNIST_LOCK = ROOT / "configs/corrected_reference_order_audit.json"
CIFAR_LOCK = ROOT / "configs/corrected_cifar_order_audit.json"


def test_mammoth_log_parser_accepts_multiline_numpy_records(tmp_path: Path) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text(
        "{'record': 'older'}\n"
        "{'custom_class_order': array([0, 1,\n"
        " 2, 3]), 'score': np.float64(7.5)}\n",
        encoding="utf-8",
    )
    assert parse_mammoth_log(path) == {
        "custom_class_order": [0, 1, 2, 3],
        "score": 7.5,
    }


def test_mammoth_log_parser_rejects_non_expression_records(tmp_path: Path) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text("{'record': 1}\nvalue = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-expression"):
        parse_mammoth_log(path)


def test_pmnist_grid_is_exactly_228_and_priority_prefix_is_132() -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    units = enumerate_pmnist_units(lock)
    assert len(units) == len({unit.unit_id for unit in units}) == 228
    assert Counter(unit.seed for unit in units) == {seed: 19 for seed in range(3801, 3813)}
    for seed in range(3801, 3813):
        panel = [unit for unit in units if unit.seed == seed]
        assert Counter(unit.arm for unit in panel) == {
            "standard_no_replay": 1,
            "fixed_current_no_replay": 1,
            "er_paired": 1,
            "er_reassigned": 16,
        }
        assert Counter(unit.lambda_bp for unit in panel if unit.arm == "er_reassigned") == {
            2500: 4,
            5000: 4,
            7500: 4,
            10000: 4,
        }
    plan = queue_plan(PMNIST_LOCK, "pmnist")
    assert plan["priority_units"] == 132
    assert plan["completion_requires_units"] == 228
    assert plan["partial_scientific_aggregate_forbidden"] is True


def test_cifar_grid_is_exactly_504_with_seven_units_per_seed_order() -> None:
    lock = load_cifar_lock(CIFAR_LOCK)
    units = enumerate_cifar_units(lock)
    assert len(units) == len({unit.unit_id for unit in units}) == 504
    assert Counter((unit.order_family, unit.order_replica) for unit in units) == {
        (family, replica): 84
        for family in ("task_block", "full_class")
        for replica in range(3)
    }
    assert Counter(
        (unit.order_family, unit.order_replica, unit.seed) for unit in units
    ) == {
        (family, replica, seed): 7
        for family in ("task_block", "full_class")
        for replica in range(3)
        for seed in range(3801, 3813)
    }
    assert Counter(unit.arm for unit in units) == {
        "standard_no_replay": 72,
        "fixed_current_no_replay": 72,
        "er_paired": 72,
        "er_reassigned": 288,
    }


def test_effective_learning_rates_and_explicit_class_order_are_locked(tmp_path: Path) -> None:
    pmnist = load_pmnist_lock(PMNIST_LOCK)
    pu = enumerate_pmnist_units(pmnist)
    standard = next(unit for unit in pu if unit.arm == "standard_no_replay")
    fixed = next(unit for unit in pu if unit.arm == "fixed_current_no_replay")
    assert effective_args(standard, pmnist, "pmnist")["lr"] == 0.1
    assert effective_args(fixed, pmnist, "pmnist")["lr"] == 0.08

    cifar = load_cifar_lock(CIFAR_LOCK)
    cu = enumerate_cifar_units(cifar)
    standard_cifar = next(unit for unit in cu if unit.arm == "standard_no_replay")
    fixed_cifar = next(unit for unit in cu if unit.arm == "fixed_current_no_replay")
    paired_cifar = next(unit for unit in cu if unit.arm == "er_paired")
    assert effective_args(standard_cifar, cifar, "cifar")["lr"] == 0.03
    assert effective_args(paired_cifar, cifar, "cifar")["lr"] == 0.03
    fixed_args = effective_args(fixed_cifar, cifar, "cifar")
    assert fixed_args["lr"] == 0.015
    assert fixed_args["permute_classes"] == 1
    assert len(fixed_args["custom_class_order"].split(",")) == 100
    command, arguments = build_command(
        fixed_cifar,
        cifar,
        "cifar",
        python=Path("/python"),
        mammoth_root=Path("/mammoth"),
        data_root=Path("/data"),
        results=tmp_path / "results",
        assignment_audit=tmp_path / "assignment.jsonl",
        bootstrap_audit=tmp_path / "bootstrap.json",
        contract=tmp_path / "contract.json",
        bank_or_order_manifest=tmp_path / "order.json",
    )
    assert command.count("--custom_class_order") == 1
    assert arguments["lr"] == 0.015


def test_assignment_summary_adapter_carries_the_locked_run_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    unit = next(
        candidate
        for candidate in enumerate_pmnist_units(lock)
        if candidate.arm == "er_paired"
    )
    arguments = effective_args(unit, lock, "pmnist")
    audit_path = tmp_path / "assignment_audit.jsonl"

    def checked_summary(adapter, path, expected):
        assert adapter.method == "experience_replay"
        assert adapter.lambda_bp == unit.lambda_bp
        assert adapter.benchmark == unit.benchmark
        assert adapter.mammoth_dataset == unit.mammoth_dataset
        assert adapter.seed == unit.seed
        assert path == audit_path
        assert expected["seed"] == unit.seed
        return {"record_count": 1}

    monkeypatch.setattr(corrected_unit_runner, "summarize_audit", checked_summary)
    assert corrected_unit_runner._assignment_summary(
        unit, audit_path, arguments
    ) == {"record_count": 1}


def test_lock_mutations_fail_closed() -> None:
    pmnist = load_pmnist_lock(PMNIST_LOCK)
    bad_pmnist = copy.deepcopy(pmnist)
    bad_pmnist["execution"]["expected_units"] = 132
    with pytest.raises(RuntimeError):
        validate_pmnist_lock(bad_pmnist)
    cifar = load_cifar_lock(CIFAR_LOCK)
    bad_cifar = copy.deepcopy(cifar)
    bad_cifar["orders"][0]["old_to_stream_label"][0] = 99
    with pytest.raises(RuntimeError):
        validate_cifar_lock(bad_cifar)


def test_pmnist_seven_test_family_and_signs_are_prelocked() -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    inference = lock["inference"]
    assert inference["confirmatory_tests"] == 7
    assert inference["confirmatory_family"] == [
        "dose_slope",
        "paired_minus_standard",
        "paired_minus_fixed",
        "lambda_07500bp_minus_standard",
        "lambda_07500bp_minus_fixed",
        "lambda_10000bp_minus_standard",
        "lambda_10000bp_minus_fixed",
    ]
    assert inference["contrast_sign_convention"]["reassigned_minus_fixed"].startswith(
        "positive favors replica-mean reassigned ER"
    )


def test_source_closure_is_sorted_complete_and_binds_both_configs() -> None:
    assert SOURCE_FILES == tuple(sorted(SOURCE_FILES))
    manifest = build_source_manifest(ROOT)
    assert len(manifest["files"]) == len(SOURCE_FILES)
    assert manifest["campaigns"] == {
        "pmnist": {"expected_units": 228},
        "cifar": {"expected_units": 504},
    }
    assert set(manifest["configuration_sha256"]) == {
        "configs/corrected_reference_order_audit.json",
        "configs/corrected_cifar_order_audit.json",
    }


def test_completed_unit_seal_is_bound_to_lock_and_campaign_provenance(tmp_path: Path) -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    unit = enumerate_pmnist_units(lock)[0]
    provenance = {
        "lock_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "source_bundle_sha256": "3" * 64,
        "mammoth_commit": lock["upstream"]["commit"],
        "data_bundle_sha256": "4" * 64,
        "environment_bundle_sha256": "5" * 64,
        "python_sha256": "6" * 64,
    }
    root = tmp_path / unit.unit_id
    root.mkdir()
    atomic_json(
        root / "UNIT.json",
        {
            "schema_version": UNIT_SCHEMA,
            "status": "PASS",
            "campaign": "pmnist",
            "unit": unit.as_dict(),
            **provenance,
            "lock_content_sha256": canonical_sha256(lock),
            "task_bank_sha256": expected_task_bank_digest(lock, unit.seed),
            "order_sha256": None,
        },
    )
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    atomic_json(
        root / "DONE.json",
        {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "unit_id": unit.unit_id,
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )
    assert validate_completed_unit(
        root,
        unit,
        campaign="pmnist",
        lock=lock,
        expected_provenance=provenance,
    )["unit"] == unit.as_dict()
    changed = dict(provenance, data_bundle_sha256="7" * 64)
    with pytest.raises(RuntimeError, match="campaign binding"):
        validate_completed_unit(
            root,
            unit,
            campaign="pmnist",
            lock=lock,
            expected_provenance=changed,
        )
