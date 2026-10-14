"""Focused integration contracts for the 816-unit exact-reference follow-up."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory,
    atomic_json,
)
from experiments.gradient_reuse_corrected_reference_order.bank import canonical_sha256
from experiments.gradient_reuse_exact_reference import unit_runner as exact_unit_runner
from experiments.gradient_reuse_exact_reference import queue
from experiments.gradient_reuse_exact_reference.cifar_protocol import (
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    order_manifest,
)
from experiments.gradient_reuse_exact_reference.mammoth_bootstrap import (
    validate_parsed_args,
)
from experiments.gradient_reuse_exact_reference.protocol import (
    enumerate_units as enumerate_pmnist_units,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
    sha256_file,
)
from experiments.gradient_reuse_exact_reference.unit_runner import (
    DONE_SCHEMA as UNIT_DONE_SCHEMA,
    UNIT_SCHEMA,
    build_command,
    effective_args,
    validate_completed_unit,
)


ROOT = Path(__file__).resolve().parents[1]
PMNIST_LOCK = ROOT / "configs" / "exact_reference_pmnist.json"
CIFAR_LOCK = ROOT / "configs" / "exact_reference_cifar.json"


def _one(units, arm: str):
    return next(unit for unit in units if unit.arm == arm)


def test_queue_plans_require_all_240_plus_576_units() -> None:
    pmnist = queue.queue_plan(PMNIST_LOCK, "pmnist")
    cifar = queue.queue_plan(CIFAR_LOCK, "cifar")
    assert pmnist["expected_units"] == pmnist["completion_requires_units"] == 240
    assert cifar["expected_units"] == cifar["completion_requires_units"] == 576
    assert len(pmnist["ordered_unit_ids"]) == len(
        set(pmnist["ordered_unit_ids"])
    ) == 240
    assert len(cifar["ordered_unit_ids"]) == len(
        set(cifar["ordered_unit_ids"])
    ) == 576
    assert pmnist["partial_scientific_aggregate_forbidden"] is True
    assert cifar["partial_scientific_aggregate_forbidden"] is True


def test_campaign_output_rejects_a_symlinked_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real-parent"
    target.mkdir()
    alias = tmp_path / "symbolic-parent"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic path component"):
        queue._output_root(alias / "campaign", ROOT)
    assert not (target / "campaign").exists()


@pytest.mark.parametrize(
    "relative", (Path("logs"), Path("logs") / "units", Path("units"))
)
def test_campaign_layout_rejects_symbolic_output_directories(
    tmp_path: Path, relative: Path
) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = output / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic path component"):
        queue._prepare_output_layout(output, create=True)


def test_campaign_write_rejects_destination_outside_result_root(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    queue._prepare_output_layout(output, create=True)
    escaped = tmp_path / "escaped.json"
    with pytest.raises(RuntimeError, match="escapes the result root"):
        queue._write_or_match(output, escaped, {"status": "PASS"}, "test seal")
    assert not escaped.exists()


def test_exact_build_args_lock_learning_rates_partial_batches_and_audit_paths(
    tmp_path: Path,
) -> None:
    pmnist_lock = load_pmnist_lock(PMNIST_LOCK)
    cifar_lock = load_cifar_lock(CIFAR_LOCK)
    pmnist_exact = _one(enumerate_pmnist_units(pmnist_lock), "exact_current_coefficient")
    cifar_exact = _one(enumerate_cifar_units(cifar_lock), "exact_current_coefficient")

    pmnist_args = effective_args(pmnist_exact, pmnist_lock, "pmnist")
    cifar_args = effective_args(cifar_exact, cifar_lock, "cifar")
    assert pmnist_args["lr"] == 0.1
    assert cifar_args["lr"] == 0.03
    assert pmnist_args["drop_last"] == cifar_args["drop_last"] == 0
    assert pmnist_args["reference_buffer_size"] == 500
    assert pmnist_args["reference_minibatch_size"] == 32
    assert cifar_args["reference_buffer_size"] == 500
    assert cifar_args["reference_minibatch_size"] == 32
    for arguments in (pmnist_args, cifar_args):
        assert "buffer_size" not in arguments
        assert "minibatch_size" not in arguments
        assert "assignment_audit_path" not in arguments

    command, built = build_command(
        cifar_exact,
        cifar_lock,
        "cifar",
        python=Path("/python"),
        mammoth_root=Path("/mammoth"),
        data_root=Path("/data"),
        results=tmp_path / "results",
        assignment_audit=tmp_path / "assignment.jsonl",
        exact_coefficient_audit=tmp_path / "exact.jsonl",
        bootstrap_audit=tmp_path / "bootstrap.json",
        contract=tmp_path / "contract.json",
        bank_or_order_manifest=tmp_path / "order.json",
    )
    assert command.count("--exact_coefficient_audit_path") == 1
    assert command.count("--drop_last") == 1
    assert built["exact_coefficient_audit_path"] == str(tmp_path / "exact.jsonl")
    assert built["drop_last"] == 0


def test_every_arm_explicitly_keeps_partial_final_minibatches() -> None:
    pmnist_lock = load_pmnist_lock(PMNIST_LOCK)
    cifar_lock = load_cifar_lock(CIFAR_LOCK)
    assert {
        effective_args(unit, pmnist_lock, "pmnist")["drop_last"]
        for unit in enumerate_pmnist_units(pmnist_lock)
    } == {0}
    assert {
        effective_args(unit, cifar_lock, "cifar")["drop_last"]
        for unit in enumerate_cifar_units(cifar_lock)
    } == {0}


def test_assignment_audit_contract_includes_explicit_drop_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    unit = _one(enumerate_pmnist_units(lock), "er_paired")
    arguments = effective_args(unit, lock, "pmnist")
    audit = tmp_path / "assignment.jsonl"

    def checked(adapter, path, expected):
        assert adapter.seed == unit.seed
        assert path == audit
        assert expected["drop_last"] == 0
        return {"record_count": 1}

    monkeypatch.setattr(exact_unit_runner, "summarize_audit", checked)
    assert exact_unit_runner._assignment_summary(unit, audit, arguments) == {
        "record_count": 1
    }


def test_real_assignment_audit_producer_emits_explicit_drop_last(
    tmp_path: Path,
) -> None:
    path = (
        ROOT
        / "assignment_strength_overlay"
        / "models"
        / "utils"
        / "assignment_strength_intervention.py"
    )
    module_name = "exact_reference_drop_last_audit_regression"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)

    audit = tmp_path / "assignment.jsonl"
    module.AssignmentStrengthIntervention(
        SimpleNamespace(
            assignment_benchmark="permuted_mnist",
            assignment_run_seed=4101,
            assignment_lambda_bp=10000,
            assignment_permutation_seed=104104,
            assignment_replica=0,
            assignment_selection_seed_salt=2779096485,
            assignment_audit_path=str(audit),
            assignment_audit_interval=100,
            assignment_audit_permutations=4,
            dataset="perm-mnist",
            model="er_assignment_strength",
            seed=4101,
            drop_last=False,
        ),
        "er_assignment_strength",
    )
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    effective = next(
        row["effective_args"]
        for row in rows
        if row["schema_version"]
        == "gradient_reuse.assignment_strength.mammoth_effective_args.v1"
    )
    assert effective["drop_last"] is False


def test_bootstrap_accepts_mammoth_boolean_for_locked_drop_last_zero() -> None:
    expected = {
        "model": "sgd_exact_current_coefficient",
        "drop_last": 0,
    }
    parsed = SimpleNamespace(
        model="sgd-exact-current-coefficient",
        drop_last=False,
        custom_task_order=None,
        noise_rate=0.0,
        validation=None,
    )
    validate_parsed_args(parsed, {"expected_args": expected})
    parsed.drop_last = True
    with pytest.raises(RuntimeError, match="drop_last"):
        validate_parsed_args(parsed, {"expected_args": expected})


@dataclass(frozen=True)
class _FakeUnit:
    unit_id: str

    def as_dict(self) -> dict[str, str]:
        return {"unit_id": self.unit_id}


def _write_complete_campaign(
    output: Path,
    *,
    binding: dict[str, object],
    aggregate: dict[str, object],
    unit: _FakeUnit,
) -> None:
    output.mkdir()
    (output / "units").mkdir()
    atomic_json(output / "CAMPAIGN.json", binding)
    atomic_json(output / "AGGREGATE.json", aggregate)
    atomic_json(
        output / "DONE.json",
        {
            "schema_version": queue.DONE_SCHEMA,
            "status": "PASS",
            "campaign": "pmnist",
            "completed_units": 1,
            "required_units": 1,
            "aggregate_sha256": sha256_file(output / "AGGREGATE.json"),
            "unit_ids_sha256": queue.canonical_sha256([unit.unit_id]),
            "campaign_binding_sha256": sha256_file(output / "CAMPAIGN.json"),
        },
    )


def test_check_only_completion_chain_binds_campaign_aggregate_and_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit = _FakeUnit("unit-0")
    binding: dict[str, object] = {
        "schema_version": queue.BINDING_SCHEMA,
        "status": "LOCKED",
        "campaign": "pmnist",
        "expected_units": 1,
        "unit_ids_sha256": queue.canonical_sha256([unit.unit_id]),
    }
    aggregate: dict[str, object] = {
        "schema_version": "gradient_reuse.exact_reference.aggregate.v1",
        "status": "PASS",
        "campaign": "pmnist",
        "unit_count": 1,
    }
    output = tmp_path / "results"
    _write_complete_campaign(
        output, binding=binding, aggregate=aggregate, unit=unit
    )
    seen: list[str] = []

    def validate(root, observed_unit, **kwargs):
        del root, kwargs
        seen.append(observed_unit.unit_id)
        return {"campaign": "pmnist", "unit": observed_unit.as_dict()}

    monkeypatch.setattr(queue, "validate_completed_unit", validate)
    monkeypatch.setattr(queue, "aggregate_pmnist", lambda records, lock: aggregate)
    monkeypatch.setattr(
        queue,
        "aggregate_cifar",
        lambda records, lock: (_ for _ in ()).throw(AssertionError("wrong campaign")),
    )
    result = queue.validate_campaign_completion(
        output,
        campaign="pmnist",
        lock={},
        units=[unit],
        expected=1,
        binding=binding,
    )
    assert result["status"] == "PASS"
    assert seen == [unit.unit_id]

    changed = dict(result, campaign_binding_sha256="0" * 64)
    atomic_json(output / "DONE.json", changed)
    with pytest.raises(RuntimeError, match="completion seal"):
        queue.validate_campaign_completion(
            output,
            campaign="pmnist",
            lock={},
            units=[unit],
            expected=1,
            binding=binding,
        )


def test_check_only_cli_never_dispatches_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        queue,
        "check_completed",
        lambda args: called.append(args.campaign) or {"status": "PASS"},
    )
    monkeypatch.setattr(
        queue,
        "execute",
        lambda args: (_ for _ in ()).throw(AssertionError("training dispatched")),
    )
    assert queue.main(
        ["--campaign", "pmnist", "--lock", str(PMNIST_LOCK), "--check-only"]
    ) == 0
    assert called == ["pmnist"]


def test_unit_done_seal_must_name_the_exact_unit(tmp_path: Path) -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    unit = enumerate_pmnist_units(lock)[0]
    root = tmp_path / unit.unit_id
    root.mkdir()
    atomic_json(
        root / "UNIT.json",
        {
            "schema_version": UNIT_SCHEMA,
            "status": "PASS",
            "campaign": "pmnist",
            "unit": unit.as_dict(),
            "lock_content_sha256": canonical_sha256(lock),
            "task_bank_sha256": expected_task_bank_digest(lock, unit.seed),
            "order_sha256": None,
        },
    )
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    atomic_json(
        root / "DONE.json",
        {
            "schema_version": UNIT_DONE_SCHEMA,
            "status": "PASS",
            "unit_id": "another-unit",
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )
    with pytest.raises(RuntimeError, match="sealed PASS"):
        validate_completed_unit(
            root,
            unit,
            campaign="pmnist",
            lock=lock,
        )


def _seal_unit(root: Path, unit_id: str, record: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "UNIT.json", record)
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    atomic_json(
        root / "DONE.json",
        {
            "schema_version": UNIT_DONE_SCHEMA,
            "status": "PASS",
            "unit_id": unit_id,
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )


@pytest.mark.parametrize(
    ("campaign", "field", "expected_updates", "expected_examples"),
    (
        ("pmnist", "updates", 9380, 1_200_000),
        ("pmnist", "seen_examples", 9380, 1_200_000),
        ("cifar", "updates", 1580, 50_000),
        ("cifar", "seen_examples", 1580, 50_000),
    ),
)
def test_completed_exact_unit_requires_campaign_locked_audit_counts(
    tmp_path: Path,
    campaign: str,
    field: str,
    expected_updates: int,
    expected_examples: int,
) -> None:
    if campaign == "pmnist":
        lock = load_pmnist_lock(PMNIST_LOCK)
        unit = _one(enumerate_pmnist_units(lock), "exact_current_coefficient")
        task_digest = expected_task_bank_digest(lock, unit.seed)
        order_digest = None
    else:
        lock = load_cifar_lock(CIFAR_LOCK)
        unit = _one(enumerate_cifar_units(lock), "exact_current_coefficient")
        task_digest = None
        order_digest = order_manifest(lock, unit)["order_sha256"]
    record: dict[str, object] = {
        "schema_version": UNIT_SCHEMA,
        "status": "PASS",
        "campaign": campaign,
        "unit": unit.as_dict(),
        "lock_content_sha256": canonical_sha256(lock),
        "task_bank_sha256": task_digest,
        "order_sha256": order_digest,
        "exact_coefficient_diagnostics": {
            "schema_version": "gradient_reuse.exact_current_coefficient.summary.v1",
            "status": "PASS",
            "updates": expected_updates,
            "seen_examples": expected_examples,
        },
    }
    root = tmp_path / "units" / unit.unit_id
    _seal_unit(root, unit.unit_id, record)
    assert validate_completed_unit(
        root, unit, campaign=campaign, lock=lock
    )["status"] == "PASS"

    diagnostics = record["exact_coefficient_diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics[field] = int(diagnostics[field]) - 1
    _seal_unit(root, unit.unit_id, record)
    with pytest.raises(RuntimeError, match="counts differ from the campaign lock"):
        validate_completed_unit(root, unit, campaign=campaign, lock=lock)


def test_completed_unit_rejects_symbolic_units_parent(tmp_path: Path) -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    unit = enumerate_pmnist_units(lock)[0]
    outside_units = tmp_path / "outside-units"
    record: dict[str, object] = {
        "schema_version": UNIT_SCHEMA,
        "status": "PASS",
        "campaign": "pmnist",
        "unit": unit.as_dict(),
        "lock_content_sha256": canonical_sha256(lock),
        "task_bank_sha256": expected_task_bank_digest(lock, unit.seed),
        "order_sha256": None,
    }
    _seal_unit(outside_units / unit.unit_id, unit.unit_id, record)
    output = tmp_path / "campaign"
    output.mkdir()
    (output / "units").symlink_to(outside_units, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic path component"):
        validate_completed_unit(
            output / "units" / unit.unit_id,
            unit,
            campaign="pmnist",
            lock=lock,
        )
