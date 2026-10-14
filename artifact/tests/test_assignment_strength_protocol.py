"""Tests for the locked assignment-strength protocol and unit runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_assignment_strength.protocol import (
    EXPECTED_UNITS,
    LAMBDA_BASIS_POINTS,
    RUN_SEEDS,
    enumerate_units,
    load_lock,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    validate_lock,
)
from experiments.gradient_reuse_assignment_strength.unit_runner import (
    _validate_logged_effective_args,
    build_command,
    effective_args,
    summarize_audit,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/assignment_strength_sweep.json"


def test_locked_panel_has_432_unique_units_and_exact_per_seed_shape() -> None:
    lock = load_lock(LOCK_PATH)
    units = enumerate_units(lock)
    assert len(units) == EXPECTED_UNITS == 432
    assert len({unit.unit_id for unit in units}) == EXPECTED_UNITS
    for benchmark in ("split_cifar100", "permuted_mnist"):
        for seed in RUN_SEEDS:
            panel = [
                unit
                for unit in units
                if unit.benchmark == benchmark and unit.seed == seed
            ]
            assert len(panel) == 18
            assert sum(unit.method == "reference" for unit in panel) == 1
            assert sum(unit.lambda_bp == 0 for unit in panel) == 1
            for strength in LAMBDA_BASIS_POINTS[1:]:
                replicas = sorted(
                    unit.permutation_replica
                    for unit in panel
                    if unit.lambda_bp == strength
                )
                assert replicas == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["runtime"].__setitem__("run_seeds", list(range(1, 13))),
        lambda value: value["runtime"].__setitem__(
            "lambda_basis_points", [0, 5000, 10000]
        ),
        lambda value: value["inference"].__setitem__("no_optional_stopping", False),
        lambda value: value["interpretation"].__setitem__(
            "no_causal_mechanism_decomposition_claim", False
        ),
    ),
)
def test_lock_rejects_scientifically_material_mutations(mutation) -> None:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    mutation(value)
    with pytest.raises(RuntimeError):
        validate_lock(value)


def test_parser_accepts_only_exact_cuda_device_repr(tmp_path: Path) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text(
        "{'device': device(type='cuda', index=0), "
        "'accuracy_1_task1': 50.0, 'accmean_task1': 50.0}\n",
        encoding="utf-8",
    )
    assert parse_mammoth_log(path)["device"] == "cuda:0"
    for expression in (
        "device(type='cuda', index=1)",
        "device(type='cpu', index=0)",
        "device(index=0, type='cuda')",
        "torch.device(type='cuda', index=0)",
    ):
        path.write_text("{'device': " + expression + "}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported expression"):
            parse_mammoth_log(path)


def test_metric_reconstruction_checks_reported_means_and_bwt() -> None:
    payload = {
        "accuracy_1_task1": 80.0,
        "accmean_task1": 80.0,
        "accuracy_1_task2": 70.0,
        "accuracy_2_task2": 90.0,
        "accmean_task2": 80.0,
        "backward_transfer": -10.0,
    }
    metrics = metrics_from_mammoth_log(payload)
    assert metrics["final_average_accuracy"] == 80.0
    assert metrics["backward_transfer"] == -10.0
    bad = copy.deepcopy(payload)
    bad["accmean_task2"] = 81.0
    with pytest.raises(RuntimeError, match="task-mean disagreement"):
        metrics_from_mammoth_log(bad)


def test_command_has_unique_well_formed_flags() -> None:
    lock = load_lock(LOCK_PATH)
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.benchmark == "split_cifar100"
        and unit.seed == 3501
        and unit.lambda_bp == 2500
        and unit.permutation_replica == 0
    )
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
    tail = command[4:]
    assert len(tail) % 2 == 0
    flags = tail[0::2]
    assert all(flag.startswith("--") for flag in flags)
    assert len(flags) == len(set(flags))
    assert "--assignment_lambda_bp" in flags


def test_logged_effective_args_authenticate_no_replay_reference() -> None:
    lock = load_lock(LOCK_PATH)
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.benchmark == "split_cifar100"
        and unit.seed == 3501
        and unit.method == "reference"
    )
    expected = effective_args(unit, lock, "cuda:0")
    payload = {
        **expected,
        "enable_other_metrics": True,
        "non_verbose": True,
        "base_path": "/data",
        "results_path": "/results",
        "debug_mode": False,
        "assignment_audit_path": None,
    }
    normalized = _validate_logged_effective_args(
        payload,
        expected,
        unit=unit,
        data_root=Path("/data"),
        results=Path("/results"),
        audit=Path("/unused.jsonl"),
    )
    assert normalized["model"] == "sgd"
    assert normalized["enable_other_metrics"] is True

    changed = dict(payload)
    changed["lr"] = float(expected["lr"]) * 2.0
    with pytest.raises(RuntimeError, match="effective argument changed"):
        _validate_logged_effective_args(
            changed,
            expected,
            unit=unit,
            data_root=Path("/data"),
            results=Path("/results"),
            audit=Path("/unused.jsonl"),
        )


def test_logged_effective_args_authenticate_replay_and_optional_config() -> None:
    lock = load_lock(LOCK_PATH)
    replay = next(
        unit
        for unit in enumerate_units(lock)
        if unit.benchmark == "permuted_mnist"
        and unit.seed == 3501
        and unit.lambda_bp == 10000
        and unit.permutation_replica == 0
    )
    expected = effective_args(replay, lock, "cuda:0")
    payload = {
        **expected,
        "model": "er-assignment-strength",
        "enable_other_metrics": True,
        "non_verbose": True,
        "base_path": "/data",
        "results_path": "/results",
        "debug_mode": False,
        "assignment_audit_path": "/audit.jsonl",
        "dataset_config": None,
    }
    normalized = _validate_logged_effective_args(
        payload,
        expected,
        unit=replay,
        data_root=Path("/data"),
        results=Path("/results"),
        audit=Path("/audit.jsonl"),
    )
    assert normalized["model"] == "er_assignment_strength"
    assert normalized["assignment_lambda_bp"] == 10000

    changed = dict(payload)
    changed["dataset_config"] = "20tasks"
    with pytest.raises(RuntimeError, match="unexpected dataset configuration"):
        _validate_logged_effective_args(
            changed,
            expected,
            unit=replay,
            data_root=Path("/data"),
            results=Path("/results"),
            audit=Path("/audit.jsonl"),
        )


def _canonical_audit_case(
    tmp_path: Path, *, lambda_bp: int = 5000
) -> tuple[object, dict, list[dict], Path]:
    lock = load_lock(LOCK_PATH)
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.benchmark == "permuted_mnist"
        and unit.seed == 3501
        and unit.lambda_bp == lambda_bp
        and unit.permutation_replica == 0
    )
    arguments = effective_args(unit, lock, "cuda:0")
    effective = dict(arguments)
    effective["runtime_patch_domain_il_fullacc"] = "v1"
    effective["debug_mode"] = False
    common = {
        "method": "er_assignment_strength",
        "legacy_counter_method": "er_pairing",
        "benchmark": "permuted_mnist",
        "dataset": "perm-mnist",
        "run_seed": 3501,
        "permutation_seed": 103504,
        "selection_seed_salt": 2779096485,
        "replica": 0,
        "lambda_bp": lambda_bp,
    }
    replay_rows = 32
    selected_rows = {
        0: 0,
        2500: 9,
        5000: 17,
        7500: 24,
        10000: 32,
    }[lambda_bp]
    selected_fraction = selected_rows / replay_rows
    realized_lambda = (
        0.0 if selected_rows == 0 else (selected_rows - 1) / (replay_rows - 1)
    )
    displaced_rows = 0 if selected_rows == 0 else max(0, selected_rows - 2)
    changed_rows = 0 if selected_rows == 0 else max(0, displaced_rows - 2)
    original_target_sha256 = "c" * 64
    used_target_sha256 = (
        original_target_sha256 if changed_rows == 0 else "d" * 64
    )
    permutation_sha256 = "e" * 64
    legacy_permutation_sha256 = (
        permutation_sha256 if lambda_bp == 10000 else "9" * 64
    )
    rows = [
        {
            **common,
            "schema_version": "gradient_reuse.assignment_strength.mammoth_effective_args.v1",
            "effective_args": effective,
        },
        {
            **common,
            "schema_version": "gradient_reuse.assignment_strength.mammoth_gradient_diagnostic.v1",
            "observation": 0,
            "event": 0,
            "channel": "replay_labels",
            "rows": replay_rows,
            "assignment_draws": 4,
            "mean_selected_fraction": selected_fraction,
            "mean_realized_lambda": realized_lambda,
            "mean_index_displaced_fraction": displaced_rows / replay_rows,
            "mean_target_value_changed_fraction": changed_rows / replay_rows,
            "true_gradient_norm": 1.0,
            "mean_strength_gradient_norm": 0.8,
            "assignment_component_norm": 0.3,
            "finite_assignment_variance": 0.2,
            "true_mean_strength_cosine": 0.5,
            "sampled_indexes_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "target_sha256": original_target_sha256,
            "target_multiset_sha256": "4" * 64,
        },
        {
            **common,
            "schema_version": "gradient_reuse.assignment_strength.mammoth_intervention.v1",
            "observation": 0,
            "event": 0,
            "channel": "replay_labels",
            "rows": replay_rows,
            "target_multiset_preserved": True,
            "counter_key_compatibility": "legacy_er_pairing_v1",
            "selected_rows": selected_rows,
            "selected_fraction": selected_fraction,
            "realized_lambda": realized_lambda,
            "index_displaced_rows": displaced_rows,
            "index_displaced_fraction": displaced_rows / replay_rows,
            "target_value_changed_rows": changed_rows,
            "target_value_changed_fraction": changed_rows / replay_rows,
            "sampled_indexes_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "original_target_sha256": original_target_sha256,
            "used_target_sha256": used_target_sha256,
            "target_multiset_sha256": "4" * 64,
            "permutation_sha256": permutation_sha256,
            "selected_support_sha256": "f" * 64,
            "support_order_sha256": "1" * 64,
            "legacy_full_permutation_sha256": legacy_permutation_sha256,
        },
    ]
    path = tmp_path / "audit.jsonl"
    return unit, arguments, rows, path


def _write_audit(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _intervention_row(rows: list[dict]) -> dict:
    return next(
        row
        for row in rows
        if row["schema_version"]
        == "gradient_reuse.assignment_strength.mammoth_intervention.v1"
    )


def test_audit_summary_accepts_canonical_overlay_records(tmp_path: Path) -> None:
    unit, arguments, rows, path = _canonical_audit_case(tmp_path)
    _write_audit(path, rows)
    summary = summarize_audit(unit, path, arguments)
    assert summary["intervention_records"] == 1
    assert summary["gradient_diagnostic_records"] == 1
    assert summary["minimum_replay_batch_rows"] == 32
    assert summary["maximum_replay_batch_rows"] == 32
    assert summary["means"]["target_value_changed_fraction"] == 13 / 32


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("selected_rows", 16, "selected-row count"),
        ("selected_fraction", 0.5, "exact reconstruction"),
        ("realized_lambda", 0.5, "exact reconstruction"),
        ("index_displaced_fraction", 0.25, "exact reconstruction"),
        ("target_value_changed_fraction", 0.25, "exact reconstruction"),
    ),
)
def test_audit_summary_rejects_tampered_counts_and_rates(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    unit, arguments, rows, path = _canonical_audit_case(tmp_path)
    _intervention_row(rows)[field] = value
    _write_audit(path, rows)
    with pytest.raises(RuntimeError, match=match):
        summarize_audit(unit, path, arguments)


@pytest.mark.parametrize(
    "field",
    (
        "sampled_indexes_sha256",
        "input_sha256",
        "original_target_sha256",
        "used_target_sha256",
        "target_multiset_sha256",
        "permutation_sha256",
        "selected_support_sha256",
        "support_order_sha256",
        "legacy_full_permutation_sha256",
    ),
)
def test_audit_summary_rejects_tampered_intervention_digests(
    tmp_path: Path, field: str
) -> None:
    unit, arguments, rows, path = _canonical_audit_case(tmp_path)
    _intervention_row(rows)[field] = "not-a-sha256"
    _write_audit(path, rows)
    with pytest.raises(RuntimeError, match="invalid assignment audit SHA-256"):
        summarize_audit(unit, path, arguments)


def test_full_strength_audit_requires_legacy_permutation_hash_equivalence(
    tmp_path: Path,
) -> None:
    unit, arguments, rows, path = _canonical_audit_case(
        tmp_path, lambda_bp=10000
    )
    _write_audit(path, rows)
    summarize_audit(unit, path, arguments)

    _intervention_row(rows)["legacy_full_permutation_sha256"] = "8" * 64
    _write_audit(path, rows)
    with pytest.raises(RuntimeError, match="differs from the legacy full permutation"):
        summarize_audit(unit, path, arguments)
