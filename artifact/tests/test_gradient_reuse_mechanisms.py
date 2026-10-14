"""Protocol-level tests for the ICLR continual-learning mechanism audit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from experiments.gradient_reuse_mechanisms.data_manifest import (
    build_data_manifest,
    write_manifest,
)
from experiments.gradient_reuse_mechanisms.protocol import (
    MechanismUnit,
    accuracy_matrix,
    compute_diagnostic_estimands,
    compute_mechanism_estimands,
    enumerate_units,
    load_and_validate_lock,
    mean_and_ci,
    metrics_from_mammoth_log,
    method_estimand_class,
    parse_mammoth_log,
    canonical_sha256,
)
from experiments.gradient_reuse_mechanisms.run import (
    _artifact_inventory,
    _diagnostic_summary,
    _validate_completed_unit,
    _validate_cross_arm_comparability,
    build_mammoth_command,
    effective_mammoth_args,
    validate_data_manifest,
    build_environment_manifest,
    validate_environment_manifest,
)
import experiments.gradient_reuse_mechanisms.run as mechanism_run


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "mechanisms.json"


@pytest.fixture(scope="module")
def lock() -> dict:
    return load_and_validate_lock(LOCK_PATH)


def test_overlay_preparation_is_serialized_across_processes(tmp_path: Path) -> None:
    mammoth_root = tmp_path / "mammoth"
    (mammoth_root / ".git").mkdir(parents=True)
    coordination = tmp_path / "coordination"
    coordination.mkdir()
    script = r"""
import sys
import time
from pathlib import Path

import experiments.gradient_reuse_mechanisms.overlay as overlay

role = sys.argv[1]
root = Path(sys.argv[2])
coordination = Path(sys.argv[3])
partial = coordination / "partial"
complete = coordination / "complete"
payload = {"transaction": "complete"}

def wait_for(path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {path.name}")
        time.sleep(0.01)

def verify_upstream(_root, *, allow_installed_overlay=True):
    if partial.exists() and not complete.exists():
        raise RuntimeError("observed a partially installed overlay")

def install_overlay(_root):
    if complete.exists():
        return payload
    if role != "first":
        raise RuntimeError("second process entered installation before the first finished")
    partial.write_text("partial", encoding="utf-8")
    wait_for(coordination / "release_first")
    complete.write_text("complete", encoding="utf-8")
    partial.unlink()
    return payload

def verify_installed_overlay(_root):
    if not complete.exists() or partial.exists():
        raise RuntimeError("overlay is not complete")
    return payload

overlay.verify_upstream = verify_upstream
overlay.install_overlay = install_overlay
overlay.verify_installed_overlay = verify_installed_overlay
(coordination / f"{role}_started").write_text("started", encoding="utf-8")
overlay.prepare_overlay(root)
"""
    environment = dict(os.environ)
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT) if not pythonpath else os.pathsep.join((str(ROOT), pythonpath))
    )

    def wait_for(path: Path, *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out waiting for {path}")
            time.sleep(0.01)

    first = subprocess.Popen(
        [sys.executable, "-c", script, "first", str(mammoth_root), str(coordination)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    second = None
    try:
        wait_for(coordination / "partial")
        second = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                "second",
                str(mammoth_root),
                str(coordination),
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for(coordination / "second_started")
        time.sleep(0.25)
        assert second.poll() is None, second.communicate()[0]
        (coordination / "release_first").write_text("release", encoding="utf-8")
        first_output, _ = first.communicate(timeout=5)
        second_output, _ = second.communicate(timeout=5)
        assert first.returncode == 0, first_output
        assert second.returncode == 0, second_output
    finally:
        (coordination / "release_first").touch()
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_lock_enumerates_the_complete_nested_design(lock: dict) -> None:
    units = enumerate_units(lock)
    # Per benchmark/seed: two matched reference controls plus six methods with
    # one paired and four nested target-permutation replicas.
    assert len(units) == 3 * 5 * (2 + 6 * 5) == 480
    assert len({unit.unit_id for unit in units}) == len(units)
    naive = [unit for unit in units if unit.method == "naive_sgd"]
    assert len(naive) == 15
    assert all(unit.intervention == "none" for unit in naive)
    mer_controls = [unit for unit in units if unit.method == "mer_current_only"]
    assert len(mer_controls) == 15
    assert all(unit.intervention == "none" for unit in mer_controls)
    permuted = [unit for unit in units if unit.intervention == "permuted"]
    assert {unit.permutation_replica for unit in permuted} == {0, 1, 2, 3}


def test_mammoth_command_has_no_duplicate_flags_and_keeps_current_labels_outside_intervention(
    lock: dict, tmp_path: Path
) -> None:
    unit = next(
        value
        for value in enumerate_units(lock)
        if value.benchmark == "split_cifar100"
        and value.method == "experience_replay"
        and value.intervention == "permuted"
        and value.seed == 3401
        and value.permutation_replica == 2
    )
    command = build_mammoth_command(
        unit,
        lock,
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        audit_path=tmp_path / "audit.jsonl",
        device="cpu",
        smoke=True,
    )
    assert Path(command[1]) == tmp_path / "mammoth" / "main.py"
    flags = [value for value in command if value.startswith("--")]
    assert max(Counter(flags).values()) == 1
    assert command[command.index("--pairing_intervention") + 1] == "permuted"
    assert command[command.index("--pairing_replica") + 1] == "2"
    assert command[command.index("--pairing_audit_interval") + 1] == "1"
    assert command[command.index("--batch_size") + 1] == "32"
    assert command[command.index("--minibatch_size") + 1] == "32"
    assert command[command.index("--pairing_benchmark") + 1] == "split_cifar100"
    assert command[command.index("--pairing_run_seed") + 1] == "3401"


def test_naive_command_is_invariant_to_the_replay_target_protocol(
    lock: dict, tmp_path: Path
) -> None:
    unit = next(value for value in enumerate_units(lock) if value.method == "naive_sgd")
    command = build_mammoth_command(
        unit,
        lock,
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        audit_path=tmp_path / "audit.jsonl",
        device="cpu",
        smoke=False,
    )
    assert not any("pairing_" in value for value in command)


def test_mammoth_subprocess_environment_disables_bytecode_writes(
    lock: dict, tmp_path: Path
) -> None:
    unit = next(value for value in enumerate_units(lock) if value.method == "naive_sgd")
    mammoth_root = tmp_path / "mammoth"
    environment = mechanism_run._mammoth_subprocess_environment(
        unit,
        lock,
        mammoth_root=mammoth_root,
    )
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPATH"] == str(mammoth_root)
    assert environment["GRADIENT_REUSE_DETERMINISTIC"] == "1"


def test_mer_replay_batch_size_survives_mammoths_stream_batch_override(
    lock: dict, tmp_path: Path
) -> None:
    unit = next(
        value
        for value in enumerate_units(lock)
        if value.method == "mer" and value.intervention == "paired"
    )
    command = build_mammoth_command(
        unit,
        lock,
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        audit_path=tmp_path / "audit.jsonl",
        device="cpu",
        smoke=True,
    )
    expected = str(lock["methods"]["mer"]["minibatch_size"])
    assert command[command.index("--mer_replay_batch_size") + 1] == expected


def test_mer_current_only_control_matches_mer_schedule_without_replay(
    lock: dict, tmp_path: Path
) -> None:
    units = enumerate_units(lock)
    mer = next(
        value
        for value in units
        if value.benchmark == "split_cifar100"
        and value.method == "mer"
        and value.intervention == "paired"
    )
    control = next(
        value
        for value in units
        if value.benchmark == "split_cifar100"
        and value.method == "mer_current_only"
    )
    mer_args = effective_mammoth_args(mer, lock, device="cuda:0", smoke=False)
    control_args = effective_mammoth_args(
        control, lock, device="cuda:0", smoke=False
    )
    assert mer_args["batch_size"] == control_args["batch_size"] == 1
    for name in ("lr", "beta", "gamma", "batch_num"):
        assert mer_args[name] == control_args[name]
    assert "buffer_size" not in control_args
    assert "minibatch_size" not in control_args
    assert "mer_replay_batch_size" not in control_args
    command = build_mammoth_command(
        control,
        lock,
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        audit_path=tmp_path / "audit.jsonl",
        device="cuda:0",
        smoke=False,
    )
    assert not any("pairing_" in value for value in command)


def test_parser_accepts_numpy_scalar_and_array_repr_without_eval(tmp_path: Path) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text(
        "{'accmean_task1': np.float64(50.0), "
        "'accuracy_1_task1': np.float32(50.0), "
        "'cpu_memory_usage': array([1.0, 2.0])}\n",
        encoding="utf-8",
    )
    payload = parse_mammoth_log(path)
    assert payload["accmean_task1"] == 50.0
    assert payload["cpu_memory_usage"] == [1.0, 2.0]


def test_accuracy_matrix_and_bwt_are_reconstructed_in_percentage_points() -> None:
    payload = {
        "accmean_task1": 50.0,
        "accmean_task2": 52.5,
        "accmean_task3": 55.0,
        "accuracy_1_task1": 50.0,
        "accuracy_1_task2": 45.0,
        "accuracy_2_task2": 60.0,
        "accuracy_1_task3": 40.0,
        "accuracy_2_task3": 55.0,
        "accuracy_3_task3": 70.0,
        "backward_transfer": -7.5,
    }
    assert accuracy_matrix(payload) == [[50.0], [45.0, 60.0], [40.0, 55.0, 70.0]]
    metrics = metrics_from_mammoth_log(payload)
    assert metrics["final_average_accuracy"] == 55.0
    assert metrics["average_incremental_accuracy"] == 52.5
    assert metrics["backward_transfer"] == -7.5
    assert metrics["accuracy_unit"] == "percentage_points"

    inconsistent = dict(payload)
    inconsistent["accmean_task2"] = 99.0
    with pytest.raises(RuntimeError, match="task-mean disagreement"):
        metrics_from_mammoth_log(inconsistent)


def _unit_record(
    *,
    method: str,
    intervention: str,
    value: float,
    replica: int | None = None,
) -> dict:
    return {
        "benchmark": "split_cifar100",
        "method": method,
        "estimand_class": method_estimand_class(method),
        "intervention": intervention,
        "seed": 3401,
        "permutation_replica": replica,
        "metrics": {
            "final_average_accuracy": value,
            "backward_transfer": value / 10.0,
            "average_incremental_accuracy": value + 1.0,
        },
    }


def test_mechanism_identity_uses_four_nested_replicas() -> None:
    records = [
        _unit_record(method="naive_sgd", intervention="none", value=50.0),
        _unit_record(method="gem", intervention="paired", value=54.0),
        *[
            _unit_record(
                method="gem",
                intervention="permuted",
                value=value,
                replica=replica,
            )
            for replica, value in enumerate([51.0, 52.0, 53.0, 54.0])
        ],
    ]
    row = compute_mechanism_estimands(records)[0]
    assert row["permuted_mean_accuracy"] == 52.5
    assert row["V_total"] == 4.0
    assert row["V_perm"] == 2.5
    assert row["V_pair"] == 1.5
    assert row["identity_residual"] == 0.0
    assert row["bwt_V_total"] == pytest.approx(0.4)
    assert row["V_perm_mc_variance"] > 0.0


def test_mer_estimands_use_the_matched_current_only_reference() -> None:
    records = [
        _unit_record(method="naive_sgd", intervention="none", value=40.0),
        _unit_record(method="mer_current_only", intervention="none", value=50.0),
        _unit_record(method="mer", intervention="paired", value=54.0),
        *[
            _unit_record(
                method="mer",
                intervention="permuted",
                value=value,
                replica=replica,
            )
            for replica, value in enumerate([50.0, 51.0, 52.0, 53.0])
        ],
    ]
    row = compute_mechanism_estimands(records)[0]
    assert row["reference_method"] == "mer_current_only"
    assert row["V_total"] == 4.0
    assert row["V_perm"] == 1.5
    assert row["V_pair"] == 2.5


def test_diagnostics_are_contrasted_with_nested_permutation_replicas() -> None:
    def record(intervention: str, value: float, replica: int | None = None) -> dict:
        row = _unit_record(
            method="gem", intervention=intervention, value=50.0, replica=replica
        )
        row["diagnostics"] = {
            "mean_pairing_component_norm": value,
            "mean_finite_permutation_variance": value / 10.0,
            "mean_true_permuted_cosine": value / 20.0,
            "mean_fixed_point_fraction": 1.0 if intervention == "paired" else 0.25,
            "projection_activation_rate": value / 100.0,
            "mean_pre_projection_constraint": -value,
            "mean_post_projection_constraint": value,
        }
        return row

    rows = compute_diagnostic_estimands(
        [
            record("paired", 4.0),
            *[
                record("permuted", value, replica)
                for replica, value in enumerate([1.0, 2.0, 3.0, 4.0])
            ],
        ]
    )
    diagnostic = rows[0]["mean_pairing_component_norm"]
    assert diagnostic["paired"] == 4.0
    assert diagnostic["permuted_mean"] == 2.5
    assert diagnostic["paired_minus_permuted"] == 1.5
    assert diagnostic["permuted_mean_mc_variance"] > 0.0


def test_replica_ids_are_taken_from_the_locked_design_not_hardcoded() -> None:
    records = [
        _unit_record(method="naive_sgd", intervention="none", value=50.0),
        _unit_record(method="gem", intervention="paired", value=54.0),
        _unit_record(
            method="gem", intervention="permuted", value=51.0, replica=2
        ),
        _unit_record(
            method="gem", intervention="permuted", value=53.0, replica=7
        ),
    ]
    row = compute_mechanism_estimands(
        records, permutation_replica_ids=[2, 7]
    )[0]
    assert row["permutation_replicas"] == [2, 7]
    assert row["permuted_mean_accuracy"] == 52.0


def test_incomplete_permutation_monte_carlo_is_rejected() -> None:
    records = [
        _unit_record(method="naive_sgd", intervention="none", value=50.0),
        _unit_record(method="gem", intervention="paired", value=54.0),
        _unit_record(method="gem", intervention="permuted", value=52.0, replica=0),
    ]
    with pytest.raises(RuntimeError, match="incomplete nested permutation"):
        compute_mechanism_estimands(records)


def test_seed_interval_uses_student_t() -> None:
    summary = mean_and_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == 3.0
    assert summary["interval"] == "two_sided_student_t_over_run_seeds"
    assert summary["ci95_low"] < 3.0 < summary["ci95_high"]


def test_seed_interval_reports_nested_permutation_mc_uncertainty() -> None:
    summary = mean_and_ci(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        nested_mc_variances=[0.25] * 5,
    )
    assert summary["interval"] == "two_sided_student_t_with_nested_permutation_mc"
    assert summary["nested_permutation_mc_standard_error"] == pytest.approx(
        (0.25 / 5) ** 0.5
    )


def test_domain_il_trajectory_fallback_never_invents_a_matrix_or_bwt() -> None:
    payload = {"accmean_task1": 60.0, "accmean_task2": 55.0}
    metrics = metrics_from_mammoth_log(payload, allow_trajectory_only=True)
    assert metrics["final_average_accuracy"] == 55.0
    assert metrics["average_incremental_accuracy"] == 57.5
    assert metrics["accuracy_matrix"] is None
    assert metrics["accuracy_matrix_available"] is False
    assert metrics["backward_transfer"] is None
    assert metrics["backward_transfer_available"] is False


def test_cross_arm_comparability_rejects_input_index_or_target_drift() -> None:
    fingerprint = [{"input_sha256": "a" * 64, "sampled_indexes_sha256": "b" * 64}]
    records = [
        {
            "benchmark": "split_cifar100",
            "method": "gem",
            "estimand_class": "assignment_only",
            "seed": 3401,
            "intervention": "paired",
            "permutation_replica": None,
            "sampled_event_fingerprints": fingerprint,
        },
        *[
            {
                "benchmark": "split_cifar100",
                "method": "gem",
                "estimand_class": "assignment_only",
                "seed": 3401,
                "intervention": "permuted",
                "permutation_replica": replica,
                "sampled_event_fingerprints": fingerprint,
            }
            for replica in range(4)
        ],
    ]
    _validate_cross_arm_comparability(records, permutation_replica_ids=range(4))
    records[-1]["sampled_event_fingerprints"] = [
        {"input_sha256": "c" * 64, "sampled_indexes_sha256": "b" * 64}
    ]
    with pytest.raises(RuntimeError, match="sampled input-index-target mismatch"):
        _validate_cross_arm_comparability(records, permutation_replica_ids=range(4))


def test_der_keeps_input_index_coupling_but_reports_endogenous_target_drift() -> None:
    base = {
        "observation": 10,
        "event": 0,
        "channel": "dark_logits",
        "rows": 4,
        "input_sha256": "1" * 64,
        "sampled_indexes_sha256": "2" * 64,
        "original_target_sha256": "3" * 64,
        "target_multiset_sha256": "4" * 64,
    }
    common = {
        "benchmark": "split_cifar100",
        "method": "der",
        "estimand_class": "path_dependent_endogenous_targets",
        "seed": 3401,
    }
    records = [
        {
            **common,
            "intervention": "paired",
            "permutation_replica": None,
            "sampled_event_fingerprints": [base],
        },
        *[
            {
                **common,
                "intervention": "permuted",
                "permutation_replica": replica,
                "sampled_event_fingerprints": [
                    {
                        **base,
                        "original_target_sha256": (
                            "5" * 64 if replica == 1 else base["original_target_sha256"]
                        ),
                        "target_multiset_sha256": (
                            "6" * 64 if replica == 1 else base["target_multiset_sha256"]
                        ),
                    }
                ],
            }
            for replica in range(4)
        ],
    ]
    report = _validate_cross_arm_comparability(
        records, permutation_replica_ids=range(4)
    )
    assert report["audit_scope"] == "sampled_events_only"
    assert report["sampled_endogenous_target_event_count"] == 4
    assert report["sampled_endogenous_target_divergence_count"] == 1
    group = report["path_dependent_endogenous_target_groups"][0]
    assert group["first_divergence"] == {
        "observation": 10,
        "event": 0,
        "channel": "dark_logits",
        "replica": 1,
    }
    records[-1]["sampled_event_fingerprints"][0]["input_sha256"] = "7" * 64
    with pytest.raises(RuntimeError, match="DER sampled replay input/index mismatch"):
        _validate_cross_arm_comparability(records, permutation_replica_ids=range(4))


def test_completed_unit_inventory_rejects_unlisted_artifacts(tmp_path: Path) -> None:
    identity = {"schema_version": "test.identity.v1", "unit": "x"}
    (tmp_path / "UNIT.json").write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.mechanism_unit.v2",
                "identity": identity,
                "benchmark": "test",
                "method": "test",
                "mammoth_model": "test",
                "estimand_class": None,
                "intervention": "none",
                "seed": 1,
                "permutation_replica": None,
                "metrics": {},
                "diagnostics": {},
                "sampled_event_fingerprints": [],
                "effective_args": {},
                "runtime_seconds": 0.0,
                "paper_eligible": False,
                "attempt": "attempt_001",
            }
        ),
        encoding="utf-8",
    )
    inventory = _artifact_inventory(tmp_path, exclude=("DONE.json",))
    (tmp_path / "DONE.json").write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.mechanism_unit_done.v2",
                "status": "PASS",
                "identity_sha256": canonical_sha256(identity),
                "artifact_count": len(inventory),
                "artifact_inventory": inventory,
                "artifact_bundle_sha256": canonical_sha256(inventory),
            }
        ),
        encoding="utf-8",
    )
    _validate_completed_unit(tmp_path, identity)
    (tmp_path / "unlisted.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact inventory changed"):
        _validate_completed_unit(tmp_path, identity)


def test_effective_args_lock_batch_size_device_and_shared_lr(lock: dict) -> None:
    for benchmark in lock["benchmarks"]:
        rates = {
            effective_mammoth_args(unit, lock, device="cuda:0", smoke=False)["lr"]
            for unit in enumerate_units(lock)
            if unit.benchmark == benchmark
        }
        assert len(rates) == 1
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.method == "experience_replay" and unit.intervention == "paired"
    )
    effective = effective_mammoth_args(unit, lock, device="cuda:0", smoke=False)
    assert effective["device"] == "cuda:0"
    assert effective["batch_size"] == 32
    assert effective["runtime_patch_domain_il_fullacc"] == "v1"


def _synthetic_audit_records(
    unit: MechanismUnit, effective: dict, *, projection: dict | None = None
) -> list[dict]:
    common = {
        "method": unit.mammoth_model,
        "benchmark": unit.benchmark,
        "dataset": unit.mammoth_dataset,
        "intervention": unit.intervention,
        "run_seed": unit.seed,
        "permutation_seed": unit.seed + 100_003,
        "replica": 0 if unit.permutation_replica is None else unit.permutation_replica,
    }
    digest = {name: character * 64 for name, character in {
        "input": "1", "index": "2", "target": "3", "multiset": "4"
    }.items()}
    rows = [
        {
            **common,
            "schema_version": "gradient_reuse.mammoth_effective_args.v1",
            "effective_args": effective,
        },
        {
            **common,
            "schema_version": "gradient_reuse.mammoth_gradient_diagnostic.v1",
            "observation": 0,
            "event": 0,
            "channel": "replay_labels",
            "rows": 4,
            "input_sha256": digest["input"],
            "sampled_indexes_sha256": digest["index"],
            "target_sha256": digest["target"],
            "target_multiset_sha256": digest["multiset"],
            "true_gradient_norm": 1.0,
            "mean_permuted_gradient_norm": 0.5,
            "pairing_component_norm": 0.5,
            "finite_permutation_variance": 0.1,
            "true_mean_permuted_cosine": 0.2,
            "permutation_draws": 4,
        },
        {
            **common,
            "schema_version": "gradient_reuse.mammoth_intervention.v1",
            "observation": 0,
            "event": 0,
            "channel": "replay_labels",
            "rows": 4,
            "input_sha256": digest["input"],
            "sampled_indexes_sha256": digest["index"],
            "original_target_sha256": digest["target"],
            "used_target_sha256": digest["target"],
            "target_multiset_sha256": digest["multiset"],
            "fixed_point_fraction": 1.0,
        },
    ]
    if projection is not None:
        rows.append({**common, **projection})
    return rows


def test_audit_validation_binds_effective_args_and_comparison_hashes(lock: dict) -> None:
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.method == "experience_replay" and unit.intervention == "paired"
    )
    effective = effective_mammoth_args(unit, lock, device="cuda:0", smoke=False)
    summary, fingerprints = _diagnostic_summary(
        _synthetic_audit_records(unit, effective),
        unit=unit,
        lock=lock,
        expected_effective_args=effective,
    )
    assert summary["gradient_diagnostic_records"] == 1
    assert fingerprints[0]["input_sha256"] == "1" * 64
    drifted = _synthetic_audit_records(unit, effective)
    drifted[0]["run_seed"] = 999
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _diagnostic_summary(
            drifted, unit=unit, lock=lock, expected_effective_args=effective
        )


def test_projection_diagnostic_fails_closed_on_unsatisfied_constraint(lock: dict) -> None:
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.method == "agem" and unit.intervention == "paired"
    )
    effective = effective_mammoth_args(unit, lock, device="cuda:0", smoke=False)
    records = _synthetic_audit_records(
        unit,
        effective,
        projection={
            "schema_version": "gradient_reuse.agem_projection.v1",
            "observation": 0,
            "event": 1,
            "channel": "reference_projection",
            "projection_active": True,
            "pre_projection_dot": -1.0,
            "post_projection_dot": -0.1,
        },
    )
    with pytest.raises(RuntimeError, match="left a violated dot product"):
        _diagnostic_summary(
            records, unit=unit, lock=lock, expected_effective_args=effective
        )


def test_data_manifest_is_immutable_and_validated(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "CIFAR100").mkdir(parents=True)
    (root / "MNIST" / "raw").mkdir(parents=True)
    (root / "MINIIMG").mkdir(parents=True)
    (root / "CIFAR100" / "archive.bin").write_bytes(b"cifar")
    (root / "MNIST" / "raw" / "train.bin").write_bytes(b"mnist")
    (root / "PUBLIC_DATA_ACQUISITION.json").write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.mechanism_public_data_acquisition.v1",
                "status": "PASS",
                "datasets": {
                    "cifar100": {
                        "root": "CIFAR100",
                        "train_rows": 50_000,
                        "test_rows": 10_000,
                        "classes": 100,
                        "torchvision_integrity_check": "PASS",
                    },
                    "mnist": {
                        "root": "MNIST",
                        "train_rows": 60_000,
                        "test_rows": 10_000,
                        "classes": 10,
                        "torchvision_integrity_check": "PASS",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "MINIIMG" / "MINIIMAGENET_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "classes": 100,
                "image_bundle_sha256": "a" * 64,
                "split_bundle_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    payload = build_data_manifest(root)
    assert payload["status"] == "LOCKED_BEFORE_SCIENTIFIC_RUN"
    output = tmp_path / "MECHANISM_DATA_MANIFEST.json"
    write_manifest(root, output)
    validate_data_manifest(output, data_root=root)
    (root / "CIFAR100" / "archive.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="data bytes changed"):
        validate_data_manifest(output, data_root=root)


def test_data_manifest_writer_requires_an_external_destination(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "CIFAR100").mkdir(parents=True)
    (root / "MNIST" / "raw").mkdir(parents=True)
    (root / "MINIIMG").mkdir(parents=True)
    (root / "CIFAR100" / "archive.bin").write_bytes(b"cifar")
    (root / "MNIST" / "raw" / "train.bin").write_bytes(b"mnist")
    (root / "PUBLIC_DATA_ACQUISITION.json").write_text(
        json.dumps(
            {
                "schema_version": "gradient_reuse.mechanism_public_data_acquisition.v1",
                "status": "PASS",
                "datasets": {
                    "cifar100": {
                        "root": "CIFAR100",
                        "train_rows": 50_000,
                        "test_rows": 10_000,
                        "classes": 100,
                        "torchvision_integrity_check": "PASS",
                    },
                    "mnist": {
                        "root": "MNIST",
                        "train_rows": 60_000,
                        "test_rows": 10_000,
                        "classes": 10,
                        "torchvision_integrity_check": "PASS",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "MINIIMG" / "MINIIMAGENET_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "classes": 100,
                "image_bundle_sha256": "a" * 64,
                "split_bundle_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="outside the data root"):
        write_manifest(root, root / "MECHANISM_DATA_MANIFEST.json")


def test_data_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "PUBLIC_DATA_ACQUISITION.json").write_text(
        '{"status":"PASS","status":"CHANGED"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        build_data_manifest(root)


def test_environment_manifest_is_bound_to_observed_versions_and_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = {
        "python": "3.12.0",
        "device": "cuda:0",
        "packages": {"torch": "2.0"},
        "runtime_patches": {
            "domain_il_full_accuracy_logger": "v1",
            "python_dont_write_bytecode": "1",
        },
    }
    monkeypatch.setattr(
        mechanism_run,
        "observed_environment",
        lambda **kwargs: dict(observed),
    )
    payload = build_environment_manifest(
        mammoth_root=tmp_path / "mammoth", device="cuda:0"
    )
    path = tmp_path / "ENVIRONMENT.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_environment_manifest(
        path, mammoth_root=tmp_path / "mammoth", device="cuda:0"
    ) == payload
    monkeypatch.setattr(
        mechanism_run,
        "observed_environment",
        lambda **kwargs: {**observed, "device": "cuda:1"},
    )
    with pytest.raises(RuntimeError, match="runtime environment differs"):
        validate_environment_manifest(
            path, mammoth_root=tmp_path / "mammoth", device="cuda:0"
        )


def test_scientific_modulo_sharding_is_canonical_and_filter_exclusive(
    lock: dict,
) -> None:
    units = enumerate_units(lock)
    selected, metadata = mechanism_run._select_campaign_units(
        units,
        benchmark=None,
        method=None,
        unit_id=None,
        max_units=None,
        shard_count=7,
        shard_index=3,
    )
    assert metadata == {
        "selection_scheme": mechanism_run.SHARD_SELECTION_SCHEME,
        "shard_count": 7,
        "shard_index": 3,
        "selected_unit_ids": [unit.unit_id for unit in selected],
    }
    assert selected == [unit for index, unit in enumerate(units) if index % 7 == 3]

    with pytest.raises(RuntimeError, match="provided together"):
        mechanism_run._select_campaign_units(
            units,
            benchmark=None,
            method=None,
            unit_id=None,
            max_units=None,
            shard_count=3,
            shard_index=None,
        )
    with pytest.raises(RuntimeError, match="positive"):
        mechanism_run._select_campaign_units(
            units,
            benchmark=None,
            method=None,
            unit_id=None,
            max_units=None,
            shard_count=0,
            shard_index=0,
        )
    with pytest.raises(RuntimeError, match="0 <= index"):
        mechanism_run._select_campaign_units(
            units,
            benchmark=None,
            method=None,
            unit_id=None,
            max_units=None,
            shard_count=3,
            shard_index=3,
        )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        mechanism_run._select_campaign_units(
            units,
            benchmark="split_cifar100",
            method=None,
            unit_id=None,
            max_units=None,
            shard_count=3,
            shard_index=0,
        )


def test_cli_requires_one_explicit_execution_mode(tmp_path: Path) -> None:
    common = ["--mammoth-root", str(tmp_path / "mammoth")]
    with pytest.raises(SystemExit):
        mechanism_run._parser().parse_args(common)
    with pytest.raises(SystemExit):
        mechanism_run._parser().parse_args([*common, "--full", "--smoke"])
    full = mechanism_run._parser().parse_args([*common, "--full"])
    smoke = mechanism_run._parser().parse_args([*common, "--smoke"])
    capture = mechanism_run._parser().parse_args(
        [*common, "--capture-environment-manifest", str(tmp_path / "env.json")]
    )
    assert full.full is True and full.smoke is False
    assert smoke.smoke is True and smoke.full is False
    assert capture.capture_environment_manifest == str(tmp_path / "env.json")


def test_runner_strict_json_and_jsonl_reject_duplicate_keys(tmp_path: Path) -> None:
    document = tmp_path / "duplicate.json"
    document.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        mechanism_run.strict_json(document)

    stream = tmp_path / "duplicate.jsonl"
    stream.write_text('{"event":1,"event":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        mechanism_run._read_jsonl(stream)


def test_attempt_allocation_uses_max_index_and_rejects_unowned_entries(
    tmp_path: Path,
) -> None:
    unit_root = tmp_path / "unit"
    (unit_root / "attempt_001").mkdir(parents=True)
    (unit_root / "attempt_003").mkdir()
    allocated = mechanism_run._allocate_attempt_directory(unit_root)
    assert allocated.name == "attempt_004"
    assert allocated.is_dir()

    unsafe_root = tmp_path / "unsafe-unit"
    unsafe_root.mkdir()
    (unsafe_root / "notes.txt").write_text("not campaign state", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected entry"):
        mechanism_run._allocate_attempt_directory(unsafe_root)


def test_campaign_root_rejects_arbitrary_preexisting_directory(
    tmp_path: Path,
) -> None:
    out = tmp_path / "campaign"
    out.mkdir()
    with pytest.raises(RuntimeError, match="without an owned RUN_MANIFEST"):
        mechanism_run._prepare_campaign_root(
            out,
            invocation={"paper_eligible": False, "selected_unit_ids": []},
        )


def test_campaign_root_lock_rejects_concurrent_same_root_writer(
    tmp_path: Path,
) -> None:
    out = tmp_path / "campaign"
    with mechanism_run._campaign_root_lock(out):
        with pytest.raises(RuntimeError, match="another launcher"):
            with mechanism_run._campaign_root_lock(out):
                raise AssertionError("unreachable")


def test_campaign_paths_are_pairwise_separated(tmp_path: Path) -> None:
    kwargs = {
        "out": tmp_path / "output",
        "data_root": tmp_path / "data",
        "mammoth_root": tmp_path / "mammoth",
        "lock_path": tmp_path / "repo" / "config.json",
        "source_manifest": tmp_path / "repo" / "source.json",
        "data_manifest": tmp_path / "repo" / "data.json",
        "environment_manifest": tmp_path / "repo" / "environment.json",
    }
    mechanism_run._validate_campaign_path_disjointness(**kwargs)

    with pytest.raises(RuntimeError, match="output root overlaps the data root"):
        mechanism_run._validate_campaign_path_disjointness(
            **{**kwargs, "out": tmp_path / "data" / "campaign"}
        )
    with pytest.raises(RuntimeError, match="outside the data root"):
        mechanism_run._validate_campaign_path_disjointness(
            **{
                **kwargs,
                "data_manifest": tmp_path / "data" / "manifest.json",
            }
        )


def test_failed_campaign_resume_clears_stale_failure_fields(tmp_path: Path) -> None:
    out = tmp_path / "campaign"
    out.mkdir()
    invocation = {"paper_eligible": False, "selected_unit_ids": ["unit-a"]}
    failed = mechanism_run._running_campaign_manifest(invocation, run_attempt=1)
    failed.update(
        {
            "status": "FAILED",
            "failed_unix": 2.0,
            "error": "RuntimeError: interrupted",
            "traceback": "stale traceback",
        }
    )
    mechanism_run.atomic_json(out / "RUN_MANIFEST.json", failed)
    _, already_complete = mechanism_run._prepare_campaign_root(
        out, invocation=invocation
    )
    assert already_complete is False
    resumed = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert set(resumed) == {
        "schema_version",
        "status",
        "started_unix",
        "run_attempt",
        "invocation",
    }
    assert resumed["status"] == "RUNNING"
    assert resumed["run_attempt"] == 2
    assert not {"failed_unix", "error", "traceback"} & set(resumed)


def test_existing_campaign_done_is_authenticated_before_resume(tmp_path: Path) -> None:
    out = tmp_path / "campaign"
    unit_done = out / "units" / "unit-a" / "DONE.json"
    unit_done.parent.mkdir(parents=True)
    unit_done.write_text('{"status":"PASS"}\n', encoding="utf-8")
    invocation = {"paper_eligible": False, "selected_unit_ids": ["unit-a"]}
    unit_done_hashes = {"unit-a": mechanism_run.sha256_file(unit_done)}
    campaign = {
        "schema_version": "gradient_reuse.mechanism_campaign.v2",
        "status": "COMPLETED",
        "started_unix": 1.0,
        "run_attempt": 1,
        "invocation": invocation,
        "completed_unix": 2.0,
        "completed_units": 1,
        "completed_unit_ids": ["unit-a"],
        "completed_unit_done_sha256": unit_done_hashes,
        "complete_protocol": False,
    }
    mechanism_run.atomic_json(out / "RUN_MANIFEST.json", campaign)
    inventory = mechanism_run._artifact_inventory(out, exclude=("DONE.json",))
    mechanism_run.atomic_json(
        out / "DONE.json",
        {
            "schema_version": "gradient_reuse.mechanism_done.v2",
            "status": "PASS",
            "paper_eligible": False,
            "complete_protocol": False,
            "completed_unit_ids": ["unit-a"],
            "completed_unit_done_sha256": unit_done_hashes,
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        },
    )
    _, already_complete = mechanism_run._prepare_campaign_root(
        out, invocation=invocation
    )
    assert already_complete is True

    unit_done.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unit-DONE digest changed"):
        mechanism_run._prepare_campaign_root(out, invocation=invocation)


def test_environment_captures_and_requires_mammoth_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mechanism_run, "_package_version", lambda name: f"locked-{name}"
    )
    observed = mechanism_run.observed_environment(
        mammoth_root=tmp_path / "mammoth", device="cpu"
    )
    assert set(observed["packages"]) == set(
        mechanism_run.REQUIRED_MAMMOTH_PACKAGES
    )
    assert "continuum" not in observed["packages"]
    payload = {"observed": observed}
    mechanism_run.validate_required_mammoth_packages(payload)

    missing = json.loads(json.dumps(payload))
    missing["observed"]["packages"]["kornia"] = None
    with pytest.raises(RuntimeError, match="lacks required packages"):
        mechanism_run.validate_required_mammoth_packages(missing)
