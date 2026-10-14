"""Contracts for the frozen reviewer-extension design and unit table."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.gradient_reuse_reviewer_extension import (
    cli_adapters,
    mammoth_bootstrap,
    protocol,
    source,
    unit_runner,
)
from experiments.gradient_reuse_reviewer_extension.protocol import (
    EXPECTED_CL_UNITS,
    enumerate_units,
    load_lock,
    validate_lock,
)
from experiments.gradient_reuse_mechanisms.protocol import method_estimand_class


EXPECTED_EXPERIMENT_COUNTS = {
    "E1_off_floor_cifar50": 18,
    "E3_E6_state_matched_bridge": 30,
    "E4_buffer2000": 36,
    "E5_pmnist_seed_extension": 54,
    "E8_miniimagenet10": 18,
}


def test_frozen_lock_enumerates_the_exact_156_unique_units() -> None:
    lock = load_lock()
    units = enumerate_units(lock)

    assert len(units) == EXPECTED_CL_UNITS == 156
    assert len({unit.unit_id for unit in units}) == len(units)
    assert Counter(unit.experiment for unit in units) == EXPECTED_EXPERIMENT_COUNTS
    assert all(unit.seed in {3401, 3402, 3403, 3404, 3405} for unit in units)
    assert all(unit.permutation_replica in {None, 0, 1, 2, 3} for unit in units)


def test_reference_and_replay_arms_have_the_prespecified_nested_structure() -> None:
    units = enumerate_units(load_lock())
    groups: dict[tuple[str, str, str, int], list] = defaultdict(list)
    for unit in units:
        groups[(unit.experiment, unit.benchmark, unit.method, unit.seed)].append(unit)

    for (_, _, method, _), group in groups.items():
        observed = {(unit.intervention, unit.permutation_replica) for unit in group}
        if method_estimand_class(method) == "reference_baseline":
            assert observed == {("none", None)}
            assert len(group) == 1
            assert group[0].buffer_size is None
        else:
            assert observed == {
                ("paired", None),
                ("permuted", 0),
                ("permuted", 1),
                ("permuted", 2),
                ("permuted", 3),
            }
            assert len(group) == 5
            assert all(unit.buffer_size in {500, 2000} for unit in group)


def test_pmnist_seed_extension_covers_all_five_assignment_methods() -> None:
    units = [
        unit
        for unit in enumerate_units(load_lock())
        if unit.experiment == "E5_pmnist_seed_extension"
    ]

    assert len(units) == 54
    assert {unit.seed for unit in units} == {3404, 3405}
    assert {unit.method for unit in units} == {
        "naive_sgd",
        "mer_current_only",
        "experience_replay",
        "gem",
        "agem",
        "mer",
        "pcgrad_replay",
    }
    for seed in (3404, 3405):
        per_seed = [unit for unit in units if unit.seed == seed]
        assert sum(unit.method == "naive_sgd" for unit in per_seed) == 1
        assert sum(unit.method == "mer_current_only" for unit in per_seed) == 1
        assert sum(unit.method == "mer" for unit in per_seed) == 5


def test_mer_current_only_is_a_true_no_replay_reference_arm(
    tmp_path: Path,
) -> None:
    lock = load_lock()
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.experiment == "E5_pmnist_seed_extension"
        and unit.method == "mer_current_only"
        and unit.seed == 3404
    )

    arguments = unit_runner.effective_args(unit, lock, device="cuda:0")
    assert arguments["model"] == "mer_current_only"
    assert arguments["batch_size"] == 1
    assert {name: arguments[name] for name in ("beta", "gamma", "batch_num")} == {
        "beta": 0.01,
        "gamma": 0.03,
        "batch_num": 1,
    }
    assert not any(
        name.startswith("pairing_") or name in {"buffer_size", "minibatch_size"}
        for name in arguments
    )

    command = unit_runner.build_command(
        unit,
        lock,
        python=Path("/campaign/python"),
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        pairing_audit_path=tmp_path / "pairing.jsonl",
        state_audit_path=tmp_path / "state.jsonl",
        device="cuda:0",
    )
    assert "--pairing_audit_path" not in command
    assert unit_runner._pairing_summary(
        unit,
        lock,
        tmp_path / "absent-pairing.jsonl",
        arguments,
    ) == ({"records": 0}, [])
    assert {"mer_current_only", "mer_pairing"} <= set(cli_adapters.ADAPTER_NAMES)


def test_mer_replay_arm_uses_the_exact_matched_meta_update_arguments(
    tmp_path: Path,
) -> None:
    lock = load_lock()
    unit = next(
        unit
        for unit in enumerate_units(lock)
        if unit.experiment == "E5_pmnist_seed_extension"
        and unit.method == "mer"
        and unit.intervention == "paired"
        and unit.seed == 3404
    )

    arguments = unit_runner.effective_args(unit, lock, device="cuda:0")
    assert arguments["model"] == "mer_pairing"
    assert {
        name: arguments[name]
        for name in (
            "batch_size",
            "buffer_size",
            "minibatch_size",
            "mer_replay_batch_size",
            "beta",
            "gamma",
            "batch_num",
        )
    } == {
        "batch_size": 1,
        "buffer_size": 500,
        "minibatch_size": 10,
        "mer_replay_batch_size": 10,
        "beta": 0.01,
        "gamma": 0.03,
        "batch_num": 1,
    }
    assert arguments["pairing_intervention"] == "paired"

    command = unit_runner.build_command(
        unit,
        lock,
        python=Path("/campaign/python"),
        mammoth_root=tmp_path / "mammoth",
        data_root=tmp_path / "data",
        results_path=tmp_path / "results",
        pairing_audit_path=tmp_path / "pairing.jsonl",
        state_audit_path=tmp_path / "state.jsonl",
        device="cuda:0",
    )
    assert command.count("--pairing_audit_path") == 1


def test_mammoth_bootstrap_parse_only_emits_a_sealed_parser_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mammoth = tmp_path / "mammoth"
    mammoth.mkdir()
    parsed_arguments: list[list[str]] = []
    extended_datasets: list[tuple[int, str]] = []

    class FakeMain:
        @staticmethod
        def parse_args(arguments):
            parsed_arguments.append(list(arguments))
            return SimpleNamespace(
                model="mer-pairing",
                dataset="perm-mnist",
                batch_size=1,
                minibatch_size=10,
            )

        @staticmethod
        def extend_args(arguments, dataset) -> None:
            extended_datasets.append((dataset.N_CLASSES, dataset.SETTING))
            # A correctly wrapped Mammoth extension retains this explicit value.
            arguments.minibatch_size = 10

    monkeypatch.setenv("MAMMOTH_TEST", "1")
    monkeypatch.setattr(mammoth_bootstrap, "prepare", lambda root: FakeMain)

    assert (
        mammoth_bootstrap.main(
            [
                "--mammoth-root",
                str(mammoth),
                "--parse-only",
                "--model",
                "mer-pairing",
                "--dataset",
                "perm-mnist",
                "--batch-size",
                "1",
                "--minibatch-size",
                "10",
            ]
        )
        == 0
    )

    probe = json.loads(capsys.readouterr().out)
    assert probe == {
        "schema_version": "gradient_reuse.reviewer_mammoth_parser_probe.v1",
        "status": "PASS",
        "model": "mer-pairing",
        "dataset": "perm-mnist",
        "batch_size": 1,
        "minibatch_size": 10,
    }
    assert parsed_arguments == [
        [
            "--model",
            "mer-pairing",
            "--dataset",
            "perm-mnist",
            "--batch-size",
            "1",
            "--minibatch-size",
            "10",
        ]
    ]
    assert extended_datasets == [(1000, "class-il")]


def test_mammoth_bootstrap_rejects_parser_overwrite_of_explicit_minibatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mammoth = tmp_path / "mammoth"
    mammoth.mkdir()

    class BrokenMain:
        @staticmethod
        def parse_args(arguments):
            return SimpleNamespace(
                model="er-pairing",
                dataset="seq-cifar100",
                batch_size=32,
                minibatch_size=32,
            )

        @staticmethod
        def extend_args(arguments, dataset) -> None:
            arguments.minibatch_size = 128

    monkeypatch.setenv("MAMMOTH_TEST", "1")
    monkeypatch.setattr(mammoth_bootstrap, "prepare", lambda root: BrokenMain)

    with pytest.raises(RuntimeError, match="explicit replay minibatch"):
        mammoth_bootstrap.main(
            [
                "--mammoth-root",
                str(mammoth),
                "--parse-only",
                "--model",
                "er-pairing",
            ]
        )


def test_state_matched_probes_are_only_on_paired_er_trajectories() -> None:
    units = enumerate_units(load_lock())
    bridge = [unit for unit in units if unit.experiment == "E3_E6_state_matched_bridge"]

    assert len(bridge) == 30
    assert {unit.benchmark for unit in bridge} == {
        "split_cifar100",
        "permuted_mnist",
    }
    assert all(unit.method == "experience_replay" for unit in bridge)
    assert all(unit.mammoth_model == "er_state_matched_audit" for unit in bridge)
    assert all(unit.final_confusion for unit in bridge)
    assert sum(unit.state_matched_probe for unit in bridge) == 6
    assert all(
        unit.state_matched_probe is (unit.intervention == "paired") for unit in bridge
    )


def test_epoch_buffer_and_resource_sensitivities_match_the_lock() -> None:
    units = enumerate_units(load_lock())

    e1 = [unit for unit in units if unit.experiment == "E1_off_floor_cifar50"]
    assert {(unit.epochs, unit.resource_weight) for unit in e1} == {(50, 6)}
    assert {unit.buffer_size for unit in e1 if unit.method == "experience_replay"} == {
        500
    }

    e4 = [unit for unit in units if unit.experiment == "E4_buffer2000"]
    assert {unit.buffer_size for unit in e4 if unit.method != "naive_sgd"} == {2000}
    assert {(unit.benchmark, unit.resource_weight) for unit in e4} == {
        ("split_cifar100", 5),
        ("permuted_mnist", 1),
    }

    e8 = [unit for unit in units if unit.experiment == "E8_miniimagenet10"]
    assert {(unit.epochs, unit.resource_weight) for unit in e8} == {(10, 6)}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lock: lock["runtime"].__setitem__("num_workers", 99),
        lambda lock: lock["runtime"]["scheduler"].__setitem__(
            "minimum_free_gpu_mib_before_launch", 1
        ),
        lambda lock: lock["experiments"]["E1_off_floor_cifar50"].__setitem__(
            "resource_weight", 1
        ),
        lambda lock: lock["state_matched_definition"].__setitem__(
            "first_order_term", "changed"
        ),
        lambda lock: lock["experiments"]["E8_miniimagenet10"].__setitem__(
            "primary_endpoint", "changed"
        ),
    ],
)
def test_lock_rejects_changes_to_execution_and_analysis_contracts(mutation) -> None:
    lock = copy.deepcopy(load_lock())
    mutation(lock)
    with pytest.raises(RuntimeError, match="lock contents changed"):
        validate_lock(lock)


def test_lock_rejects_undeclared_top_level_fields() -> None:
    lock = copy.deepcopy(load_lock())
    lock["outcome_dependent_override"] = True
    with pytest.raises(RuntimeError, match="keys|unexpected|lock"):
        validate_lock(lock)


def test_parent_completion_rejects_a_symbolic_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "RTX_7DAY_PROFILE_DONE.json"
    target.write_text("{}\n", encoding="utf-8")
    marker = tmp_path / "completion-link.json"
    marker.symlink_to(target)
    monkeypatch.setattr(
        protocol,
        "sha256_file",
        lambda path: protocol.EXPECTED_PARENT_COMPLETION_SHA256,
    )
    monkeypatch.setattr(
        protocol,
        "strict_json",
        lambda path: {
            "schema_version": "gradient_reuse.iclr_rtx_7day_queue_done.v1",
            "status": "PASS",
            "profile_id": "rtx_7day_v1",
            "mechanism_unit_count": 288,
        },
    )

    with pytest.raises(RuntimeError, match="regular|symbolic"):
        protocol.validate_parent_completion(marker)


def test_public_reviewer_extension_source_closure_fails_closed() -> None:
    omissions = source.public_source_closure_omissions()
    assert omissions == source.PUBLIC_CLOSURE_REQUIRED_EXTERNAL_FILES
    with pytest.raises(RuntimeError, match="implementation/reference only"):
        source.build_source_manifest()
