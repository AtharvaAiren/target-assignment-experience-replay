"""Deterministic scheduling and external-campaign gate tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.gradient_reuse_reviewer_extension import queue
from experiments.gradient_reuse_reviewer_extension.protocol import (
    ExtensionUnit,
    enumerate_units,
    load_lock,
)


def _unit(index: int, weight: int, *, priority: int = 1) -> ExtensionUnit:
    return ExtensionUnit(
        experiment=f"experiment_{index}",
        priority=priority,
        benchmark="benchmark",
        method=f"method_{index}",
        mammoth_dataset="dataset",
        mammoth_model="model",
        intervention="none",
        seed=3400 + index,
        permutation_replica=None,
        epochs=1,
        buffer_size=None,
        state_matched_probe=False,
        final_confusion=False,
        resource_weight=weight,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _seal(payload: dict) -> dict:
    value = dict(payload)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    value["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def _bm_root(tmp_path: Path, *, gate: str = "PASS") -> Path:
    root = tmp_path / "bm"
    verification_bindings = []
    shard_sizes = (41, 41, 40, 40)
    for index in range(4):
        panel = root / "units" / f"SHARD_{index:03d}_DONE.json"
        verification = root / "verification" / f"shard_{index}.json"
        _write_json(
            panel,
            _seal(
                {
                    "schema_version": (
                        "marginal_alignment.iclr_extension."
                        "play3_optimization_panel_shard.v1"
                    ),
                    "status": "PASS",
                    "shard_count": 4,
                    "shard_index": index,
                    "expected_units": shard_sizes[index],
                    "completed_units": shard_sizes[index],
                }
            ),
        )
        _write_json(
            verification,
            _seal(
                {
                    "schema_version": (
                        "marginal_alignment.iclr_extension."
                        "play3_optimization_verification_shard.v1"
                    ),
                    "status": "PASS",
                    "shard_count": 4,
                    "shard_index": index,
                    "expected_units": shard_sizes[index],
                    "verified_units": shard_sizes[index],
                    "full_endpoint_feature_reconstruction": True,
                }
            ),
        )
        verification_bindings.append(
            {
                "shard_index": index,
                "manifest_file_sha256": queue.sha256_file(verification),
            }
        )
    _write_json(
        root / "PLAY3_OPTIMIZATION_AGGREGATE.json",
        _seal(
            {
                "schema_version": (
                    "marginal_alignment.iclr_extension."
                    "play3_optimization_aggregate.v1"
                ),
                "status": "PASS",
                "artifact_status": "PASS",
                "status_interpretation": "artifact_integrity_and_completeness_only",
                "expected_units": 162,
                "observed_units": 162,
                "zero_silent_drops": True,
                "scientific_gate_status": gate,
                "all_required_gates_pass": gate == "PASS",
                "verification_manifests": verification_bindings,
            }
        ),
    )
    return root


def test_scheduler_selection_is_deterministic_and_respects_both_caps() -> None:
    pending = [_unit(2, 6), _unit(0, 6), _unit(3, 1, priority=2), _unit(1, 6)]

    first = queue.choose_launchable(
        pending,
        used_capacity=0,
        active_processes=0,
        capacity=16,
        maximum_processes=6,
    )
    second = queue.choose_launchable(
        list(reversed(pending)),
        used_capacity=0,
        active_processes=0,
        capacity=16,
        maximum_processes=6,
    )

    assert [unit.unit_id for unit in first] == [unit.unit_id for unit in second]
    assert sum(unit.resource_weight for unit in first) <= 16
    assert len(first) <= 6
    assert [unit.priority for unit in first[:2]] == [1, 1]
    assert first[-1].priority == 2


def test_scheduler_accounts_for_existing_occupancy_and_process_slots() -> None:
    pending = [_unit(index, 1) for index in range(8)]
    selected = queue.choose_launchable(
        pending,
        used_capacity=13,
        active_processes=4,
        capacity=16,
        maximum_processes=6,
    )
    assert len(selected) == 2
    assert sum(unit.resource_weight for unit in selected) == 2

    assert (
        queue.choose_launchable(
            pending,
            used_capacity=16,
            active_processes=2,
            capacity=16,
            maximum_processes=6,
        )
        == []
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"used_capacity": -1, "active_processes": 0},
        {"used_capacity": 17, "active_processes": 0},
        {"used_capacity": 0, "active_processes": 7},
    ],
)
def test_scheduler_rejects_invalid_occupancy(kwargs: dict[str, int]) -> None:
    with pytest.raises(RuntimeError, match="capacity|occupancy"):
        queue.choose_launchable(
            [_unit(0, 1)],
            capacity=16,
            maximum_processes=6,
            **kwargs,
        )


def test_scheduler_rejects_a_unit_heavier_than_total_capacity() -> None:
    with pytest.raises(RuntimeError, match="invalid resource weight"):
        queue.choose_launchable(
            [_unit(0, 17)],
            used_capacity=0,
            active_processes=0,
            capacity=16,
            maximum_processes=6,
        )


class _EndSchedulerCycle(Exception):
    pass


class _FakeProcess:
    pid = 12345

    def poll(self):
        return None


class _FakeLogHandle:
    def close(self) -> None:
        pass


def test_external_campaign_wait_surfaces_an_early_e2_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = SimpleNamespace(poll=lambda: 7)
    monkeypatch.setattr(
        queue,
        "external_campaign_processes",
        lambda _: pytest.fail("BM polling must stop after the E2 failure"),
    )

    with pytest.raises(RuntimeError, match="E2 secondary-endpoint"):
        queue._wait_for_external_campaign(
            tmp_path,
            poll_seconds=10,
            update=lambda *args: None,
            watched_e2_process=process,
        )


@pytest.mark.parametrize(
    ("free_mib", "expected_launches", "expected_capacity"),
    [
        (10_799, 0, 5),
        (10_800, 1, 6),
    ],
)
def test_live_gpu_capacity_reserves_1800_mib_per_launched_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_mib: int,
    expected_launches: int,
    expected_capacity: int,
) -> None:
    unit = _unit(0, 6)
    launched: list[str] = []
    updates: list[tuple[str, dict]] = []

    monkeypatch.setattr(queue, "_nvidia_free_mib", lambda device: free_mib)

    def fake_launch(selected, args, *, repo, out):
        launched.append(selected.unit_id)
        return queue.ActiveUnit(
            unit=selected,
            process=_FakeProcess(),
            log_path=tmp_path / "unit.log",
            log_handle=_FakeLogHandle(),
        )

    monkeypatch.setattr(queue, "_launch_unit", fake_launch)
    monkeypatch.setattr(
        queue.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(_EndSchedulerCycle()),
    )

    lock = {
        "runtime": {
            "scheduler": {
                "resource_capacity": 16,
                "maximum_parallel_processes": 6,
                "minimum_free_gpu_mib_before_launch": 1800,
                "resource_weight_mib": 1800,
                "poll_seconds": 10,
            }
        }
    }
    with pytest.raises(_EndSchedulerCycle):
        queue._run_cl_scheduler(
            [unit],
            SimpleNamespace(device="cuda:0"),
            lock=lock,
            repo=tmp_path,
            out=tmp_path / "out",
            update=lambda stage, state: updates.append((stage, state)),
        )

    assert len(launched) == expected_launches
    assert updates[-1][0] == "CONTINUAL_LEARNING"
    assert updates[-1][1]["effective_resource_capacity"] == expected_capacity


def test_parser_preflight_covers_every_distinct_model_dataset_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = load_lock()
    units = enumerate_units(lock)
    mammoth = tmp_path / "mammoth"
    mammoth.mkdir()
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def capture(command, *, log, cwd, environment) -> None:
        assert log == tmp_path / "out/logs/mammoth_parser_preflight.log"
        assert cwd == tmp_path
        commands.append(list(command))
        environments.append(dict(environment))

    monkeypatch.setattr(queue, "_run_logged", capture)
    queue._probe_mammoth_cli(
        units,
        SimpleNamespace(
            mechanism_python=tmp_path / "python",
            mammoth_root=mammoth,
            data_root=tmp_path / "data",
            device="cuda:0",
        ),
        lock=lock,
        repo=tmp_path,
        out=tmp_path / "out",
    )

    expected_pairs = {(unit.mammoth_model, unit.mammoth_dataset) for unit in units}
    assert len(commands) == len(expected_pairs) == 13
    observed_pairs = set()
    for command, environment in zip(commands, environments, strict=True):
        assert command[1:3] == [
            "-m",
            "experiments.gradient_reuse_reviewer_extension.mammoth_bootstrap",
        ]
        assert command[3:6] == ["--mammoth-root", str(mammoth), "--parse-only"]
        arguments = dict(zip(command[6::2], command[7::2], strict=True))
        observed_pairs.add((arguments["--model"], arguments["--dataset"]))
        assert environment["MAMMOTH_TEST"] == "1"
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert observed_pairs == expected_pairs


@pytest.mark.parametrize("gate", ["PASS", "FAIL"])
def test_external_campaign_terminal_gate_accepts_complete_artifacts_independent_of_sign(
    tmp_path: Path, gate: str
) -> None:
    root = _bm_root(tmp_path, gate=gate)
    binding = queue.validate_external_campaign_completion(root)

    assert binding["scientific_gate_status"] == gate
    assert binding["all_required_gates_pass"] is (gate == "PASS")
    assert len(binding["aggregate_sha256"]) == 64


def test_external_campaign_gate_rejects_inconsistent_scientific_boolean(
    tmp_path: Path,
) -> None:
    root = _bm_root(tmp_path, gate="FAIL")
    aggregate = root / "PLAY3_OPTIMIZATION_AGGREGATE.json"
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["all_required_gates_pass"] = True
    payload.pop("manifest_sha256")
    payload = _seal(payload)
    _write_json(aggregate, payload)

    with pytest.raises(RuntimeError, match="terminal aggregate contract"):
        queue.validate_external_campaign_completion(root)


def test_external_campaign_gate_rejects_the_wrong_terminal_schema(
    tmp_path: Path,
) -> None:
    root = _bm_root(tmp_path)
    aggregate = root / "PLAY3_OPTIMIZATION_AGGREGATE.json"
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["schema_version"] = "changed"
    payload.pop("manifest_sha256")
    _write_json(aggregate, _seal(payload))

    with pytest.raises(RuntimeError, match="terminal aggregate contract"):
        queue.validate_external_campaign_completion(root)


def test_external_campaign_gate_rejects_incomplete_shard_totals(
    tmp_path: Path,
) -> None:
    root = _bm_root(tmp_path)
    panel = root / "units" / "SHARD_003_DONE.json"
    panel_payload = json.loads(panel.read_text(encoding="utf-8"))
    panel_payload["expected_units"] = 39
    panel_payload["completed_units"] = 39
    panel_payload.pop("manifest_sha256")
    _write_json(panel, _seal(panel_payload))

    with pytest.raises(RuntimeError, match="162|count|units|shard"):
        queue.validate_external_campaign_completion(root)


def test_external_campaign_gate_rejects_symbolic_shard_markers(tmp_path: Path) -> None:
    root = _bm_root(tmp_path)
    marker = root / "units" / "SHARD_003_DONE.json"
    marker.unlink()
    outside = tmp_path / "outside-marker.json"
    _write_json(outside, {"status": "PASS"})
    marker.symlink_to(outside)

    with pytest.raises(RuntimeError, match="shard|symbolic|regular"):
        queue.validate_external_campaign_completion(root)


def test_process_filter_requires_both_bound_root_and_known_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bound-run"
    own = os.getpid()
    monkeypatch.setattr(
        queue,
        "_process_rows",
        lambda: [
            (own, f"external_campaign_runner {root}"),
            (101, f"python prepare_external_campaign_stores.py --out {root}"),
            (102, "python prepare_external_campaign_stores.py --out /other"),
            (103, f"python unrelated.py --out {root}"),
            (
                104,
                f"bash external_campaign_runner.sh {root}",
            ),
            (105, f"python scripts/external_campaign/run.py --out {root}"),
        ],
    )

    assert queue.external_campaign_processes(root) == [
        (101, f"python prepare_external_campaign_stores.py --out {root}"),
        (104, f"bash external_campaign_runner.sh {root}"),
        (105, f"python scripts/external_campaign/run.py --out {root}"),
    ]


def test_orphan_filter_is_scoped_to_exact_output_and_scientific_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "extension"
    monkeypatch.setattr(
        queue,
        "_process_rows",
        lambda: [
            (
                201,
                "python -m experiments.gradient_reuse_reviewer_extension.unit_runner "
                f"--out {out}",
            ),
            (
                202,
                "python transformer_bootstrap.py "
                f"--out {out}/transformer_stream2/attempt_0/RESULT.json",
            ),
            (
                203,
                "python -m experiments.gradient_reuse_reviewer_extension.unit_runner "
                "--out /other",
            ),
            (204, f"python unrelated.py --out {out}"),
            (
                205,
                "python -m "
                "experiments.gradient_reuse_reviewer_extension.secondary_endpoints "
                f"--out {out}/secondary.json",
            ),
            (
                206,
                "python -m experiments.gradient_reuse_reviewer_extension.aggregate "
                f"--out {out}",
            ),
        ],
    )
    assert [pid for pid, _ in queue.reviewer_extension_orphan_processes(out)] == [
        201,
        202,
        205,
        206,
    ]


@pytest.mark.parametrize(
    ("free_gib", "expected_status"),
    [(99, "FAIL"), (100, "PASS"), (151, "PASS")],
)
def test_disk_gate_is_frozen_at_one_hundred_gib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gib: int,
    expected_status: str,
) -> None:
    block = 4096
    monkeypatch.setattr(
        queue.os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=free_gib * (1024**3) // block,
            f_frsize=block,
        ),
    )
    assert queue.disk_gate_status(tmp_path, required_gib=100) == {
        "observed_free_disk_gib": free_gib,
        "required_free_disk_gib": 100,
        "status": expected_status,
    }


def test_queue_lock_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.write_text("preserve\n", encoding="utf-8")
    symbolic = tmp_path / "symbolic.lock"
    symbolic.symlink_to(target)
    with pytest.raises(RuntimeError, match="unsafe|safely"):
        queue._open_queue_lock(symbolic)

    hardlink = tmp_path / "hardlink.lock"
    hardlink.hardlink_to(target)
    with pytest.raises(RuntimeError, match="singly linked"):
        queue._open_queue_lock(hardlink)
    assert target.read_text(encoding="utf-8") == "preserve\n"


def test_existing_e7_success_requires_live_bundle_parent_and_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.gradient_reuse_reviewer_extension import transformer_stream2

    result = tmp_path / "attempt_0" / "RESULT.json"
    result.parent.mkdir()
    payload = {
        "parent": {"stream_sha256": "a" * 64},
        "correction_provenance": {"patch_sha256": "b" * 64},
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    expected_bundle = {"bundle_sha256": "c" * 64}
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        transformer_stream2,
        "validate_stream2_result",
        lambda value, *, expected_input_bundle: observed.append(
            dict(expected_input_bundle)
        ),
    )
    context = {
        "input_bundle": expected_bundle,
        "parent_binding": payload["parent"],
        "correction_provenance": payload["correction_provenance"],
    }
    assert queue._existing_transformer_success(tmp_path, live_context=context) == result
    assert observed == [expected_bundle]

    context["parent_binding"] = {"stream_sha256": "d" * 64}
    with pytest.raises(RuntimeError, match="stale parent"):
        queue._existing_transformer_success(tmp_path, live_context=context)


def test_clone_gate_rejects_a_symbolic_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "actual-target"
    source.mkdir()
    target.mkdir()
    destination = tmp_path / "mammoth"
    destination.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        queue.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "clone must not run for unsafe destination"
        ),
    )

    with pytest.raises(RuntimeError, match="symbolic|symlink|destination"):
        queue._clone_pinned(source, destination)
