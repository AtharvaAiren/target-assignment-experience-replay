"""Authenticated, resumable queue for the reviewer-extension campaign.

The CPU-only E2 re-analysis starts immediately.  New GPU work waits until the
separately running external campaign has published its terminal
aggregate and no related worker remains.  Continual-learning units then share
one accelerator under a fixed resource-weight scheduler; the second Pythia
trajectory runs alone after every CL unit is sealed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_mechanisms.protocol import (
    UPSTREAM_COMMIT,
    canonical_sha256,
    sha256_file,
    strict_json,
)

from .overlay import BASE_MANIFEST_NAME, install_overlay, verify_installed_overlay
from .protocol import (
    DEFAULT_LOCK,
    EXPECTED_CL_UNITS,
    ExtensionUnit,
    enumerate_units,
    load_lock,
    lock_identity,
    validate_parent_completion,
)
from .source import validate_source_manifest
from .unit_runner import atomic_json, build_command, subprocess_environment


QUEUE_SCHEMA = "gradient_reuse.reviewer_extension_queue.v1"
DONE_SCHEMA = "gradient_reuse.reviewer_extension_done.v1"
EXTERNAL_CAMPAIGN_PROCESS_MARKERS = (
    "external_campaign_runner",
    "external_campaign_runner.sh",
    "prepare_external_campaign_stores.py",
    "run_external_campaign_gate.py",
    "scripts/external_campaign/run.py",
)
ORPHAN_PROCESS_MARKERS = (
    "experiments.gradient_reuse_reviewer_extension.secondary_endpoints",
    "experiments.gradient_reuse_reviewer_extension.unit_runner",
    "experiments.gradient_reuse_reviewer_extension.mammoth_bootstrap",
    "transformer_bootstrap.py",
    "experiments.gradient_reuse_reviewer_extension.transformer_stream2",
    "experiments.gradient_reuse_reviewer_extension.aggregate",
)


@dataclass
class ActiveUnit:
    unit: ExtensionUnit
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: Any


def scheduler_key(unit: ExtensionUnit) -> tuple[Any, ...]:
    """Outcome-independent, fully deterministic launch order."""

    replica = -1 if unit.permutation_replica is None else unit.permutation_replica
    intervention_order = {"none": 0, "paired": 1, "permuted": 2}
    return (
        unit.priority,
        unit.experiment,
        unit.benchmark,
        unit.seed,
        unit.method,
        intervention_order[unit.intervention],
        replica,
        unit.unit_id,
    )


def choose_launchable(
    pending: Sequence[ExtensionUnit],
    *,
    used_capacity: int,
    active_processes: int,
    capacity: int,
    maximum_processes: int,
    available_launch_capacity: int | None = None,
) -> list[ExtensionUnit]:
    """Choose a maximal deterministic subset that fits the frozen resource cap."""

    if min(used_capacity, active_processes) < 0:
        raise RuntimeError("scheduler occupancy cannot be negative")
    if capacity < 1 or maximum_processes < 1:
        raise RuntimeError("scheduler limits must be positive")
    if used_capacity > capacity or active_processes > maximum_processes:
        raise RuntimeError("scheduler is already over capacity")
    remaining_capacity = capacity - used_capacity
    if available_launch_capacity is not None:
        if available_launch_capacity < 0:
            raise RuntimeError("available launch capacity cannot be negative")
        remaining_capacity = min(remaining_capacity, available_launch_capacity)
    remaining_slots = maximum_processes - active_processes
    selected: list[ExtensionUnit] = []
    for unit in sorted(pending, key=scheduler_key):
        weight = int(unit.resource_weight)
        if weight < 1 or weight > capacity:
            raise RuntimeError(f"unit has an invalid resource weight: {unit.unit_id}")
        if len(selected) >= remaining_slots:
            break
        if weight <= remaining_capacity:
            selected.append(unit)
            remaining_capacity -= weight
    return selected


def _environment(repo: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(repo),
        }
    )
    return value


def _process_rows() -> list[tuple[int, str]]:
    completed = subprocess.run(
        ["ps", "-ww", "-eo", "pid=,args="],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, _, command = stripped.partition(" ")
        if head.isdigit():
            rows.append((int(head), command))
    return rows


def external_campaign_processes(root: Path) -> list[tuple[int, str]]:
    root_text = str(Path(root).resolve())
    own_pid = os.getpid()
    return [
        (pid, command)
        for pid, command in _process_rows()
        if pid != own_pid
        and root_text in command
        and any(marker in command for marker in EXTERNAL_CAMPAIGN_PROCESS_MARKERS)
    ]


def reviewer_extension_orphan_processes(out: Path) -> list[tuple[int, str]]:
    """Find scientific children bound to OUT but not owned by this queue PID."""

    out_text = str(Path(out).resolve())
    own_pid = os.getpid()
    return [
        (pid, command)
        for pid, command in _process_rows()
        if pid != own_pid
        and out_text in command
        and any(marker in command for marker in ORPHAN_PROCESS_MARKERS)
    ]


def free_disk_gib(path: Path) -> int:
    status = os.statvfs(Path(path))
    return int(status.f_bavail * status.f_frsize // (1024**3))


def disk_gate_status(path: Path, *, required_gib: int) -> dict[str, Any]:
    required = int(required_gib)
    if required < 1:
        raise RuntimeError("reviewer-extension disk requirement must be positive")
    observed = free_disk_gib(path)
    return {
        "observed_free_disk_gib": observed,
        "required_free_disk_gib": required,
        "status": "PASS" if observed >= required else "FAIL",
    }


def _open_queue_lock(path: Path) -> int:
    """Open a singly linked regular queue lock without following symlinks."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe reviewer-extension queue lock: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(
            f"cannot safely open reviewer-extension queue lock: {path}"
        ) from error
    try:
        lock_status = os.fstat(descriptor)
        if not stat.S_ISREG(lock_status.st_mode) or lock_status.st_nlink != 1:
            raise RuntimeError(
                f"reviewer-extension queue lock is not a singly linked file: {path}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _external_campaign_manifest_checksum(
    value: Mapping[str, Any], *, context: str
) -> None:
    expected = value.get("manifest_sha256")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    encoded = (
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    observed = hashlib.sha256(encoded).hexdigest()
    if not isinstance(expected, str) or observed != expected:
        raise RuntimeError(f"external campaign {context} checksum failed")


def _external_campaign_shards(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    panel: dict[str, str] = {}
    verification: dict[str, str] = {}
    panel_total = 0
    verification_total = 0
    for index in range(4):
        panel_path = root / "units" / f"SHARD_{index:03d}_DONE.json"
        verification_path = root / "verification" / f"shard_{index}.json"
        if not verification_path.exists():
            # The locked runner normally uses three-digit names; accept that
            # exact alternative if present, never an arbitrary glob match.
            verification_path = root / "verification" / f"shard_{index:03d}.json"
        for role, path in (
            ("panel shard", panel_path),
            ("verification shard", verification_path),
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"external campaign {role} is absent: {path}")
        panel_value = strict_json(panel_path)
        verification_value = strict_json(verification_path)
        if not isinstance(panel_value, dict) or not (
            panel_value.get("schema_version")
            == "marginal_alignment.iclr_extension.play3_optimization_panel_shard.v1"
            and panel_value.get("status") == "PASS"
            and panel_value.get("shard_count") == 4
            and panel_value.get("shard_index") == index
            and panel_value.get("expected_units") == panel_value.get("completed_units")
        ):
            raise RuntimeError(f"external campaign panel shard {index} is invalid")
        if not isinstance(verification_value, dict) or not (
            verification_value.get("schema_version")
            == "marginal_alignment.iclr_extension.play3_optimization_verification_shard.v1"
            and verification_value.get("status") == "PASS"
            and verification_value.get("shard_count") == 4
            and verification_value.get("shard_index") == index
            and verification_value.get("expected_units")
            == verification_value.get("verified_units")
            and verification_value.get("full_endpoint_feature_reconstruction") is True
        ):
            raise RuntimeError(
                f"external campaign verification shard {index} is invalid"
            )
        panel_count = panel_value.get("expected_units")
        verification_count = verification_value.get("expected_units")
        if (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
            or isinstance(verification_count, bool)
            or not isinstance(verification_count, int)
            or verification_count < 1
            or panel_count != verification_count
        ):
            raise RuntimeError(
                f"external campaign shard {index} unit counts do not agree"
            )
        panel_total += panel_count
        verification_total += verification_count
        _external_campaign_manifest_checksum(
            panel_value, context=f"panel shard {index}"
        )
        _external_campaign_manifest_checksum(
            verification_value, context=f"verification shard {index}"
        )
        panel[panel_path.name] = sha256_file(panel_path)
        verification[verification_path.name] = sha256_file(verification_path)
    if panel_total != 162 or verification_total != 162:
        raise RuntimeError(
            "external campaign shard totals changed: "
            f"panel={panel_total}, verification={verification_total}"
        )
    return panel, verification


def validate_external_campaign_completion(root: Path) -> dict[str, Any]:
    """Authenticate external-campaign completeness, independent of gate sign."""

    aggregate_path = Path(root) / "PLAY3_OPTIMIZATION_AGGREGATE.json"
    if not aggregate_path.is_file() or aggregate_path.is_symlink():
        raise RuntimeError("external campaign terminal aggregate is absent")
    payload = strict_json(aggregate_path)
    if not isinstance(payload, dict) or not (
        payload.get("schema_version")
        == "marginal_alignment.iclr_extension.play3_optimization_aggregate.v1"
        and payload.get("status") == "PASS"
        and payload.get("artifact_status") == "PASS"
        and payload.get("status_interpretation")
        == "artifact_integrity_and_completeness_only"
        and payload.get("expected_units") == 162
        and payload.get("observed_units") == 162
        and payload.get("zero_silent_drops") is True
        and payload.get("scientific_gate_status") in {"PASS", "FAIL"}
        and payload.get("all_required_gates_pass")
        is (payload.get("scientific_gate_status") == "PASS")
    ):
        raise RuntimeError("external campaign terminal aggregate contract failed")
    _external_campaign_manifest_checksum(payload, context="terminal aggregate")
    panel, verification = _external_campaign_shards(Path(root))
    aggregate_verification = payload.get("verification_manifests")
    if not isinstance(aggregate_verification, list) or len(aggregate_verification) != 4:
        raise RuntimeError("external campaign aggregate lacks verification bindings")
    by_index = {
        int(row.get("shard_index", -1)): row
        for row in aggregate_verification
        if isinstance(row, dict)
    }
    for index, (name, digest) in enumerate(sorted(verification.items())):
        row = by_index.get(index)
        if not isinstance(row, dict) or row.get("manifest_file_sha256") != digest:
            raise RuntimeError(
                f"external campaign aggregate does not bind verification shard {index}"
            )
    return {
        "aggregate_path": str(aggregate_path.resolve()),
        "aggregate_sha256": sha256_file(aggregate_path),
        "panel_shard_sha256": panel,
        "verification_shard_sha256": verification,
        "scientific_gate_status": payload["scientific_gate_status"],
        "all_required_gates_pass": payload["all_required_gates_pass"],
    }


def _wait_for_external_campaign(
    root: Path,
    *,
    poll_seconds: int,
    update: Any,
    watched_e2_process: subprocess.Popen[Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    stopped_without_aggregate_since: float | None = None
    while True:
        e2_status = None if watched_e2_process is None else watched_e2_process.poll()
        if e2_status not in {None, 0}:
            raise RuntimeError(
                "E2 secondary-endpoint re-analysis failed while waiting for "
                "external campaign"
            )
        processes = external_campaign_processes(root)
        aggregate = root / "PLAY3_OPTIMIZATION_AGGREGATE.json"
        if aggregate.is_file() and not aggregate.is_symlink() and not processes:
            return validate_external_campaign_completion(root)
        if not processes and root.exists() and not aggregate.exists():
            if stopped_without_aggregate_since is None:
                stopped_without_aggregate_since = time.monotonic()
            elif time.monotonic() - stopped_without_aggregate_since >= 180.0:
                raise RuntimeError(
                    "external campaign stopped for three minutes without its "
                    "terminal aggregate"
                )
        else:
            stopped_without_aggregate_since = None
        update(
            "WAITING_FOR_EXTERNAL_CAMPAIGN",
            {
                "external_campaign_processes": [
                    {"pid": pid, "command": command} for pid, command in processes
                ],
                "terminal_aggregate_present": aggregate.is_file(),
                "e2_status": "RUNNING" if e2_status is None else "PASS",
            },
        )
        time.sleep(poll_seconds)


def _nvidia_free_mib(device: str) -> int:
    if device != "cuda:0":
        raise RuntimeError("reviewer extension is locked to cuda:0")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    text = completed.stdout.strip()
    if not text.isdigit():
        raise RuntimeError(f"could not parse free GPU memory: {text!r}")
    return int(text)


def _nvidia_compute_processes(device: str) -> list[dict[str, int]]:
    if device != "cuda:0":
        raise RuntimeError("reviewer extension is locked to cuda:0")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.strip().split(",")]
        if fields == [""]:
            continue
        if len(fields) != 2 or not all(field.isdigit() for field in fields):
            raise RuntimeError(f"could not parse GPU compute process: {line!r}")
        output.append({"pid": int(fields[0]), "used_gpu_mib": int(fields[1])})
    return output


def _wait_for_gpu_free(
    *,
    device: str,
    required_mib: int,
    poll_seconds: int,
    stage: str,
    update: Any,
) -> int:
    if required_mib < 1:
        raise RuntimeError("GPU free-memory gate must be positive")
    while True:
        observed = _nvidia_free_mib(device)
        update(
            stage,
            {"free_gpu_mib": observed, "required_free_gpu_mib": required_mib},
        )
        if observed >= required_mib:
            return observed
        time.sleep(poll_seconds)


def _wait_for_exclusive_pythia(
    *, device: str, required_mib: int, poll_seconds: int, update: Any
) -> int:
    while True:
        free_mib = _nvidia_free_mib(device)
        compute_processes = _nvidia_compute_processes(device)
        update(
            "WAITING_FOR_EXCLUSIVE_PYTHIA_GPU",
            {
                "free_gpu_mib": free_mib,
                "required_free_gpu_mib": required_mib,
                "compute_processes": compute_processes,
                "zero_compute_processes_required": True,
            },
        )
        if free_mib >= required_mib and not compute_processes:
            return free_mib
        time.sleep(poll_seconds)


def _clone_pinned(source: Path, destination: Path) -> None:
    source = Path(source).resolve(strict=True)
    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise RuntimeError(
            f"reviewer Mammoth root must not be symbolic: {raw_destination}"
        )
    destination = raw_destination.resolve()
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(
                f"reviewer Mammoth root is not a directory: {destination}"
            )
        observed = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if observed != UPSTREAM_COMMIT:
            raise RuntimeError("existing reviewer Mammoth root is at another commit")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.clone.{os.getpid()}.{time.time_ns()}"
    )
    try:
        subprocess.run(
            ["git", "clone", "--no-hardlinks", str(source), str(temporary)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--detach", UPSTREAM_COMMIT],
            check=True,
        )
        observed = subprocess.run(
            ["git", "-C", str(temporary), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if observed != UPSTREAM_COMMIT:
            raise RuntimeError("fresh reviewer checkout resolved to another commit")
        os.replace(temporary, destination)
    except BaseException:
        # Preserve a failed clone for diagnosis; never remove evidence silently.
        if temporary.exists():
            preserved = temporary.with_name(temporary.name + ".failed")
            if not preserved.exists():
                os.replace(temporary, preserved)
        raise


def _run_logged(
    command: Sequence[str], *, log: Path, cwd: Path, environment: Mapping[str, str]
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command exited {completed.returncode}; diagnostic log: {log}"
        )


def _probe_mammoth_cli(
    units: Sequence[ExtensionUnit],
    args: argparse.Namespace,
    *,
    lock: Mapping[str, Any],
    repo: Path,
    out: Path,
) -> None:
    """Parse every model/dataset combination before launching scientific work."""

    representatives: dict[tuple[str, str], ExtensionUnit] = {}
    for unit in sorted(units, key=scheduler_key):
        representatives.setdefault((unit.mammoth_model, unit.mammoth_dataset), unit)
    log = out / "logs" / "mammoth_parser_preflight.log"
    for unit in representatives.values():
        command = build_command(
            unit,
            lock,
            python=Path(args.mechanism_python),
            mammoth_root=Path(args.mammoth_root),
            data_root=Path(args.data_root),
            results_path=out / "preflight_only" / unit.unit_id,
            pairing_audit_path=out / "preflight_only" / "pairing.jsonl",
            state_audit_path=out / "preflight_only" / "state.jsonl",
            device=str(args.device),
            parse_only=True,
        )
        _run_logged(
            command,
            log=log,
            cwd=repo,
            environment=subprocess_environment(
                unit, lock, Path(args.mammoth_root).resolve(strict=True)
            ),
        )
    if len(representatives) != 13:
        raise RuntimeError(
            "reviewer Mammoth parser-preflight inventory changed: "
            f"{len(representatives)}"
        )


def _start_e2(args: argparse.Namespace, repo: Path, out: Path) -> subprocess.Popen[Any]:
    log = out / "logs" / "e2_secondary_endpoints.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    command = [
        str(args.mechanism_python),
        "-m",
        "experiments.gradient_reuse_reviewer_extension.secondary_endpoints",
        "--campaign",
        str(args.parent_campaign),
        "--parent-completion",
        str(args.parent_completion),
        "--output",
        str(out / "secondary" / "E2_EXISTING_RUN_REANALYSIS.json"),
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=_environment(repo),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Keep the descriptor reachable for the lifetime of the child.
    process._reviewer_log_handle = handle  # type: ignore[attr-defined]
    return process


def _finish_e2(process: subprocess.Popen[Any], out: Path) -> Path:
    status = process.wait()
    handle = getattr(process, "_reviewer_log_handle", None)
    if handle is not None:
        handle.close()
    if status:
        raise RuntimeError(
            "E2 secondary-endpoint re-analysis failed; see "
            f"{out / 'logs/e2_secondary_endpoints.log'}"
        )
    output = out / "secondary" / "E2_EXISTING_RUN_REANALYSIS.json"
    payload = strict_json(output)
    if not isinstance(payload, dict) or not (
        payload.get("schema_version") == "gradient_reuse.reviewer_extension_e2.v1"
        and payload.get("status") == "PASS"
        and payload.get("raw_unit_count") == 288
    ):
        raise RuntimeError("E2 terminal output failed authentication")
    return output


def _unit_command(
    unit: ExtensionUnit,
    args: argparse.Namespace,
    *,
    out: Path,
) -> list[str]:
    return [
        str(args.mechanism_python),
        "-m",
        "experiments.gradient_reuse_reviewer_extension.unit_runner",
        "--lock",
        str(args.lock),
        "--source-manifest",
        str(args.source_manifest),
        "--parent-completion",
        str(args.parent_completion),
        "--mammoth-root",
        str(args.mammoth_root),
        "--data-root",
        str(args.data_root),
        "--data-manifest",
        str(args.data_manifest),
        "--environment-manifest",
        str(args.environment_manifest),
        "--python",
        str(args.mechanism_python),
        "--out",
        str(out),
        "--unit-id",
        unit.unit_id,
        "--device",
        str(args.device),
    ]


def _launch_unit(
    unit: ExtensionUnit,
    args: argparse.Namespace,
    *,
    repo: Path,
    out: Path,
) -> ActiveUnit:
    log = out / "logs" / "units" / f"{unit.unit_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        _unit_command(unit, args, out=out),
        cwd=repo,
        env=_environment(repo),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveUnit(unit=unit, process=process, log_path=log, log_handle=handle)


def _count_completed(out: Path) -> int:
    return len(list((out / "units").glob("*/DONE.json")))


def _run_cl_scheduler(
    units: Sequence[ExtensionUnit],
    args: argparse.Namespace,
    *,
    lock: Mapping[str, Any],
    repo: Path,
    out: Path,
    update: Any,
) -> None:
    scheduler = lock["runtime"]["scheduler"]
    capacity = int(scheduler["resource_capacity"])
    maximum = int(scheduler["maximum_parallel_processes"])
    minimum_free = int(scheduler["minimum_free_gpu_mib_before_launch"])
    weight_mib = int(scheduler["resource_weight_mib"])
    if weight_mib != minimum_free:
        raise RuntimeError("resource-weight memory binding differs from launch gate")
    poll_seconds = int(scheduler["poll_seconds"])
    pending = list(sorted(units, key=scheduler_key))
    active: list[ActiveUnit] = []
    failed: list[dict[str, Any]] = []
    while pending or active:
        still_active: list[ActiveUnit] = []
        for running in active:
            status = running.process.poll()
            if status is None:
                still_active.append(running)
                continue
            running.log_handle.close()
            if status:
                failed.append(
                    {
                        "unit_id": running.unit.unit_id,
                        "returncode": status,
                        "log": str(running.log_path),
                    }
                )
        active = still_active
        if failed:
            # Stop scheduling, but allow already-running scientific units to
            # finish and seal rather than killing valid work mid-trajectory.
            if active:
                update(
                    "DRAINING_AFTER_UNIT_FAILURE",
                    {
                        "failed_units": failed,
                        "active_units": [x.unit.unit_id for x in active],
                    },
                )
                time.sleep(poll_seconds)
                continue
            raise RuntimeError(f"reviewer CL unit failure(s): {failed}")
        used = sum(item.unit.resource_weight for item in active)
        free_mib = _nvidia_free_mib(str(args.device))
        launched: list[str] = []
        if free_mib >= minimum_free:
            # The frozen ``minimum_free`` quantity is a per-resource-weight
            # launch reserve.  Coupling the abstract capacity to live free
            # memory prevents an unrelated process that appears after the
            # 28-GiB campaign gate from turning a valid parallel launch into
            # an avoidable OOM race.
            free_weight = free_mib // weight_mib
            selected = choose_launchable(
                pending,
                used_capacity=used,
                active_processes=len(active),
                capacity=capacity,
                maximum_processes=maximum,
                available_launch_capacity=free_weight,
            )
            for unit in selected:
                active.append(_launch_unit(unit, args, repo=repo, out=out))
                pending.remove(unit)
                launched.append(unit.unit_id)
        update(
            "CONTINUAL_LEARNING",
            {
                "completed_units": _count_completed(out),
                "expected_units": EXPECTED_CL_UNITS,
                "pending_units": len(pending),
                "active_units": [
                    {
                        "unit_id": row.unit.unit_id,
                        "pid": row.process.pid,
                        "weight": row.unit.resource_weight,
                        "log": str(row.log_path),
                    }
                    for row in active
                ],
                "used_resource_capacity": sum(
                    row.unit.resource_weight for row in active
                ),
                "effective_resource_capacity": (
                    min(capacity, used + free_mib // weight_mib)
                    if free_mib >= minimum_free
                    else used
                ),
                "free_gpu_mib": free_mib,
                "launched_this_cycle": launched,
            },
        )
        if pending or active:
            time.sleep(poll_seconds)
    if _count_completed(out) != EXPECTED_CL_UNITS:
        raise RuntimeError(
            "CL scheduler exited without " f"{EXPECTED_CL_UNITS} sealed unit markers"
        )


def _binding(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).resolve(strict=True)
        for name in (
            "lock",
            "source_manifest",
            "parent_campaign",
            "parent_completion",
            "mammoth_source",
            "data_root",
            "data_manifest",
            "environment_manifest",
            "mechanism_python",
            "transformer_python",
            "transformer_config",
            "transformer_correction_manifest",
            "transformer_correction_patch",
            "asset_manifest",
            "parent_transformer_result",
        )
    }
    files = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    return {
        "schema_version": "gradient_reuse.reviewer_extension_run_binding.v1",
        "status": "LOCKED_BEFORE_EXTENSION_OUTCOMES",
        "repo": str(repo),
        "paths": {name: str(path) for name, path in paths.items()},
        "file_sha256": files,
        "mammoth_root": str(Path(args.mammoth_root).resolve()),
        "external_campaign_root": str(Path(args.external_campaign_root).resolve()),
        "out": str(Path(args.out).resolve()),
        "device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lock_identity": lock_identity(paths["lock"]),
    }


def _write_or_validate_binding(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError("reviewer-extension run binding changed")
    else:
        atomic_json(path, dict(value))


def _validate_terminal(path: Path, schema: str) -> dict[str, Any]:
    payload = strict_json(path)
    if not isinstance(payload, dict) or not (
        payload.get("schema_version") == schema and payload.get("status") == "PASS"
    ):
        raise RuntimeError(f"terminal artifact failed authentication: {path}")
    return payload


def _e7_live_validation_context(
    args: argparse.Namespace,
    *,
    lock_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    """Rebuild E7's exact live input, parent, and correction bindings."""

    from experiments.gradient_reuse_transformer.run import read_config

    from .transformer_stream2 import (
        _validate_parent_result,
        activated_correction_provenance,
        authenticate_correction_files,
        build_e7_input_bundle,
    )

    config_path = Path(args.transformer_config)
    asset_manifest = Path(args.asset_manifest)
    parent_completion = Path(args.parent_completion)
    parent_result_path = Path(args.parent_transformer_result)
    correction_manifest = Path(args.transformer_correction_manifest)
    correction_patch = Path(args.transformer_correction_patch)
    input_bundle = build_e7_input_bundle(
        {
            "reviewer_extension_lock": lock_path,
            "reviewer_extension_source_manifest": source_path,
            "transformer_configuration": config_path,
            "transformer_asset_manifest": asset_manifest,
            "parent_completion": parent_completion,
            "parent_transformer_result": parent_result_path,
            "transformer_correction_manifest": correction_manifest,
            "transformer_correction_patch": correction_patch,
        }
    )
    _, parent_binding = _validate_parent_result(
        parent_completion,
        parent_result_path,
        config=read_config(config_path),
        config_sha256=sha256_file(config_path),
    )
    correction = activated_correction_provenance(
        authenticate_correction_files(correction_manifest, correction_patch)
    )
    return {
        "input_bundle": input_bundle,
        "parent_binding": parent_binding,
        "correction_provenance": correction,
    }


def _existing_transformer_success(
    stream_root: Path,
    *,
    live_context: Mapping[str, Any],
) -> Path | None:
    """Return one sealed E7 attempt, while preserving every failed attempt."""

    from .transformer_stream2 import validate_stream2_result

    successes: list[Path] = []
    for attempt in sorted(stream_root.glob("attempt_*")):
        if not attempt.is_dir() or attempt.is_symlink():
            raise RuntimeError(f"unsafe E7 attempt path: {attempt}")
        result = attempt / "RESULT.json"
        if result.exists() or result.is_symlink():
            if result.is_symlink() or not result.is_file():
                raise RuntimeError(f"unsafe E7 result path: {result}")
            payload = strict_json(result)
            if not isinstance(payload, dict):
                raise RuntimeError(f"E7 result is not an object: {result}")
            validate_stream2_result(
                payload,
                expected_input_bundle=live_context["input_bundle"],
            )
            if payload.get("parent") != live_context["parent_binding"]:
                raise RuntimeError("existing E7 result has a stale parent binding")
            if (
                payload.get("correction_provenance")
                != live_context["correction_provenance"]
            ):
                raise RuntimeError("existing E7 result has stale correction provenance")
            successes.append(result)
    if len(successes) > 1:
        raise RuntimeError("multiple successful E7 trajectories exist")
    return successes[0] if successes else None


def _next_transformer_attempt(stream_root: Path) -> tuple[int, Path, Path]:
    indices: list[int] = []
    if stream_root.exists() and (not stream_root.is_dir() or stream_root.is_symlink()):
        raise RuntimeError("E7 attempt root is not a real directory")
    for path in stream_root.glob("attempt_*"):
        suffix = path.name.removeprefix("attempt_")
        if not suffix.isdigit():
            raise RuntimeError(f"unexpected E7 attempt path: {path}")
        indices.append(int(suffix))
    index = max(indices, default=-1) + 1
    attempt = stream_root / f"attempt_{index}"
    return index, attempt / "RESULT.json", attempt / "HEARTBEAT.json"


def _run_or_reuse_transformer(
    args: argparse.Namespace,
    *,
    repo: Path,
    lock_path: Path,
    source_path: Path,
    out: Path,
    update: Any,
) -> Path:
    stream_root = out / "transformer_stream2"
    live_context = _e7_live_validation_context(
        args, lock_path=lock_path, source_path=source_path
    )
    existing = _existing_transformer_success(stream_root, live_context=live_context)
    if existing is not None:
        update("SECOND_PYTHIA_STREAM_REUSED", {"result": str(existing)})
        return existing

    scheduler = load_lock(lock_path)["runtime"]["scheduler"]
    _wait_for_exclusive_pythia(
        device="cuda:0",
        required_mib=int(scheduler["minimum_free_gpu_mib_before_pythia"]),
        poll_seconds=int(scheduler["poll_seconds"]),
        update=update,
    )

    attempt_index, transformer_output, heartbeat = _next_transformer_attempt(
        stream_root
    )
    update(
        "SECOND_PYTHIA_STREAM",
        {
            "attempt_index": attempt_index,
            "result": str(transformer_output),
            "heartbeat": str(heartbeat),
        },
    )
    transformer_command = [
        str(args.transformer_python),
        "-I",
        str(
            repo
            / "experiments/gradient_reuse_reviewer_extension/transformer_bootstrap.py"
        ),
        "--correction-manifest",
        str(args.transformer_correction_manifest),
        "--correction-patch",
        str(args.transformer_correction_patch),
        "--",
        "--lock",
        str(lock_path),
        "--source-manifest",
        str(source_path),
        "--config",
        str(args.transformer_config),
        "--parent-completion",
        str(args.parent_completion),
        "--parent-transformer-result",
        str(args.parent_transformer_result),
        "--asset-manifest",
        str(args.asset_manifest),
        "--out",
        str(transformer_output),
        "--heartbeat",
        str(heartbeat),
        "--device",
        "cuda",
    ]
    transformer_environment = _environment(repo)
    transformer_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _run_logged(
        transformer_command,
        log=out / "logs" / f"transformer_stream2_attempt_{attempt_index}.log",
        cwd=repo,
        environment=transformer_environment,
    )
    _validate_terminal(
        transformer_output,
        "gradient_reuse.reviewer_extension.transformer_stream2.v1",
    )
    # Apply the complete semantic validator before downstream aggregation.
    final_live_context = _e7_live_validation_context(
        args, lock_path=lock_path, source_path=source_path
    )
    if final_live_context != live_context:
        raise RuntimeError("an E7 live input changed during Transformer execution")
    existing = _existing_transformer_success(
        stream_root, live_context=final_live_context
    )
    if existing != transformer_output:
        raise RuntimeError("new E7 result was not the unique authenticated success")
    return transformer_output


def _validate_existing_queue_done(
    args: argparse.Namespace,
    *,
    lock_path: Path,
    source_path: Path,
    units: Sequence[ExtensionUnit],
    out: Path,
) -> Path:
    """Rebuild every terminal binding before accepting a prior queue PASS."""

    done_path = out / "DONE.json"
    done = _validate_terminal(done_path, DONE_SCHEMA)
    e2_path = out / "secondary" / "E2_EXISTING_RUN_REANALYSIS.json"
    _validate_terminal(e2_path, "gradient_reuse.reviewer_extension_e2.v1")
    transformer_output = _existing_transformer_success(
        out / "transformer_stream2",
        live_context=_e7_live_validation_context(
            args, lock_path=lock_path, source_path=source_path
        ),
    )
    if transformer_output is None:
        raise RuntimeError("queue DONE exists without an authenticated E7 attempt")
    external = validate_external_campaign_completion(
        Path(args.external_campaign_root)
    )
    overlay = verify_installed_overlay(Path(args.mammoth_root))

    from .aggregate import run as run_aggregate

    run_aggregate(
        out=out,
        lock_path=lock_path,
        source_manifest=source_path,
        parent_completion=Path(args.parent_completion),
        secondary_output=e2_path,
        transformer_output=transformer_output,
        transformer_config=Path(args.transformer_config),
        asset_manifest=Path(args.asset_manifest),
        parent_transformer_result=Path(args.parent_transformer_result),
        transformer_correction_manifest=Path(args.transformer_correction_manifest),
        transformer_correction_patch=Path(args.transformer_correction_patch),
    )
    aggregate_done = out / "aggregate" / "DONE.json"
    _validate_terminal(
        aggregate_done,
        "gradient_reuse.reviewer_extension_aggregate_done.v1",
    )
    artifacts = {
        "run_binding": sha256_file(out / "RUN_BINDING.json"),
        "e2_secondary": sha256_file(e2_path),
        "transformer_stream2": sha256_file(transformer_output),
        "aggregate_done": sha256_file(aggregate_done),
        "base_overlay_manifest": sha256_file(
            Path(args.mammoth_root) / BASE_MANIFEST_NAME
        ),
        "external_campaign_aggregate": external["aggregate_sha256"],
        "unit_done_bundle": canonical_sha256(
            {
                unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json")
                for unit in units
            }
        ),
    }
    if not (
        done.get("profile_id") == "iclr_reviewer_extension_v1"
        and done.get("completed_cl_units") == EXPECTED_CL_UNITS
        and done.get("completed_transformer_trajectories") == 1
        and done.get("cpu_reanalysis_status") == "PASS"
        and done.get("external_campaign_wait_binding") == external
        and done.get("overlay_binding") == overlay
        and done.get("artifact_sha256") == artifacts
        and done.get("artifact_bundle_sha256") == canonical_sha256(artifacts)
        and isinstance(done.get("completed_unix"), (int, float))
    ):
        raise RuntimeError("existing reviewer-extension queue DONE changed")
    return done_path


def run(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[2]
    raw_lock = Path(args.lock).expanduser()
    raw_source = Path(args.source_manifest).expanduser()
    lock = load_lock(raw_lock)
    validate_source_manifest(raw_source, lock_path=raw_lock, repo=repo)
    lock_path = raw_lock.resolve(strict=True)
    source_path = raw_source.resolve(strict=True)
    validate_parent_completion(Path(args.parent_completion))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        lock["runtime"]["cuda_visible_devices"]
    ):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the frozen GPU binding")
    for name in (
        "parent_campaign",
        "parent_completion",
        "mammoth_source",
        "data_root",
        "data_manifest",
        "environment_manifest",
        "transformer_config",
        "transformer_correction_manifest",
        "transformer_correction_patch",
        "asset_manifest",
        "parent_transformer_result",
        "external_campaign_root",
    ):
        raw_path = Path(getattr(args, name)).expanduser()
        if raw_path.is_symlink() or not raw_path.exists():
            raise RuntimeError(f"bound {name} path is absent or symbolic: {raw_path}")
    if str(args.device) != str(lock["runtime"]["device"]):
        raise RuntimeError("queue device differs from the frozen lock")
    units = enumerate_units(lock)
    raw_out = Path(args.out).expanduser()
    if raw_out.is_symlink():
        raise RuntimeError("reviewer-extension output root must not be symbolic")
    out = raw_out.resolve()
    orphans = reviewer_extension_orphan_processes(out)
    if orphans:
        details = "\n".join(f"{pid} {command}" for pid, command in orphans)
        raise RuntimeError(
            "orphan reviewer-extension scientific process is still active for "
            f"{out}:\n{details}"
        )
    out.mkdir(parents=True, exist_ok=True)
    binding = _binding(args, repo)
    _write_or_validate_binding(out / "RUN_BINDING.json", binding)
    state_path = out / "QUEUE_STATE.json"

    def update(stage: str, details: Mapping[str, Any] | None = None) -> None:
        atomic_json(
            state_path,
            {
                "schema_version": QUEUE_SCHEMA,
                "status": "RUNNING",
                "stage": stage,
                "updated_unix": time.time(),
                "queue_pid": os.getpid(),
                "completed_cl_units": _count_completed(out),
                "expected_cl_units": EXPECTED_CL_UNITS,
                "details": dict(details or {}),
            },
        )

    done_path = out / "DONE.json"
    if done_path.exists() or done_path.is_symlink():
        return _validate_existing_queue_done(
            args,
            lock_path=lock_path,
            source_path=source_path,
            units=units,
            out=out,
        )
    e2_process: subprocess.Popen[Any] | None = None
    try:
        update("STARTING_E2")
        e2_process = _start_e2(args, repo, out)
        external = _wait_for_external_campaign(
            Path(args.external_campaign_root),
            poll_seconds=int(lock["runtime"]["scheduler"]["poll_seconds"]),
            update=update,
            watched_e2_process=e2_process,
        )
        update("AUTHENTICATING_E2", {"external_campaign": external})
        e2_path = _finish_e2(e2_process, out)
        e2_process = None

        scheduler = lock["runtime"]["scheduler"]
        disk_gate = disk_gate_status(
            out,
            required_gib=int(scheduler["minimum_free_disk_gib_before_campaign"]),
        )
        update("DISK_GATE", disk_gate)
        if disk_gate["status"] != "PASS":
            raise RuntimeError(
                "reviewer-extension disk gate failed: "
                f"{disk_gate['observed_free_disk_gib']} GiB free; "
                f"{disk_gate['required_free_disk_gib']} GiB required"
            )
        _wait_for_gpu_free(
            device=str(args.device),
            required_mib=int(scheduler["minimum_free_gpu_mib_before_campaign"]),
            poll_seconds=int(scheduler["poll_seconds"]),
            stage="WAITING_FOR_GRADIENT_GPU_GATE",
            update=update,
        )

        update("PREPARING_PINNED_MAMMOTH")
        _clone_pinned(Path(args.mammoth_source), Path(args.mammoth_root))
        overlay = install_overlay(Path(args.mammoth_root))
        verify_installed_overlay(Path(args.mammoth_root))

        update("MAMMOTH_PARSER_PREFLIGHT")
        _probe_mammoth_cli(
            units,
            args,
            lock=lock,
            repo=repo,
            out=out,
        )

        _run_cl_scheduler(
            units,
            args,
            lock=lock,
            repo=repo,
            out=out,
            update=update,
        )

        transformer_output = _run_or_reuse_transformer(
            args,
            repo=repo,
            lock_path=lock_path,
            source_path=source_path,
            out=out,
            update=update,
        )

        update("AGGREGATING")
        aggregate_command = [
            str(args.mechanism_python),
            "-m",
            "experiments.gradient_reuse_reviewer_extension.aggregate",
            "--lock",
            str(lock_path),
            "--source-manifest",
            str(source_path),
            "--parent-completion",
            str(args.parent_completion),
            "--secondary-output",
            str(e2_path),
            "--transformer-output",
            str(transformer_output),
            "--transformer-config",
            str(args.transformer_config),
            "--asset-manifest",
            str(args.asset_manifest),
            "--parent-transformer-result",
            str(args.parent_transformer_result),
            "--transformer-correction-manifest",
            str(args.transformer_correction_manifest),
            "--transformer-correction-patch",
            str(args.transformer_correction_patch),
            "--out",
            str(out),
        ]
        _run_logged(
            aggregate_command,
            log=out / "logs" / "aggregate.log",
            cwd=repo,
            environment=_environment(repo),
        )
        aggregate_done = out / "aggregate" / "DONE.json"
        _validate_terminal(
            aggregate_done,
            "gradient_reuse.reviewer_extension_aggregate_done.v1",
        )
        artifacts = {
            "run_binding": sha256_file(out / "RUN_BINDING.json"),
            "e2_secondary": sha256_file(e2_path),
            "transformer_stream2": sha256_file(transformer_output),
            "aggregate_done": sha256_file(aggregate_done),
            "base_overlay_manifest": sha256_file(
                Path(args.mammoth_root) / BASE_MANIFEST_NAME
            ),
            "external_campaign_aggregate": external["aggregate_sha256"],
            "unit_done_bundle": canonical_sha256(
                {
                    unit.unit_id: sha256_file(
                        out / "units" / unit.unit_id / "DONE.json"
                    )
                    for unit in units
                }
            ),
        }
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "profile_id": "iclr_reviewer_extension_v1",
            "completed_cl_units": EXPECTED_CL_UNITS,
            "completed_transformer_trajectories": 1,
            "cpu_reanalysis_status": "PASS",
            "external_campaign_wait_binding": external,
            "overlay_binding": overlay,
            "artifact_sha256": artifacts,
            "artifact_bundle_sha256": canonical_sha256(artifacts),
            "completed_unix": time.time(),
        }
        atomic_json(done_path, done)
        atomic_json(
            state_path,
            {
                "schema_version": QUEUE_SCHEMA,
                "status": "PASS",
                "stage": "COMPLETE",
                "updated_unix": time.time(),
                "queue_pid": os.getpid(),
                "completed_cl_units": EXPECTED_CL_UNITS,
                "expected_cl_units": EXPECTED_CL_UNITS,
                "details": {"done": str(done_path)},
            },
        )
        return done_path
    except BaseException as error:
        if e2_process is not None:
            # Avoid an untracked duplicate re-analysis on a later queue retry.
            # E2 is read-only and bounded, so wait for it to seal or fail.
            e2_status: Any = e2_process.wait()
            handle = getattr(e2_process, "_reviewer_log_handle", None)
            if handle is not None:
                handle.close()
        else:
            e2_status = None
        failure = {
            "schema_version": "gradient_reuse.reviewer_extension_failure.v1",
            "status": "FAILED",
            "failed_unix": time.time(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "e2": e2_status,
            "completed_cl_units": _count_completed(out),
        }
        atomic_json(out / "FAILURE.json", failure)
        atomic_json(
            state_path,
            {
                "schema_version": QUEUE_SCHEMA,
                "status": "FAILED",
                "stage": "FAILED",
                "updated_unix": time.time(),
                "queue_pid": os.getpid(),
                "completed_cl_units": _count_completed(out),
                "expected_cl_units": EXPECTED_CL_UNITS,
                "details": {"failure": str(out / "FAILURE.json")},
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--parent-campaign", type=Path, required=True)
    parser.add_argument("--parent-completion", type=Path, required=True)
    parser.add_argument("--parent-transformer-result", type=Path, required=True)
    parser.add_argument("--mammoth-source", type=Path, required=True)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--mechanism-python", type=Path, required=True)
    parser.add_argument("--transformer-python", type=Path, required=True)
    parser.add_argument("--transformer-config", type=Path, required=True)
    public_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--transformer-correction-manifest",
        type=Path,
        default=public_root / "provenance/transformer_correction_portable.json",
    )
    parser.add_argument(
        "--transformer-correction-patch",
        type=Path,
        default=(
            public_root
            / "experiments/gradient_reuse_reviewer_extension/"
            "transformer_correction.py"
        ),
    )
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--external-campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    out = Path(arguments.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.parent / f".{out.name}.reviewer-extension-queue.lock"
    descriptor = _open_queue_lock(lock_path)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another queue already owns {out}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\nout={out}\n".encode("utf-8"))
        os.fsync(descriptor)
        done = run(arguments)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    print(done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
