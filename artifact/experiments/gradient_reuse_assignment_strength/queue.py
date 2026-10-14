"""Authenticated, resumable queue for the assignment-strength sweep.

The queue is deliberately operational: the scientific unit definition lives in
``protocol.py`` and every trajectory is executed and sealed by
``unit_runner.py``.  Scheduling depends only on the frozen unit identities and
resource availability.  No scientific outcome is read to choose a later arm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from .integrity import (
    atomic_json,
    executable_entry,
    git_output,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from .protocol import (
    BENCHMARKS,
    EXPECTED_UNITS,
    SweepUnit,
    canonical_sha256,
    enumerate_units,
    load_lock,
    sha256_file,
    strict_json,
)
from .source import validate_source_manifest
from .unit_runner import effective_args, validate_completed_unit


STATE_SCHEMA = "gradient_reuse.assignment_strength_queue_state.v1"
FAILURE_SCHEMA = "gradient_reuse.assignment_strength_queue_failure.v1"
DONE_SCHEMA = "gradient_reuse.assignment_strength_queue_done.v1"
AGGREGATE_SCHEMA = "gradient_reuse.assignment_strength_aggregate.v1"
BINDING_SCHEMA = "gradient_reuse.assignment_strength_run_binding.v1"

POLL_SECONDS = 10
DEVICE = "cuda:0"
CUDA_VISIBLE_DEVICE = "0"

ORPHAN_MARKERS = (
    "experiments.gradient_reuse_assignment_strength.unit_runner",
    "experiments.gradient_reuse_assignment_strength.mammoth_bootstrap",
    "experiments.gradient_reuse_assignment_strength.aggregate",
)


@dataclass
class ActiveUnit:
    unit: SweepUnit
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: Any
    started_unix: float


def scheduler_key(unit: SweepUnit) -> tuple[Any, ...]:
    """A frozen, outcome-independent launch order."""

    benchmark_order = {name: index for index, name in enumerate(BENCHMARKS)}
    method_order = {"reference": 0, "experience_replay": 1}
    strength = -1 if unit.lambda_bp is None else int(unit.lambda_bp)
    replica = -1 if unit.permutation_replica is None else int(unit.permutation_replica)
    return (
        int(unit.seed),
        benchmark_order[unit.benchmark],
        method_order[unit.method],
        strength,
        replica,
        unit.unit_id,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _output_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise RuntimeError("assignment-strength output root must be absolute")
    if raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError(f"unsafe assignment-strength output root: {raw}")
    parent = real_directory(raw.parent, "assignment-strength output parent")
    resolved = parent / raw.name
    if resolved in {Path("/"), parent}:
        raise RuntimeError("assignment-strength output root is too broad")
    resolved.mkdir(mode=0o700, exist_ok=True)
    return resolved.resolve(strict=True)


def _open_queue_lock(path: Path) -> int:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe assignment-strength queue lock: {path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RuntimeError("assignment-strength queue lock is not a singly linked file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another assignment-strength queue owns this output") from error
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


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


def _orphan_processes(out: Path) -> list[tuple[int, str]]:
    own_pid = os.getpid()
    identity = str(out)
    return [
        (pid, command)
        for pid, command in _process_rows()
        if pid != own_pid
        and identity in command
        and any(marker in command for marker in ORPHAN_MARKERS)
    ]


def _validate_output_topology(out: Path) -> None:
    allowed_directories = {"units", "logs", "failures"}
    allowed_files = {
        "RUN_BINDING.json",
        "QUEUE_STATE.json",
        "FAILURE.json",
        "ASSIGNMENT_STRENGTH_AGGREGATE.json",
        "DONE.json",
    }
    for path in out.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"assignment-strength output contains a symlink: {path}")
        if path.name in allowed_directories and path.is_dir():
            continue
        if path.name in allowed_files and path.is_file():
            continue
        raise RuntimeError(f"unexpected assignment-strength output artifact: {path}")


def _free_disk_gib(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail * value.f_frsize // (1024**3))


def _free_gpu_mib() -> int:
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
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"unexpected nvidia-smi memory output: {rows!r}")
    try:
        value = int(rows[0])
    except ValueError as error:
        raise RuntimeError(f"unexpected nvidia-smi memory output: {rows!r}") from error
    if value < 0:
        raise RuntimeError("nvidia-smi reported negative free memory")
    return value


def _environment(project: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICE,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    return value


def _expected_binding(
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    data: Mapping[str, Any],
    environment_manifest_path: Path,
    environment: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    # This is intentionally identical to the object independently rebuilt by
    # unit_runner.py.  Adding operational fields here would invalidate units.
    return {
        "schema_version": BINDING_SCHEMA,
        "status": "PASS",
        "project": str(project),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_root": str(mammoth),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_root": str(data_root),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "device": DEVICE,
        "python_entry": str(python),
        "expected_units": EXPECTED_UNITS,
    }


def _write_or_validate(path: Path, value: Mapping[str, Any], role: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError(f"existing {role} changed")
        return
    atomic_json(path, dict(value))


def _unit_identity(
    unit: SweepUnit,
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    mammoth: Path,
    data_manifest_path: Path,
    data: Mapping[str, Any],
    environment_manifest_path: Path,
    environment: Mapping[str, Any],
    python: Path,
    binding_path: Path,
) -> dict[str, Any]:
    effective = effective_args(unit, lock, DEVICE)
    return {
        "schema_version": "gradient_reuse.assignment_strength_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "binding_sha256": sha256_file(binding_path),
        "effective_args_sha256": canonical_sha256(effective),
        "device": DEVICE,
        "paper_eligible": True,
    }


def _validate_partial_unit(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"unsafe partial unit root: {root}")
    indexes: list[int] = []
    for path in root.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"partial unit contains a symlink: {path}")
        if path.name == "UNIT.json" and path.is_file():
            continue
        if (
            path.is_dir()
            and path.name.startswith("attempt_")
            and path.name[8:].isdigit()
            and path.name == f"attempt_{int(path.name[8:]):03d}"
        ):
            indexes.append(int(path.name[8:]))
            for nested in path.rglob("*"):
                if nested.is_symlink():
                    raise RuntimeError(f"partial unit contains a nested symlink: {nested}")
            continue
        raise RuntimeError(f"unexpected partial unit artifact: {path}")
    if indexes and sorted(indexes) != list(range(1, max(indexes) + 1)):
        raise RuntimeError(f"partial unit attempts are noncontiguous: {root}")


def _completed_units(
    units: Sequence[SweepUnit],
    *,
    units_root: Path,
    identities: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    units_root.mkdir(parents=True, exist_ok=True)
    expected = {unit.unit_id for unit in units}
    lock_names = {f".{identifier}.lock" for identifier in expected}
    for path in units_root.iterdir():
        if path.name in expected:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"unsafe assignment-strength unit root: {path}")
        elif path.name in lock_names:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe assignment-strength unit lock: {path}")
        else:
            raise RuntimeError(f"unexpected assignment-strength units entry: {path}")
    completed: set[str] = set()
    for unit in units:
        root = units_root / unit.unit_id
        if not root.exists() and not root.is_symlink():
            continue
        if (root / "DONE.json").exists() or (root / "DONE.json").is_symlink():
            validate_completed_unit(root, identities[unit.unit_id])
            completed.add(unit.unit_id)
        else:
            _validate_partial_unit(root)
    return completed


def _unit_command(
    unit: SweepUnit,
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    environment_manifest_path: Path,
    binding_path: Path,
    python: Path,
    out: Path,
) -> list[str]:
    return [
        str(python),
        "-B",
        "-m",
        "experiments.gradient_reuse_assignment_strength.unit_runner",
        "--project",
        str(project),
        "--lock",
        str(lock_path),
        "--source-manifest",
        str(source_manifest_path),
        "--mammoth-root",
        str(mammoth),
        "--data-root",
        str(data_root),
        "--data-manifest",
        str(data_manifest_path),
        "--environment-manifest",
        str(environment_manifest_path),
        "--binding",
        str(binding_path),
        "--python",
        str(python),
        "--out",
        str(out),
        "--unit-id",
        unit.unit_id,
        "--device",
        DEVICE,
    ]


def _launch_unit(
    unit: SweepUnit,
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    environment_manifest_path: Path,
    binding_path: Path,
    python: Path,
    out: Path,
) -> ActiveUnit:
    log_path = out / "logs" / "units" / f"{unit.unit_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
        raise RuntimeError(f"unsafe assignment-strength unit log: {log_path}")
    handle = log_path.open("a", encoding="utf-8")
    command = _unit_command(
        unit,
        project=project,
        lock_path=lock_path,
        source_manifest_path=source_manifest_path,
        mammoth=mammoth,
        data_root=data_root,
        data_manifest_path=data_manifest_path,
        environment_manifest_path=environment_manifest_path,
        binding_path=binding_path,
        python=python,
        out=out,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=project,
            env=_environment(project),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        handle.close()
        raise
    return ActiveUnit(unit, process, log_path, handle, time.time())


def _next_attempt(state_path: Path) -> int:
    attempts: list[int] = []
    if state_path.exists() or state_path.is_symlink():
        if state_path.is_symlink() or not state_path.is_file():
            raise RuntimeError(f"unsafe queue state path: {state_path}")
        value = strict_json(state_path)
        attempt = value.get("queue_attempt") if isinstance(value, Mapping) else None
        if not (
            isinstance(value, Mapping)
            and value.get("schema_version") == STATE_SCHEMA
            and isinstance(attempt, int)
            and not isinstance(attempt, bool)
            and attempt >= 1
        ):
            raise RuntimeError("existing assignment-strength queue state changed")
        attempts.append(attempt)
    failures = state_path.parent / "failures"
    if failures.exists() or failures.is_symlink():
        if failures.is_symlink() or not failures.is_dir():
            raise RuntimeError(f"unsafe queue failure directory: {failures}")
        for path in failures.iterdir():
            if path.is_symlink() or not path.is_file() or not (
                path.name.startswith("FAILURE_")
                and path.name.endswith(".json")
                and path.name[8:-5].isdigit()
            ):
                raise RuntimeError(f"unexpected queue failure artifact: {path}")
            value = strict_json(path)
            attempt = value.get("queue_attempt") if isinstance(value, Mapping) else None
            if not (
                isinstance(value, Mapping)
                and value.get("schema_version") == FAILURE_SCHEMA
                and value.get("status") == "FAILED"
                and isinstance(attempt, int)
                and not isinstance(attempt, bool)
                and int(path.name[8:-5]) == attempt
            ):
                raise RuntimeError(f"queue failure record changed: {path}")
            attempts.append(attempt)
    return max(attempts, default=0) + 1


def _clear_current_failure(out: Path) -> None:
    path = out / "FAILURE.json"
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe current failure path: {path}")
    value = strict_json(path)
    attempt = value.get("queue_attempt") if isinstance(value, Mapping) else None
    numbered = out / "failures" / f"FAILURE_{int(attempt):03d}.json" if isinstance(attempt, int) and not isinstance(attempt, bool) else None
    if not (
        isinstance(value, Mapping)
        and value.get("schema_version") == FAILURE_SCHEMA
        and value.get("status") == "FAILED"
        and numbered is not None
        and numbered.is_file()
        and not numbered.is_symlink()
        and strict_json(numbered) == value
    ):
        raise RuntimeError("current failure is not backed by an authenticated numbered record")
    path.unlink()


def _record_failure(
    out: Path,
    *,
    queue_attempt: int,
    completed: set[str],
    error: BaseException,
) -> Path:
    failure = {
        "schema_version": FAILURE_SCHEMA,
        "status": "FAILED",
        "queue_attempt": queue_attempt,
        "failed_unix": time.time(),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "completed_units": len(completed),
        "expected_units": EXPECTED_UNITS,
        "completed_unit_ids": sorted(completed),
    }
    path = out / "failures" / f"FAILURE_{queue_attempt:03d}.json"
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"numbered failure record already exists: {path}")
    atomic_json(path, failure)
    atomic_json(out / "FAILURE.json", failure)
    return path


def _counts_by_benchmark(units: Sequence[SweepUnit], completed: set[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for benchmark in BENCHMARKS:
        rows = [unit for unit in units if unit.benchmark == benchmark]
        output[benchmark] = {
            "completed": sum(unit.unit_id in completed for unit in rows),
            "expected": len(rows),
        }
    return output


def _run_scheduler(
    units: Sequence[SweepUnit],
    completed: set[str],
    *,
    project: Path,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    data: Mapping[str, Any],
    environment_manifest_path: Path,
    environment: Mapping[str, Any],
    binding_path: Path,
    python: Path,
    out: Path,
    max_processes: int,
    update: Any,
) -> None:
    identities = {
        unit.unit_id: _unit_identity(
            unit,
            lock=lock,
            lock_path=lock_path,
            source_manifest_path=source_manifest_path,
            source=source,
            mammoth=mammoth,
            data_manifest_path=data_manifest_path,
            data=data,
            environment_manifest_path=environment_manifest_path,
            environment=environment,
            python=python,
            binding_path=binding_path,
        )
        for unit in units
    }
    pending = [
        unit for unit in sorted(units, key=scheduler_key) if unit.unit_id not in completed
    ]
    active: list[ActiveUnit] = []
    failed: list[dict[str, Any]] = []
    initial_gate = int(lock["runtime"]["minimum_free_gpu_memory_mib_before_first_launch"])
    additional_gate = int(lock["runtime"]["minimum_free_gpu_memory_mib_per_additional_launch"])
    minimum_disk = int(lock["runtime"]["minimum_free_disk_gib"])

    def validate(unit: SweepUnit) -> None:
        validate_completed_unit(out / "units" / unit.unit_id, identities[unit.unit_id])

    try:
        while pending or active:
            survivors: list[ActiveUnit] = []
            for running in active:
                returncode = running.process.poll()
                if returncode is None:
                    survivors.append(running)
                    continue
                running.log_handle.close()
                if returncode != 0:
                    failed.append(
                        {
                            "unit_id": running.unit.unit_id,
                            "returncode": returncode,
                            "log": str(running.log_path),
                        }
                    )
                    continue
                try:
                    validate(running.unit)
                    completed.add(running.unit.unit_id)
                except BaseException as error:
                    failed.append(
                        {
                            "unit_id": running.unit.unit_id,
                            "returncode": returncode,
                            "log": str(running.log_path),
                            "validation_error": f"{type(error).__name__}: {error}",
                        }
                    )
            active = survivors
            if failed:
                if active:
                    update(
                        "DRAINING_AFTER_UNIT_FAILURE",
                        {
                            "failed_units": failed,
                            "active_units": [row.unit.unit_id for row in active],
                        },
                    )
                    time.sleep(POLL_SECONDS)
                    continue
                raise RuntimeError(f"assignment-strength unit failure(s): {failed}")

            if not pending and not active:
                update("UNITS_COMPLETE", {"pending_units": 0, "active_units": []})
                break

            free_disk = _free_disk_gib(out)
            if free_disk < minimum_disk:
                raise RuntimeError(
                    f"disk gate failed: {free_disk} GiB free; {minimum_disk} GiB required"
                )
            free_gpu = _free_gpu_mib()
            # If this campaign currently owns no GPU process, require the
            # stronger admission gate again.  This prevents a later external
            # process from turning a resumed launch into accidental sharing.
            required_gpu = additional_gate if active else initial_gate
            launched: list[str] = []
            if pending and len(active) < max_processes and free_gpu >= required_gpu:
                # Launch at most one child per fresh GPU-memory observation.
                unit = pending.pop(0)
                active.append(
                    _launch_unit(
                        unit,
                        project=project,
                        lock_path=lock_path,
                        source_manifest_path=source_manifest_path,
                        mammoth=mammoth,
                        data_root=data_root,
                        data_manifest_path=data_manifest_path,
                        environment_manifest_path=environment_manifest_path,
                        binding_path=binding_path,
                        python=python,
                        out=out,
                    )
                )
                launched.append(unit.unit_id)
            stage = "RUNNING_UNITS" if active else "WAITING_FOR_GPU_GATE"
            update(
                stage,
                {
                    "pending_units": len(pending),
                    "active_units": [
                        {
                            "unit_id": row.unit.unit_id,
                            "pid": row.process.pid,
                            "log": str(row.log_path),
                            "started_unix": row.started_unix,
                        }
                        for row in active
                    ],
                    "launched_this_cycle": launched,
                    "free_gpu_mib": free_gpu,
                    "required_free_gpu_mib_for_next_launch": (
                        additional_gate if active else initial_gate
                    ),
                    "free_disk_gib": free_disk,
                    "required_free_disk_gib": minimum_disk,
                    "maximum_concurrent_gpu_processes": max_processes,
                },
            )
            time.sleep(POLL_SECONDS)
    except BaseException:
        # Never terminate an already-running scientific trajectory because the
        # queue itself or a sibling unit failed.  Drain and seal what survives.
        while active:
            survivors = []
            for running in active:
                returncode = running.process.poll()
                if returncode is None:
                    survivors.append(running)
                    continue
                running.log_handle.close()
                if returncode == 0:
                    try:
                        validate(running.unit)
                        completed.add(running.unit.unit_id)
                    except BaseException:
                        pass
            active = survivors
            if active:
                update(
                    "DRAINING_AFTER_QUEUE_ERROR",
                    {"active_units": [row.unit.unit_id for row in active]},
                )
                time.sleep(POLL_SECONDS)
        raise
    if len(completed) != EXPECTED_UNITS:
        raise RuntimeError(
            f"scheduler sealed {len(completed)} units, expected {EXPECTED_UNITS}"
        )


def _aggregate_command(
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    environment_manifest_path: Path,
    binding_path: Path,
    python: Path,
    out: Path,
    output: Path,
) -> list[str]:
    return [
        str(python),
        "-B",
        "-m",
        "experiments.gradient_reuse_assignment_strength.aggregate",
        "--project",
        str(project),
        "--lock",
        str(lock_path),
        "--source-manifest",
        str(source_manifest_path),
        "--mammoth-root",
        str(mammoth),
        "--data-root",
        str(data_root),
        "--data-manifest",
        str(data_manifest_path),
        "--environment-manifest",
        str(environment_manifest_path),
        "--binding",
        str(binding_path),
        "--out",
        str(out),
        "--output",
        str(output),
    ]


def _validate_aggregate(
    path: Path,
    *,
    out: Path | None = None,
    units: Sequence[SweepUnit] | None = None,
) -> dict[str, Any]:
    value = strict_json(real_file(path, "assignment-strength aggregate"))
    if not isinstance(value, Mapping) or not (
        value.get("schema_version") == AGGREGATE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("complete_protocol") is True
        and value.get("paper_eligible") is True
        and value.get("unit_count") == EXPECTED_UNITS
    ):
        raise RuntimeError("assignment-strength aggregate failed authentication")
    if (out is None) != (units is None):
        raise RuntimeError("aggregate unit validation requires both out and units")
    if out is not None and units is not None:
        unit_done = {
            unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json")
            for unit in units
        }
        if not (
            value.get("unit_done_sha256") == unit_done
            and value.get("unit_done_bundle_sha256") == canonical_sha256(unit_done)
        ):
            raise RuntimeError("assignment-strength aggregate unit inventory changed")
    return dict(value)


def _run_aggregate(
    *,
    project: Path,
    lock_path: Path,
    source_manifest_path: Path,
    mammoth: Path,
    data_root: Path,
    data_manifest_path: Path,
    environment_manifest_path: Path,
    binding_path: Path,
    python: Path,
    out: Path,
) -> Path:
    output = out / "ASSIGNMENT_STRENGTH_AGGREGATE.json"
    log = out / "logs" / "aggregate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_symlink() or (log.exists() and not log.is_file()):
        raise RuntimeError(f"unsafe assignment-strength aggregate log: {log}")
    command = _aggregate_command(
        project=project,
        lock_path=lock_path,
        source_manifest_path=source_manifest_path,
        mammoth=mammoth,
        data_root=data_root,
        data_manifest_path=data_manifest_path,
        environment_manifest_path=environment_manifest_path,
        binding_path=binding_path,
        python=python,
        out=out,
        output=output,
    )
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=project,
            env=_environment(project),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"assignment-strength aggregate exited {completed.returncode}; see {log}"
        )
    _validate_aggregate(output)
    return output


def _scientific_inventory(out: Path, units: Sequence[SweepUnit], aggregate: Path) -> dict[str, str]:
    paths = [out / "RUN_BINDING.json", aggregate]
    for unit in units:
        paths.extend(
            (
                out / "units" / unit.unit_id / "UNIT.json",
                out / "units" / unit.unit_id / "DONE.json",
            )
        )
    inventory: dict[str, str] = {}
    for path in paths:
        checked = real_file(path, "terminal scientific artifact")
        relative = checked.relative_to(out).as_posix()
        inventory[relative] = sha256_file(checked)
    return dict(sorted(inventory.items()))


def _terminal_done(
    *,
    out: Path,
    units: Sequence[SweepUnit],
    binding_path: Path,
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    aggregate: Path,
    completed_unix: float,
) -> dict[str, Any]:
    ordered_ids = [unit.unit_id for unit in units]
    unit_done = {
        unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json")
        for unit in units
    }
    inventory = _scientific_inventory(out, units, aggregate)
    return {
        "schema_version": DONE_SCHEMA,
        "status": "PASS",
        "scientific_role": "outcome_informed_prospectively_locked_assignment_strength_followup",
        "unit_count": EXPECTED_UNITS,
        "expected_units": EXPECTED_UNITS,
        "completed_units": EXPECTED_UNITS,
        "validated_unit_ids": ordered_ids,
        "validated_unit_ids_sha256": canonical_sha256(ordered_ids),
        "unit_done_sha256": unit_done,
        "unit_done_bundle_sha256": canonical_sha256(unit_done),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "run_binding_sha256": sha256_file(binding_path),
        "aggregate_sha256": sha256_file(aggregate),
        "scientific_artifact_count": len(inventory),
        "scientific_artifact_inventory": inventory,
        "scientific_artifact_bundle_sha256": canonical_sha256(inventory),
        "completed_unix": completed_unix,
    }


def _validate_done(
    path: Path,
    *,
    out: Path,
    units: Sequence[SweepUnit],
    binding_path: Path,
    lock_path: Path,
    source_manifest_path: Path,
    source: Mapping[str, Any],
    aggregate: Path,
) -> dict[str, Any]:
    value = strict_json(real_file(path, "assignment-strength terminal DONE"))
    completed_unix = value.get("completed_unix") if isinstance(value, Mapping) else None
    if not (
        isinstance(completed_unix, (int, float))
        and not isinstance(completed_unix, bool)
        and math.isfinite(float(completed_unix))
        and float(completed_unix) > 0.0
    ):
        raise RuntimeError("assignment-strength terminal completion time is invalid")
    expected = _terminal_done(
        out=out,
        units=units,
        binding_path=binding_path,
        lock_path=lock_path,
        source_manifest_path=source_manifest_path,
        source=source,
        aggregate=aggregate,
        completed_unix=float(completed_unix),
    )
    if value != expected:
        raise RuntimeError("assignment-strength terminal DONE failed authentication")
    return dict(value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--lock", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--mammoth-root", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--data-manifest", type=Path, required=True)
    result.add_argument("--environment-manifest", type=Path, required=True)
    result.add_argument("--device", default=DEVICE)
    result.add_argument("--max-processes", type=int, default=2)
    return result


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError("assignment-strength queue must run through Python -B")
    if arguments.device != DEVICE:
        raise RuntimeError("assignment-strength queue requires cuda:0")

    project = real_directory(arguments.project, "assignment-strength project")
    lock_path = real_file(arguments.lock, "assignment-strength lock")
    source_manifest_path = real_file(
        arguments.source_manifest, "assignment-strength source manifest"
    )
    expected_lock = (project / "configs/assignment_strength_sweep.json").resolve(strict=True)
    expected_source = (
        project / "provenance/assignment_strength_source_manifest.json"
    ).resolve(strict=True)
    if lock_path != expected_lock or source_manifest_path != expected_source:
        raise RuntimeError("assignment-strength lock/source paths are not the installed canonical paths")
    python = executable_entry(arguments.python, "assignment-strength Python")
    if not python.is_absolute():
        raise RuntimeError("assignment-strength Python entry must be absolute")
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_manifest_path = real_file(arguments.data_manifest, "mechanism data manifest")
    environment_manifest_path = real_file(
        arguments.environment_manifest, "mechanism environment manifest"
    )
    out = _output_root(arguments.out)

    protected = {
        "project": project,
        "Mammoth": mammoth,
        "data root": data_root,
        "lock": lock_path,
        "source manifest": source_manifest_path,
        "data manifest": data_manifest_path,
        "environment manifest": environment_manifest_path,
    }
    for role, path in protected.items():
        if _paths_overlap(out, path):
            raise RuntimeError(f"assignment-strength output overlaps {role}: {path}")

    queue_lock_path = out.parent / f".{out.name}.queue.lock"
    descriptor = _open_queue_lock(queue_lock_path)
    try:
        orphans = _orphan_processes(out)
        if orphans:
            detail = "\n".join(f"{pid} {command}" for pid, command in orphans)
            raise RuntimeError("orphan assignment-strength process is active:\n" + detail)
        _validate_output_topology(out)

        # These are the full pre-outcome gates.  They run before RUN_BINDING is
        # published and before any completed unit is accepted.
        lock = load_lock(lock_path)
        source = validate_source_manifest(source_manifest_path, project)
        data = validate_data_manifest(data_manifest_path, data_root)
        environment = validate_environment_manifest(
            environment_manifest_path, mammoth, DEVICE
        )
        units = sorted(enumerate_units(lock), key=scheduler_key)
        if len(units) != EXPECTED_UNITS or arguments.max_processes not in {1, 2}:
            raise RuntimeError("assignment-strength execution dimensions changed")
        if int(arguments.max_processes) > int(
            lock["runtime"]["maximum_concurrent_gpu_processes"]
        ):
            raise RuntimeError("requested assignment-strength process count exceeds the lock")

        binding = _expected_binding(
            project=project,
            lock_path=lock_path,
            source_manifest_path=source_manifest_path,
            source=source,
            mammoth=mammoth,
            data_root=data_root,
            data_manifest_path=data_manifest_path,
            data=data,
            environment_manifest_path=environment_manifest_path,
            environment=environment,
            python=python,
        )
        binding_path = out / "RUN_BINDING.json"
        _write_or_validate(binding_path, binding, "assignment-strength RUN_BINDING")
        identities = {
            unit.unit_id: _unit_identity(
                unit,
                lock=lock,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                source=source,
                mammoth=mammoth,
                data_manifest_path=data_manifest_path,
                data=data,
                environment_manifest_path=environment_manifest_path,
                environment=environment,
                python=python,
                binding_path=binding_path,
            )
            for unit in units
        }
        completed = _completed_units(
            units, units_root=out / "units", identities=identities
        )
        state_path = out / "QUEUE_STATE.json"
        queue_attempt = _next_attempt(state_path)

        def update(stage: str, details: Mapping[str, Any] | None = None) -> None:
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "RUNNING",
                    "stage": stage,
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": len(completed),
                    "expected_units": EXPECTED_UNITS,
                    "completed_by_benchmark": _counts_by_benchmark(units, completed),
                    "details": dict(details or {}),
                },
            )

        done_path = out / "DONE.json"
        if done_path.exists() or done_path.is_symlink():
            if len(completed) != EXPECTED_UNITS:
                raise RuntimeError("terminal DONE exists without all 432 sealed units")
            aggregate = _run_aggregate(
                project=project,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                mammoth=mammoth,
                data_root=data_root,
                data_manifest_path=data_manifest_path,
                environment_manifest_path=environment_manifest_path,
                binding_path=binding_path,
                python=python,
                out=out,
            )
            _validate_aggregate(aggregate, out=out, units=units)
            _validate_done(
                done_path,
                out=out,
                units=units,
                binding_path=binding_path,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                source=source,
                aggregate=aggregate,
            )
            _clear_current_failure(out)
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "PASS",
                    "stage": "COMPLETE",
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": EXPECTED_UNITS,
                    "expected_units": EXPECTED_UNITS,
                    "completed_by_benchmark": _counts_by_benchmark(units, completed),
                    "details": {"done": str(done_path), "aggregate": str(aggregate)},
                },
            )
            return done_path

        _clear_current_failure(out)
        try:
            update("AUTHENTICATED_INPUTS")
            _run_scheduler(
                units,
                completed,
                project=project,
                lock=lock,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                source=source,
                mammoth=mammoth,
                data_root=data_root,
                data_manifest_path=data_manifest_path,
                data=data,
                environment_manifest_path=environment_manifest_path,
                environment=environment,
                binding_path=binding_path,
                python=python,
                out=out,
                max_processes=int(arguments.max_processes),
                update=update,
            )
            completed = _completed_units(
                units, units_root=out / "units", identities=identities
            )
            if len(completed) != EXPECTED_UNITS:
                raise RuntimeError("terminal assignment-strength unit validation is incomplete")
            update("AGGREGATING")
            aggregate = _run_aggregate(
                project=project,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                mammoth=mammoth,
                data_root=data_root,
                data_manifest_path=data_manifest_path,
                environment_manifest_path=environment_manifest_path,
                binding_path=binding_path,
                python=python,
                out=out,
            )
            _validate_aggregate(aggregate, out=out, units=units)
            done = _terminal_done(
                out=out,
                units=units,
                binding_path=binding_path,
                lock_path=lock_path,
                source_manifest_path=source_manifest_path,
                source=source,
                aggregate=aggregate,
                completed_unix=time.time(),
            )
            atomic_json(done_path, done)
            _clear_current_failure(out)
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "PASS",
                    "stage": "COMPLETE",
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": EXPECTED_UNITS,
                    "expected_units": EXPECTED_UNITS,
                    "completed_by_benchmark": _counts_by_benchmark(units, completed),
                    "details": {"done": str(done_path), "aggregate": str(aggregate)},
                },
            )
            return done_path
        except BaseException as error:
            failure_path = _record_failure(
                out,
                queue_attempt=queue_attempt,
                completed=completed,
                error=error,
            )
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "FAILED",
                    "stage": "FAILED",
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": len(completed),
                    "expected_units": EXPECTED_UNITS,
                    "completed_by_benchmark": _counts_by_benchmark(units, completed),
                    "details": {"failure": str(failure_path)},
                },
            )
            raise
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parser().parse_args(argv))
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
