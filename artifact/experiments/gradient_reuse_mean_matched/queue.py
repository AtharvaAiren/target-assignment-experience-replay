"""Outcome-blind, resumable queue for the mean-matched companion."""

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

from experiments.gradient_reuse_assignment_strength.integrity import (
    atomic_json, executable_entry, git_output, real_directory, real_file,
    validate_data_manifest, validate_environment_manifest, validate_pristine_mammoth,
)
from .aggregate import AGGREGATE_NAME, SCHEMA as AGGREGATE_SCHEMA, aggregate_campaign
from .protocol import (
    BENCHMARKS, EXPECTED_UNITS, MeanMatchedUnit, canonical_sha256,
    enumerate_units, load_lock, sha256_file, strict_json,
)
from .source import validate_source_manifest
from .unit_runner import effective_args, expected_identity, validate_completed_unit


STATE_SCHEMA = "gradient_reuse.mean_matched_queue_state.v1"
FAILURE_SCHEMA = "gradient_reuse.mean_matched_queue_failure.v1"
DONE_SCHEMA = "gradient_reuse.mean_matched_queue_done.v1"
BINDING_SCHEMA = "gradient_reuse.mean_matched_run_binding.v1"
POLL_SECONDS = 10


@dataclass
class Active:
    unit: MeanMatchedUnit
    process: subprocess.Popen[Any]
    log: Path
    handle: Any
    started: float


def scheduler_key(unit: MeanMatchedUnit) -> tuple[Any, ...]:
    return (unit.seed, BENCHMARKS.index(unit.benchmark), unit.lambda_bp, unit.unit_id)


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _output_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError("unsafe mean-matched output root")
    parent = real_directory(raw.parent, "mean-matched output parent")
    output = parent / raw.name
    if output in {Path("/"), parent}:
        raise RuntimeError("mean-matched output root is too broad")
    output.mkdir(mode=0o700, exist_ok=True)
    return output.resolve(strict=True)


def _queue_lock(path: Path) -> int:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError("unsafe mean-matched queue lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("mean-matched queue lock is not singly linked")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _process_rows() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-ww", "-eo", "pid=,args="], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        head, _, command = line.strip().partition(" ")
        if head.isdigit():
            rows.append((int(head), command))
    return rows


def _orphans(out: Path) -> list[tuple[int, str]]:
    markers = (
        "experiments.gradient_reuse_mean_matched.unit_runner",
        "experiments.gradient_reuse_mean_matched.mammoth_bootstrap",
        "experiments.gradient_reuse_mean_matched.aggregate",
    )
    return [
        (pid, command) for pid, command in _process_rows()
        if pid != os.getpid() and str(out) in command
        and any(marker in command for marker in markers)
    ]


def _parent_processes(parent_out: Path) -> list[tuple[int, str]]:
    markers = (
        "gradient_reuse_assignment_strength.unit_runner",
        "gradient_reuse_assignment_strength.mammoth_bootstrap",
        "gradient_reuse_assignment_strength.queue",
    )
    return [
        (pid, command) for pid, command in _process_rows()
        if pid != os.getpid() and str(parent_out) in command
        and any(marker in command for marker in markers)
    ]


def _parent_terminal_ready(lock: Mapping[str, Any]) -> bool:
    parent_out = Path(str(lock["parent"]["output_path"]))
    done_path = parent_out / str(lock["parent"]["done_name"])
    aggregate_path = parent_out / str(lock["parent"]["aggregate_name"])
    if any(path.is_symlink() for path in (parent_out, done_path, aggregate_path)):
        raise RuntimeError("parent terminal path is symbolic")
    if not parent_out.is_dir() or not done_path.is_file() or not aggregate_path.is_file():
        return False
    done = strict_json(done_path)
    if not (
        isinstance(done, Mapping)
        and done.get("schema_version") == "gradient_reuse.assignment_strength_queue_done.v1"
        and done.get("status") == "PASS"
        and done.get("expected_units") == 432
        and done.get("completed_units") == 432
        and done.get("aggregate_sha256") == sha256_file(aggregate_path)
    ):
        raise RuntimeError("parent terminal marker is present but invalid")
    return not _parent_processes(parent_out)


def _validate_output_topology(out: Path) -> None:
    directories = {"units", "logs", "failures"}
    files = {"RUN_BINDING.json", "QUEUE_STATE.json", "FAILURE.json", AGGREGATE_NAME, "DONE.json"}
    for path in out.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"mean-matched output contains a symlink: {path}")
        if path.name in directories and path.is_dir():
            continue
        if path.name in files and path.is_file():
            continue
        raise RuntimeError(f"unexpected mean-matched output artifact: {path}")


def _environment(project: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    })
    return value


def _free_gpu_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError("unexpected nvidia-smi output")
    return int(rows[0])


def _free_disk_gib(path: Path) -> int:
    info = os.statvfs(path)
    return int(info.f_bavail * info.f_frsize // (1024 ** 3))


def _binding(*, project: Path, lock_path: Path, source_path: Path,
             source: Mapping[str, Any], mammoth: Path, data_root: Path,
             data_path: Path, data: Mapping[str, Any], environment_path: Path,
             environment: Mapping[str, Any], python: Path) -> dict[str, Any]:
    return {
        "schema_version": BINDING_SCHEMA, "status": "PASS", "project": str(project),
        "lock_sha256": sha256_file(lock_path), "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"], "mammoth_root": str(mammoth),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"), "data_root": str(data_root),
        "data_manifest_sha256": sha256_file(data_path), "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python), "python_entry": str(python),
        "device": "cuda:0", "expected_units": EXPECTED_UNITS,
    }


def _write_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError(f"existing authenticated file changed: {path}")
    else:
        atomic_json(path, dict(value))


def _identities(units: Sequence[MeanMatchedUnit], *, lock: Mapping[str, Any],
                lock_path: Path, source_path: Path, source: Mapping[str, Any],
                mammoth: Path, data_path: Path, data: Mapping[str, Any],
                environment_path: Path, environment: Mapping[str, Any],
                python: Path, binding_path: Path) -> dict[str, dict[str, Any]]:
    return {
        unit.unit_id: expected_identity(
            unit, lock_path=lock_path, source_manifest_path=source_path,
            source=source, mammoth=mammoth, data_manifest_path=data_path, data=data,
            environment_manifest_path=environment_path, environment=environment,
            python=python, binding_path=binding_path, lock=lock,
        ) for unit in units
    }


def _partial(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("unsafe partial mean-matched unit")
    indexes = []
    for path in root.iterdir():
        if path.is_symlink():
            raise RuntimeError("partial mean-matched unit contains a symlink")
        if path.name == "UNIT.json" and path.is_file():
            continue
        if path.is_dir() and path.name.startswith("attempt_") and path.name[8:].isdigit():
            indexes.append(int(path.name[8:]))
            if any(nested.is_symlink() for nested in path.rglob("*")):
                raise RuntimeError("partial attempt contains a symlink")
            continue
        raise RuntimeError(f"unexpected partial unit artifact: {path}")
    if indexes and sorted(indexes) != list(range(1, max(indexes) + 1)):
        raise RuntimeError("partial attempt numbers are noncontiguous")


def _completed(units: Sequence[MeanMatchedUnit], out: Path,
               identities: Mapping[str, Mapping[str, Any]]) -> set[str]:
    root = out / "units"
    root.mkdir(parents=True, exist_ok=True)
    expected = {unit.unit_id for unit in units}
    allowed_locks = {f".{name}.lock" for name in expected}
    for path in root.iterdir():
        if path.name in expected and path.is_dir() and not path.is_symlink():
            continue
        if path.name in allowed_locks and path.is_file() and not path.is_symlink():
            continue
        raise RuntimeError(f"unexpected mean-matched units entry: {path}")
    done = set()
    for unit in units:
        unit_root = root / unit.unit_id
        if not unit_root.exists() and not unit_root.is_symlink():
            continue
        if (unit_root / "DONE.json").exists() or (unit_root / "DONE.json").is_symlink():
            validate_completed_unit(unit_root, identities[unit.unit_id])
            done.add(unit.unit_id)
        else:
            _partial(unit_root)
    return done


def _unit_command(unit: MeanMatchedUnit, *, project: Path, lock_path: Path,
                  source_path: Path, mammoth: Path, data_root: Path,
                  data_path: Path, environment_path: Path, binding_path: Path,
                  python: Path, out: Path) -> list[str]:
    return [
        str(python), "-B", "-m", "experiments.gradient_reuse_mean_matched.unit_runner",
        "--project", str(project), "--lock", str(lock_path),
        "--source-manifest", str(source_path), "--mammoth-root", str(mammoth),
        "--data-root", str(data_root), "--data-manifest", str(data_path),
        "--environment-manifest", str(environment_path), "--binding", str(binding_path),
        "--python", str(python), "--out", str(out), "--unit-id", unit.unit_id,
        "--device", "cuda:0",
    ]


def _launch(unit: MeanMatchedUnit, **paths: Any) -> Active:
    out = paths["out"]
    log = out / "logs" / "units" / f"{unit.unit_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_symlink() or (log.exists() and not log.is_file()):
        raise RuntimeError("unsafe mean-matched unit log")
    handle = log.open("a", encoding="utf-8")
    command = _unit_command(unit, **paths)
    try:
        process = subprocess.Popen(
            command, cwd=paths["project"], env=_environment(paths["project"]),
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    except BaseException:
        handle.close()
        raise
    return Active(unit, process, log, handle, time.time())


def _attempt(state_path: Path) -> int:
    attempts = []
    if state_path.exists():
        state = strict_json(real_file(state_path, "queue state"))
        if not isinstance(state, Mapping) or state.get("schema_version") != STATE_SCHEMA:
            raise RuntimeError("existing mean-matched queue state changed")
        attempts.append(int(state["queue_attempt"]))
    failures = state_path.parent / "failures"
    if failures.exists():
        if failures.is_symlink() or not failures.is_dir():
            raise RuntimeError("unsafe mean-matched failures directory")
        for path in failures.iterdir():
            if path.is_symlink() or not path.is_file() or not path.name.startswith("FAILURE_") or not path.name.endswith(".json"):
                raise RuntimeError("unexpected mean-matched failure artifact")
            value = strict_json(path)
            if not isinstance(value, Mapping) or value.get("schema_version") != FAILURE_SCHEMA:
                raise RuntimeError("mean-matched failure record changed")
            attempts.append(int(value["queue_attempt"]))
    return max(attempts, default=0) + 1


def _clear_failure(out: Path) -> None:
    path = out / "FAILURE.json"
    if not path.exists() and not path.is_symlink():
        return
    value = strict_json(real_file(path, "current failure"))
    numbered = out / "failures" / f"FAILURE_{int(value['queue_attempt']):03d}.json"
    if strict_json(real_file(numbered, "numbered failure")) != value:
        raise RuntimeError("current failure is not backed by numbered failure")
    path.unlink()


def _record_failure(out: Path, attempt: int, completed: set[str], error: BaseException) -> Path:
    value = {
        "schema_version": FAILURE_SCHEMA, "status": "FAILED", "queue_attempt": attempt,
        "failed_unix": time.time(), "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(), "completed_units": len(completed),
        "expected_units": EXPECTED_UNITS, "completed_unit_ids": sorted(completed),
    }
    path = out / "failures" / f"FAILURE_{attempt:03d}.json"
    if path.exists() or path.is_symlink():
        raise RuntimeError("numbered failure already exists")
    atomic_json(path, value)
    atomic_json(out / "FAILURE.json", value)
    return path


def _counts(units: Sequence[MeanMatchedUnit], completed: set[str]) -> dict[str, Any]:
    return {
        benchmark: {
            "completed": sum(unit.unit_id in completed for unit in units if unit.benchmark == benchmark),
            "expected": sum(unit.benchmark == benchmark for unit in units),
        } for benchmark in BENCHMARKS
    }


def _scheduler(units: Sequence[MeanMatchedUnit], completed: set[str], *, lock: Mapping[str, Any],
               identities: Mapping[str, Mapping[str, Any]], update: Any,
               max_processes: int, **paths: Any) -> None:
    pending = [unit for unit in sorted(units, key=scheduler_key) if unit.unit_id not in completed]
    active: list[Active] = []
    failed = []
    runtime = lock["runtime"]
    try:
        while pending or active:
            survivors = []
            for row in active:
                code = row.process.poll()
                if code is None:
                    survivors.append(row)
                    continue
                row.handle.close()
                if code:
                    failed.append({"unit_id": row.unit.unit_id, "returncode": code, "log": str(row.log)})
                else:
                    try:
                        validate_completed_unit(paths["out"] / "units" / row.unit.unit_id, identities[row.unit.unit_id])
                        completed.add(row.unit.unit_id)
                    except BaseException as error:
                        failed.append({"unit_id": row.unit.unit_id, "returncode": 0, "log": str(row.log), "validation_error": f"{type(error).__name__}: {error}"})
            active = survivors
            if failed:
                if active:
                    update("DRAINING_AFTER_UNIT_FAILURE", {"failed_units": failed, "active_units": [item.unit.unit_id for item in active]})
                    time.sleep(POLL_SECONDS)
                    continue
                raise RuntimeError(f"mean-matched unit failure(s): {failed}")
            if not pending and not active:
                break
            disk = _free_disk_gib(paths["out"])
            if disk < int(runtime["minimum_free_disk_gib"]):
                raise RuntimeError(f"disk gate failed: {disk} GiB")
            gpu = _free_gpu_mib()
            required = int(runtime["minimum_free_gpu_memory_mib_per_additional_launch"] if active else runtime["minimum_free_gpu_memory_mib_before_first_launch"])
            launched = []
            if pending and len(active) < max_processes and gpu >= required:
                unit = pending.pop(0)
                active.append(_launch(unit, **paths))
                launched.append(unit.unit_id)
            update("RUNNING_UNITS" if active else "WAITING_FOR_GPU_GATE", {
                "pending_units": len(pending),
                "active_units": [{"unit_id": row.unit.unit_id, "pid": row.process.pid, "log": str(row.log), "started_unix": row.started} for row in active],
                "launched_this_cycle": launched, "free_gpu_mib": gpu,
                "required_free_gpu_mib_for_next_launch": int(runtime["minimum_free_gpu_memory_mib_per_additional_launch"] if active else runtime["minimum_free_gpu_memory_mib_before_first_launch"]),
                "free_disk_gib": disk, "maximum_concurrent_gpu_processes": max_processes,
            })
            time.sleep(POLL_SECONDS)
    except BaseException:
        while active:
            survivors = []
            for row in active:
                code = row.process.poll()
                if code is None:
                    survivors.append(row)
                else:
                    row.handle.close()
                    if code == 0:
                        try:
                            validate_completed_unit(paths["out"] / "units" / row.unit.unit_id, identities[row.unit.unit_id])
                            completed.add(row.unit.unit_id)
                        except BaseException:
                            pass
            active = survivors
            if active:
                update("DRAINING_AFTER_QUEUE_ERROR", {"active_units": [row.unit.unit_id for row in active]})
                time.sleep(POLL_SECONDS)
        raise
    if len(completed) != EXPECTED_UNITS:
        raise RuntimeError("mean-matched scheduler ended with incomplete coverage")


def _done(*, out: Path, units: Sequence[MeanMatchedUnit], binding_path: Path,
          lock_path: Path, source_path: Path, source: Mapping[str, Any],
          aggregate_path: Path, completed_unix: float) -> dict[str, Any]:
    unit_done = {unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json") for unit in units}
    inventory_paths = [binding_path, aggregate_path]
    for unit in units:
        inventory_paths.extend((out / "units" / unit.unit_id / "UNIT.json", out / "units" / unit.unit_id / "DONE.json"))
    inventory = {path.relative_to(out).as_posix(): sha256_file(real_file(path, "terminal artifact")) for path in inventory_paths}
    inventory = dict(sorted(inventory.items()))
    return {
        "schema_version": DONE_SCHEMA, "status": "PASS",
        "scientific_role": "prospectively_locked_mean_matched_companion",
        "unit_count": EXPECTED_UNITS, "expected_units": EXPECTED_UNITS,
        "completed_units": EXPECTED_UNITS,
        "validated_unit_ids": [unit.unit_id for unit in units],
        "validated_unit_ids_sha256": canonical_sha256([unit.unit_id for unit in units]),
        "unit_done_sha256": unit_done, "unit_done_bundle_sha256": canonical_sha256(unit_done),
        "lock_sha256": sha256_file(lock_path), "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"], "run_binding_sha256": sha256_file(binding_path),
        "aggregate_sha256": sha256_file(aggregate_path), "scientific_artifact_count": len(inventory),
        "scientific_artifact_inventory": inventory, "scientific_artifact_bundle_sha256": canonical_sha256(inventory),
        "completed_unix": completed_unix,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "lock", "source_manifest", "mammoth_root", "data_root", "data_manifest", "environment_manifest", "python", "out"):
        result.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    result.add_argument("--max-processes", type=int, default=2)
    return result


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError("mean-matched queue requires Python -B")
    project = real_directory(arguments.project, "mean-matched project")
    lock_path = real_file(arguments.lock, "mean-matched lock")
    source_path = real_file(arguments.source_manifest, "mean-matched source manifest")
    if lock_path != (project / "configs/mean_matched_assignment.json").resolve(strict=True):
        raise RuntimeError("mean-matched lock path is noncanonical")
    if source_path != (project / "provenance/mean_matched_source_manifest.json").resolve(strict=True):
        raise RuntimeError("mean-matched source path is noncanonical")
    source = validate_source_manifest(source_path, project)
    lock = load_lock(lock_path)
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "mechanism environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "mean-matched Python")
    out = _output_root(arguments.out)
    for path in (project, lock_path, source_path, mammoth, data_root, data_path, environment_path):
        if _paths_overlap(out, path):
            raise RuntimeError(f"mean-matched output overlaps protected input: {path}")
    descriptor = _queue_lock(out.parent / f".{out.name}.queue.lock")
    try:
        if _orphans(out):
            raise RuntimeError(f"orphan mean-matched processes are active: {_orphans(out)}")
        _validate_output_topology(out)
        units = sorted(enumerate_units(lock), key=scheduler_key)
        if arguments.max_processes not in (1, 2) or arguments.max_processes > int(lock["runtime"]["maximum_concurrent_gpu_processes"]):
            raise RuntimeError("invalid mean-matched scheduler width")
        binding = _binding(
            project=project, lock_path=lock_path, source_path=source_path,
            source=source, mammoth=mammoth, data_root=data_root, data_path=data_path,
            data=data, environment_path=environment_path, environment=environment,
            python=python,
        )
        binding_path = out / "RUN_BINDING.json"
        _write_or_validate(binding_path, binding)
        identities = _identities(
            units, lock=lock, lock_path=lock_path, source_path=source_path,
            source=source, mammoth=mammoth, data_path=data_path, data=data,
            environment_path=environment_path, environment=environment,
            python=python, binding_path=binding_path,
        )
        completed = _completed(units, out, identities)
        state_path = out / "QUEUE_STATE.json"
        attempt = _attempt(state_path)

        def update(stage: str, details: Mapping[str, Any] | None = None) -> None:
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "RUNNING", "stage": stage,
                "queue_attempt": attempt, "queue_pid": os.getpid(), "updated_unix": time.time(),
                "completed_units": len(completed), "expected_units": EXPECTED_UNITS,
                "completed_by_benchmark": _counts(units, completed), "details": dict(details or {}),
            })

        done_path = out / "DONE.json"
        try:
            update("WAITING_FOR_PARENT_TERMINAL_PASS", {
                "parent_output": str(lock["parent"]["output_path"]),
                "scientific_outcomes_read": False,
            })
            while not _parent_terminal_ready(lock):
                time.sleep(int(lock["runtime"]["parent_wait_poll_seconds"]))
                update("WAITING_FOR_PARENT_TERMINAL_PASS", {
                    "parent_output": str(lock["parent"]["output_path"]),
                    "scientific_outcomes_read": False,
                })
            if done_path.exists() or done_path.is_symlink():
                if len(completed) != EXPECTED_UNITS:
                    raise RuntimeError("mean-matched DONE exists without 96 units")
            else:
                _clear_failure(out)
                update("PARENT_TERMINAL_PASS_CONFIRMED")
                _scheduler(
                    units, completed, lock=lock, identities=identities,
                    update=update, max_processes=arguments.max_processes,
                    project=project, lock_path=lock_path, source_path=source_path,
                    mammoth=mammoth, data_root=data_root, data_path=data_path,
                    environment_path=environment_path, binding_path=binding_path,
                    python=python, out=out,
                )
                completed = _completed(units, out, identities)
            update("AGGREGATING_AND_REAUTHENTICATING_PARENT")
            aggregate_path = aggregate_campaign(
                project=project, lock_path=lock_path,
                source_manifest_path=source_path, mammoth_root=mammoth,
                data_root=data_root, data_manifest_path=data_path,
                environment_manifest_path=environment_path,
                binding_path=binding_path, out=out,
                output=out / AGGREGATE_NAME,
            )
            aggregate = strict_json(aggregate_path)
            if not (
                aggregate.get("schema_version") == AGGREGATE_SCHEMA
                and aggregate.get("status") == "PASS"
                and aggregate.get("complete_protocol") is True
                and aggregate.get("unit_count") == EXPECTED_UNITS
            ):
                raise RuntimeError("mean-matched aggregate failed terminal gate")
            terminal_time = time.time()
            expected_done = _done(
                out=out, units=units, binding_path=binding_path,
                lock_path=lock_path, source_path=source_path, source=source,
                aggregate_path=aggregate_path, completed_unix=terminal_time,
            )
            if done_path.exists() or done_path.is_symlink():
                observed = strict_json(real_file(done_path, "mean-matched DONE"))
                expected_done = _done(
                    out=out, units=units, binding_path=binding_path,
                    lock_path=lock_path, source_path=source_path, source=source,
                    aggregate_path=aggregate_path,
                    completed_unix=float(observed.get("completed_unix", -1)),
                )
                if observed != expected_done:
                    raise RuntimeError("mean-matched terminal DONE changed")
            else:
                atomic_json(done_path, expected_done)
            _clear_failure(out)
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "PASS", "stage": "COMPLETE",
                "queue_attempt": attempt, "queue_pid": os.getpid(), "updated_unix": time.time(),
                "completed_units": EXPECTED_UNITS, "expected_units": EXPECTED_UNITS,
                "completed_by_benchmark": _counts(units, completed),
                "details": {"done": str(done_path), "aggregate": str(aggregate_path)},
            })
            return done_path
        except BaseException as error:
            failure = _record_failure(out, attempt, completed, error)
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "FAILED", "stage": "FAILED",
                "queue_attempt": attempt, "queue_pid": os.getpid(), "updated_unix": time.time(),
                "completed_units": len(completed), "expected_units": EXPECTED_UNITS,
                "completed_by_benchmark": _counts(units, completed),
                "details": {"failure": str(failure)},
            })
            raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    print(run(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
