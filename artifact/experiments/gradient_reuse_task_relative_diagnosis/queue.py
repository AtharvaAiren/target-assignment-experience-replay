"""Outcome-blind resumable two-slot queue for the 36-unit diagnostic rerun."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_task_structure.integrity import (
    atomic_json,
    executable_entry,
    git_output,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from experiments.gradient_reuse_task_structure.protocol import canonical_sha256, sha256_file, strict_json
from experiments.gradient_reuse_task_structure.queue import _free_disk_gib, _free_gpu_mib, _open_lock
from experiments.gradient_reuse_task_structure.protocol import load_lock as load_original_lock

from .aggregate import NAME as AGGREGATE_NAME, SCHEMA as AGGREGATE_SCHEMA
from .protocol import EXPECTED_UNITS, DiagnosticUnit, enumerate_units, load_lock, validate_parent
from .source import validate_source_manifest
from .training_semantics import _validate_unit_seal, verify_training_semantics
from .unit_runner import build_identity, validate_completed_unit


STATE_SCHEMA = "gradient_reuse.task_relative_diagnosis_queue_state.v1"
DONE_SCHEMA = "gradient_reuse.task_relative_diagnosis_queue_done.v1"
FAILURE_SCHEMA = "gradient_reuse.task_relative_diagnosis_queue_failure.v1"


@dataclass
class Active:
    unit: DiagnosticUnit
    process: subprocess.Popen[Any]
    handle: Any
    started: float


def _output_root(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError("task-relative output must be an absolute real directory path")
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    return raw.resolve(strict=True)


def _environment(project: Path, mammoth: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONPATH": os.pathsep.join((str(project), str(mammoth))),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
        }
    )
    return value


def _state(path: Path, stage: str, completed: set[str], active: Sequence[Active], started: float) -> None:
    atomic_json(
        path,
        {
            "schema_version": STATE_SCHEMA,
            "status": "PASS" if stage == "COMPLETE" else "RUNNING",
            "stage": stage, "queue_pid": os.getpid(), "updated_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "completed_units": len(completed), "expected_units": EXPECTED_UNITS,
            "active_units": [
                {"unit_id": item.unit.unit_id, "pid": item.process.pid, "started_unix": item.started}
                for item in active
            ],
            "remaining_units": EXPECTED_UNITS - len(completed),
        },
    )


def _unit_command(arguments: argparse.Namespace, unit: DiagnosticUnit, binding: Path) -> list[str]:
    return [
        str(arguments.python), "-B", "-m", "experiments.gradient_reuse_task_relative_diagnosis.unit_runner",
        "--project", str(arguments.project), "--lock", str(arguments.lock),
        "--source-manifest", str(arguments.source_manifest),
        "--mammoth-root", str(arguments.mammoth_root), "--data-root", str(arguments.data_root),
        "--data-manifest", str(arguments.data_manifest),
        "--environment-manifest", str(arguments.environment_manifest),
        "--binding", str(binding), "--python", str(arguments.python),
        "--parent-root", str(arguments.parent_root), "--out", str(arguments.out),
        "--unit-id", unit.unit_id,
    ]


def _write_or_validate_done(path: Path, expected: Mapping[str, Any]) -> None:
    """Publish DONE once; later runs authenticate it without changing its bytes."""

    raw = Path(path).expanduser()
    if raw.is_symlink() or (raw.exists() and not raw.is_file()):
        raise RuntimeError(f"unsafe task-relative DONE destination: {raw}")
    if raw.exists():
        observed = strict_json(real_file(raw, "existing task-relative DONE"))
        if not isinstance(observed, Mapping):
            raise RuntimeError("existing task-relative DONE is not an object")
        completed_unix = observed.get("completed_unix")
        static = {key: value for key, value in observed.items() if key != "completed_unix"}
        if not (
            static == dict(expected)
            and isinstance(completed_unix, (int, float))
            and not isinstance(completed_unix, bool)
            and math.isfinite(float(completed_unix))
            and float(completed_unix) > 0.0
        ):
            raise RuntimeError("existing task-relative DONE differs from full reconstruction")
        return
    atomic_json(raw, {**dict(expected), "completed_unix": time.time()})


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError("task-relative queue requires Python -B")
    project = real_directory(arguments.project, "task-relative project")
    lock_path = real_file(arguments.lock, "task-relative lock")
    source_path = real_file(arguments.source_manifest, "task-relative source manifest")
    lock = load_lock(lock_path)
    source = validate_source_manifest(source_path, project)
    original_manifest = real_file(arguments.original_source_manifest, "original task source manifest")
    original_lock_path = real_file(project / "configs/task_structure_equal_severity.json", "original task lock")
    original_lock = load_original_lock(original_lock_path)
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "mechanism environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "task-relative Python")
    parent_root = real_directory(arguments.parent_root, "authenticated task-structure parent")
    parent_evidence = validate_parent(parent_root)
    out = _output_root(arguments.out)
    descriptor = _open_lock(out.parent / f".{out.name}.queue.lock")
    started = time.time()
    active: list[Active] = []
    try:
        semantics_path = out / "TRAINING_SEMANTICS.json"
        semantics = verify_training_semantics(
            project=project, source_manifest=original_manifest,
            parent_root=parent_root, output=semantics_path,
        )
        binding = {
            "schema_version": "gradient_reuse.task_relative_diagnosis_run_binding.v1",
            "status": "PASS", "project": str(project),
            "lock_sha256": sha256_file(lock_path),
            "source_manifest_sha256": sha256_file(source_path),
            "source_bundle_sha256": source["source_bundle_sha256"],
            "original_lock_sha256": sha256_file(original_lock_path),
            "original_source_manifest_sha256": sha256_file(original_manifest),
            "mammoth_root": str(mammoth), "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
            "data_root": str(data_root), "data_manifest_sha256": sha256_file(data_path),
            "data_bundle_sha256": data["data_bundle_sha256"],
            "environment_manifest_sha256": sha256_file(environment_path),
            "environment_bundle_sha256": environment["environment_bundle_sha256"],
            "python_entry": str(python), "python_sha256": sha256_file(python),
            "parent_root": str(parent_root), "parent_file_sha256": parent_evidence,
            "training_semantics_sha256": sha256_file(semantics_path),
            "training_semantics_status": semantics["status"],
            "expected_units": EXPECTED_UNITS, "device": "cuda:0",
        }
        binding_path = out / "RUN_BINDING.json"
        if binding_path.exists():
            if strict_json(binding_path) != binding:
                raise RuntimeError("existing task-relative run binding changed")
        else:
            atomic_json(binding_path, binding)
        parent_aggregate = strict_json(parent_root / "TASK_STRUCTURE_AGGREGATE.json")
        parent_done_hashes = parent_aggregate["unit_done_sha256"]
        parent_units = {
            unit.unit_id: _validate_unit_seal(
                parent_root / "units" / unit.unit_id,
                str(parent_done_hashes[unit.unit_id]),
            )
            for unit in enumerate_units()
        }
        identities = {
            unit.unit_id: build_identity(
                unit, lock_path=lock_path, source_path=source_path, source=source,
                original_lock_path=original_lock_path, original_lock=original_lock,
                mammoth=mammoth, data_path=data_path, data=data,
                environment_path=environment_path, environment=environment, python=python,
                binding_path=binding_path,
                parent_unit_done_sha256=str(parent_done_hashes[unit.unit_id]),
            )
            for unit in enumerate_units()
        }
        completed: set[str] = set()
        for unit in enumerate_units():
            root = out / "units" / unit.unit_id
            if (root / "DONE.json").exists():
                validate_completed_unit(
                    root,
                    identities[unit.unit_id],
                    parent_unit=parent_units[unit.unit_id],
                )
                completed.add(unit.unit_id)
        logs = out / "logs"
        logs.mkdir(exist_ok=True)
        state_path = out / "QUEUE_STATE.json"
        _state(state_path, "AUTHENTICATED", completed, active, started)
        units = list(enumerate_units())
        while len(completed) < EXPECTED_UNITS:
            for item in list(active):
                status = item.process.poll()
                if status is None:
                    continue
                item.handle.close()
                active.remove(item)
                if status:
                    raise RuntimeError(f"task-relative unit exited {status}: {item.unit.unit_id}")
                validate_completed_unit(
                    out / "units" / item.unit.unit_id,
                    identities[item.unit.unit_id],
                    parent_unit=parent_units[item.unit.unit_id],
                )
                completed.add(item.unit.unit_id)
            pending = [
                unit for unit in units
                if unit.unit_id not in completed and all(item.unit != unit for item in active)
            ]
            launched = False
            # Launch at most one CUDA process per polling cycle.  A ten-second
            # observation window lets its allocation become visible before a
            # second process is considered against the remaining-memory gate.
            if pending and len(active) < arguments.max_processes:
                runtime = lock["runtime"]
                threshold = int(
                    runtime[
                        "minimum_free_gpu_memory_mib_before_first_launch"
                        if not active
                        else "minimum_free_gpu_memory_mib_per_additional_launch"
                    ]
                )
                free_gpu_mib = _free_gpu_mib()
                if free_gpu_mib >= threshold:
                    if _free_disk_gib(out) < int(runtime["minimum_free_disk_gib"]):
                        raise RuntimeError("task-relative disk gate failed")
                    unit = pending[0]
                    handle = (logs / f"{unit.unit_id}.log").open("a", encoding="utf-8")
                    process = subprocess.Popen(
                        _unit_command(arguments, unit, binding_path), cwd=project,
                        env=_environment(project, mammoth), stdout=handle,
                        stderr=subprocess.STDOUT, text=True,
                    )
                    active.append(Active(unit, process, handle, time.time()))
                    launched = True
            _state(state_path, "RUNNING_UNITS", completed, active, started)
            time.sleep(
                int(lock["runtime"]["post_launch_observation_seconds"])
                if launched or not active
                else 2
            )
        aggregate_path = out / AGGREGATE_NAME
        with (logs / "aggregate.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(python), "-B", "-m", "experiments.gradient_reuse_task_relative_diagnosis.aggregate",
                 "--out", str(out), "--lock", str(lock_path), "--output", str(aggregate_path)],
                cwd=project, env=_environment(project, mammoth), stdout=handle,
                stderr=subprocess.STDOUT, text=True, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"task-relative aggregate exited {result.returncode}")
        aggregate = strict_json(real_file(aggregate_path, "task-relative aggregate"))
        if not isinstance(aggregate, Mapping) or not (
            aggregate.get("schema_version") == AGGREGATE_SCHEMA
            and aggregate.get("status") == "PASS"
            and aggregate.get("unit_count") == EXPECTED_UNITS
        ):
            raise RuntimeError("task-relative aggregate failed authentication")
        unit_ids = [unit.unit_id for unit in units]
        done = {
            "schema_version": DONE_SCHEMA, "status": "PASS",
            "expected_units": EXPECTED_UNITS, "completed_units": EXPECTED_UNITS,
            "validated_unit_ids": unit_ids,
            "validated_unit_ids_sha256": canonical_sha256(unit_ids),
            "lock_sha256": sha256_file(lock_path),
            "source_manifest_sha256": sha256_file(source_path),
            "source_bundle_sha256": source["source_bundle_sha256"],
            "run_binding_sha256": sha256_file(binding_path),
            "aggregate_sha256": sha256_file(aggregate_path),
        }
        _write_or_validate_done(out / "DONE.json", done)
        _state(state_path, "COMPLETE", completed, [], started)
        return out / "DONE.json"
    except BaseException as error:
        for item in active:
            if item.process.poll() is None:
                item.process.terminate()
        for item in active:
            try:
                item.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait(timeout=10)
            item.handle.close()
        atomic_json(
            out / "FAILURE.json",
            {"schema_version": FAILURE_SCHEMA, "status": "FAILED", "failed_unix": time.time(),
             "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project", "lock", "source_manifest", "original_source_manifest", "mammoth_root",
        "data_root", "data_manifest", "environment_manifest", "python", "parent_root", "out",
    ):
        result.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    result.add_argument("--max-processes", type=int, choices=(1, 2), default=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(run(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_write_or_validate_done", "main", "run"]
