"""Rerun one authenticated task-structure unit and seal terminal logits."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.gradient_reuse_task_structure.integrity import (
    artifact_inventory,
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
from experiments.gradient_reuse_task_structure.unit_runner import (
    allocate_attempt,
    effective_args,
    read_metrics,
    subprocess_environment,
    summarize_audit,
    unit_lock,
)

from .metrics import SCHEMA as METRIC_SCHEMA, compute_metrics
from .protocol import DiagnosticUnit, enumerate_units, load_lock, validate_parent
from .source import validate_source_manifest
from .training_semantics import _validate_unit_seal


UNIT_SCHEMA = "gradient_reuse.task_relative_diagnosis_unit.v1"
DONE_SCHEMA = "gradient_reuse.task_relative_diagnosis_unit_done.v1"


def build_identity(
    unit: DiagnosticUnit,
    *,
    lock_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    original_lock_path: Path,
    mammoth: Path,
    data_path: Path,
    data: Mapping[str, Any],
    environment_path: Path,
    environment: Mapping[str, Any],
    python: Path,
    binding_path: Path,
    parent_unit_done_sha256: str,
    original_lock: Mapping[str, Any],
) -> dict[str, Any]:
    effective = effective_args(unit, original_lock, "cuda:0")
    return {
        "schema_version": "gradient_reuse.task_relative_diagnosis_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "original_lock_sha256": sha256_file(original_lock_path),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "binding_sha256": sha256_file(binding_path),
        "parent_unit_done_sha256": parent_unit_done_sha256,
        "effective_args_sha256": canonical_sha256(effective),
        "device": "cuda:0",
    }


def build_command(
    unit: DiagnosticUnit,
    original_lock: Mapping[str, Any],
    *,
    python: Path,
    mammoth: Path,
    data_root: Path,
    results: Path,
    audit: Path,
    diagnostic: Path,
) -> list[str]:
    command = [
        str(python), "-B", "-m",
        "experiments.gradient_reuse_task_relative_diagnosis.mammoth_bootstrap",
        "--mammoth-root", str(mammoth),
        "--diagnostic-output", str(diagnostic),
    ]
    for name, value in effective_args(unit, original_lock, "cuda:0").items():
        command.extend((f"--{name}", str(value)))
    command.extend(
        (
            "--base_path", str(data_root),
            "--results_path", str(results),
            "--task_structure_audit_path", str(audit),
        )
    )
    flags = command[4:][0::2]
    if any(not flag.startswith("--") for flag in flags) or len(flags) != len(set(flags)):
        raise RuntimeError("task-relative Mammoth command has malformed flags")
    return command


def _load_raw(path: Path, expected: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = real_file(path, "terminal raw logits")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"schema_version", "logits", "labels", "indexes"}:
            raise RuntimeError("terminal raw-logit inventory changed")
        if str(archive["schema_version"].item()) != "gradient_reuse.task_relative_raw_logits.v1":
            raise RuntimeError("terminal raw-logit schema changed")
        arrays = {name: np.array(archive[name], copy=True) for name in ("logits", "labels", "indexes")}
    observed = compute_metrics(
        arrays["logits"], arrays["labels"], arrays["indexes"],
        input_stream_sha256=str(expected["input_stream_sha256"]),
    )
    if {**observed, "raw_logits_file": path.name} != dict(expected):
        raise RuntimeError("terminal task-relative summary does not reconstruct from logits")
    return arrays


def validate_completed_unit(
    root: Path,
    identity: Mapping[str, Any],
    *,
    parent_unit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    done = strict_json(real_file(root / "DONE.json", "task-relative unit DONE"))
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == DONE_SCHEMA
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"task-relative unit completion identity changed: {root}")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == inventory
        and done.get("artifact_bundle_sha256") == canonical_sha256(inventory)
    ):
        raise RuntimeError(f"task-relative unit artifact inventory changed: {root}")
    record = strict_json(real_file(root / "UNIT.json", "task-relative unit record"))
    if not isinstance(record, dict) or record.get("identity") != dict(identity):
        raise RuntimeError(f"task-relative unit record changed: {root}")
    if parent_unit is not None:
        _scientific_parent_match(record, parent_unit)
    attempt = root / str(record.get("attempt"))
    summary = strict_json(real_file(attempt / "FINAL_TASK_RELATIVE_METRICS.json", "unit summary"))
    if not isinstance(summary, Mapping) or summary.get("schema_version") != METRIC_SCHEMA:
        raise RuntimeError("task-relative unit summary changed")
    _load_raw(attempt / "FINAL_LOGITS.npz", summary)
    return record


def _scientific_parent_match(observed: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    if not (
        parent.get("schema_version") == "gradient_reuse.task_structure_unit.v1"
        and observed.get("unit") == parent.get("unit")
        and observed.get("effective_args") == parent.get("effective_args")
    ):
        raise RuntimeError("diagnostic rerun and parent protocol identities differ")
    observed_metrics = observed.get("metrics", {})
    parent_metrics = parent.get("metrics", {})
    # Log-file hashes and absolute results/audit paths must differ in the new
    # namespace.  Every scientific endpoint, including full 20x20 triangular
    # accuracy matrices, must remain byte-for-JSON identical.
    for name in (
        "benchmark_default_setting", "followup_primary_setting", "class_il", "task_il"
    ):
        if observed_metrics.get(name) != parent_metrics.get(name):
            raise RuntimeError(
                f"diagnostic rerun did not exactly reproduce original Mammoth endpoint: {name}"
            )
    if (
        observed_metrics.get("effective_argument_evidence", {}).get("normalized")
        != parent_metrics.get("effective_argument_evidence", {}).get("normalized")
    ):
        raise RuntimeError("diagnostic rerun and parent logged arguments differ")
    observed_audit = observed.get("task_structure_diagnostics", {})
    parent_audit = parent.get("task_structure_diagnostics", {})
    observed_audit_without_path_digest = {
        name: value for name, value in observed_audit.items() if name != "audit_sha256"
    }
    parent_audit_without_path_digest = {
        name: value for name, value in parent_audit.items() if name != "audit_sha256"
    }
    if observed_audit_without_path_digest != parent_audit_without_path_digest:
        raise RuntimeError("diagnostic rerun did not reproduce original task-structure trajectory evidence")


def run_one(arguments: argparse.Namespace) -> Path:
    if not __import__("sys").dont_write_bytecode:
        raise RuntimeError("task-relative unit runner requires Python -B")
    project = real_directory(arguments.project, "task-relative project")
    lock_path = real_file(arguments.lock, "task-relative lock")
    if lock_path != (project / "configs/task_relative_diagnosis.json").resolve(strict=True):
        raise RuntimeError("task-relative lock is not installed at its canonical path")
    load_lock(lock_path)
    source_path = real_file(arguments.source_manifest, "task-relative source manifest")
    source = validate_source_manifest(source_path, project)
    original_lock_path = real_file(project / "configs/task_structure_equal_severity.json", "original task lock")
    from experiments.gradient_reuse_task_structure.protocol import load_lock as load_original_lock
    original_lock = load_original_lock(original_lock_path)
    units = {unit.unit_id: unit for unit in enumerate_units()}
    if arguments.unit_id not in units:
        raise RuntimeError(f"unknown task-relative unit: {arguments.unit_id}")
    unit = units[arguments.unit_id]
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "task-relative Python")
    parent_root = real_directory(arguments.parent_root, "task-structure parent")
    validate_parent(parent_root)
    binding_path = real_file(arguments.binding, "task-relative run binding")
    binding = strict_json(binding_path)
    if not isinstance(binding, Mapping) or binding.get("status") != "PASS":
        raise RuntimeError("task-relative run binding changed")
    parent_aggregate = strict_json(parent_root / "TASK_STRUCTURE_AGGREGATE.json")
    parent_hashes = parent_aggregate.get("unit_done_sha256", {})
    parent_unit = _validate_unit_seal(parent_root / "units" / unit.unit_id, str(parent_hashes[unit.unit_id]))
    effective = effective_args(unit, original_lock, "cuda:0")
    identity = build_identity(
        unit, lock_path=lock_path, source_path=source_path, source=source,
        original_lock_path=original_lock_path, original_lock=original_lock,
        mammoth=mammoth, data_path=data_path, data=data,
        environment_path=environment_path, environment=environment,
        python=python, binding_path=binding_path,
        parent_unit_done_sha256=str(parent_hashes[unit.unit_id]),
    )
    out = Path(arguments.out).expanduser().resolve()
    unit_root = out / "units" / unit.unit_id
    with unit_lock(unit_root):
        if (unit_root / "DONE.json").exists():
            validate_completed_unit(unit_root, identity, parent_unit=parent_unit)
            return unit_root
        if (unit_root / "UNIT.json").exists():
            raise RuntimeError(f"unsealed task-relative unit requires explicit recovery: {unit_root}")
        attempt = allocate_attempt(unit_root)
        results = attempt / "mammoth_results"
        audit = attempt / "task_structure_audit.jsonl"
        stdout = attempt / "stdout.log"
        command = build_command(
            unit, original_lock, python=python, mammoth=mammoth, data_root=data_root,
            results=results, audit=audit, diagnostic=attempt,
        )
        started = time.time()
        manifest = {
            "schema_version": "gradient_reuse.task_relative_diagnosis_attempt.v1",
            "status": "RUNNING", "identity": identity,
            "identity_sha256": canonical_sha256(identity), "command": command,
            "attempt": attempt.name, "started_unix": started,
        }
        atomic_json(attempt / "RUN_MANIFEST.json", manifest)
        try:
            with stdout.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command, cwd=mammoth,
                    env=subprocess_environment(unit, original_lock, project, mammoth),
                    stdout=handle, stderr=subprocess.STDOUT, check=False, text=True,
                )
            if completed.returncode:
                raise RuntimeError(f"Mammoth returned {completed.returncode}; see {stdout}")
            metrics = read_metrics(results, original_lock, expected_args=effective, data_root=data_root, audit=audit)
            diagnostics = summarize_audit(unit, audit)
            summary = strict_json(real_file(attempt / "FINAL_TASK_RELATIVE_METRICS.json", "unit summary"))
            crosscheck = strict_json(real_file(attempt / "MAMMOTH_ENDPOINT_CROSSCHECK.json", "endpoint crosscheck"))
            if not isinstance(summary, Mapping) or summary.get("status") != "PASS" or not (
                isinstance(crosscheck, Mapping) and crosscheck.get("status") == "PASS"
            ):
                raise RuntimeError("task-relative inference hook did not authenticate")
            _load_raw(attempt / "FINAL_LOGITS.npz", summary)
            record = {
                "schema_version": UNIT_SCHEMA,
                "identity": identity, "unit": unit.as_dict(), "effective_args": effective,
                "metrics": metrics, "task_structure_diagnostics": diagnostics,
                "task_relative_metrics": summary,
                "runtime_seconds": time.time() - started, "attempt": attempt.name,
                "original_endpoint_reproduction": "PASS",
            }
            _scientific_parent_match(record, parent_unit)
            manifest.update({"status": "COMPLETED", "completed_unix": time.time(), "returncode": 0})
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            atomic_json(unit_root / "UNIT.json", record)
            inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
            atomic_json(
                unit_root / "DONE.json",
                {
                    "schema_version": DONE_SCHEMA, "status": "PASS",
                    "identity_sha256": canonical_sha256(identity),
                    "artifact_count": len(inventory), "artifact_inventory": inventory,
                    "artifact_bundle_sha256": canonical_sha256(inventory),
                },
            )
            return unit_root
        except BaseException as error:
            manifest.update(
                {"status": "FAILED", "failed_unix": time.time(),
                 "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            )
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project", "lock", "source_manifest", "mammoth_root", "data_root",
        "data_manifest", "environment_manifest", "binding", "python", "parent_root",
        "out", "unit_id",
    ):
        result.add_argument("--" + name.replace("_", "-"), dest=name, type=Path if name != "unit_id" else str, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(run_one(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_command", "build_identity", "run_one", "validate_completed_unit"]
