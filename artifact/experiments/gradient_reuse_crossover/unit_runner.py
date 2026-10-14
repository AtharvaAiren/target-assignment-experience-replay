"""Execute and seal one crossover-localization follow-up unit.

The follow-up deliberately contains only the seven newly locked reassignment
strengths.  Historical controls and the 0.75/1.00 endpoints are authenticated
and joined only by the campaign aggregator.  This runner therefore has no
code path that can rerun or overwrite a historical scientific unit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory,
    atomic_json,
    executable_entry,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from experiments.gradient_reuse_assignment_strength.protocol import (
    parse_mammoth_log,
)
from experiments.gradient_reuse_assignment_strength.unit_runner import (
    build_command as build_assignment_command,
    effective_args as assignment_effective_args,
    read_metrics as read_assignment_metrics,
    subprocess_environment as assignment_environment,
    summarize_audit,
)
from experiments.gradient_reuse_corrected_reference_order.bank import (
    build_task_bank,
    validate_task_bank_manifest,
)
from experiments.gradient_reuse_corrected_reference_order.mammoth_bootstrap import (
    AUDIT_SCHEMA as CORRECTED_BOOTSTRAP_AUDIT_SCHEMA,
)
from experiments.gradient_reuse_corrected_reference_order.protocol import (
    expected_task_bank_digest,
    load_lock as load_task_bank_lock,
)
from experiments.gradient_reuse_corrected_reference_order.unit_runner import (
    _contract as corrected_contract,
    _environment as corrected_environment,
    _metrics as corrected_metrics,
    build_command as build_corrected_command,
    effective_args as corrected_effective_args,
)

from .protocol import (
    CAMPAIGNS,
    CrossoverUnit,
    UPSTREAM_COMMIT,
    canonical_sha256,
    load_lock,
    sha256_file,
    strict_json,
    unit_by_id,
    validate_runtime_provenance,
)


UNIT_SCHEMA = "gradient_reuse.crossover_unit.v1"
DONE_SCHEMA = "gradient_reuse.crossover_unit_done.v1"
ATTEMPT_SCHEMA = "gradient_reuse.crossover_attempt.v1"
STREAM_SCHEMA = "gradient_reuse.crossover_stream_identity.v1"
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


@contextmanager
def unit_lock(unit_root: Path):
    """Acquire a nonblocking, no-follow, per-unit filesystem lock."""

    unit_root.parent.mkdir(parents=True, exist_ok=True)
    if unit_root.parent.is_symlink() or not unit_root.parent.is_dir():
        raise RuntimeError("unsafe crossover units directory")
    path = unit_root.parent / f".{unit_root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe crossover unit lock: {path}")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise RuntimeError("crossover unit lock is not a singly linked file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another process owns crossover unit {unit_root.name}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def allocate_attempt(unit_root: Path) -> Path:
    """Allocate a fresh attempt while preserving all prior failed attempts."""

    if unit_root.is_symlink() or (unit_root.exists() and not unit_root.is_dir()):
        raise RuntimeError(f"unsafe crossover unit root: {unit_root}")
    unit_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in unit_root.iterdir():
        if (
            path.name in {"UNIT.json", "DONE.json"}
            and path.is_file()
            and not path.is_symlink()
        ):
            continue
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("attempt_")
            and path.name[8:].isdigit()
            and path.name == f"attempt_{int(path.name[8:]):03d}"
        ):
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected crossover unit artifact: {path}")
    attempt = unit_root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def _pmnist_compatible_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the crossover lock to the corrected-bank runner interface."""

    benchmark = lock["benchmarks"]["pmnist"]
    return {
        "runtime": dict(lock["runtime"]),
        "benchmarks": {
            "permuted_mnist": {
                "backbone": benchmark["backbone"],
                "standard_learning_rate": benchmark["learning_rate"],
                # The crossover grid has no fixed-current arm, but retaining a
                # total adapter makes accidental access fail deterministically.
                "fixed_current_learning_rate": benchmark["learning_rate"],
                "epochs": benchmark["epochs"],
                "batch_size": benchmark["batch_size"],
                "dataset_config": benchmark["dataset_config"],
                "replay_batch_size": benchmark["replay_batch_size"],
            }
        },
    }


def _cifar_compatible_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the lock to the original assignment-strength runner interface."""

    benchmark = lock["benchmarks"]["cifar"]
    name = str(benchmark["benchmark"])
    return {
        "runtime": dict(lock["runtime"]),
        "benchmarks": {
            name: {
                "backbone": benchmark["backbone"],
                "learning_rate": benchmark["learning_rate"],
                "epochs": benchmark["epochs"],
                "batch_size": benchmark["batch_size"],
                "dataset_config": benchmark["dataset_config"],
                "setting": benchmark["setting"],
            }
        },
        "methods": {
            "experience_replay": {
                "buffer_size": lock["method"]["buffer_size"],
                "minibatch_size": lock["method"]["minibatch_size"],
            }
        },
    }


def effective_args(
    unit: CrossoverUnit,
    lock: Mapping[str, Any],
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Return the exact branch-native Mammoth argument contract."""

    if device != "cuda:0":
        raise RuntimeError("crossover paper units require cuda:0")
    if unit.campaign == "pmnist":
        arguments = corrected_effective_args(
            unit, _pmnist_compatible_lock(lock), "pmnist", device=device
        )
    elif unit.campaign == "cifar":
        arguments = assignment_effective_args(
            unit, _cifar_compatible_lock(lock), device
        )
        forbidden = {
            "permute_classes",
            "custom_class_order",
            "custom_task_order",
        }
        if forbidden & set(arguments):
            raise RuntimeError("CIFAR crossover arguments request a nondefault task order")
    else:
        raise RuntimeError(f"unsupported crossover campaign: {unit.campaign}")
    if int(arguments.get("minibatch_size", -1)) != 32:
        raise RuntimeError("crossover support geometry requires replay minibatches of 32")
    return arguments


def _task_bank_parent(
    lock: Mapping[str, Any], project: Path
) -> tuple[Path, dict[str, Any]]:
    benchmark = lock["benchmarks"]["pmnist"]
    relative_text = str(benchmark["task_bank_lock_path"])
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_text
    ):
        raise RuntimeError("unsafe PMNIST task-bank parent-lock path")
    path = real_file(project / relative, "PMNIST task-bank parent lock")
    path.relative_to(project.resolve(strict=True))
    expected_sha = str(benchmark["task_bank_lock_sha256"])
    if sha256_file(path) != expected_sha:
        raise RuntimeError("PMNIST task-bank parent lock bytes changed")
    return path, load_task_bank_lock(path)


def stream_identity(
    lock: Mapping[str, Any], unit: CrossoverUnit, project: Path
) -> dict[str, Any]:
    """Bind the exact PMNIST bank or original CIFAR stream order."""

    if unit.campaign == "pmnist":
        parent_path, parent = _task_bank_parent(lock, project)
        digest = expected_task_bank_digest(parent, unit.seed)
        return {
            "schema_version": STREAM_SCHEMA,
            "campaign": "pmnist",
            "kind": "fixed_task_bank_from_authenticated_parent_lock",
            "run_seed": unit.seed,
            "parent_lock_path": str(parent_path.relative_to(project)),
            "parent_lock_sha256": sha256_file(parent_path),
            "task_bank_sha256": digest,
            "order_sha256": None,
        }
    if unit.campaign != "cifar":
        raise RuntimeError(f"unsupported crossover campaign: {unit.campaign}")
    order_core = {
        "kind": "pinned_mammoth_default_split_cifar100_order",
        "mammoth_commit": UPSTREAM_COMMIT,
        "dataset_config": "20tasks",
        "permute_classes": False,
        "custom_class_order": None,
        "custom_task_order": None,
        "task_order": list(range(20)),
        "class_order": list(range(100)),
        "task_sets": [list(range(5 * task, 5 * task + 5)) for task in range(20)],
    }
    return {
        "schema_version": STREAM_SCHEMA,
        "campaign": "cifar",
        **order_core,
        "task_bank_sha256": None,
        "order_sha256": canonical_sha256(order_core),
    }


def _find_log(results: Path, unit: CrossoverUnit, setting: str) -> Path:
    names = {unit.mammoth_model, unit.mammoth_model.replace("_", "-")}
    candidates = [
        results / setting / unit.mammoth_dataset / name / "logs.pyd"
        for name in sorted(names)
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected one Mammoth {setting} log, found {present}")
    return present[0]


def _validate_cifar_original_order(results: Path, unit: CrossoverUnit) -> None:
    expected_defaults = {
        "permute_classes": False,
        "custom_class_order": None,
        "custom_task_order": None,
        "noise_rate": 0.0,
    }
    normalized: dict[str, Any] | None = None
    for setting in ("class-il", "task-il"):
        payload = parse_mammoth_log(_find_log(results, unit, setting))
        observed = {name: payload.get(name) for name in expected_defaults}
        if observed != expected_defaults:
            raise RuntimeError(
                f"CIFAR {setting} log does not use the original task order: {observed}"
            )
        if normalized is not None and observed != normalized:
            raise RuntimeError("CIFAR endpoint logs disagree on their task order")
        normalized = observed


def _validate_assignment_geometry(
    unit: CrossoverUnit, diagnostics: Mapping[str, Any]
) -> None:
    if not (
        int(diagnostics.get("intervention_records", 0)) > 0
        and int(diagnostics.get("gradient_diagnostic_records", 0)) > 0
        and diagnostics.get("minimum_replay_batch_rows") == 32
        and diagnostics.get("maximum_replay_batch_rows") == 32
    ):
        raise RuntimeError("assignment audit does not cover exact 32-row replay batches")
    expected = {
        "realized_lambda": unit.induced_strength,
        "selected_fraction": unit.selected_rows / 32.0,
    }
    for group in ("means", "minima", "maxima"):
        values = diagnostics.get(group)
        if not isinstance(values, Mapping):
            raise RuntimeError(f"assignment audit omitted {group}")
        for name, wanted in expected.items():
            try:
                observed = float(values[name])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"assignment audit omitted exact {group}.{name}"
                ) from error
            if not math.isfinite(observed) or not math.isclose(
                observed, wanted, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise RuntimeError(
                    f"assignment audit changed locked geometry: "
                    f"{group}.{name}={observed!r}, expected={wanted!r}"
                )


def _final_accuracy(metrics: Mapping[str, Any], campaign: str) -> dict[str, float]:
    if campaign == "pmnist":
        try:
            values = {
                "domain_il": float(
                    metrics["domain_il"]["metrics"]["final_average_accuracy"]
                )
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("PMNIST metrics omit the domain-IL endpoint") from error
    elif campaign == "cifar":
        try:
            values = {
                "task_il": float(metrics["task_il"]["final_average_accuracy"]),
                "class_il": float(metrics["primary"]["final_average_accuracy"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("CIFAR metrics omit a mandatory endpoint") from error
    else:
        raise RuntimeError(f"unsupported crossover campaign: {campaign}")
    if any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in values.values()):
        raise RuntimeError("crossover endpoint accuracy is invalid")
    return values


def _expected_record_stream_fields(
    lock: Mapping[str, Any], unit: CrossoverUnit, project: Path
) -> tuple[dict[str, Any], str | None, str | None]:
    identity = stream_identity(lock, unit, project)
    return (
        identity,
        identity.get("task_bank_sha256"),
        identity.get("order_sha256"),
    )


def validate_completed_unit(
    root: Path,
    unit: CrossoverUnit,
    *,
    campaign: str,
    lock: Mapping[str, Any],
    expected_provenance: Mapping[str, Any] | None = None,
    project: Path | None = None,
) -> dict[str, Any]:
    """Authenticate a completed unit and attach verified seal digests."""

    project = (
        Path(__file__).resolve().parents[2]
        if project is None
        else real_directory(project, "crossover project")
    )
    if campaign not in CAMPAIGNS or unit.campaign != campaign:
        raise RuntimeError("crossover completed-unit campaign changed")
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"completed crossover unit is absent or unsafe: {root}")
    done_path = real_file(root / "DONE.json", "crossover DONE")
    done = strict_json(done_path)
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == DONE_SCHEMA
        and done.get("status") == "PASS"
        and done.get("unit_id") == unit.unit_id
    ):
        raise RuntimeError("crossover unit is not a sealed PASS")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == inventory
        and done.get("artifact_bundle_sha256") == canonical_sha256(inventory)
        and done.get("artifact_count") == len(inventory)
    ):
        raise RuntimeError("crossover unit artifact inventory changed")
    unit_path = real_file(root / "UNIT.json", "crossover UNIT")
    record = strict_json(unit_path)
    if not isinstance(record, dict) or not (
        record.get("schema_version") == UNIT_SCHEMA
        and record.get("status") == "PASS"
        and record.get("campaign") == campaign
        and record.get("unit") == unit.as_dict()
        and record.get("lock_content_sha256") == canonical_sha256(lock)
    ):
        raise RuntimeError("crossover completed-unit identity changed")
    provenance_names = (
        "lock_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "mammoth_commit",
        "data_manifest_sha256",
        "data_bundle_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "python_sha256",
    )
    if record.get("mammoth_commit") != UPSTREAM_COMMIT or any(
        not _is_sha256(record.get(name))
        for name in provenance_names
        if name != "mammoth_commit"
    ):
        raise RuntimeError("completed crossover unit has malformed provenance")
    if expected_provenance is not None:
        if any(
            record.get(name) != expected_provenance.get(name)
            for name in provenance_names
        ):
            raise RuntimeError("completed crossover provenance differs from campaign binding")
    expected_stream, task_digest, order_digest = _expected_record_stream_fields(
        lock, unit, project
    )
    if not (
        record.get("stream_identity") == expected_stream
        and record.get("task_bank_sha256") == task_digest
        and record.get("order_sha256") == order_digest
    ):
        raise RuntimeError("completed crossover unit carries another data stream")
    effective = effective_args(unit, lock)
    if not (
        record.get("effective_args") == effective
        and record.get("effective_args_sha256") == canonical_sha256(effective)
    ):
        raise RuntimeError("completed crossover effective arguments changed")
    diagnostics = record.get("assignment_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("completed crossover unit omits assignment diagnostics")
    _validate_assignment_geometry(unit, diagnostics)
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("completed crossover unit omits native metrics")
    if record.get("endpoint_values") != _final_accuracy(metrics, campaign):
        raise RuntimeError("completed crossover endpoints disagree with native metrics")
    attempt_name = record.get("attempt")
    if not (
        isinstance(attempt_name, str)
        and attempt_name.startswith("attempt_")
        and attempt_name[8:].isdigit()
        and attempt_name == f"attempt_{int(attempt_name[8:]):03d}"
    ):
        raise RuntimeError("completed crossover unit names an invalid attempt")
    attempt = real_directory(root / attempt_name, "completed crossover attempt")
    run_manifest = strict_json(
        real_file(attempt / "RUN_MANIFEST.json", "crossover attempt manifest")
    )
    if not isinstance(run_manifest, Mapping) or not (
        run_manifest.get("schema_version") == ATTEMPT_SCHEMA
        and run_manifest.get("status") == "COMPLETED"
        and run_manifest.get("campaign") == campaign
        and run_manifest.get("unit") == unit.as_dict()
        and run_manifest.get("lock_content_sha256") == canonical_sha256(lock)
        and run_manifest.get("stream_identity") == expected_stream
        and run_manifest.get("effective_args_sha256") == canonical_sha256(effective)
        and run_manifest.get("attempt") == attempt_name
        and run_manifest.get("returncode") == 0
        and isinstance(run_manifest.get("command"), list)
        and bool(run_manifest.get("command"))
        and all(
            run_manifest.get(name) == record.get(name)
            for name in provenance_names
        )
    ):
        raise RuntimeError("completed crossover attempt manifest changed")
    try:
        runtime_seconds = float(record.get("runtime_seconds"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("completed crossover runtime is invalid") from error
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        raise RuntimeError("completed crossover runtime is invalid")
    if campaign == "pmnist":
        manifest = strict_json(real_file(attempt / "TASK_BANK.json", "PMNIST task bank"))
        if not isinstance(manifest, Mapping):
            raise RuntimeError("completed PMNIST task bank is not an object")
        validate_task_bank_manifest(
            manifest,
            expected_seed=unit.seed,
            expected_digest=str(task_digest),
        )
        bootstrap_path = real_file(
            attempt / "bootstrap_audit.json", "PMNIST bootstrap audit"
        )
        bootstrap = strict_json(bootstrap_path)
        contract_path = real_file(
            attempt / "BOOTSTRAP_CONTRACT.json", "PMNIST bootstrap contract"
        )
        if not isinstance(bootstrap, Mapping) or not (
            bootstrap.get("schema_version") == CORRECTED_BOOTSTRAP_AUDIT_SCHEMA
            and bootstrap.get("status") == "PASS"
            and bootstrap.get("unit_id") == unit.unit_id
            and bootstrap.get("benchmark") == unit.benchmark
            and bootstrap.get("seed") == unit.seed
            and bootstrap.get("mammoth_commit") == UPSTREAM_COMMIT
            and bootstrap.get("contract_sha256") == sha256_file(contract_path)
            and bootstrap.get("task_bank_sha256") == task_digest
            and bootstrap.get("task_permutations_consumed") == 20
            and bootstrap.get("order_sha256") is None
            and bootstrap.get("observed_task_sets") is None
            and record.get("bootstrap_audit_sha256") == sha256_file(bootstrap_path)
        ):
            raise RuntimeError("completed PMNIST bootstrap evidence changed")
    else:
        if record.get("bootstrap_audit_sha256") is not None:
            raise RuntimeError("CIFAR unit unexpectedly names a task-bank bootstrap audit")
        _validate_cifar_original_order(attempt / "mammoth_results", unit)
    sealed = dict(record)
    sealed["seal_digests"] = {
        "unit_json_sha256": sha256_file(unit_path),
        "unit_done_sha256": sha256_file(done_path),
        "artifact_bundle_sha256": done["artifact_bundle_sha256"],
    }
    return sealed


def _safe_output_root(raw: Path, *protected: Path) -> Path:
    output = raw.expanduser()
    if not output.is_absolute() or output.is_symlink() or (
        output.exists() and not output.is_dir()
    ):
        raise RuntimeError("crossover output root must be a safe absolute directory")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    for protected_root in protected:
        protected_root = protected_root.resolve(strict=True)
        if (
            output == protected_root
            or output in protected_root.parents
            or protected_root in output.parents
        ):
            raise RuntimeError("crossover output overlaps a protected input tree")
    units = output / "units"
    if units.is_symlink() or (units.exists() and not units.is_dir()):
        raise RuntimeError("unsafe crossover units directory")
    units.mkdir(exist_ok=True)
    return output


def _provenance(
    *,
    lock_path: Path,
    source_manifest: Path,
    source: Mapping[str, Any],
    data_manifest: Path,
    data: Mapping[str, Any],
    environment_manifest: Path,
    environment: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    return {
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": UPSTREAM_COMMIT,
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python.resolve(strict=True)),
    }


def _write_failure(manifest_path: Path, manifest: dict[str, Any], error: BaseException) -> None:
    manifest.update(
        {
            "status": "FAILED",
            "failed_unix": time.time(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    )
    atomic_json(manifest_path, manifest)


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "crossover lock")
    lock = load_lock(lock_path)
    unit = unit_by_id(lock, args.campaign, args.unit_id)

    # Import lazily so source-manifest construction can include this runner
    # without an import cycle during source-closure generation.
    from .source import validate_source_manifest

    source_manifest = real_file(args.source_manifest, "crossover source manifest")
    source = validate_source_manifest(source_manifest, project)
    mammoth = validate_pristine_mammoth(args.mammoth_root)
    data_root = real_directory(args.data_root, "mechanism data root")
    data_manifest = real_file(args.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_manifest, data_root)
    environment_manifest = real_file(
        args.environment_manifest, "mechanism environment manifest"
    )
    environment = validate_environment_manifest(
        environment_manifest, mammoth, "cuda:0"
    )
    python = executable_entry(args.python, "crossover Python")
    output = _safe_output_root(args.output_root, project, mammoth, data_root)
    root = output / "units" / unit.unit_id
    provenance = _provenance(
        lock_path=lock_path,
        source_manifest=source_manifest,
        source=source,
        data_manifest=data_manifest,
        data=data,
        environment_manifest=environment_manifest,
        environment=environment,
        python=python,
    )
    validate_runtime_provenance(lock, provenance)
    with unit_lock(root):
        if (root / "DONE.json").exists():
            return validate_completed_unit(
                root,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=provenance,
                project=project,
            )
        if (root / "UNIT.json").exists():
            raise RuntimeError(
                f"unsealed crossover unit requires explicit recovery: {root}"
            )
        attempt = allocate_attempt(root)
        results = attempt / "mammoth_results"
        assignment_audit = attempt / "assignment_audit.jsonl"
        stdout = attempt / "stdout.log"
        stderr = attempt / "stderr.log"
        stream, task_digest, order_digest = _expected_record_stream_fields(
            lock, unit, project
        )
        base_arguments = effective_args(unit, lock)
        command: list[str]
        bootstrap_audit: Path | None = None
        if args.campaign == "pmnist":
            _, parent_lock = _task_bank_parent(lock, project)
            manifest = build_task_bank(unit.seed)
            expected_digest = expected_task_bank_digest(parent_lock, unit.seed)
            if not (
                task_digest == expected_digest
                and manifest.get("task_bank_sha256") == expected_digest
            ):
                raise RuntimeError("generated PMNIST bank differs from its parent lock")
            task_bank_path = attempt / "TASK_BANK.json"
            atomic_json(task_bank_path, manifest)
            contract_path = attempt / "BOOTSTRAP_CONTRACT.json"
            bootstrap_audit = attempt / "bootstrap_audit.json"
            expected_effective = {
                **base_arguments,
                "base_path": str(data_root),
                "results_path": str(results),
                "assignment_audit_path": str(assignment_audit),
            }
            contract = corrected_contract(
                unit,
                "pmnist",
                expected_effective,
                task_bank_sha256=task_digest,
                order_sha256=None,
            )
            atomic_json(contract_path, contract)
            command, observed_effective = build_corrected_command(
                unit,
                _pmnist_compatible_lock(lock),
                "pmnist",
                python=python,
                mammoth_root=mammoth,
                data_root=data_root,
                results=results,
                assignment_audit=assignment_audit,
                bootstrap_audit=bootstrap_audit,
                contract=contract_path,
                bank_or_order_manifest=task_bank_path,
            )
            if observed_effective != expected_effective:
                raise RuntimeError("PMNIST command differs from its bootstrap contract")
            environment_values = corrected_environment(
                unit, _pmnist_compatible_lock(lock), project, mammoth
            )
        else:
            compatible = _cifar_compatible_lock(lock)
            command = build_assignment_command(
                unit,
                compatible,
                python=python,
                mammoth_root=mammoth,
                data_root=data_root,
                results=results,
                audit=assignment_audit,
                device="cuda:0",
            )
            environment_values = assignment_environment(
                unit, compatible, project, mammoth
            )

        started = time.time()
        attempt_manifest = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "RUNNING",
            "campaign": args.campaign,
            "unit": unit.as_dict(),
            "lock_content_sha256": canonical_sha256(lock),
            "stream_identity": stream,
            "effective_args_sha256": canonical_sha256(base_arguments),
            **provenance,
            "command": command,
            "attempt": attempt.name,
            "started_unix": started,
        }
        attempt_manifest_path = attempt / "RUN_MANIFEST.json"
        atomic_json(attempt_manifest_path, attempt_manifest)
        sealed = False
        try:
            with stdout.open("wb") as out_handle, stderr.open("wb") as err_handle:
                completed = subprocess.run(
                    command,
                    cwd=project,
                    env=environment_values,
                    stdout=out_handle,
                    stderr=err_handle,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Mammoth returned {completed.returncode}; see {stdout} and {stderr}"
                )
            if args.campaign == "pmnist":
                if bootstrap_audit is None:
                    raise RuntimeError("internal PMNIST bootstrap-audit path is absent")
                bootstrap = strict_json(
                    real_file(bootstrap_audit, "PMNIST bootstrap audit")
                )
                contract_path = attempt / "BOOTSTRAP_CONTRACT.json"
                if not isinstance(bootstrap, Mapping) or not (
                    bootstrap.get("schema_version")
                    == CORRECTED_BOOTSTRAP_AUDIT_SCHEMA
                    and bootstrap.get("status") == "PASS"
                    and bootstrap.get("unit_id") == unit.unit_id
                    and bootstrap.get("benchmark") == unit.benchmark
                    and bootstrap.get("seed") == unit.seed
                    and bootstrap.get("mammoth_commit") == UPSTREAM_COMMIT
                    and bootstrap.get("contract_sha256")
                    == sha256_file(contract_path)
                    and bootstrap.get("task_bank_sha256") == task_digest
                    and bootstrap.get("task_permutations_consumed") == 20
                    and bootstrap.get("order_sha256") is None
                    and bootstrap.get("observed_task_sets") is None
                ):
                    raise RuntimeError("PMNIST bootstrap audit differs from its contract")
                native_metrics = corrected_metrics(results, unit, {
                    **base_arguments,
                    "base_path": str(data_root),
                    "results_path": str(results),
                    "assignment_audit_path": str(assignment_audit),
                })
                bootstrap_digest: str | None = sha256_file(bootstrap_audit)
            else:
                compatible = _cifar_compatible_lock(lock)
                native_metrics = read_assignment_metrics(
                    results,
                    unit,
                    compatible,
                    expected_args=base_arguments,
                    data_root=data_root,
                    audit=assignment_audit,
                )
                _validate_cifar_original_order(results, unit)
                bootstrap_digest = None
            diagnostics = summarize_audit(unit, assignment_audit, base_arguments)
            _validate_assignment_geometry(unit, diagnostics)
            endpoints = _final_accuracy(native_metrics, args.campaign)
            record = {
                "schema_version": UNIT_SCHEMA,
                "status": "PASS",
                "campaign": args.campaign,
                "unit": unit.as_dict(),
                **provenance,
                "lock_content_sha256": canonical_sha256(lock),
                "stream_identity": stream,
                "task_bank_sha256": task_digest,
                "order_sha256": order_digest,
                "effective_args": base_arguments,
                "effective_args_sha256": canonical_sha256(base_arguments),
                "metrics": native_metrics,
                "endpoint_values": endpoints,
                "assignment_diagnostics": diagnostics,
                "bootstrap_audit_sha256": bootstrap_digest,
                "attempt": attempt.name,
                "runtime_seconds": time.time() - started,
            }
            attempt_manifest.update(
                {
                    "status": "COMPLETED",
                    "completed_unix": time.time(),
                    "returncode": completed.returncode,
                }
            )
            atomic_json(attempt_manifest_path, attempt_manifest)
            atomic_json(root / "UNIT.json", record)
            inventory = artifact_inventory(root, exclude=("DONE.json",))
            atomic_json(
                root / "DONE.json",
                {
                    "schema_version": DONE_SCHEMA,
                    "status": "PASS",
                    "unit_id": unit.unit_id,
                    "artifact_count": len(inventory),
                    "artifact_inventory": inventory,
                    "artifact_bundle_sha256": canonical_sha256(inventory),
                },
            )
            sealed = True
        except BaseException as error:
            if not sealed:
                _write_failure(attempt_manifest_path, attempt_manifest, error)
            raise
        return validate_completed_unit(
            root,
            unit,
            campaign=args.campaign,
            lock=lock,
            expected_provenance=provenance,
            project=project,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_unit(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
