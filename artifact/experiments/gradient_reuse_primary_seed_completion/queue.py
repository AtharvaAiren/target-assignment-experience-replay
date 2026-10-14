#!/usr/bin/env python3
"""Run the two missing image-benchmark seeds as immutable single-unit campaigns.

This is an operational queue, not a second scientific implementation.  It
imports the authenticated mechanism protocol from ``--project``, selects the
108 units fixed by the post-hoc completion lock, and delegates every outcome
to ``experiments.gradient_reuse_mechanisms.run``.  A completed unit is counted
only after the original runner's full campaign and unit validators accept it.

The queue is restartable.  Partial single-unit campaign roots are handed back
to the original runner, while completed roots are immutable and revalidated.
On a unit failure no new unit is launched; already-running units are allowed to
finish and seal before the queue records a failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


EXPECTED_LOCK_SHA256 = (
    "be5247f6f94ba5835be7b87377fda106487306b2215ba30e3955377b4a402012"
)
EXPECTED_PARENT_SUMMARY_SHA256 = (
    "06015597a934d8dba380b8e2e5b278eaf220c96e7c4288444cc0145bd84b7e86"
)
EXPECTED_PARENT_E2_SHA256 = (
    "14d6cc2b32422b993986e86d37a1068d77b26b53cd4ee0a19efaa1e2946e5f4b"
)
EXPECTED_E5_INVENTORY_SHA256 = (
    "d0b5b77defe49badff91db0861ee04cef642634c4a52ec013ce39e61cacc1c6e"
)
EXPECTED_PARENT_OVERLAY_BUNDLE_SHA256 = (
    "15586d828e1388653328c4e1a971ff3e5839411da7c1cbca3a9b59a82dfbf8d6"
)
EXPECTED_NEW_UNITS = 108
EXPECTED_E5_UNITS = 54
COMPLETION_SEEDS = (3404, 3405)
BENCHMARKS = ("split_cifar100", "split_miniimagenet")
REFERENCE_METHODS = ("naive_sgd", "mer_current_only")
ASSIGNMENT_METHODS = (
    "experience_replay",
    "gem",
    "agem",
    "mer",
    "pcgrad_replay",
)
ALL_METHODS = REFERENCE_METHODS + ASSIGNMENT_METHODS
PERMUTATION_REPLICAS = (0, 1, 2, 3)
DEVICE = "cuda:0"
CUDA_VISIBLE_DEVICE = "0"
INITIAL_EXCLUSIVE_GPU_GATE_MIB = 28_000
PER_UNIT_LAUNCH_GPU_GATE_MIB = 14_000
MINIMUM_FREE_DISK_GIB = 100
POLL_SECONDS = 10
MAXIMUM_PROCESSES = 2

STATE_SCHEMA = "gradient_reuse.primary_seed_completion_queue_state.v1"
FAILURE_SCHEMA = "gradient_reuse.primary_seed_completion_queue_failure.v1"
DONE_SCHEMA = "gradient_reuse.primary_seed_completion_queue_done.v1"
AGGREGATE_SCHEMA = "gradient_reuse.primary_seed_completion_aggregate.v1"
RUN_BINDING_SCHEMA = "gradient_reuse.primary_seed_completion_run_binding.v1"

# Outcome-independent ordering informed only by runtimes from completed seeds.
# Running the longest families first reduces the two-slot makespan.  Dataset,
# seed, intervention, and replica are deterministic tie breakers.
METHOD_PRIORITY = {
    "mer_current_only": 0,
    "mer": 1,
    "gem": 2,
    "pcgrad_replay": 3,
    "agem": 4,
    "experience_replay": 5,
    "naive_sgd": 6,
}
BENCHMARK_PRIORITY = {"split_miniimagenet": 0, "split_cifar100": 1}
INTERVENTION_PRIORITY = {"none": 0, "paired": 1, "permuted": 2}


@dataclass(frozen=True)
class ScientificModules:
    protocol: ModuleType
    runner: ModuleType
    overlay: ModuleType
    notice_identity: ModuleType


@dataclass
class ActiveUnit:
    unit: Any
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: Any
    started_unix: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid strict JSON: {path}") from error


def atomic_json(path: Path, value: Any) -> None:
    """Atomically publish finite JSON and durably sync its directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe JSON destination: {path}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _real_file(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"{role} is absent, symbolic, or not a file: {raw}")
    return raw.resolve(strict=True)


def _real_directory(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{role} is absent, symbolic, or not a directory: {raw}")
    return raw.resolve(strict=True)


def _python_entry(path: Path) -> tuple[Path, Path]:
    """Validate an executable entry while retaining its lexical venv identity."""

    lexical = Path(path).expanduser()
    if not lexical.is_absolute() or not lexical.exists() or not os.access(lexical, os.X_OK):
        raise RuntimeError(f"Python entry is absent or not executable: {lexical}")
    target = lexical.resolve(strict=True)
    if not target.is_file():
        raise RuntimeError(f"Python entry target is not a file: {target}")
    return lexical, target


def _completion_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise RuntimeError("completion root must be an absolute path")
    if raw.is_symlink():
        raise RuntimeError("completion root must not be symbolic")
    parent = _real_directory(raw.parent, "completion-root parent")
    resolved = parent / raw.name
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_dir()):
        raise RuntimeError(f"completion root is unsafe: {resolved}")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _open_queue_lock(completion_root: Path) -> int:
    path = completion_root.parent / f".{completion_root.name}.primary-seed-queue.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe queue-lock path: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot safely open queue lock: {path}") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise RuntimeError(f"queue lock is not a singly linked file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another primary-seed queue owns {completion_root}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()}\ncompletion_root={completion_root}\n".encode(),
        )
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _load_scientific(project: Path) -> ScientificModules:
    expected = {
        "protocol": project / "experiments/gradient_reuse_mechanisms/protocol.py",
        "runner": project / "experiments/gradient_reuse_mechanisms/run.py",
        "overlay": project / "experiments/gradient_reuse_mechanisms/overlay.py",
        "notice_identity": (
            project / "experiments/gradient_reuse_mechanisms/notice_identity.py"
        ),
    }
    for role, path in expected.items():
        _real_file(path, f"scientific {role} module")
    sys.path.insert(0, str(project))
    modules = ScientificModules(
        protocol=importlib.import_module(
            "experiments.gradient_reuse_mechanisms.protocol"
        ),
        runner=importlib.import_module("experiments.gradient_reuse_mechanisms.run"),
        overlay=importlib.import_module(
            "experiments.gradient_reuse_mechanisms.overlay"
        ),
        notice_identity=importlib.import_module(
            "experiments.gradient_reuse_mechanisms.notice_identity"
        ),
    )
    for role, module in (
        ("protocol", modules.protocol),
        ("runner", modules.runner),
        ("overlay", modules.overlay),
        ("notice_identity", modules.notice_identity),
    ):
        observed = Path(str(module.__file__)).resolve(strict=True)
        if observed != expected[role].resolve(strict=True):
            raise RuntimeError(
                f"scientific {role} imported from another project: {observed}"
            )
    return modules


def _selection_key(unit: Any) -> tuple[Any, ...]:
    replica = -1 if unit.permutation_replica is None else int(unit.permutation_replica)
    return (
        METHOD_PRIORITY[unit.method],
        BENCHMARK_PRIORITY[unit.benchmark],
        int(unit.seed),
        INTERVENTION_PRIORITY[unit.intervention],
        replica,
        unit.unit_id,
    )


def _select_units(protocol: ModuleType, lock: Mapping[str, Any]) -> list[Any]:
    units = [
        unit
        for unit in protocol.enumerate_units(lock)
        if unit.benchmark in BENCHMARKS
        and unit.method in ALL_METHODS
        and int(unit.seed) in COMPLETION_SEEDS
    ]
    if len(units) != EXPECTED_NEW_UNITS:
        raise RuntimeError(
            f"primary seed completion selected {len(units)} units, expected "
            f"{EXPECTED_NEW_UNITS}"
        )
    identifiers = [unit.unit_id for unit in units]
    if len(set(identifiers)) != EXPECTED_NEW_UNITS:
        raise RuntimeError("primary seed completion unit IDs are not unique")
    for benchmark in BENCHMARKS:
        for seed in COMPLETION_SEEDS:
            selected = [
                unit
                for unit in units
                if unit.benchmark == benchmark and int(unit.seed) == seed
            ]
            if len(selected) != 27:
                raise RuntimeError(
                    f"{benchmark}/seed {seed} has {len(selected)} units, expected 27"
                )
            for method in REFERENCE_METHODS:
                rows = [unit for unit in selected if unit.method == method]
                if len(rows) != 1 or not (
                    rows[0].intervention == "none"
                    and rows[0].permutation_replica is None
                ):
                    raise RuntimeError(f"reference arm coverage changed: {method}")
            for method in ASSIGNMENT_METHODS:
                rows = [unit for unit in selected if unit.method == method]
                paired = [unit for unit in rows if unit.intervention == "paired"]
                permuted = [unit for unit in rows if unit.intervention == "permuted"]
                if not (
                    len(rows) == 5
                    and len(paired) == 1
                    and paired[0].permutation_replica is None
                    and tuple(sorted(x.permutation_replica for x in permuted))
                    == PERMUTATION_REPLICAS
                ):
                    raise RuntimeError(f"assignment-arm coverage changed: {method}")
    return sorted(units, key=_selection_key)


def _validate_execution_lock(path: Path) -> dict[str, Any]:
    path = _real_file(path, "primary seed completion execution lock")
    value = strict_json(path)
    if not isinstance(value, Mapping) or not (
        value.get("schema_version")
        == "gradient_reuse.primary_seed_completion_lock.v1"
        and value.get("status")
        == "POST_HOC_EXECUTION_PLAN_LOCKED_BEFORE_NEW_IMAGE_OUTCOMES"
    ):
        raise RuntimeError("primary seed completion execution lock identity changed")
    original = value.get("original_protocol", {})
    execution = value.get("new_execution", {})
    reused = value.get("reused_evidence", {})
    if not (
        original.get("lock_relative_path") == "configs/mechanisms.json"
        and original.get("lock_sha256") == EXPECTED_LOCK_SHA256
        and tuple(original.get("completion_seeds", ())) == COMPLETION_SEEDS
        and tuple(original.get("permutation_replicas", ())) == PERMUTATION_REPLICAS
        and tuple(execution.get("benchmarks", ())) == BENCHMARKS
        and tuple(execution.get("assignment_only_methods", ()))
        == ASSIGNMENT_METHODS
        and tuple(execution.get("reference_methods", ())) == REFERENCE_METHODS
        and execution.get("expected_units_per_benchmark_seed") == 27
        and execution.get("expected_new_units") == EXPECTED_NEW_UNITS
        and execution.get("maximum_concurrent_gpu_processes")
        == MAXIMUM_PROCESSES
        and execution.get("gpu_device") == DEVICE
        and execution.get("initial_exclusive_free_gpu_memory_mib")
        == INITIAL_EXCLUSIVE_GPU_GATE_MIB
        and execution.get("minimum_free_gpu_memory_mib_per_unit_launch")
        == PER_UNIT_LAUNCH_GPU_GATE_MIB
        and execution.get("minimum_free_disk_gib_before_launch")
        == MINIMUM_FREE_DISK_GIB
        and reused.get("parent_three_seed_summary", {}).get("sha256")
        == EXPECTED_PARENT_SUMMARY_SHA256
        and reused.get("parent_three_seed_secondary_endpoints", {}).get("sha256")
        == EXPECTED_PARENT_E2_SHA256
        and reused.get("pmnist_seed_extension", {}).get(
            "unit_and_done_inventory_sha256"
        )
        == EXPECTED_E5_INVENTORY_SHA256
        and value.get("operational_correction", {}).get(
            "runtime_overlay_bundle_sha256"
        )
        == EXPECTED_PARENT_OVERLAY_BUNDLE_SHA256
    ):
        raise RuntimeError("primary seed completion execution lock contents changed")
    return dict(value)


def _validate_parent_inputs(parent_summary: Path, parent_e2: Path) -> None:
    if sha256_file(parent_summary) != EXPECTED_PARENT_SUMMARY_SHA256:
        raise RuntimeError("parent three-seed SUMMARY.json digest changed")
    if sha256_file(parent_e2) != EXPECTED_PARENT_E2_SHA256:
        raise RuntimeError("parent E2 secondary-endpoint digest changed")
    summary = strict_json(parent_summary)
    e2 = strict_json(parent_e2)
    if not isinstance(summary, Mapping) or not (
        summary.get("schema_version")
        == "gradient_reuse.mechanism_aggregate_summary.v3"
        and summary.get("paper_eligible") is True
        and summary.get("unit_count") == 288
    ):
        raise RuntimeError("parent three-seed summary identity changed")
    if not isinstance(e2, Mapping) or not (
        e2.get("schema_version") == "gradient_reuse.reviewer_extension_e2.v1"
        and e2.get("status") == "PASS"
        and e2.get("raw_unit_count") == 288
        and e2.get("permutation_replicas_are_nested_monte_carlo") is True
    ):
        raise RuntimeError("parent E2 secondary-endpoint identity changed")


def _artifact_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"artifact inventory forbids symlinks: {path}")
        if path.is_file() and relative not in excluded:
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                raise RuntimeError(f"unfinished temporary artifact exists: {path}")
            output[relative] = sha256_file(path)
    return output


def _validate_e5(e5_root: Path, protocol: ModuleType, lock: Mapping[str, Any]) -> dict[str, str]:
    expected_base = [
        unit
        for unit in protocol.enumerate_units(lock)
        if unit.benchmark == "permuted_mnist"
        and unit.method in ALL_METHODS
        and int(unit.seed) in COMPLETION_SEEDS
    ]
    if len(expected_base) != EXPECTED_E5_UNITS:
        raise RuntimeError("original protocol no longer yields exactly 54 E5 units")
    expected = {
        f"E5_pmnist_seed_extension__{unit.unit_id}": unit
        for unit in expected_base
    }
    observed_directories = {
        path.name
        for path in (e5_root / "units").glob("E5_pmnist_seed_extension__*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_directories != set(expected):
        raise RuntimeError(
            "E5 unit coverage changed: "
            f"missing={sorted(set(expected) - observed_directories)}, "
            f"unexpected={sorted(observed_directories - set(expected))}"
        )
    inventory: dict[str, str] = {}
    for extension_id, unit in sorted(expected.items()):
        root = e5_root / "units" / extension_id
        unit_path = _real_file(root / "UNIT.json", f"E5 UNIT {extension_id}")
        done_path = _real_file(root / "DONE.json", f"E5 DONE {extension_id}")
        record = strict_json(unit_path)
        done = strict_json(done_path)
        expected_unit = {
            "experiment": "E5_pmnist_seed_extension",
            "priority": 4,
            "benchmark": unit.benchmark,
            "method": unit.method,
            "mammoth_dataset": unit.mammoth_dataset,
            "mammoth_model": unit.mammoth_model,
            "intervention": unit.intervention,
            "seed": unit.seed,
            "permutation_replica": unit.permutation_replica,
            "epochs": 1,
            "buffer_size": None if unit.method in REFERENCE_METHODS else 500,
            "state_matched_probe": False,
            "final_confusion": False,
            "resource_weight": 1,
        }
        if not isinstance(record, Mapping) or not (
            record.get("schema_version")
            == "gradient_reuse.reviewer_extension_unit.v1"
            and record.get("paper_eligible") is True
            and record.get("unit") == expected_unit
            and isinstance(record.get("identity"), Mapping)
            and record["identity"].get("unit") == expected_unit
            and record["identity"].get("paper_eligible") is True
        ):
            raise RuntimeError(f"E5 UNIT identity changed: {extension_id}")
        if not isinstance(done, Mapping) or not (
            done.get("schema_version")
            == "gradient_reuse.reviewer_extension_unit_done.v1"
            and done.get("status") == "PASS"
            and done.get("identity_sha256") == canonical_sha256(record["identity"])
        ):
            raise RuntimeError(f"E5 DONE identity changed: {extension_id}")
        claimed = done.get("artifact_inventory")
        observed = _artifact_inventory(root, exclude=("DONE.json",))
        if not isinstance(claimed, Mapping) or dict(claimed) != observed:
            raise RuntimeError(f"E5 artifact inventory changed: {extension_id}")
        if not (
            done.get("artifact_bundle_sha256") == canonical_sha256(observed)
            and done.get("artifact_count") == len(observed)
        ):
            raise RuntimeError(f"E5 DONE bundle changed: {extension_id}")
        inventory[f"{extension_id}/UNIT.json"] = sha256_file(unit_path)
        inventory[f"{extension_id}/DONE.json"] = sha256_file(done_path)
    if len(inventory) != 108 or canonical_sha256(inventory) != (
        EXPECTED_E5_INVENTORY_SHA256
    ):
        raise RuntimeError("E5 UNIT/DONE inventory digest changed")
    return inventory


def _expected_context(
    scientific: ScientificModules,
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_manifest: Path,
    mammoth_root: Path,
    data_root: Path,
    data_manifest: Path,
    environment_manifest: Path,
    units: Sequence[Any],
) -> dict[str, Any]:
    source_bundle, source_files = scientific.protocol.validate_source_manifest(
        source_manifest, repo=Path(scientific.runner.REPO), lock_path=lock_path
    )
    data = scientific.runner.validate_data_manifest(
        data_manifest, data_root=data_root
    )
    environment = scientific.runner.validate_environment_manifest(
        environment_manifest, mammoth_root=mammoth_root, device=DEVICE
    )
    scientific.runner.validate_required_mammoth_packages(environment)
    overlay_files = scientific.overlay.overlay_files()
    overlay_bundle = canonical_sha256(overlay_files)
    installed_overlay_manifest = {
        "schema_version": "gradient_reuse.mammoth_overlay.v1",
        "upstream_commit": scientific.protocol.UPSTREAM_COMMIT,
        "overlay_bundle_sha256": overlay_bundle,
        "files": overlay_files,
    }
    installed_bytes = (
        json.dumps(installed_overlay_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    effective = {
        unit.unit_id: canonical_sha256(
            scientific.runner.effective_mammoth_args(
                unit, lock, device=DEVICE, smoke=False
            )
        )
        for unit in units
    }
    return {
        "source_bundle_sha256": source_bundle,
        "source_files": source_files,
        "overlay_files": overlay_files,
        "overlay_bundle_sha256": overlay_bundle,
        "overlay_manifest_sha256": hashlib.sha256(installed_bytes).hexdigest(),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "effective_args_sha256": effective,
    }


def _invocation(
    unit: Any,
    context: Mapping[str, Any],
    *,
    lock_path: Path,
    source_manifest: Path,
    data_manifest: Path,
    environment_manifest: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.mechanism_invocation.v2",
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_bundle_sha256": context["source_bundle_sha256"],
        "overlay_manifest_sha256": context["overlay_manifest_sha256"],
        "overlay_bundle_sha256": context["overlay_bundle_sha256"],
        "upstream_commit": "e75a491c69fd729edeb01431afb753d9157d9a81",
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_bundle_sha256": context["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest),
        "environment_bundle_sha256": context["environment_bundle_sha256"],
        "smoke": False,
        "paper_eligible": False,
        "device": DEVICE,
        "selection_scheme": "canonical_filter_v1",
        "shard_count": None,
        "shard_index": None,
        "selected_unit_ids": [unit.unit_id],
        "effective_args_sha256": {
            unit.unit_id: context["effective_args_sha256"][unit.unit_id]
        },
    }


def _validate_completed_campaign(
    unit: Any,
    root: Path,
    scientific: ScientificModules,
    context: Mapping[str, Any],
    *,
    lock_path: Path,
    source_manifest: Path,
    data_manifest: Path,
    environment_manifest: Path,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"completed unit campaign root is unsafe: {root}")
    invocation = _invocation(
        unit,
        context,
        lock_path=lock_path,
        source_manifest=source_manifest,
        data_manifest=data_manifest,
        environment_manifest=environment_manifest,
    )
    scientific.runner._validate_completed_campaign_root(root, invocation=invocation)
    identity = scientific.protocol.unit_identity(
        unit,
        lock_sha256=invocation["lock_sha256"],
        source_bundle_sha256=context["source_bundle_sha256"],
        overlay_bundle_sha256=context["overlay_bundle_sha256"],
        data_manifest_sha256=invocation["data_manifest_sha256"],
        data_bundle_sha256=context["data_bundle_sha256"],
        environment_manifest_sha256=invocation["environment_manifest_sha256"],
        environment_bundle_sha256=context["environment_bundle_sha256"],
        effective_args_sha256=context["effective_args_sha256"][unit.unit_id],
        device=DEVICE,
        smoke=False,
    )
    record = scientific.runner._validate_completed_unit(
        root / "units" / unit.unit_id, identity
    )
    if not (
        record.get("paper_eligible") is True
        and record.get("benchmark") == unit.benchmark
        and record.get("method") == unit.method
        and record.get("intervention") == unit.intervention
        and record.get("seed") == unit.seed
        and record.get("permutation_replica") == unit.permutation_replica
    ):
        raise RuntimeError(f"completed scientific UNIT changed: {unit.unit_id}")
    return record


def _validate_partial_campaign(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"partial unit campaign root is unsafe: {root}")
    manifest = root / "RUN_MANIFEST.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(
            f"preexisting unit campaign lacks a real RUN_MANIFEST.json: {root}"
        )
    value = strict_json(manifest)
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "gradient_reuse.mechanism_campaign.v2"
    ):
        raise RuntimeError(f"partial unit campaign schema changed: {root}")
    if value.get("status") not in {"RUNNING", "FAILED"}:
        raise RuntimeError(f"partial unit campaign has an invalid status: {root}")


def _completed_units(
    units: Sequence[Any],
    unit_campaigns: Path,
    validator: Callable[[Any, Path], dict[str, Any]],
) -> set[str]:
    completed: set[str] = set()
    expected_names = {unit.unit_id for unit in units}
    if unit_campaigns.exists():
        if unit_campaigns.is_symlink() or not unit_campaigns.is_dir():
            raise RuntimeError(f"unit-campaign directory is unsafe: {unit_campaigns}")
        expected_locks = {
            f".{unit_id}.mechanism-launch.lock" for unit_id in expected_names
        }
        unexpected: list[str] = []
        for path in unit_campaigns.iterdir():
            if path.name in expected_names:
                if path.is_symlink() or not path.is_dir():
                    unexpected.append(path.name)
            elif path.name in expected_locks:
                if path.is_symlink() or not path.is_file():
                    unexpected.append(path.name)
            else:
                unexpected.append(path.name)
        if unexpected:
            raise RuntimeError(f"unexpected unit-campaign roots: {sorted(unexpected)}")
    for unit in units:
        root = unit_campaigns / unit.unit_id
        if not root.exists() and not root.is_symlink():
            continue
        done = root / "DONE.json"
        if done.exists() or done.is_symlink():
            validator(unit, root)
            completed.add(unit.unit_id)
        else:
            _validate_partial_campaign(root)
    return completed


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
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError(f"unexpected nvidia-smi memory output: {rows}")
    return int(rows[0])


def _process_rows() -> list[tuple[int, str]]:
    completed = subprocess.run(
        ["ps", "-ww", "-eo", "pid=,args="],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, command = stripped.partition(" ")
        if pid.isdigit():
            output.append((int(pid), command))
    return output


def _orphan_processes(completion_root: Path, aggregate_script: Path) -> list[tuple[int, str]]:
    root_text = str(completion_root)
    aggregate_text = str(aggregate_script)
    own_pid = os.getpid()
    return [
        (pid, command)
        for pid, command in _process_rows()
        if pid != own_pid
        and root_text in command
        and (
            "experiments.gradient_reuse_mechanisms.run" in command
            or "mammoth_cli_identity_bridge.py" in command
            or "/main.py" in command
            or aggregate_text in command
        )
    ]


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


def _unit_command(
    unit: Any,
    *,
    project: Path,
    lock_path: Path,
    completion_root: Path,
    python_wrapper: Path,
    mammoth_root: Path,
    data_root: Path,
    data_manifest: Path,
    environment_manifest: Path,
) -> list[str]:
    return [
        str(python_wrapper),
        "-B",
        str(
            project
            / "experiments/gradient_reuse_primary_seed_completion"
            / "runner.py"
        ),
        "--full",
        "--lock",
        str(lock_path),
        "--source-manifest",
        str(project / "provenance/source/mechanisms.json"),
        "--mammoth-root",
        str(mammoth_root),
        "--data-root",
        str(data_root),
        "--data-manifest",
        str(data_manifest),
        "--environment-manifest",
        str(environment_manifest),
        "--out",
        str(completion_root / "unit_campaigns" / unit.unit_id),
        "--device",
        DEVICE,
        "--unit-id",
        unit.unit_id,
    ]


def _launch_unit(
    unit: Any,
    *,
    project: Path,
    lock_path: Path,
    completion_root: Path,
    python_wrapper: Path,
    mammoth_root: Path,
    data_root: Path,
    data_manifest: Path,
    environment_manifest: Path,
) -> ActiveUnit:
    log = completion_root / "logs" / "units" / f"{unit.unit_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_symlink() or (log.exists() and not log.is_file()):
        raise RuntimeError(f"unsafe unit log: {log}")
    handle = log.open("a", encoding="utf-8")
    command = _unit_command(
        unit,
        project=project,
        lock_path=lock_path,
        completion_root=completion_root,
        python_wrapper=python_wrapper,
        mammoth_root=mammoth_root,
        data_root=data_root,
        data_manifest=data_manifest,
        environment_manifest=environment_manifest,
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
    return ActiveUnit(
        unit=unit,
        process=process,
        log_path=log,
        log_handle=handle,
        started_unix=time.time(),
    )


def _next_queue_attempt(state_path: Path) -> int:
    observed_attempts: list[int] = []
    if state_path.exists() or state_path.is_symlink():
        if state_path.is_symlink() or not state_path.is_file():
            raise RuntimeError(f"queue state path is unsafe: {state_path}")
        value = strict_json(state_path)
        if not isinstance(value, Mapping) or value.get("schema_version") != STATE_SCHEMA:
            raise RuntimeError("existing queue state schema changed")
        attempt = value.get("queue_attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise RuntimeError("existing queue attempt is invalid")
        observed_attempts.append(attempt)

    failures = state_path.parent / "failures"
    if failures.exists() or failures.is_symlink():
        if failures.is_symlink() or not failures.is_dir():
            raise RuntimeError(f"queue failure directory is unsafe: {failures}")
        for path in sorted(failures.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe queue failure artifact: {path}")
            if not (
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
                and attempt >= 1
                and int(path.name[8:-5]) == attempt
            ):
                raise RuntimeError(f"queue failure record changed: {path}")
            observed_attempts.append(attempt)
    return max(observed_attempts, default=0) + 1


def _write_or_validate_binding(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError("primary seed completion RUN_BINDING.json changed")
        return
    atomic_json(path, dict(value))


def _record_failure(
    completion_root: Path,
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
        "expected_units": EXPECTED_NEW_UNITS,
        "completed_unit_ids": sorted(completed),
    }
    numbered = completion_root / "failures" / f"FAILURE_{queue_attempt:03d}.json"
    if numbered.exists() or numbered.is_symlink():
        raise RuntimeError(f"numbered failure record already exists: {numbered}")
    atomic_json(numbered, failure)
    atomic_json(completion_root / "FAILURE.json", failure)
    return numbered


def _run_scheduler(
    units: Sequence[Any],
    completed: set[str],
    *,
    project: Path,
    lock_path: Path,
    completion_root: Path,
    python_wrapper: Path,
    mammoth_root: Path,
    data_root: Path,
    data_manifest: Path,
    environment_manifest: Path,
    max_processes: int,
    validate: Callable[[Any, Path], dict[str, Any]],
    update: Callable[[str, Mapping[str, Any] | None], None],
) -> None:
    pending = [unit for unit in units if unit.unit_id not in completed]
    active: list[ActiveUnit] = []
    failed: list[dict[str, Any]] = []
    initial_gate_passed = False
    try:
        while pending or active:
            survivors: list[ActiveUnit] = []
            for running in active:
                status = running.process.poll()
                if status is None:
                    survivors.append(running)
                    continue
                running.log_handle.close()
                if status != 0:
                    failed.append(
                        {
                            "unit_id": running.unit.unit_id,
                            "returncode": status,
                            "log": str(running.log_path),
                        }
                    )
                    continue
                try:
                    validate(
                        running.unit,
                        completion_root / "unit_campaigns" / running.unit.unit_id,
                    )
                    completed.add(running.unit.unit_id)
                except BaseException as error:
                    failed.append(
                        {
                            "unit_id": running.unit.unit_id,
                            "returncode": status,
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
                raise RuntimeError(f"primary seed completion unit failure(s): {failed}")

            if not pending and not active:
                update(
                    "UNITS_COMPLETE",
                    {
                        "completed_units": len(completed),
                        "pending_units": 0,
                        "active_units": [],
                    },
                )
                break

            free_disk = _free_disk_gib(completion_root)
            if free_disk < MINIMUM_FREE_DISK_GIB:
                raise RuntimeError(
                    f"disk gate failed: {free_disk} GiB free, "
                    f"{MINIMUM_FREE_DISK_GIB} GiB required"
                )
            free_gpu = _free_gpu_mib()
            launched: list[str] = []
            available_slots = max_processes - len(active)
            required_gpu = (
                PER_UNIT_LAUNCH_GPU_GATE_MIB
                if initial_gate_passed
                else INITIAL_EXCLUSIVE_GPU_GATE_MIB
            )
            if pending and available_slots and free_gpu >= required_gpu:
                # Launch only one process from each fresh memory observation.
                # This gives the first child time to create its CUDA context
                # before the queue decides whether a second 14-GiB reserve is
                # genuinely available.
                unit = pending.pop(0)
                active.append(
                    _launch_unit(
                        unit,
                        project=project,
                        lock_path=lock_path,
                        completion_root=completion_root,
                        python_wrapper=python_wrapper,
                        mammoth_root=mammoth_root,
                        data_root=data_root,
                        data_manifest=data_manifest,
                        environment_manifest=environment_manifest,
                    )
                )
                launched.append(unit.unit_id)
                initial_gate_passed = True
            stage = (
                "RUNNING_UNITS"
                if active
                else "WAITING_FOR_GPU_GATE"
                if pending
                else "UNITS_COMPLETE"
            )
            update(
                stage,
                {
                    "completed_units": len(completed),
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
                    "launch_gate_mib_used_this_cycle": (
                        required_gpu if launched else None
                    ),
                    "required_free_gpu_mib_for_next_launch": (
                        PER_UNIT_LAUNCH_GPU_GATE_MIB
                        if initial_gate_passed
                        else INITIAL_EXCLUSIVE_GPU_GATE_MIB
                    ),
                    "initial_exclusive_gpu_gate_mib": (
                        INITIAL_EXCLUSIVE_GPU_GATE_MIB
                    ),
                    "per_unit_launch_gpu_gate_mib": PER_UNIT_LAUNCH_GPU_GATE_MIB,
                    "free_disk_gib": free_disk,
                    "required_free_disk_gib": MINIMUM_FREE_DISK_GIB,
                    "initial_gpu_gate_passed": initial_gate_passed,
                },
            )
            if pending or active:
                time.sleep(POLL_SECONDS)
    except BaseException:
        # Scientific children may have already produced valuable work.  Do not
        # terminate them mid-trajectory; drain and close every log descriptor.
        while active:
            survivors = []
            for running in active:
                status = running.process.poll()
                if status is None:
                    survivors.append(running)
                    continue
                running.log_handle.close()
                if status == 0:
                    try:
                        validate(
                            running.unit,
                            completion_root / "unit_campaigns" / running.unit.unit_id,
                        )
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
    if len(completed) != EXPECTED_NEW_UNITS:
        raise RuntimeError(
            f"scheduler sealed {len(completed)} units, expected {EXPECTED_NEW_UNITS}"
        )


def _validate_aggregate(path: Path) -> dict[str, Any]:
    path = _real_file(path, "primary seed completion aggregate")
    value = strict_json(path)
    if not isinstance(value, Mapping) or not (
        value.get("schema_version") == AGGREGATE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("new_unit_count") == EXPECTED_NEW_UNITS
    ):
        raise RuntimeError("primary seed completion aggregate failed authentication")
    return dict(value)


def _validate_queue_done(
    path: Path,
    *,
    binding_path: Path,
    aggregate_path: Path,
    units: Sequence[Any],
    ordered_ids: Sequence[str],
) -> dict[str, Any]:
    value = strict_json(_real_file(path, "queue DONE"))
    expected_keys = {
        "schema_version",
        "status",
        "scientific_role",
        "new_unit_count",
        "completion_seeds",
        "benchmarks",
        "methods",
        "run_binding_sha256",
        "ordered_unit_ids_sha256",
        "unit_campaign_done_sha256",
        "unit_campaign_done_bundle_sha256",
        "aggregate_sha256",
        "completed_unix",
    }
    unit_done = {
        unit.unit_id: sha256_file(
            path.parent / "unit_campaigns" / unit.unit_id / "DONE.json"
        )
        for unit in units
    }
    completed_unix = value.get("completed_unix") if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping) or not (
        set(value) == expected_keys
        and value.get("schema_version") == DONE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("scientific_role") == "post_hoc_seed_completion"
        and value.get("new_unit_count") == EXPECTED_NEW_UNITS
        and value.get("completion_seeds") == list(COMPLETION_SEEDS)
        and value.get("benchmarks") == list(BENCHMARKS)
        and value.get("methods") == list(ALL_METHODS)
        and value.get("run_binding_sha256") == sha256_file(binding_path)
        and value.get("ordered_unit_ids_sha256")
        == canonical_sha256(list(ordered_ids))
        and value.get("unit_campaign_done_sha256") == unit_done
        and value.get("unit_campaign_done_bundle_sha256")
        == canonical_sha256(unit_done)
        and value.get("aggregate_sha256") == sha256_file(aggregate_path)
        and isinstance(completed_unix, (int, float))
        and not isinstance(completed_unix, bool)
        and 0.0 < float(completed_unix) < float("inf")
    ):
        raise RuntimeError("existing queue DONE failed authentication")
    return dict(value)


def _run_aggregate(
    *,
    project: Path,
    completion_root: Path,
    python_wrapper: Path,
    parent_summary: Path,
    parent_e2: Path,
    e5_root: Path,
    aggregate_script: Path,
) -> Path:
    output = completion_root / "PRIMARY_SEED_COMPLETION_AGGREGATE.json"
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise RuntimeError(f"aggregate output path is unsafe: {output}")
    log = completion_root / "logs" / "aggregate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_symlink() or (log.exists() and not log.is_file()):
        raise RuntimeError(f"aggregate log is unsafe: {log}")
    command = [
        str(python_wrapper),
        "-B",
        "-I",
        str(aggregate_script),
        "--project",
        str(project),
        "--completion-root",
        str(completion_root),
        "--output",
        str(output),
        "--parent-summary",
        str(parent_summary),
        "--parent-e2",
        str(parent_e2),
        "--e5-root",
        str(e5_root),
    ]
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"aggregate exited {completed.returncode}; diagnostic log: {log}"
        )
    _validate_aggregate(output)
    return output


def _remove_current_failure(completion_root: Path) -> None:
    path = completion_root / "FAILURE.json"
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"cannot clear unsafe current failure path: {path}")
    value = strict_json(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != FAILURE_SCHEMA:
        raise RuntimeError("cannot clear an unauthenticated current failure record")
    path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--python-entry", type=Path, required=True)
    parser.add_argument("--python-wrapper", type=Path, required=True)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--parent-e2", type=Path, required=True)
    parser.add_argument("--e5-root", type=Path, required=True)
    parser.add_argument("--aggregate-script", type=Path, required=True)
    parser.add_argument("--max-processes", type=int, default=MAXIMUM_PROCESSES)
    return parser


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "primary seed completion queue must run through a Python -B entry"
        )
    project = _real_directory(arguments.project, "installed scientific project")
    completion_root = _completion_root(arguments.completion_root)
    lock_path = _real_file(arguments.lock, "original mechanism lock")
    expected_lock = project / "configs/mechanisms.json"
    if lock_path != expected_lock.resolve(strict=True):
        raise RuntimeError("--lock must be exactly <project>/configs/mechanisms.json")
    if sha256_file(lock_path) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("original mechanism lock digest changed")
    source_manifest = _real_file(
        project / "provenance/source/mechanisms.json",
        "mechanism source manifest",
    )
    overlay = _real_directory(arguments.overlay, "mechanism source overlay")
    if overlay != (project / "mammoth_overlay").resolve(strict=True):
        raise RuntimeError("--overlay must be exactly <project>/mammoth_overlay")
    python_entry, python_target = _python_entry(arguments.python_entry)
    python_wrapper = _real_file(arguments.python_wrapper, "Python wrapper")
    if not os.access(python_wrapper, os.X_OK):
        raise RuntimeError(f"Python wrapper is not executable: {python_wrapper}")
    mammoth_root = _real_directory(arguments.mammoth_root, "Mammoth checkout")
    data_root = _real_directory(arguments.data_root, "mechanism data root")
    data_manifest = _real_file(arguments.data_manifest, "mechanism data manifest")
    environment_manifest = _real_file(
        arguments.environment_manifest, "mechanism environment manifest"
    )
    parent_summary = _real_file(arguments.parent_summary, "parent summary")
    parent_e2 = _real_file(arguments.parent_e2, "parent E2")
    e5_root = _real_directory(arguments.e5_root, "E5 reviewer-extension root")
    aggregate_script = _real_file(arguments.aggregate_script, "aggregate script")
    if arguments.max_processes not in {1, 2}:
        raise RuntimeError("--max-processes must be 1 or 2")
    max_processes = int(arguments.max_processes)

    protected = {
        "scientific project": project,
        "Mammoth checkout": mammoth_root,
        "data root": data_root,
        "E5 root": e5_root,
        "original lock": lock_path,
        "data manifest": data_manifest,
        "environment manifest": environment_manifest,
        "parent summary": parent_summary,
        "parent E2": parent_e2,
        "aggregate script": aggregate_script,
    }
    for role, path in protected.items():
        if _paths_overlap(completion_root, path):
            raise RuntimeError(f"completion root overlaps {role}: {path}")

    queue_lock = _open_queue_lock(completion_root)
    try:
        orphans = _orphan_processes(completion_root, aggregate_script)
        if orphans:
            lines = "\n".join(f"{pid} {command}" for pid, command in orphans)
            raise RuntimeError(
                "orphan primary-seed scientific process is still active:\n" + lines
            )
        completion_root.mkdir(parents=True, exist_ok=True)
        scientific = _load_scientific(project)
        lock = scientific.protocol.load_and_validate_lock(lock_path)
        if tuple(lock["runtime"]["seeds"]) != (3401, 3402, 3403, 3404, 3405):
            raise RuntimeError("original full lock no longer contains all five seeds")
        units = _select_units(scientific.protocol, lock)
        execution_lock_path = project / "configs/primary_seed_completion.json"
        execution_lock = _validate_execution_lock(execution_lock_path)
        _validate_parent_inputs(parent_summary, parent_e2)
        e5_inventory = _validate_e5(e5_root, scientific.protocol, lock)
        notice_identity = scientific.notice_identity.activate(mammoth_root)
        if not (
            notice_identity.get("overlay_file_count") == 19
            and notice_identity.get("overlay_bundle_sha256")
            == EXPECTED_PARENT_OVERLAY_BUNDLE_SHA256
        ):
            raise RuntimeError("historical NOTICE identity activation failed")
        scientific.overlay.verify_upstream(mammoth_root)
        context = _expected_context(
            scientific,
            lock=lock,
            lock_path=lock_path,
            source_manifest=source_manifest,
            mammoth_root=mammoth_root,
            data_root=data_root,
            data_manifest=data_manifest,
            environment_manifest=environment_manifest,
            units=units,
        )

        ordered_ids = [unit.unit_id for unit in units]
        binding = {
            "schema_version": RUN_BINDING_SCHEMA,
            "status": "LOCKED_BEFORE_NEW_IMAGE_OUTCOMES",
            "paths": {
                "project": str(project),
                "lock": str(lock_path),
                "source_manifest": str(source_manifest),
                "completion_root": str(completion_root),
                "python_entry_lexical": str(python_entry),
                "python_entry_target": str(python_target),
                "python_wrapper": str(python_wrapper),
                "mammoth_root": str(mammoth_root),
                "data_root": str(data_root),
                "data_manifest": str(data_manifest),
                "environment_manifest": str(environment_manifest),
                "overlay": str(overlay),
                "parent_summary": str(parent_summary),
                "parent_e2": str(parent_e2),
                "e5_root": str(e5_root),
                "aggregate_script": str(aggregate_script),
                "execution_lock": str(execution_lock_path),
            },
            "file_sha256": {
                "queue": sha256_file(Path(__file__).resolve()),
                "execution_lock": sha256_file(execution_lock_path),
                "lock": sha256_file(lock_path),
                "source_manifest": sha256_file(source_manifest),
                "python_entry_target": sha256_file(python_target),
                "python_wrapper": sha256_file(python_wrapper),
                "data_manifest": sha256_file(data_manifest),
                "environment_manifest": sha256_file(environment_manifest),
                "parent_summary": sha256_file(parent_summary),
                "parent_e2": sha256_file(parent_e2),
                "aggregate_script": sha256_file(aggregate_script),
            },
            "source_bundle_sha256": context["source_bundle_sha256"],
            "source_files": context["source_files"],
            "overlay_bundle_sha256": context["overlay_bundle_sha256"],
            "overlay_files": context["overlay_files"],
            "e5_unit_and_done_inventory_sha256": canonical_sha256(e5_inventory),
            "e5_unit_and_done_inventory": e5_inventory,
            "ordered_unit_ids": ordered_ids,
            "ordered_unit_ids_sha256": canonical_sha256(ordered_ids),
            "expected_new_units": EXPECTED_NEW_UNITS,
            "maximum_concurrent_gpu_processes": max_processes,
            "initial_exclusive_free_gpu_memory_mib": (
                INITIAL_EXCLUSIVE_GPU_GATE_MIB
            ),
            "minimum_free_gpu_memory_mib_per_unit_launch": (
                PER_UNIT_LAUNCH_GPU_GATE_MIB
            ),
            "minimum_free_disk_gib_before_launch": MINIMUM_FREE_DISK_GIB,
            "device": DEVICE,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICE,
            "execution_lock_scientific_role": execution_lock["scientific_role"],
        }
        binding_path = completion_root / "RUN_BINDING.json"
        _write_or_validate_binding(binding_path, binding)
        state_path = completion_root / "STATE.json"
        queue_attempt = _next_queue_attempt(state_path)

        def validate(unit: Any, root: Path) -> dict[str, Any]:
            return _validate_completed_campaign(
                unit,
                root,
                scientific,
                context,
                lock_path=lock_path,
                source_manifest=source_manifest,
                data_manifest=data_manifest,
                environment_manifest=environment_manifest,
            )

        completed = _completed_units(
            units, completion_root / "unit_campaigns", validate
        )

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
                    "expected_units": EXPECTED_NEW_UNITS,
                    "details": dict(details or {}),
                },
            )

        done_path = completion_root / "DONE.json"
        if done_path.exists() or done_path.is_symlink():
            # Re-run the deterministic aggregator even on terminal resume.  It
            # recomputes every input binding and accepts an existing output
            # only when the complete bytes are identical.
            aggregate_path = _run_aggregate(
                project=project,
                completion_root=completion_root,
                python_wrapper=python_wrapper,
                parent_summary=parent_summary,
                parent_e2=parent_e2,
                e5_root=e5_root,
                aggregate_script=aggregate_script,
            )
            _validate_queue_done(
                done_path,
                binding_path=binding_path,
                aggregate_path=aggregate_path,
                units=units,
                ordered_ids=ordered_ids,
            )
            _remove_current_failure(completion_root)
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "PASS",
                    "stage": "COMPLETE",
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": EXPECTED_NEW_UNITS,
                    "expected_units": EXPECTED_NEW_UNITS,
                    "details": {"done": str(done_path)},
                },
            )
            return done_path

        # A numbered copy of every prior failure is immutable under
        # ``failures/``.  Clear only the convenience pointer before a genuine
        # resume so monitors do not label the active retry as failed.
        _remove_current_failure(completion_root)
        try:
            update("AUTHENTICATED_INPUTS")
            _run_scheduler(
                units,
                completed,
                project=project,
                lock_path=lock_path,
                completion_root=completion_root,
                python_wrapper=python_wrapper,
                mammoth_root=mammoth_root,
                data_root=data_root,
                data_manifest=data_manifest,
                environment_manifest=environment_manifest,
                max_processes=max_processes,
                validate=validate,
                update=update,
            )
            # Rebuild every unit binding before producing pooled statistics.
            completed = _completed_units(
                units, completion_root / "unit_campaigns", validate
            )
            if len(completed) != EXPECTED_NEW_UNITS:
                raise RuntimeError("terminal unit revalidation is incomplete")
            update("AGGREGATING")
            aggregate_path = _run_aggregate(
                project=project,
                completion_root=completion_root,
                python_wrapper=python_wrapper,
                parent_summary=parent_summary,
                parent_e2=parent_e2,
                e5_root=e5_root,
                aggregate_script=aggregate_script,
            )
            unit_done = {
                unit.unit_id: sha256_file(
                    completion_root / "unit_campaigns" / unit.unit_id / "DONE.json"
                )
                for unit in units
            }
            done = {
                "schema_version": DONE_SCHEMA,
                "status": "PASS",
                "scientific_role": "post_hoc_seed_completion",
                "new_unit_count": EXPECTED_NEW_UNITS,
                "completion_seeds": list(COMPLETION_SEEDS),
                "benchmarks": list(BENCHMARKS),
                "methods": list(ALL_METHODS),
                "run_binding_sha256": sha256_file(binding_path),
                "ordered_unit_ids_sha256": canonical_sha256(ordered_ids),
                "unit_campaign_done_sha256": unit_done,
                "unit_campaign_done_bundle_sha256": canonical_sha256(unit_done),
                "aggregate_sha256": sha256_file(aggregate_path),
                "completed_unix": time.time(),
            }
            atomic_json(done_path, done)
            _remove_current_failure(completion_root)
            atomic_json(
                state_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "PASS",
                    "stage": "COMPLETE",
                    "queue_attempt": queue_attempt,
                    "queue_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "completed_units": EXPECTED_NEW_UNITS,
                    "expected_units": EXPECTED_NEW_UNITS,
                    "details": {
                        "done": str(done_path),
                        "aggregate": str(aggregate_path),
                    },
                },
            )
            return done_path
        except BaseException as error:
            failure_path = _record_failure(
                completion_root,
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
                    "expected_units": EXPECTED_NEW_UNITS,
                    "details": {"failure": str(failure_path)},
                },
            )
            raise
    finally:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_UN)
        finally:
            os.close(queue_lock)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(arguments)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
