"""Fail-closed aggregation of independently executed mechanism shards.

Shard summaries are never trusted.  This module authenticates every shard and
unit artifact from bytes, reconstructs the complete locked unit set, recomputes
the nested-permutation estimands, and writes a new immutable aggregate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from experiments.source_provenance import (
    validate_source_manifest as validate_shared_source_manifest,
)

from .protocol import (
    FULL_PROTOCOL_UNIT_COUNT,
    MechanismUnit,
    UPSTREAM_COMMIT,
    canonical_sha256,
    compute_diagnostic_estimands,
    compute_mechanism_estimands,
    enumerate_units,
    expected_paper_unit_count,
    load_and_validate_lock,
    mean_and_ci,
    mechanism_profile_id,
    mechanism_source_experiment,
    method_estimand_class,
    sha256_file,
    summarize_diagnostic_estimands,
    unit_identity,
)
from .run import (
    SHARD_SELECTION_SCHEME,
    _validate_cross_arm_comparability,
    effective_mammoth_args,
)


REPO = Path(__file__).resolve().parents[2]
# Backward-compatible name for the original, unreduced paper lock. Aggregation
# itself derives the count from the validated profile and never rewrites this.
EXPECTED_PAPER_UNITS = FULL_PROTOCOL_UNIT_COUNT
_HEX = frozenset("0123456789abcdef")


class MechanismAggregationError(RuntimeError):
    """Raised when shard evidence cannot support a scientific aggregate."""


@dataclass(frozen=True)
class AggregationIdentity:
    lock_sha256: str
    source_manifest_sha256: str
    source_bundle_sha256: str
    data_manifest_sha256: str
    data_bundle_sha256: str
    environment_manifest_sha256: str
    environment_bundle_sha256: str
    overlay_manifest_sha256: str
    overlay_bundle_sha256: str
    upstream_commit: str
    device: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name.endswith("_sha256"):
                _digest(getattr(self, name), context=f"aggregation identity {name}")
        if self.upstream_commit != UPSTREAM_COMMIT:
            raise MechanismAggregationError("aggregation identity uses another Mammoth commit")
        if self.device != "cuda:0":
            raise MechanismAggregationError("paper aggregation identity requires cuda:0")

    def as_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in self.__dataclass_fields__
        }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise MechanismAggregationError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except MechanismAggregationError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise MechanismAggregationError(f"artifact is not strict JSON: {path}") from error


def _digest(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MechanismAggregationError(f"invalid SHA-256 for {context}")
    return value


def _relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MechanismAggregationError(f"{context} must be a relative path")
    if "\\" in value or "\x00" in value:
        raise MechanismAggregationError(f"unsafe {context}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MechanismAggregationError(f"unsafe {context}: {value!r}")
    if path.as_posix() != value:
        raise MechanismAggregationError(f"noncanonical {context}: {value!r}")
    return value


def _mapping_bundle(
    payload: Mapping[str, Any],
    *,
    field: str,
    bundle_field: str,
    context: str,
) -> tuple[str, dict[str, str]]:
    raw = payload.get(field)
    if not isinstance(raw, Mapping) or not raw:
        raise MechanismAggregationError(f"{context} has no {field} inventory")
    inventory: dict[str, str] = {}
    for relative, digest in raw.items():
        path = _relative(relative, context=f"{context} inventory path")
        if path in inventory:
            raise MechanismAggregationError(f"duplicate {context} inventory path: {path}")
        inventory[path] = _digest(digest, context=f"{context}/{path}")
    bundle = canonical_sha256(inventory)
    if payload.get(bundle_field) != bundle:
        raise MechanismAggregationError(f"{context} bundle hash mismatch")
    return bundle, inventory


def load_aggregation_identity(
    *,
    lock_path: Path,
    source_manifest: Path,
    data_manifest: Path,
    environment_manifest: Path,
    overlay_manifest: Path,
    repo: Path = REPO,
) -> tuple[dict[str, Any], AggregationIdentity]:
    """Authenticate the five external identities used by every shard."""

    lock_path = lock_path.resolve(strict=True)
    lock = load_and_validate_lock(lock_path)
    source_manifest = source_manifest.resolve(strict=True)
    source_bundle, _ = validate_shared_source_manifest(
        source_manifest, mechanism_source_experiment(lock), repo=repo
    )

    data_manifest = data_manifest.resolve(strict=True)
    data = strict_json(data_manifest)
    if not isinstance(data, Mapping):
        raise MechanismAggregationError("data manifest must be an object")
    if data.get("schema_version") != "gradient_reuse.mechanism_data_manifest.v1":
        raise MechanismAggregationError("data manifest schema mismatch")
    if data.get("status") != "LOCKED_BEFORE_SCIENTIFIC_RUN":
        raise MechanismAggregationError("data manifest is not locked")
    data_bundle, _ = _mapping_bundle(
        data,
        field="files",
        bundle_field="data_bundle_sha256",
        context="data manifest",
    )

    environment_manifest = environment_manifest.resolve(strict=True)
    environment = strict_json(environment_manifest)
    if not isinstance(environment, Mapping):
        raise MechanismAggregationError("environment manifest must be an object")
    if environment.get("schema_version") != "gradient_reuse.mechanism_environment.v1":
        raise MechanismAggregationError("environment manifest schema mismatch")
    if environment.get("status") != "LOCKED_BEFORE_SCIENTIFIC_RUN":
        raise MechanismAggregationError("environment manifest is not locked")
    observed = environment.get("observed")
    if not isinstance(observed, Mapping):
        raise MechanismAggregationError("environment manifest lacks observations")
    environment_bundle = canonical_sha256(observed)
    if environment.get("environment_bundle_sha256") != environment_bundle:
        raise MechanismAggregationError("environment manifest bundle mismatch")
    device = observed.get("device")
    if device != "cuda:0":
        raise MechanismAggregationError("paper aggregate requires locked device cuda:0")

    overlay_manifest = overlay_manifest.resolve(strict=True)
    overlay = strict_json(overlay_manifest)
    if not isinstance(overlay, Mapping):
        raise MechanismAggregationError("overlay manifest must be an object")
    if overlay.get("schema_version") != "gradient_reuse.mammoth_overlay.v1":
        raise MechanismAggregationError("overlay manifest schema mismatch")
    if overlay.get("upstream_commit") != UPSTREAM_COMMIT:
        raise MechanismAggregationError("overlay manifest uses another Mammoth commit")
    overlay_bundle, _ = _mapping_bundle(
        overlay,
        field="files",
        bundle_field="overlay_bundle_sha256",
        context="overlay manifest",
    )

    identity = AggregationIdentity(
        lock_sha256=sha256_file(lock_path),
        source_manifest_sha256=sha256_file(source_manifest),
        source_bundle_sha256=source_bundle,
        data_manifest_sha256=sha256_file(data_manifest),
        data_bundle_sha256=data_bundle,
        environment_manifest_sha256=sha256_file(environment_manifest),
        environment_bundle_sha256=environment_bundle,
        overlay_manifest_sha256=sha256_file(overlay_manifest),
        overlay_bundle_sha256=overlay_bundle,
        upstream_commit=UPSTREAM_COMMIT,
        device=str(device),
    )
    return lock, identity


def expected_unit_identity(
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
) -> dict[str, Any]:
    effective = effective_mammoth_args(
        unit, lock, device=identity.device, smoke=False
    )
    return unit_identity(
        unit,
        lock_sha256=identity.lock_sha256,
        source_bundle_sha256=identity.source_bundle_sha256,
        overlay_bundle_sha256=identity.overlay_bundle_sha256,
        data_manifest_sha256=identity.data_manifest_sha256,
        data_bundle_sha256=identity.data_bundle_sha256,
        environment_manifest_sha256=identity.environment_manifest_sha256,
        environment_bundle_sha256=identity.environment_bundle_sha256,
        effective_args_sha256=canonical_sha256(effective),
        device=identity.device,
        smoke=False,
    )


def deterministic_shard_assignments(
    lock: Mapping[str, Any], shard_count: int
) -> dict[int, list[str]]:
    """Partition the canonical unit order by ``index modulo shard_count``."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise MechanismAggregationError("shard_count must be a positive integer")
    output = {index: [] for index in range(shard_count)}
    for unit_index, unit in enumerate(enumerate_units(lock)):
        output[unit_index % shard_count].append(unit.unit_id)
    return output


def _artifact_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MechanismAggregationError(f"unit artifacts forbid symlinks: {path}")
        if not path.is_file() or path.name == "DONE.json" and path.parent == root:
            continue
        relative = str(path.relative_to(root))
        _relative(relative, context="unit artifact path")
        inventory[relative] = sha256_file(path)
    return inventory


def _validate_metrics(metrics: Any, unit: MechanismUnit, lock: Mapping[str, Any]) -> None:
    if not isinstance(metrics, Mapping):
        raise MechanismAggregationError(f"unit metrics are not an object: {unit.unit_id}")
    if metrics.get("accuracy_unit") != "percentage_points":
        raise MechanismAggregationError(f"unit accuracy scale changed: {unit.unit_id}")
    if int(metrics.get("evaluation_tasks", -1)) != int(
        lock["benchmarks"][unit.benchmark]["tasks"]
    ):
        raise MechanismAggregationError(f"unit task coverage changed: {unit.unit_id}")
    for name in ("final_average_accuracy", "average_incremental_accuracy"):
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise MechanismAggregationError(f"invalid {name}: {unit.unit_id}")
    trajectory = metrics.get("mean_accuracy_trajectory")
    tasks = int(metrics["evaluation_tasks"])
    if (
        not isinstance(trajectory, list)
        or len(trajectory) != tasks
        or any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0 for value in trajectory)
    ):
        raise MechanismAggregationError(f"invalid accuracy trajectory: {unit.unit_id}")
    if abs(float(metrics["average_incremental_accuracy"]) - sum(map(float, trajectory)) / tasks) > 1e-8:
        raise MechanismAggregationError(f"incremental accuracy mismatch: {unit.unit_id}")

    matrix_available = metrics.get("accuracy_matrix_available") is True
    matrix = metrics.get("accuracy_matrix")
    if not matrix_available or not isinstance(matrix, list) or len(matrix) != tasks:
        raise MechanismAggregationError(
            f"paper unit lacks a full accuracy matrix: {unit.unit_id}"
        )
    for row_number, row in enumerate(matrix, 1):
        if (
            not isinstance(row, list)
            or len(row) != row_number
            or any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0 for value in row)
        ):
            raise MechanismAggregationError(f"invalid accuracy matrix row: {unit.unit_id}")
    final_average = sum(map(float, matrix[-1])) / tasks
    if abs(final_average - float(metrics["final_average_accuracy"])) > 1e-8:
        raise MechanismAggregationError(f"final accuracy mismatch: {unit.unit_id}")
    bwt = sum(
        float(matrix[-1][index]) - float(matrix[index][index])
        for index in range(tasks - 1)
    ) / (tasks - 1)
    if metrics.get("backward_transfer_available") is not True:
        raise MechanismAggregationError(f"matrix unit lacks BWT: {unit.unit_id}")
    if abs(float(metrics.get("backward_transfer")) - bwt) > 1e-8:
        raise MechanismAggregationError(f"BWT mismatch: {unit.unit_id}")


def _validate_unit(
    root: Path,
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_identity = expected_unit_identity(unit, lock, identity)
    done_path = root / "DONE.json"
    unit_path = root / "UNIT.json"
    if not done_path.is_file() or not unit_path.is_file():
        raise MechanismAggregationError(f"unit is incomplete: {unit.unit_id}")
    done = strict_json(done_path)
    expected_done_keys = {
        "schema_version",
        "status",
        "identity_sha256",
        "artifact_count",
        "artifact_inventory",
        "artifact_bundle_sha256",
    }
    if not isinstance(done, Mapping) or set(done) != expected_done_keys:
        raise MechanismAggregationError(f"unit DONE schema changed: {unit.unit_id}")
    if done["schema_version"] != "gradient_reuse.mechanism_unit_done.v2" or done["status"] != "PASS":
        raise MechanismAggregationError(f"unit DONE is not a v2 pass: {unit.unit_id}")
    if done["identity_sha256"] != canonical_sha256(expected_identity):
        raise MechanismAggregationError(f"unit identity digest mismatch: {unit.unit_id}")
    claimed = done["artifact_inventory"]
    if not isinstance(claimed, Mapping) or not claimed:
        raise MechanismAggregationError(f"unit artifact inventory is empty: {unit.unit_id}")
    normalized: dict[str, str] = {}
    for relative, digest in claimed.items():
        path = _relative(relative, context="unit DONE artifact path")
        normalized[path] = _digest(digest, context=f"{unit.unit_id}/{path}")
    observed = _artifact_inventory(root)
    if normalized != observed:
        raise MechanismAggregationError(f"unit artifact inventory mismatch: {unit.unit_id}")
    if int(done["artifact_count"]) != len(observed):
        raise MechanismAggregationError(f"unit artifact count mismatch: {unit.unit_id}")
    if done["artifact_bundle_sha256"] != canonical_sha256(observed):
        raise MechanismAggregationError(f"unit artifact bundle mismatch: {unit.unit_id}")

    record = strict_json(unit_path)
    required_record_keys = {
        "schema_version",
        "identity",
        "benchmark",
        "method",
        "mammoth_model",
        "estimand_class",
        "intervention",
        "seed",
        "permutation_replica",
        "metrics",
        "diagnostics",
        "sampled_event_fingerprints",
        "effective_args",
        "runtime_seconds",
        "paper_eligible",
        "attempt",
    }
    if not isinstance(record, Mapping) or set(record) != required_record_keys:
        raise MechanismAggregationError(f"UNIT schema changed: {unit.unit_id}")
    if record["schema_version"] != "gradient_reuse.mechanism_unit.v2":
        raise MechanismAggregationError(f"UNIT version changed: {unit.unit_id}")
    observed_identity = record["identity"]
    if (
        not isinstance(observed_identity, Mapping)
        or set(observed_identity) != set(expected_identity)
        or observed_identity != expected_identity
        or record["paper_eligible"] is not True
    ):
        raise MechanismAggregationError(f"UNIT identity/paper status mismatch: {unit.unit_id}")
    locked_estimand_class = lock["methods"][unit.method].get("estimand_class")
    expected_estimand_class = method_estimand_class(unit.method)
    if locked_estimand_class != expected_estimand_class:
        raise MechanismAggregationError(
            f"config estimand class mismatch: {unit.method}"
        )
    expected_fields = {
        "benchmark": unit.benchmark,
        "method": unit.method,
        "mammoth_model": unit.mammoth_model,
        "estimand_class": expected_estimand_class,
        "intervention": unit.intervention,
        "seed": unit.seed,
        "permutation_replica": unit.permutation_replica,
    }
    for name, expected in expected_fields.items():
        if record[name] != expected:
            raise MechanismAggregationError(f"UNIT metadata mismatch {name}: {unit.unit_id}")
    effective = effective_mammoth_args(unit, lock, device=identity.device, smoke=False)
    if record["effective_args"] != effective:
        raise MechanismAggregationError(f"UNIT effective arguments changed: {unit.unit_id}")
    runtime = float(record["runtime_seconds"])
    if not math.isfinite(runtime) or runtime < 0.0:
        raise MechanismAggregationError(f"UNIT runtime is invalid: {unit.unit_id}")
    _validate_metrics(record["metrics"], unit, lock)

    attempt_name = _relative(record["attempt"], context="unit attempt")
    if "/" in attempt_name or not attempt_name.startswith("attempt_"):
        raise MechanismAggregationError(f"UNIT attempt is invalid: {unit.unit_id}")
    required_paths = {
        "UNIT.json",
        f"{attempt_name}/RUN_MANIFEST.json",
        f"{attempt_name}/stdout.log",
    }
    if unit.intervention != "none":
        required_paths.add(f"{attempt_name}/pairing_audit.jsonl")
    if not required_paths.issubset(observed):
        raise MechanismAggregationError(f"unit lacks required artifacts: {unit.unit_id}")
    if not any(
        path.startswith(f"{attempt_name}/mammoth_results/") and path.endswith("/logs.pyd")
        for path in observed
    ):
        raise MechanismAggregationError(f"unit lacks Mammoth result log: {unit.unit_id}")
    attempt = strict_json(root / attempt_name / "RUN_MANIFEST.json")
    if (
        not isinstance(attempt, Mapping)
        or attempt.get("schema_version") != "gradient_reuse.mechanism_attempt.v1"
        or attempt.get("status") != "COMPLETED"
        or attempt.get("identity") != expected_identity
        or attempt.get("identity_sha256") != canonical_sha256(expected_identity)
        or attempt.get("attempt") != attempt_name
        or int(attempt.get("returncode", -1)) != 0
    ):
        raise MechanismAggregationError(f"attempt manifest mismatch: {unit.unit_id}")
    return dict(record), {
        "unit_sha256": sha256_file(unit_path),
        "done_sha256": sha256_file(done_path),
        "artifact_bundle_sha256": str(done["artifact_bundle_sha256"]),
    }


def _validate_shard(
    shard: Path,
    expected_units: Mapping[str, MechanismUnit],
    order: Mapping[str, int],
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard = shard.resolve(strict=True)
    if not shard.is_dir() or shard.is_symlink():
        raise MechanismAggregationError(f"shard is not a regular directory: {shard}")
    run_path = shard / "RUN_MANIFEST.json"
    done_path = shard / "DONE.json"
    if not run_path.is_file() or not done_path.is_file():
        raise MechanismAggregationError(f"shard lacks RUN_MANIFEST/DONE: {shard}")
    run = strict_json(run_path)
    expected_run_keys = {
        "schema_version",
        "status",
        "started_unix",
        "run_attempt",
        "completed_unix",
        "completed_units",
        "completed_unit_ids",
        "completed_unit_done_sha256",
        "complete_protocol",
        "invocation",
    }
    if not isinstance(run, Mapping) or set(run) != expected_run_keys:
        raise MechanismAggregationError(f"shard RUN_MANIFEST schema changed: {shard}")
    if (
        run["schema_version"] != "gradient_reuse.mechanism_campaign.v2"
        or run["status"] != "COMPLETED"
        or isinstance(run["run_attempt"], bool)
        or not isinstance(run["run_attempt"], int)
        or run["run_attempt"] < 1
    ):
        raise MechanismAggregationError(f"shard campaign is not a completed v2 run: {shard}")
    invocation = run.get("invocation")
    expected_invocation_keys = {
        "schema_version",
        "lock_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "overlay_manifest_sha256",
        "overlay_bundle_sha256",
        "upstream_commit",
        "data_manifest_sha256",
        "data_bundle_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "smoke",
        "paper_eligible",
        "device",
        "selection_scheme",
        "shard_count",
        "shard_index",
        "selected_unit_ids",
        "effective_args_sha256",
    }
    if not isinstance(invocation, Mapping) or set(invocation) != expected_invocation_keys:
        raise MechanismAggregationError(f"shard invocation schema changed: {shard}")
    if invocation["schema_version"] != "gradient_reuse.mechanism_invocation.v2":
        raise MechanismAggregationError(f"shard invocation is not v2: {shard}")
    expected_invocation = {
        "lock_sha256": identity.lock_sha256,
        "source_manifest_sha256": identity.source_manifest_sha256,
        "source_bundle_sha256": identity.source_bundle_sha256,
        "overlay_manifest_sha256": identity.overlay_manifest_sha256,
        "overlay_bundle_sha256": identity.overlay_bundle_sha256,
        "data_manifest_sha256": identity.data_manifest_sha256,
        "data_bundle_sha256": identity.data_bundle_sha256,
        "environment_manifest_sha256": identity.environment_manifest_sha256,
        "environment_bundle_sha256": identity.environment_bundle_sha256,
        "upstream_commit": identity.upstream_commit,
        "device": identity.device,
        "smoke": False,
        "paper_eligible": False,
    }
    for name, expected in expected_invocation.items():
        if invocation.get(name) != expected:
            raise MechanismAggregationError(f"mixed shard identity {name}: {shard}")
    if invocation["selection_scheme"] != SHARD_SELECTION_SCHEME:
        raise MechanismAggregationError(f"input is not a scientific modulo shard: {shard}")
    shard_count = invocation["shard_count"]
    shard_index = invocation["shard_index"]
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise MechanismAggregationError(f"invalid shard partition metadata: {shard}")
    selected = invocation.get("selected_unit_ids")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise MechanismAggregationError(f"invalid shard unit selection: {shard}")
    if any(unit_id not in expected_units for unit_id in selected):
        raise MechanismAggregationError(f"shard contains an unexpected unit ID: {shard}")
    if selected != sorted(selected, key=order.__getitem__):
        raise MechanismAggregationError(f"shard unit order is noncanonical: {shard}")
    expected_selection = [
        unit_id
        for unit_id, canonical_index in sorted(order.items(), key=lambda item: item[1])
        if canonical_index % shard_count == shard_index
    ]
    if selected != expected_selection:
        raise MechanismAggregationError(f"shard selection differs from canonical modulo: {shard}")
    effective_hashes = invocation["effective_args_sha256"]
    if not isinstance(effective_hashes, Mapping) or set(effective_hashes) != set(selected):
        raise MechanismAggregationError(f"shard effective-argument inventory changed: {shard}")
    for unit_id in selected:
        observed_hash = _digest(
            effective_hashes[unit_id], context=f"shard effective args {unit_id}"
        )
        expected_hash = canonical_sha256(
            effective_mammoth_args(
                expected_units[unit_id], lock, device=identity.device, smoke=False
            )
        )
        if observed_hash != expected_hash:
            raise MechanismAggregationError(
                f"shard effective arguments changed: {unit_id}"
            )

    done = strict_json(done_path)
    expected_done_keys = {
        "schema_version",
        "status",
        "paper_eligible",
        "complete_protocol",
        "completed_unit_ids",
        "completed_unit_done_sha256",
        "artifact_count",
        "artifact_inventory",
        "artifact_bundle_sha256",
    }
    if not isinstance(done, Mapping) or set(done) != expected_done_keys:
        raise MechanismAggregationError(f"shard DONE schema changed: {shard}")
    if (
        done["schema_version"] != "gradient_reuse.mechanism_done.v2"
        or done["status"] != "PASS"
    ):
        raise MechanismAggregationError(f"shard DONE is not a v2 pass: {shard}")
    entries = list(shard.iterdir())
    if any(path.is_symlink() for path in entries):
        raise MechanismAggregationError(f"shard root forbids symlinks: {shard}")
    unexpected_directories = {
        path.name for path in entries if path.is_dir() and path.name != "units"
    }
    if unexpected_directories:
        raise MechanismAggregationError(
            f"shard root contains unexpected directories: {sorted(unexpected_directories)}"
        )

    units_root = shard / "units"
    if not units_root.is_dir() or units_root.is_symlink():
        raise MechanismAggregationError(f"shard units directory is missing: {shard}")
    observed_dirs = {
        path.name for path in units_root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if observed_dirs != set(selected):
        raise MechanismAggregationError(f"shard selected/observed units differ: {shard}")
    if any(path.is_symlink() or not path.is_dir() for path in units_root.iterdir()):
        raise MechanismAggregationError(f"shard units root contains a non-directory: {shard}")
    if int(run.get("completed_units", -1)) != len(selected):
        raise MechanismAggregationError(f"shard completed-unit count mismatch: {shard}")
    if run["complete_protocol"] is not False or done["complete_protocol"] is not False:
        raise MechanismAggregationError(f"a shard may not claim protocol completeness: {shard}")
    if done["paper_eligible"] is not False:
        raise MechanismAggregationError(f"a shard may not claim paper eligibility: {shard}")
    if run["completed_unit_ids"] != selected or done["completed_unit_ids"] != selected:
        raise MechanismAggregationError(f"shard completed-unit IDs changed: {shard}")

    claimed_unit_done = run["completed_unit_done_sha256"]
    if (
        not isinstance(claimed_unit_done, Mapping)
        or set(claimed_unit_done) != set(selected)
        or done["completed_unit_done_sha256"] != claimed_unit_done
    ):
        raise MechanismAggregationError(f"shard unit-DONE inventory changed: {shard}")
    for unit_id in selected:
        observed = sha256_file(units_root / unit_id / "DONE.json")
        if _digest(claimed_unit_done[unit_id], context=f"unit DONE {unit_id}") != observed:
            raise MechanismAggregationError(f"shard unit-DONE digest mismatch: {unit_id}")

    claimed_artifacts = done["artifact_inventory"]
    if not isinstance(claimed_artifacts, Mapping) or not claimed_artifacts:
        raise MechanismAggregationError(f"shard artifact inventory is empty: {shard}")
    normalized_artifacts = {
        _relative(relative, context="shard artifact path"): _digest(
            digest, context=f"shard artifact {relative}"
        )
        for relative, digest in claimed_artifacts.items()
    }
    observed_artifacts = _artifact_inventory(shard)
    if normalized_artifacts != observed_artifacts:
        raise MechanismAggregationError(f"shard artifact inventory mismatch: {shard}")
    if int(done["artifact_count"]) != len(observed_artifacts):
        raise MechanismAggregationError(f"shard artifact count mismatch: {shard}")
    if done["artifact_bundle_sha256"] != canonical_sha256(observed_artifacts):
        raise MechanismAggregationError(f"shard artifact bundle mismatch: {shard}")

    records: list[dict[str, Any]] = []
    unit_artifacts: dict[str, Any] = {}
    for unit_id in selected:
        record, artifacts = _validate_unit(
            units_root / unit_id, expected_units[unit_id], lock, identity
        )
        records.append(record)
        unit_artifacts[unit_id] = artifacts
    return records, {
        "run_manifest_sha256": sha256_file(run_path),
        "done_sha256": sha256_file(done_path),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selection_scheme": SHARD_SELECTION_SCHEME,
        "artifact_bundle_sha256": str(done["artifact_bundle_sha256"]),
        "unit_count": len(records),
        "unit_artifacts": unit_artifacts,
    }


def _validate_cross_arm(
    records: Sequence[Mapping[str, Any]], replica_ids: Sequence[int]
) -> dict[str, Any]:
    try:
        return _validate_cross_arm_comparability(
            records, permutation_replica_ids=replica_ids
        )
    except RuntimeError as error:
        raise MechanismAggregationError(str(error)) from error


def _optional_interval(values: Iterable[Mapping[str, Any]], key: str) -> Any:
    rows = list(values)
    availability_key = key.rsplit("_V_", 1)[0] + "_available"
    availability = [bool(row.get(availability_key)) for row in rows]
    if any(availability) and not all(availability):
        raise MechanismAggregationError(f"mixed endpoint availability: {key}")
    return (
        mean_and_ci(
            (float(row[key]) for row in rows),
            nested_mc_variances=(
                float(row[f"{key}_mc_variance"]) for row in rows
            ),
        )
        if all(availability)
        else None
    )


def summarize_complete_records(
    records: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    *,
    cross_arm_comparability: Mapping[str, Any],
) -> dict[str, Any]:
    replica_ids = [int(value) for value in lock["runtime"]["permutation_replicas"]]
    try:
        rows = compute_mechanism_estimands(
            records, permutation_replica_ids=replica_ids
        )
        diagnostic_rows = compute_diagnostic_estimands(
            records, permutation_replica_ids=replica_ids
        )
    except RuntimeError as error:
        raise MechanismAggregationError(str(error)) from error
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["benchmark"]), str(row["method"])), []).append(row)
    expected_groups = {
        (benchmark, method)
        for benchmark in lock["benchmarks"]
        for method in lock["methods"]
        if method_estimand_class(method) != "reference_baseline"
    }
    if set(grouped) != expected_groups:
        raise MechanismAggregationError("estimand groups do not cover the locked design")
    diagnostic_grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in diagnostic_rows:
        diagnostic_grouped.setdefault(
            (str(row["benchmark"]), str(row["method"])), []
        ).append(row)
    if set(diagnostic_grouped) != expected_groups or any(
        len(values) != len(lock["runtime"]["seeds"])
        for values in diagnostic_grouped.values()
    ):
        raise MechanismAggregationError(
            "diagnostic groups do not cover the locked seed design"
        )
    try:
        diagnostic_summaries = summarize_diagnostic_estimands(diagnostic_rows)
    except RuntimeError as error:
        raise MechanismAggregationError(str(error)) from error
    summaries = []
    for (benchmark, method), values in sorted(grouped.items()):
        if len(values) != len(lock["runtime"]["seeds"]):
            raise MechanismAggregationError(f"seed coverage changed: {benchmark}/{method}")
        reference_methods = {str(value["reference_method"]) for value in values}
        if len(reference_methods) != 1:
            raise MechanismAggregationError(
                f"reference-method routing changed: {benchmark}/{method}"
            )
        summaries.append(
            {
                "benchmark": benchmark,
                "method": method,
                "reference_method": next(iter(reference_methods)),
                "estimand_class": method_estimand_class(method),
                "accuracy_unit": "percentage_points",
                "V_total": _optional_interval(values, "accuracy_V_total"),
                "V_perm": _optional_interval(values, "accuracy_V_perm"),
                "V_pair": _optional_interval(values, "accuracy_V_pair"),
                "bwt_V_total": _optional_interval(values, "bwt_V_total"),
                "bwt_V_perm": _optional_interval(values, "bwt_V_perm"),
                "bwt_V_pair": _optional_interval(values, "bwt_V_pair"),
                "incremental_accuracy_V_total": _optional_interval(
                    values, "incremental_accuracy_V_total"
                ),
                "incremental_accuracy_V_perm": _optional_interval(
                    values, "incremental_accuracy_V_perm"
                ),
                "incremental_accuracy_V_pair": _optional_interval(
                    values, "incremental_accuracy_V_pair"
                ),
                "max_abs_accuracy_identity_residual": max(
                    abs(float(value["accuracy_identity_residual"])) for value in values
                ),
            }
        )
    profile_id = mechanism_profile_id(lock)
    original_protocol = profile_id == "full_v1"
    return {
        "schema_version": "gradient_reuse.mechanism_aggregate_summary.v3",
        "paper_eligible": True,
        "profile_id": profile_id,
        "protocol_profile_id": profile_id,
        "complete_selected_profile": True,
        "complete_original_protocol": original_protocol,
        "unit_count": len(records),
        "permutation_replica_ids": replica_ids,
        "permutation_replicas_are_nested": True,
        "primary_endpoint": str(lock["evaluation"]["primary"]),
        "primary_inference": str(lock["evaluation"]["primary_inference"]),
        "primary_estimand": "V_pair",
        "cross_method_accuracy_ranking_supported": bool(
            lock["evaluation"]["cross_method_accuracy_ranking_supported"]
        ),
        "methods_pooled_across_estimand_classes": False,
        "estimand_classes": {
            method: str(specification["estimand_class"])
            for method, specification in lock["methods"].items()
        },
        "cross_arm_comparability": dict(cross_arm_comparability),
        "seed_rows": rows,
        "method_summaries": summaries,
        "diagnostic_seed_rows": diagnostic_rows,
        "diagnostic_summaries": diagnostic_summaries,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def aggregate_shards(
    shards: Sequence[Path],
    *,
    output: Path,
    lock: Mapping[str, Any],
    identity: AggregationIdentity,
) -> Path:
    """Authenticate shards and atomically materialize a complete aggregate."""

    if len(shards) < 1:
        raise MechanismAggregationError("at least one shard is required")
    output = output.resolve()
    if output.exists():
        raise MechanismAggregationError(f"aggregate output must be fresh: {output}")
    raw_shards = [Path(path) for path in shards]
    if any(path.is_symlink() for path in raw_shards):
        raise MechanismAggregationError("shard inputs must not be symlinks")
    resolved_shards = [path.resolve(strict=True) for path in raw_shards]
    if len(resolved_shards) != len(set(resolved_shards)):
        raise MechanismAggregationError("the same shard was supplied more than once")
    units = enumerate_units(lock)
    expected_unit_count = expected_paper_unit_count(lock)
    if len(units) != expected_unit_count:
        raise MechanismAggregationError(
            f"selected paper profile must contain exactly {expected_unit_count} units"
        )
    expected = {unit.unit_id: unit for unit in units}
    order = {unit.unit_id: index for index, unit in enumerate(units)}
    records_by_id: dict[str, dict[str, Any]] = {}
    shard_evidence: list[dict[str, Any]] = []
    for input_index, shard in enumerate(resolved_shards):
        records, evidence = _validate_shard(shard, expected, order, lock, identity)
        selected_ids = [record["identity"]["unit"] for record in records]
        unit_ids = []
        for unit_payload, record in zip(selected_ids, records):
            unit_id = MechanismUnit(**unit_payload).unit_id
            if unit_id in records_by_id:
                raise MechanismAggregationError(f"duplicate unit across shards: {unit_id}")
            records_by_id[unit_id] = record
            unit_ids.append(unit_id)
        shard_evidence.append(
            {
                "input_index": input_index,
                "input_name": shard.name,
                "unit_ids": unit_ids,
                **evidence,
            }
        )
    shard_counts = {int(evidence["shard_count"]) for evidence in shard_evidence}
    if len(shard_counts) != 1:
        raise MechanismAggregationError("shards use mixed partition counts")
    declared_shard_count = next(iter(shard_counts))
    shard_indices = [int(evidence["shard_index"]) for evidence in shard_evidence]
    if (
        len(shard_evidence) != declared_shard_count
        or len(shard_indices) != len(set(shard_indices))
        or set(shard_indices) != set(range(declared_shard_count))
    ):
        raise MechanismAggregationError(
            "aggregate requires each canonical shard index exactly once"
        )
    shard_evidence.sort(key=lambda evidence: int(evidence["shard_index"]))
    observed_ids = set(records_by_id)
    if observed_ids != set(expected):
        raise MechanismAggregationError(
            "aggregate unit coverage is incomplete or unexpected; "
            f"missing={len(set(expected) - observed_ids)}, "
            f"unexpected={len(observed_ids - set(expected))}"
        )
    records = [records_by_id[unit.unit_id] for unit in units]
    comparability = _validate_cross_arm(
        records, lock["runtime"]["permutation_replicas"]
    )
    summary = summarize_complete_records(
        records, lock, cross_arm_comparability=comparability
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary_path = temporary / "SUMMARY.json"
        _atomic_json(summary_path, summary)
        identity_payload = identity.as_dict()
        unit_artifacts = {
            unit_id: evidence
            for shard in shard_evidence
            for unit_id, evidence in shard["unit_artifacts"].items()
        }
        profile_id = mechanism_profile_id(lock)
        original_protocol = profile_id == "full_v1"
        manifest = {
            "schema_version": "gradient_reuse.mechanism_aggregate_campaign.v3",
            "status": "COMPLETED",
            "paper_eligible": True,
            "complete_protocol": True,
            "profile_id": profile_id,
            "protocol_profile_id": profile_id,
            "complete_selected_profile": True,
            "complete_original_protocol": original_protocol,
            "completed_unix": time.time(),
            "identity": identity_payload,
            "identity_sha256": canonical_sha256(identity_payload),
            "shard_count": len(shard_evidence),
            "shards": shard_evidence,
            "unit_count": len(records),
            "unit_artifacts": unit_artifacts,
            "unit_artifact_bundle_sha256": canonical_sha256(unit_artifacts),
            "summary_sha256": sha256_file(summary_path),
        }
        manifest_path = temporary / "RUN_MANIFEST.json"
        _atomic_json(manifest_path, manifest)
        outputs = {
            "RUN_MANIFEST.json": sha256_file(manifest_path),
            "SUMMARY.json": sha256_file(summary_path),
        }
        done = {
            "schema_version": "gradient_reuse.mechanism_aggregate_done.v3",
            "status": "PASS",
            "paper_eligible": True,
            "complete_protocol": True,
            "profile_id": profile_id,
            "protocol_profile_id": profile_id,
            "complete_selected_profile": True,
            "complete_original_protocol": original_protocol,
            "unit_count": len(records),
            "identity_sha256": canonical_sha256(identity_payload),
            "outputs": outputs,
            "output_bundle_sha256": canonical_sha256(outputs),
        }
        _atomic_json(temporary / "DONE.json", done)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def aggregate_from_manifests(
    shards: Sequence[Path],
    *,
    output: Path,
    lock_path: Path,
    source_manifest: Path,
    data_manifest: Path,
    environment_manifest: Path,
    overlay_manifest: Path,
    repo: Path = REPO,
) -> Path:
    lock, identity = load_aggregation_identity(
        lock_path=lock_path,
        source_manifest=source_manifest,
        data_manifest=data_manifest,
        environment_manifest=environment_manifest,
        overlay_manifest=overlay_manifest,
        repo=repo,
    )
    return aggregate_shards(shards, output=output, lock=lock, identity=identity)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = aggregate_from_manifests(
        arguments.shard,
        output=arguments.output,
        lock_path=arguments.lock,
        source_manifest=arguments.source_manifest,
        data_manifest=arguments.data_manifest,
        environment_manifest=arguments.environment_manifest,
        overlay_manifest=arguments.overlay_manifest,
        repo=arguments.repo,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
