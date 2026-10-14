"""Fail-closed runner for the Mammoth mechanism audit.

Completed Mammoth units are immutable and resumable at unit granularity.  An
interrupted unit is retried in a new attempt directory, so partial files are
never mistaken for evidence.  Full runs require a data manifest frozen before
training; smoke runs are explicitly paper-ineligible.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .overlay import (
    MANIFEST_NAME,
    overlay_checkout_lock,
    prepare_overlay,
    verify_upstream,
)
from .protocol import (
    MechanismUnit,
    UPSTREAM_COMMIT,
    canonical_sha256,
    compute_diagnostic_estimands,
    compute_mechanism_estimands,
    enumerate_units,
    load_and_validate_lock,
    mean_and_ci,
    method_estimand_class,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    sha256_file,
    strict_json,
    strict_json_text,
    summarize_diagnostic_estimands,
    unit_identity,
    validate_source_manifest,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "configs" / "mechanisms.json"
DEFAULT_SOURCE_MANIFEST = REPO / "provenance" / "source" / "mechanisms.json"
SHARD_SELECTION_SCHEME = "canonical_unit_index_modulo_v1"
FILTER_SELECTION_SCHEME = "canonical_filter_v1"
REQUIRED_MAMMOTH_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "quadprog",
    "kornia",
    "timm",
    "tqdm",
    "PyYAML",
    "wandb",
    "setproctitle",
    "Pillow",
)
_ATTEMPT_NAME = re.compile(r"attempt_([0-9]{3,})")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _validate_campaign_path_disjointness(
    *,
    out: Path,
    data_root: Path,
    mammoth_root: Path,
    lock_path: Path,
    source_manifest: Path,
    data_manifest: Path | None,
    environment_manifest: Path | None,
) -> None:
    """Keep mutable output/install roots away from inputs and one another."""

    out = out.resolve()
    data_root = data_root.resolve()
    mammoth_root = mammoth_root.resolve()
    if _paths_overlap(data_root, mammoth_root):
        raise RuntimeError("data and Mammoth roots must be disjoint")
    protected = {
        "data root": data_root,
        "Mammoth root": mammoth_root,
        "lock": lock_path.resolve(),
        "source manifest": source_manifest.resolve(),
    }
    if data_manifest is not None:
        protected["data manifest"] = data_manifest.resolve()
    if environment_manifest is not None:
        protected["environment manifest"] = environment_manifest.resolve()
    for name, path in protected.items():
        if _paths_overlap(out, path):
            raise RuntimeError(f"output root overlaps the {name}")
    manifest_paths = [
        lock_path.resolve(),
        source_manifest.resolve(),
        *([data_manifest.resolve()] if data_manifest is not None else []),
        *(
            [environment_manifest.resolve()]
            if environment_manifest is not None
            else []
        ),
    ]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise RuntimeError("configuration and manifest paths must be distinct")
    for path in manifest_paths:
        if _paths_overlap(path, mammoth_root):
            raise RuntimeError("configuration/manifests must be outside Mammoth root")
        if _paths_overlap(path, data_root):
            raise RuntimeError("configuration/manifests must be outside the data root")


@contextmanager
def _campaign_root_lock(out: Path):
    """Hold one nonblocking advisory lock per resolved campaign root."""

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.parent / f".{out.name}.mechanism-launch.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another launcher is writing the mechanism output root: {out}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\nout={out}\n".encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _allocate_attempt_directory(unit_root: Path) -> Path:
    """Allocate max(existing attempt index)+1 after strict directory checks."""

    unit_root.mkdir(parents=True, exist_ok=True)
    if unit_root.is_symlink() or not unit_root.is_dir():
        raise RuntimeError(f"mechanism unit root is unsafe: {unit_root}")
    indexes: list[int] = []
    for path in unit_root.iterdir():
        match = _ATTEMPT_NAME.fullmatch(path.name)
        if match is not None:
            index = int(match.group(1))
            if index < 1 or path.name != f"attempt_{index:03d}":
                raise RuntimeError(f"attempt directory has a noncanonical name: {path}")
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"attempt path is not a regular directory: {path}")
            indexes.append(index)
            continue
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        raise RuntimeError(f"unexpected entry in mechanism unit root: {path.name}")
    next_index = max(indexes, default=0) + 1
    attempt = unit_root / f"attempt_{next_index:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def _artifact_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = {str(value) for value in exclude}
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise RuntimeError(f"artifact inventory forbids symlinks: {path}")
        if path.is_file() and relative not in excluded:
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                raise RuntimeError(f"artifact inventory found an unfinished file: {path}")
            inventory[relative] = sha256_file(path)
    return inventory


def _copy_locked_file(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"locked manifest copy changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def observed_environment(*, mammoth_root: Path, device: str) -> dict[str, Any]:
    import torch

    cuda_requested = str(device).startswith("cuda")
    gpu: dict[str, Any] | None = None
    if cuda_requested:
        if not torch.cuda.is_available():
            raise RuntimeError("the locked CUDA device is unavailable")
        index = int(str(device).split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"the locked CUDA device does not exist: {device}")
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
        }
    packages = {
        name: _package_version(name)
        for name in REQUIRED_MAMMOTH_PACKAGES
    }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "mammoth_commit": UPSTREAM_COMMIT,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu": gpu,
        "packages": packages,
        "runtime_patches": {
            "domain_il_full_accuracy_logger": "v1",
            "python_dont_write_bytecode": "1",
        },
    }


def build_environment_manifest(*, mammoth_root: Path, device: str) -> dict[str, Any]:
    observed = observed_environment(mammoth_root=mammoth_root, device=device)
    return {
        "schema_version": "gradient_reuse.mechanism_environment.v1",
        "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
        "observed": observed,
        "environment_bundle_sha256": canonical_sha256(observed),
    }


def validate_environment_manifest(
    path: Path, *, mammoth_root: Path, device: str
) -> dict[str, Any]:
    payload = strict_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError("mechanism environment manifest must be a JSON object")
    if payload.get("schema_version") != "gradient_reuse.mechanism_environment.v1":
        raise RuntimeError("mechanism environment manifest schema mismatch")
    if payload.get("status") != "LOCKED_BEFORE_SCIENTIFIC_RUN":
        raise RuntimeError("mechanism environment manifest is not locked")
    observed = payload.get("observed")
    if not isinstance(observed, dict):
        raise RuntimeError("mechanism environment manifest lacks observations")
    if payload.get("environment_bundle_sha256") != canonical_sha256(observed):
        raise RuntimeError("mechanism environment manifest bundle mismatch")
    current = observed_environment(mammoth_root=mammoth_root, device=device)
    if current != observed:
        raise RuntimeError("runtime environment differs from the locked manifest")
    return payload


def validate_required_mammoth_packages(
    environment_payload: Mapping[str, Any],
) -> None:
    observed = environment_payload.get("observed")
    packages = observed.get("packages") if isinstance(observed, Mapping) else None
    if not isinstance(packages, Mapping):
        raise RuntimeError("mechanism environment lacks a package inventory")
    if set(packages) != set(REQUIRED_MAMMOTH_PACKAGES):
        raise RuntimeError("mechanism environment package inventory changed")
    missing = [
        name
        for name in REQUIRED_MAMMOTH_PACKAGES
        if not isinstance(packages.get(name), str) or not packages[name]
    ]
    if missing:
        raise RuntimeError(
            f"paper-eligible environment lacks required packages: {missing}"
        )


def _validate_device(device: str, *, smoke: bool) -> None:
    if smoke:
        if device not in {"cpu", "cuda:0"}:
            raise RuntimeError("smoke device must be exactly cpu or cuda:0")
    elif device != "cuda:0":
        raise RuntimeError("paper-eligible mechanism runs are locked to cuda:0")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = strict_json_text(line, context=f"{path}:{line_number}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"non-object audit record at {path}:{line_number}")
        records.append(payload)
    return records


def _mammoth_subprocess_environment(
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    *,
    mammoth_root: Path,
) -> dict[str, str]:
    """Build the authenticated, bytecode-free Mammoth subprocess environment."""

    environment = os.environ.copy()
    environment.update(
        {
            "MAMMOTH_TEST": "1",
            "PYTHONHASHSEED": str(unit.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "MKL_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "OPENBLAS_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "GRADIENT_REUSE_DETERMINISTIC": "1",
            "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH": "v1",
            "PYTHONPATH": str(mammoth_root),
        }
    )
    return environment


def validate_data_manifest(path: Path, *, data_root: Path) -> dict[str, Any]:
    payload = strict_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError("mechanism data manifest must be a JSON object")
    if payload.get("schema_version") != "gradient_reuse.mechanism_data_manifest.v1":
        raise RuntimeError("mechanism data manifest schema mismatch")
    if payload.get("status") != "LOCKED_BEFORE_SCIENTIFIC_RUN":
        raise RuntimeError("mechanism data manifest is not locked")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("mechanism data manifest is empty")
    root = data_root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe data-manifest path: {relative}")
        source = (root / relative_path).resolve()
        source.relative_to(root)
        if not source.is_file():
            raise RuntimeError(f"data-manifest file is missing: {relative}")
        observed[str(relative_path)] = sha256_file(source)
        if observed[str(relative_path)] != str(expected):
            raise RuntimeError(f"data bytes changed after prespecification: {relative}")
    if canonical_sha256(observed) != payload.get("data_bundle_sha256"):
        raise RuntimeError("mechanism data bundle hash mismatch")
    required = {"split_cifar100", "split_miniimagenet", "permuted_mnist"}
    if set(payload.get("benchmarks", {})) != required:
        raise RuntimeError("data manifest does not authenticate all benchmarks")
    return payload


def _argument(command: list[str], name: str, value: Any) -> None:
    command.extend((f"--{name}", str(value)))


def effective_mammoth_args(
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    *,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    """Return every outcome-relevant effective Mammoth argument except paths."""

    benchmark = lock["benchmarks"][unit.benchmark]
    method = lock["methods"][unit.method]
    hyperparameters = method["hyperparameters"][unit.benchmark]
    arguments: dict[str, Any] = {
        "model": unit.mammoth_model,
        "dataset": unit.mammoth_dataset,
        "model_config": "default",
        "backbone": benchmark["backbone"],
        "lr": hyperparameters["lr"],
        "n_epochs": benchmark["epochs"],
        "seed": unit.seed,
        "device": device,
        "num_workers": 0 if smoke else lock["runtime"]["num_workers"],
        "enable_other_metrics": 1,
        "non_verbose": 1,
        "distributed": "no",
        "code_optimization": 0,
        "batch_size": (
            int(method["stream_batch_size"])
            if unit.method in {"mer", "mer_current_only"}
            else int(benchmark["batch_size"])
        ),
        "runtime_patch_domain_il_fullacc": "v1",
    }
    if benchmark.get("dataset_config"):
        arguments["dataset_config"] = benchmark["dataset_config"]
    if method.get("buffer_size") is not None:
        arguments["buffer_size"] = method["buffer_size"]
        arguments["minibatch_size"] = method["minibatch_size"]
        if unit.method == "mer":
            arguments["mer_replay_batch_size"] = method["minibatch_size"]
    for name, value in sorted(hyperparameters.items()):
        if name != "lr":
            arguments[name] = value
    if unit.intervention != "none":
        arguments.update(
            {
                "pairing_benchmark": unit.benchmark,
                "pairing_run_seed": unit.seed,
                "pairing_intervention": unit.intervention,
                "pairing_permutation_seed": unit.seed + 100_003,
                "pairing_replica": (
                    0
                    if unit.permutation_replica is None
                    else unit.permutation_replica
                ),
                "pairing_audit_interval": (
                    1 if smoke else lock["runtime"]["audit_interval"]
                ),
                "pairing_audit_permutations": lock["runtime"][
                    "audit_permutations"
                ],
            }
        )
    if smoke:
        arguments.update({"stop_after": 2, "debug_mode": 1})
    return arguments


def build_mammoth_command(
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    *,
    mammoth_root: Path,
    data_root: Path,
    results_path: Path,
    audit_path: Path,
    device: str,
    smoke: bool,
) -> list[str]:
    # Invoke Mammoth's canonical repository-root entry point.  The pinned
    # checkout also carries ``utils/main.py`` as a symlink, but entering
    # through that alias makes Mammoth's own ``__file__``-based root discovery
    # depend on symlink handling by the transfer/filesystem layer.
    command = [sys.executable, str(mammoth_root / "main.py")]
    for name, value in effective_mammoth_args(
        unit, lock, device=device, smoke=smoke
    ).items():
        if name.startswith("runtime_patch_"):
            continue
        _argument(command, name, value)
    _argument(command, "base_path", data_root.resolve())
    _argument(command, "results_path", results_path.resolve())
    if unit.intervention != "none":
        _argument(command, "pairing_audit_path", audit_path.resolve())
    tail = command[2:]
    if len(tail) % 2:
        raise RuntimeError("Mammoth command does not contain flag/value pairs")
    flags: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        flag, value = tail[index : index + 2]
        if not flag.startswith("--") or flag in flags:
            raise RuntimeError(f"invalid or duplicate Mammoth flag: {flag}")
        flags[flag] = value
    expected = {
        f"--{name}": value
        for name, value in effective_mammoth_args(
            unit, lock, device=device, smoke=smoke
        ).items()
        if not name.startswith("runtime_patch_")
    }
    expected.update(
        {
            "--base_path": data_root.resolve(),
            "--results_path": results_path.resolve(),
        }
    )
    if unit.intervention != "none":
        expected["--pairing_audit_path"] = audit_path.resolve()
    if set(flags) != set(expected) or any(
        not _same_scalar(flags[name], value) for name, value in expected.items()
    ):
        raise RuntimeError("constructed Mammoth command differs from effective arguments")
    return command


def _mammoth_log_path(results_path: Path, unit: MechanismUnit, setting: str) -> Path:
    return results_path / setting / unit.mammoth_dataset / unit.mammoth_model / "logs.pyd"


def _same_scalar(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected or str(observed).lower() == str(expected).lower()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(observed) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(observed) == str(expected)


def _require_digest(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"audit record lacks {name}")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"audit record has invalid {name}") from error
    return value


def _audit_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["observation"]),
        int(record["event"]),
        str(record["channel"]),
    )


def _diagnostic_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    expected_effective_args: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_replica = (
        0 if unit.permutation_replica is None else int(unit.permutation_replica)
    )
    allowed_projection = {
        "gem": "gradient_reuse.gem_projection.v1",
        "agem": "gradient_reuse.agem_projection.v1",
        "pcgrad_replay": "gradient_reuse.pcgrad_replay_projection.v1",
    }
    allowed_schemas = {
        "gradient_reuse.mammoth_effective_args.v1",
        "gradient_reuse.mammoth_gradient_diagnostic.v1",
        "gradient_reuse.mammoth_intervention.v1",
    }
    if unit.method in allowed_projection:
        allowed_schemas.add(allowed_projection[unit.method])
    for record in records:
        schema = str(record.get("schema_version", ""))
        if schema not in allowed_schemas:
            raise RuntimeError(f"unexpected audit schema for {unit.method}: {schema}")
        expected_common = {
            "method": unit.mammoth_model,
            "benchmark": unit.benchmark,
            "dataset": unit.mammoth_dataset,
            "intervention": unit.intervention,
            "run_seed": unit.seed,
            "permutation_seed": unit.seed + 100_003,
            "replica": expected_replica,
        }
        for name, expected in expected_common.items():
            if not _same_scalar(record.get(name), expected):
                raise RuntimeError(
                    f"audit metadata mismatch for {unit.unit_id}/{name}: "
                    f"{record.get(name)!r} != {expected!r}"
                )

    effective = [
        record
        for record in records
        if record.get("schema_version")
        == "gradient_reuse.mammoth_effective_args.v1"
    ]
    if len(effective) != 1 or not isinstance(effective[0].get("effective_args"), dict):
        raise RuntimeError("intervention arm must emit exactly one effective-args record")
    observed_effective = effective[0]["effective_args"]
    for name, expected in expected_effective_args.items():
        if not _same_scalar(observed_effective.get(name), expected):
            raise RuntimeError(
                f"Mammoth effective argument mismatch for {unit.unit_id}/{name}: "
                f"{observed_effective.get(name)!r} != {expected!r}"
            )

    gradient = [
        record
        for record in records
        if record.get("schema_version")
        == "gradient_reuse.mammoth_gradient_diagnostic.v1"
    ]
    intervention = [
        record
        for record in records
        if record.get("schema_version") == "gradient_reuse.mammoth_intervention.v1"
    ]
    projection = [
        record
        for record in records
        if str(record.get("schema_version", "")).endswith("_projection.v1")
    ]
    if not gradient or not intervention:
        raise RuntimeError("intervention arm lacks sampled intervention diagnostics")

    intervention_by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    changed = False
    sampled_event_fingerprints: list[dict[str, Any]] = []
    for record in intervention:
        key = _audit_key(record)
        if key in intervention_by_key:
            raise RuntimeError(f"duplicate intervention audit key: {key}")
        intervention_by_key[key] = record
        input_digest = _require_digest(record, "input_sha256")
        index_digest = _require_digest(record, "sampled_indexes_sha256")
        original_digest = _require_digest(record, "original_target_sha256")
        used_digest = _require_digest(record, "used_target_sha256")
        multiset_digest = _require_digest(record, "target_multiset_sha256")
        rows = int(record.get("rows", 0))
        if rows < 1:
            raise RuntimeError("intervention audit contains no rows")
        fixed = float(record.get("fixed_point_fraction", float("nan")))
        if not 0.0 <= fixed <= 1.0:
            raise RuntimeError("intervention fixed-point fraction is invalid")
        if unit.intervention == "paired":
            if used_digest != original_digest or fixed != 1.0:
                raise RuntimeError("paired arm did not consume the original targets")
        else:
            changed = changed or used_digest != original_digest
        sampled_event_fingerprints.append(
            {
                "observation": key[0],
                "event": key[1],
                "channel": key[2],
                "rows": rows,
                "method": unit.mammoth_model,
                "benchmark": unit.benchmark,
                "dataset": unit.mammoth_dataset,
                "run_seed": unit.seed,
                "input_sha256": input_digest,
                "sampled_indexes_sha256": index_digest,
                "original_target_sha256": original_digest,
                "target_multiset_sha256": multiset_digest,
            }
        )
    if unit.intervention == "permuted" and not changed:
        raise RuntimeError("permuted arm never changed a replay-target assignment")

    for record in gradient:
        key = _audit_key(record)
        matched = intervention_by_key.get(key)
        if matched is None:
            raise RuntimeError(f"gradient diagnostic has no intervention event: {key}")
        if _require_digest(record, "input_sha256") != matched["input_sha256"]:
            raise RuntimeError(f"gradient/intervention input mismatch: {key}")
        if (
            _require_digest(record, "sampled_indexes_sha256")
            != matched["sampled_indexes_sha256"]
        ):
            raise RuntimeError(f"gradient/intervention index mismatch: {key}")
        if _require_digest(record, "target_sha256") != matched["original_target_sha256"]:
            raise RuntimeError(f"gradient/intervention target mismatch: {key}")
        if int(record.get("permutation_draws", 0)) != int(
            lock["runtime"]["audit_permutations"]
        ):
            raise RuntimeError("finite-permutation diagnostic draw count changed")
        for name in (
            "true_gradient_norm",
            "mean_permuted_gradient_norm",
            "pairing_component_norm",
            "finite_permutation_variance",
            "true_mean_permuted_cosine",
        ):
            value = float(record.get(name, float("nan")))
            if not (-float("inf") < value < float("inf")):
                raise RuntimeError(f"non-finite gradient diagnostic: {name}")
        if float(record["pairing_component_norm"]) < 0.0:
            raise RuntimeError("pairing component norm is negative")
        if float(record["finite_permutation_variance"]) < 0.0:
            raise RuntimeError("finite-permutation variance is negative")

    expected_projection_schema = allowed_projection.get(unit.method)
    if expected_projection_schema is None and projection:
        raise RuntimeError(f"{unit.method} emitted an inapplicable projection record")
    if expected_projection_schema is not None:
        if not projection or any(
            record.get("schema_version") != expected_projection_schema
            for record in projection
        ):
            raise RuntimeError(f"{unit.method} lacks its required projection diagnostics")
        tolerance = 1.0e-6
        for record in projection:
            active = bool(record.get("projection_active"))
            if unit.method == "gem":
                constraints = int(record.get("constraints", 0))
                pre = float(record.get("pre_min_dot", float("nan")))
                post = float(record.get("post_min_dot", float("nan")))
                if constraints < 1 or not all(map(lambda x: -float("inf") < x < float("inf"), (pre, post))):
                    raise RuntimeError("invalid GEM projection diagnostic")
                margin = float(lock["methods"]["gem"]["hyperparameters"][unit.benchmark]["gamma"])
                if post < -margin - tolerance:
                    raise RuntimeError("GEM projection violates its locked margin")
                if not active and (pre < -tolerance or abs(post - pre) > tolerance):
                    raise RuntimeError("inactive GEM projection changed its constraint")
            else:
                pre = float(record.get("pre_projection_dot", float("nan")))
                post = float(record.get("post_projection_dot", float("nan")))
                if not all(map(lambda x: -float("inf") < x < float("inf"), (pre, post))):
                    raise RuntimeError("invalid projection dot product")
                if post < -tolerance:
                    raise RuntimeError(f"{unit.method} projection left a violated dot product")
                if not active and (pre < -tolerance or abs(post - pre) > tolerance):
                    raise RuntimeError("inactive projection changed its gradient geometry")

    def mean(name: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        return sum(values) / len(values) if values else None

    sampled_event_fingerprints.sort(
        key=lambda row: (row["observation"], row["event"], row["channel"])
    )

    def projection_mean(
        standard_name: str, gem_name: str
    ) -> float | None:
        values = [
            float(
                record[standard_name]
                if record.get(standard_name) is not None
                else record[gem_name]
            )
            for record in projection
            if record.get(standard_name) is not None
            or record.get(gem_name) is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "records": len(records),
        "gradient_diagnostic_records": len(gradient),
        "intervention_records": len(intervention),
        "projection_records": len(projection),
        "mean_pairing_component_norm": mean("pairing_component_norm", gradient),
        "mean_finite_permutation_variance": mean(
            "finite_permutation_variance", gradient
        ),
        "mean_true_permuted_cosine": mean(
            "true_mean_permuted_cosine", gradient
        ),
        "mean_fixed_point_fraction": mean("fixed_point_fraction", intervention),
        "projection_activation_rate": mean("projection_active", projection),
        "mean_pre_projection_constraint": projection_mean(
            "pre_projection_dot", "pre_min_dot"
        ),
        "mean_post_projection_constraint": projection_mean(
            "post_projection_dot", "post_min_dot"
        ),
    }, sampled_event_fingerprints


def _validate_cross_arm_comparability(
    records: Sequence[Mapping[str, Any]],
    *,
    permutation_replica_ids: Sequence[int],
) -> dict[str, Any]:
    paired: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    permuted: dict[tuple[str, str, int], dict[int, Mapping[str, Any]]] = {}
    for record in records:
        if method_estimand_class(str(record["method"])) == "reference_baseline":
            continue
        key = (str(record["benchmark"]), str(record["method"]), int(record["seed"]))
        if record["intervention"] == "paired":
            if key in paired:
                raise RuntimeError(f"duplicate paired comparability row: {key}")
            paired[key] = record
        elif record["intervention"] == "permuted":
            replica = int(record["permutation_replica"])
            if replica in permuted.setdefault(key, {}):
                raise RuntimeError(f"duplicate permuted comparability row: {key}/{replica}")
            permuted[key][replica] = record
    expected = [int(value) for value in permutation_replica_ids]
    assignment_only_groups: list[dict[str, Any]] = []
    endogenous_groups: list[dict[str, Any]] = []
    for key, paired_record in paired.items():
        replicas = permuted.get(key, {})
        if sorted(replicas) != expected:
            raise RuntimeError(f"incomplete cross-arm comparison for {key}")
        reference = paired_record.get("sampled_event_fingerprints")
        if not isinstance(reference, list) or not reference:
            raise RuntimeError(f"paired arm lacks sampled-event fingerprints: {key}")
        estimand_class = method_estimand_class(key[1])
        if paired_record.get("estimand_class") != estimand_class:
            raise RuntimeError(f"paired arm estimand class mismatch: {key}")
        replica_rows: list[dict[str, Any]] = []
        for replica in expected:
            candidate_record = replicas[replica]
            if candidate_record.get("estimand_class") != estimand_class:
                raise RuntimeError(f"permuted arm estimand class mismatch: {key}")
            candidate = candidate_record.get("sampled_event_fingerprints")
            if not isinstance(candidate, list) or len(candidate) != len(reference):
                raise RuntimeError(
                    f"cross-arm sampled-event coverage mismatch: {key}/{replica}"
                )
            if estimand_class == "assignment_only":
                if candidate != reference:
                    raise RuntimeError(
                        f"paired/permuted sampled input-index-target mismatch: "
                        f"{key}/replica={replica}"
                    )
                continue
            if estimand_class != "path_dependent_endogenous_targets" or key[1] != "der":
                raise RuntimeError(f"unsupported path-dependent estimand: {key}")
            events: list[dict[str, Any]] = []
            for paired_event, candidate_event in zip(reference, candidate):
                target_fields = {"original_target_sha256", "target_multiset_sha256"}
                paired_structure = {
                    name: value
                    for name, value in paired_event.items()
                    if name not in target_fields
                }
                candidate_structure = {
                    name: value
                    for name, value in candidate_event.items()
                    if name not in target_fields
                }
                if candidate_structure != paired_structure:
                    raise RuntimeError(
                        f"DER sampled replay input/index mismatch: {key}/{replica}"
                    )
                original_equal = (
                    candidate_event["original_target_sha256"]
                    == paired_event["original_target_sha256"]
                )
                multiset_equal = (
                    candidate_event["target_multiset_sha256"]
                    == paired_event["target_multiset_sha256"]
                )
                events.append(
                    {
                        "observation": int(paired_event["observation"]),
                        "event": int(paired_event["event"]),
                        "channel": str(paired_event["channel"]),
                        "original_target_equal": original_equal,
                        "target_multiset_equal": multiset_equal,
                        "endogenous_target_diverged": not (
                            original_equal and multiset_equal
                        ),
                    }
                )
            divergent = [event for event in events if event["endogenous_target_diverged"]]
            replica_rows.append(
                {
                    "replica": replica,
                    "events": events,
                    "sampled_event_count": len(events),
                    "sampled_divergence_count": len(divergent),
                    "sampled_divergence_rate": len(divergent) / len(events),
                    "first_divergence": (
                        {
                            "observation": divergent[0]["observation"],
                            "event": divergent[0]["event"],
                            "channel": divergent[0]["channel"],
                        }
                        if divergent
                        else None
                    ),
                }
            )
        if estimand_class == "assignment_only":
            assignment_only_groups.append(
                {
                    "benchmark": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "sampled_events_per_arm": len(reference),
                    "replicas": expected,
                    "sampled_input_index_original_target_match": True,
                }
            )
        else:
            divergence_count = sum(
                row["sampled_divergence_count"] for row in replica_rows
            )
            event_count = sum(row["sampled_event_count"] for row in replica_rows)
            first_candidates = [
                {**row["first_divergence"], "replica": row["replica"]}
                for row in replica_rows
                if row["first_divergence"] is not None
            ]
            first_candidates.sort(
                key=lambda row: (
                    row["observation"], row["event"], row["channel"], row["replica"]
                )
            )
            endogenous_groups.append(
                {
                    "benchmark": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "estimand_class": estimand_class,
                    "replicas": replica_rows,
                    "sampled_event_count": event_count,
                    "sampled_divergence_count": divergence_count,
                    "sampled_divergence_rate": divergence_count / event_count,
                    "first_divergence": first_candidates[0] if first_candidates else None,
                }
            )
    total_events = sum(row["sampled_event_count"] for row in endogenous_groups)
    total_divergences = sum(
        row["sampled_divergence_count"] for row in endogenous_groups
    )
    return {
        "schema_version": "gradient_reuse.mechanism_cross_arm_comparability.v2",
        "audit_scope": "sampled_events_only",
        "sampling_interval_source": "unit.effective_args.pairing_audit_interval",
        "assignment_only_groups": assignment_only_groups,
        "path_dependent_endogenous_target_groups": endogenous_groups,
        "sampled_endogenous_target_event_count": total_events,
        "sampled_endogenous_target_divergence_count": total_divergences,
        "sampled_endogenous_target_divergence_rate": (
            total_divergences / total_events if total_events else None
        ),
    }


def _validate_completed_unit(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    done = strict_json(path / "DONE.json")
    expected_done_keys = {
        "schema_version",
        "status",
        "identity_sha256",
        "artifact_count",
        "artifact_inventory",
        "artifact_bundle_sha256",
    }
    if not isinstance(done, Mapping) or set(done) != expected_done_keys:
        raise RuntimeError(f"unit DONE schema changed: {path}")
    if (
        done.get("schema_version") != "gradient_reuse.mechanism_unit_done.v2"
        or done.get("status") != "PASS"
    ):
        raise RuntimeError(f"unit DONE status is not PASS: {path}")
    if done.get("identity_sha256") != canonical_sha256(identity):
        raise RuntimeError(f"completed unit identity changed: {path}")
    expected_inventory = done.get("artifact_inventory")
    if not isinstance(expected_inventory, dict) or not expected_inventory:
        raise RuntimeError(f"completed unit lacks an exact artifact inventory: {path}")
    observed_inventory = _artifact_inventory(path, exclude=("DONE.json",))
    if observed_inventory != expected_inventory:
        raise RuntimeError(f"completed unit artifact inventory changed: {path}")
    if done.get("artifact_bundle_sha256") != canonical_sha256(observed_inventory):
        raise RuntimeError(f"completed unit artifact bundle changed: {path}")
    if int(done.get("artifact_count", -1)) != len(observed_inventory):
        raise RuntimeError(f"completed unit artifact count changed: {path}")
    record = strict_json(path / "UNIT.json")
    expected_record_keys = {
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
    if not isinstance(record, Mapping) or set(record) != expected_record_keys:
        raise RuntimeError(f"completed UNIT schema changed: {path}")
    if (
        record.get("schema_version") != "gradient_reuse.mechanism_unit.v2"
        or not isinstance(record.get("identity"), Mapping)
        or set(record["identity"]) != set(identity)
        or record["identity"] != dict(identity)
        or record.get("estimand_class") != identity.get("estimand_class")
    ):
        raise RuntimeError(f"completed unit record identity changed: {path}")
    return dict(record)


def run_unit(
    unit: MechanismUnit,
    lock: Mapping[str, Any],
    *,
    mammoth_root: Path,
    data_root: Path,
    out: Path,
    identity: Mapping[str, Any],
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    unit_root = out / "units" / unit.unit_id
    if (unit_root / "DONE.json").exists():
        return _validate_completed_unit(unit_root, identity)
    attempt = _allocate_attempt_directory(unit_root)
    results_path = attempt / "mammoth_results"
    audit_path = attempt / "pairing_audit.jsonl"
    stdout_path = attempt / "stdout.log"
    command = build_mammoth_command(
        unit,
        lock,
        mammoth_root=mammoth_root,
        data_root=data_root,
        results_path=results_path,
        audit_path=audit_path,
        device=device,
        smoke=smoke,
    )
    expected_effective_args = effective_mammoth_args(
        unit, lock, device=device, smoke=smoke
    )
    if identity.get("effective_args_sha256") != canonical_sha256(
        expected_effective_args
    ):
        raise RuntimeError("unit identity does not authenticate its effective arguments")
    start = time.time()
    manifest = {
        "schema_version": "gradient_reuse.mechanism_attempt.v1",
        "status": "RUNNING",
        "identity": dict(identity),
        "identity_sha256": canonical_sha256(identity),
        "attempt": attempt.name,
        "command": command,
        "started_unix": start,
    }
    atomic_json(attempt / "RUN_MANIFEST.json", manifest)
    environment = _mammoth_subprocess_environment(
        unit,
        lock,
        mammoth_root=mammoth_root,
    )
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout:
            completed = subprocess.run(
                command,
                cwd=mammoth_root,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Mammoth returned {completed.returncode}; see {stdout_path}"
            )
        setting = str(lock["benchmarks"][unit.benchmark]["setting"])
        mammoth_log = _mammoth_log_path(results_path, unit, setting)
        if not mammoth_log.is_file():
            raise RuntimeError(f"Mammoth did not create its result log: {mammoth_log}")
        raw = parse_mammoth_log(mammoth_log)
        metrics = metrics_from_mammoth_log(
            raw, allow_trajectory_only=bool(smoke) and setting == "domain-il"
        )
        expected_tasks = 2 if smoke else int(lock["benchmarks"][unit.benchmark]["tasks"])
        if int(metrics["evaluation_tasks"]) != expected_tasks:
            raise RuntimeError(
                f"task coverage mismatch: expected {expected_tasks}, "
                f"observed {metrics['evaluation_tasks']}"
            )
        audit_records = _read_jsonl(audit_path)
        if unit.intervention != "none" and not audit_records:
            raise RuntimeError("intervention arm produced no pairing audit records")
        if unit.intervention == "none":
            diagnostics: dict[str, Any] = {
                "records": 0,
                "gradient_diagnostic_records": 0,
                "intervention_records": 0,
                "projection_records": 0,
            }
            sampled_event_fingerprints: list[dict[str, Any]] = []
        else:
            diagnostics, sampled_event_fingerprints = _diagnostic_summary(
                audit_records,
                unit=unit,
                lock=lock,
                expected_effective_args=expected_effective_args,
            )
        record = {
            "schema_version": "gradient_reuse.mechanism_unit.v2",
            "identity": dict(identity),
            "benchmark": unit.benchmark,
            "method": unit.method,
            "mammoth_model": unit.mammoth_model,
            "estimand_class": method_estimand_class(unit.method),
            "intervention": unit.intervention,
            "seed": unit.seed,
            "permutation_replica": unit.permutation_replica,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "sampled_event_fingerprints": sampled_event_fingerprints,
            "effective_args": expected_effective_args,
            "runtime_seconds": time.time() - start,
            "paper_eligible": not smoke,
            "attempt": attempt.name,
        }
        manifest.update(
            {
                "status": "COMPLETED",
                "completed_unix": time.time(),
                "returncode": completed.returncode,
            }
        )
        atomic_json(attempt / "RUN_MANIFEST.json", manifest)
        atomic_json(unit_root / "UNIT.json", record)
        inventory = _artifact_inventory(unit_root, exclude=("DONE.json",))
        done = {
            "schema_version": "gradient_reuse.mechanism_unit_done.v2",
            "status": "PASS",
            "identity_sha256": canonical_sha256(identity),
            "artifact_count": len(inventory),
            "artifact_inventory": inventory,
            "artifact_bundle_sha256": canonical_sha256(inventory),
        }
        atomic_json(unit_root / "DONE.json", done)
        return record
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(attempt / "RUN_MANIFEST.json", manifest)
        raise


def _effect_summary(
    values: Sequence[Mapping[str, Any]], field: str, mc_variance_field: str
) -> dict[str, Any]:
    observed = [value.get(field) for value in values]
    available = [value is not None for value in observed]
    if not any(available):
        return {
            "available": False,
            "reason": "metric_not_emitted_and_not_reconstructible",
        }
    if not all(available):
        raise RuntimeError(f"mixed summary availability for {field}")
    result = mean_and_ci(
        (float(value) for value in observed),
        nested_mc_variances=(
            float(value.get(mc_variance_field, 0.0)) for value in values
        ),
    )
    result["available"] = True
    return result


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    lock: Mapping[str, Any],
    smoke: bool,
    permutation_replica_ids: Sequence[int],
) -> dict[str, Any]:
    comparability = _validate_cross_arm_comparability(
        records, permutation_replica_ids=permutation_replica_ids
    )
    rows = compute_mechanism_estimands(
        records, permutation_replica_ids=permutation_replica_ids
    )
    diagnostic_rows = compute_diagnostic_estimands(
        records, permutation_replica_ids=permutation_replica_ids
    )
    diagnostic_summaries = summarize_diagnostic_estimands(diagnostic_rows)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["benchmark"]), str(row["method"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (benchmark, method), values in sorted(grouped.items()):
        reference_methods = {str(value["reference_method"]) for value in values}
        if len(reference_methods) != 1:
            raise RuntimeError(f"mixed reference methods for {benchmark}/{method}")
        summaries.append(
            {
                "benchmark": benchmark,
                "method": method,
                "reference_method": next(iter(reference_methods)),
                "estimand_class": method_estimand_class(method),
                "accuracy_unit": "percentage_points",
                "V_total": _effect_summary(values, "V_total", "V_total_mc_variance"),
                "V_perm": _effect_summary(values, "V_perm", "V_perm_mc_variance"),
                "V_pair": _effect_summary(values, "V_pair", "V_pair_mc_variance"),
                "bwt_V_total": _effect_summary(
                    values, "bwt_V_total", "bwt_V_total_mc_variance"
                ),
                "bwt_V_perm": _effect_summary(
                    values, "bwt_V_perm", "bwt_V_perm_mc_variance"
                ),
                "bwt_V_pair": _effect_summary(
                    values, "bwt_V_pair", "bwt_V_pair_mc_variance"
                ),
                "incremental_accuracy_V_total": _effect_summary(
                    values,
                    "incremental_accuracy_V_total",
                    "incremental_accuracy_V_total_mc_variance",
                ),
                "incremental_accuracy_V_perm": _effect_summary(
                    values,
                    "incremental_accuracy_V_perm",
                    "incremental_accuracy_V_perm_mc_variance",
                ),
                "incremental_accuracy_V_pair": _effect_summary(
                    values,
                    "incremental_accuracy_V_pair",
                    "incremental_accuracy_V_pair_mc_variance",
                ),
                "max_abs_identity_residual": max(
                    abs(float(value["identity_residual"])) for value in values
                ),
            }
        )
    return {
        "schema_version": "gradient_reuse.mechanism_summary.v2",
        "paper_eligible": not smoke,
        "permutation_replicas_are_nested": True,
        "permutation_replica_ids": [int(value) for value in permutation_replica_ids],
        "primary_endpoint": str(lock["evaluation"]["primary"]),
        "primary_inference": str(lock["evaluation"]["primary_inference"]),
        "primary_estimand": "V_pair",
        "cross_method_accuracy_ranking_supported": bool(
            lock["evaluation"]["cross_method_accuracy_ranking_supported"]
        ),
        "cross_arm_comparability": comparability,
        "seed_rows": rows,
        "method_summaries": summaries,
        "diagnostic_seed_rows": diagnostic_rows,
        "diagnostic_summaries": diagnostic_summaries,
    }


def _selected_units(
    units: Sequence[MechanismUnit],
    *,
    benchmark: str | None,
    method: str | None,
    unit_id: str | None,
    max_units: int | None,
) -> list[MechanismUnit]:
    selected = [
        unit
        for unit in units
        if (benchmark is None or unit.benchmark == benchmark)
        and (method is None or unit.method == method)
        and (unit_id is None or unit.unit_id == unit_id)
    ]
    if unit_id is not None and len(selected) != 1:
        raise RuntimeError(f"unit selection did not resolve exactly once: {unit_id}")
    if max_units is not None:
        if isinstance(max_units, bool) or int(max_units) < 1:
            raise RuntimeError("--max-units must be a positive integer")
        selected = selected[: int(max_units)]
    if not selected:
        raise RuntimeError("mechanism unit selection is empty")
    return selected


def _select_campaign_units(
    units: Sequence[MechanismUnit],
    *,
    benchmark: str | None,
    method: str | None,
    unit_id: str | None,
    max_units: int | None,
    shard_count: int | None,
    shard_index: int | None,
) -> tuple[list[MechanismUnit], dict[str, Any]]:
    """Resolve either canonical filters or one deterministic modulo shard."""

    if (shard_count is None) != (shard_index is None):
        raise RuntimeError("--shard-count and --shard-index must be provided together")
    if shard_count is not None:
        if isinstance(shard_count, bool) or int(shard_count) < 1:
            raise RuntimeError("--shard-count must be a positive integer")
        if (
            isinstance(shard_index, bool)
            or int(shard_index) < 0
            or int(shard_index) >= int(shard_count)
        ):
            raise RuntimeError("--shard-index must satisfy 0 <= index < shard-count")
        filters = {
            "--benchmark": benchmark,
            "--method": method,
            "--unit-id": unit_id,
            "--max-units": max_units,
        }
        conflicts = [name for name, value in filters.items() if value is not None]
        if conflicts:
            raise RuntimeError(
                "scientific sharding is mutually exclusive with unit filters: "
                + ", ".join(conflicts)
            )
        count = int(shard_count)
        index = int(shard_index)
        selected = [
            unit
            for canonical_index, unit in enumerate(units)
            if canonical_index % count == index
        ]
        if not selected:
            raise RuntimeError("mechanism shard selection is empty")
        return selected, {
            "selection_scheme": SHARD_SELECTION_SCHEME,
            "shard_count": count,
            "shard_index": index,
            "selected_unit_ids": [unit.unit_id for unit in selected],
        }

    selected = _selected_units(
        units,
        benchmark=benchmark,
        method=method,
        unit_id=unit_id,
        max_units=max_units,
    )
    return selected, {
        "selection_scheme": FILTER_SELECTION_SCHEME,
        "shard_count": None,
        "shard_index": None,
        "selected_unit_ids": [unit.unit_id for unit in selected],
    }


def _running_campaign_manifest(
    invocation: Mapping[str, Any], *, run_attempt: int
) -> dict[str, Any]:
    if isinstance(run_attempt, bool) or int(run_attempt) < 1:
        raise RuntimeError("campaign run attempt must be a positive integer")
    return {
        "schema_version": "gradient_reuse.mechanism_campaign.v2",
        "status": "RUNNING",
        "started_unix": time.time(),
        "run_attempt": int(run_attempt),
        "invocation": dict(invocation),
    }


def _validate_completed_campaign_root(
    out: Path,
    *,
    invocation: Mapping[str, Any],
) -> None:
    """Authenticate an existing root DONE before treating it as resumable output."""

    manifest_path = out / "RUN_MANIFEST.json"
    done_path = out / "DONE.json"
    if not manifest_path.is_file() or not done_path.is_file():
        raise RuntimeError("completed campaign root lacks RUN_MANIFEST.json or DONE.json")
    campaign = strict_json(manifest_path)
    expected_campaign_keys = {
        "schema_version",
        "status",
        "started_unix",
        "run_attempt",
        "invocation",
        "completed_unix",
        "completed_units",
        "completed_unit_ids",
        "completed_unit_done_sha256",
        "complete_protocol",
    }
    if not isinstance(campaign, Mapping) or set(campaign) != expected_campaign_keys:
        raise RuntimeError("completed campaign RUN_MANIFEST schema changed")
    if (
        campaign["schema_version"] != "gradient_reuse.mechanism_campaign.v2"
        or campaign["status"] != "COMPLETED"
        or campaign["invocation"] != dict(invocation)
        or isinstance(campaign["run_attempt"], bool)
        or int(campaign["run_attempt"]) < 1
    ):
        raise RuntimeError("completed campaign RUN_MANIFEST identity changed")

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
        raise RuntimeError("completed campaign DONE schema changed")
    if (
        done["schema_version"] != "gradient_reuse.mechanism_done.v2"
        or done["status"] != "PASS"
        or done["paper_eligible"] is not invocation["paper_eligible"]
        or done["complete_protocol"] is not campaign["complete_protocol"]
        or done["completed_unit_ids"] != campaign["completed_unit_ids"]
        or done["completed_unit_done_sha256"]
        != campaign["completed_unit_done_sha256"]
    ):
        raise RuntimeError("completed campaign DONE identity changed")
    selected = invocation["selected_unit_ids"]
    if campaign["completed_unit_ids"] != selected:
        raise RuntimeError("completed campaign unit IDs differ from its invocation")
    unit_done = campaign["completed_unit_done_sha256"]
    if not isinstance(unit_done, Mapping) or set(unit_done) != set(selected):
        raise RuntimeError("completed campaign unit-DONE inventory changed")
    for unit_id in selected:
        path = out / "units" / unit_id / "DONE.json"
        if not path.is_file() or unit_done[unit_id] != sha256_file(path):
            raise RuntimeError(f"completed campaign unit-DONE digest changed: {unit_id}")
    claimed = done["artifact_inventory"]
    if not isinstance(claimed, Mapping) or not claimed:
        raise RuntimeError("completed campaign artifact inventory is empty")
    observed = _artifact_inventory(out, exclude=("DONE.json",))
    if claimed != observed:
        raise RuntimeError("completed campaign artifact inventory changed")
    if int(done["artifact_count"]) != len(observed):
        raise RuntimeError("completed campaign artifact count changed")
    if done["artifact_bundle_sha256"] != canonical_sha256(observed):
        raise RuntimeError("completed campaign artifact bundle changed")


def _prepare_campaign_root(
    out: Path,
    *,
    invocation: Mapping[str, Any],
) -> tuple[Path, bool]:
    """Create or resume RUN_MANIFEST, clearing stale failure state exactly."""

    if out.exists():
        if out.is_symlink() or not out.is_dir():
            raise RuntimeError(f"mechanism output root is unsafe: {out}")
        if not (out / "RUN_MANIFEST.json").is_file():
            raise RuntimeError(
                "refusing a preexisting mechanism output root without an owned "
                "RUN_MANIFEST.json"
            )
    else:
        out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "RUN_MANIFEST.json"
    done_path = out / "DONE.json"
    if done_path.exists():
        _validate_completed_campaign_root(out, invocation=invocation)
        return manifest_path, True

    run_attempt = 1
    if manifest_path.exists():
        existing = strict_json(manifest_path)
        if not isinstance(existing, Mapping):
            raise RuntimeError("existing campaign manifest is not an object")
        base_keys = {
            "schema_version",
            "status",
            "started_unix",
            "run_attempt",
            "invocation",
        }
        status_keys = {
            "RUNNING": base_keys,
            "FAILED": base_keys
            | {"failed_unix", "error", "traceback"},
            "COMPLETED": base_keys
            | {
                "completed_unix",
                "completed_units",
                "completed_unit_ids",
                "completed_unit_done_sha256",
                "complete_protocol",
            },
        }
        status = existing.get("status")
        if (
            existing.get("schema_version")
            != "gradient_reuse.mechanism_campaign.v2"
            or not isinstance(status, str)
            or status not in status_keys
            or set(existing) != status_keys[status]
            or existing.get("invocation") != dict(invocation)
        ):
            raise RuntimeError("output directory belongs to a different invocation")
        prior_attempt = existing.get("run_attempt")
        if (
            isinstance(prior_attempt, bool)
            or not isinstance(prior_attempt, int)
            or prior_attempt < 1
        ):
            raise RuntimeError("existing campaign has an invalid run attempt")
        run_attempt = prior_attempt + 1
    atomic_json(
        manifest_path,
        _running_campaign_manifest(invocation, run_attempt=run_attempt),
    )
    return manifest_path, False


def run_campaign(args: argparse.Namespace) -> Path:
    """Validate campaign roots, then serialize all writes to one output root."""

    if args.data_root is None or args.out is None:
        raise RuntimeError("campaign execution requires --data-root and --out")
    out = Path(args.out).resolve()
    data_manifest = (
        Path(args.data_manifest).resolve()
        if args.data_manifest is not None
        else None
    )
    environment_manifest = (
        Path(args.environment_manifest).resolve()
        if args.environment_manifest is not None
        else None
    )
    _validate_campaign_path_disjointness(
        out=out,
        data_root=Path(args.data_root).resolve(),
        mammoth_root=Path(args.mammoth_root).resolve(),
        lock_path=Path(args.lock).resolve(),
        source_manifest=Path(args.source_manifest).resolve(),
        data_manifest=data_manifest,
        environment_manifest=environment_manifest,
    )
    with _campaign_root_lock(out):
        return _run_campaign_locked(args)


def _run_campaign_locked(args: argparse.Namespace) -> Path:
    _validate_device(str(args.device), smoke=bool(args.smoke))
    lock_path = Path(args.lock).resolve()
    source_path = Path(args.source_manifest).resolve()
    mammoth_root = Path(args.mammoth_root).resolve()
    data_root = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    lock = load_and_validate_lock(lock_path)
    source_bundle, _ = validate_source_manifest(
        source_path, repo=REPO, lock_path=lock_path
    )
    overlay = prepare_overlay(mammoth_root)
    data_payload: dict[str, Any] | None = None
    data_path: Path | None = None
    data_manifest_sha256: str | None = None
    data_bundle_sha256: str | None = None
    if not args.smoke:
        if args.data_manifest is None:
            raise RuntimeError("paper-eligible mechanism run requires --data-manifest")
        data_path = Path(args.data_manifest).resolve()
        data_payload = validate_data_manifest(data_path, data_root=data_root)
        data_manifest_sha256 = sha256_file(data_path)
        data_bundle_sha256 = str(data_payload["data_bundle_sha256"])
    elif args.data_manifest is not None:
        data_path = Path(args.data_manifest).resolve()
        data_payload = validate_data_manifest(data_path, data_root=data_root)
        data_manifest_sha256 = sha256_file(data_path)
        data_bundle_sha256 = str(data_payload["data_bundle_sha256"])

    if not args.smoke and args.environment_manifest is None:
        raise RuntimeError(
            "paper-eligible mechanism run requires --environment-manifest"
        )
    if args.environment_manifest is not None:
        environment_path = Path(args.environment_manifest).resolve()
        environment_payload = validate_environment_manifest(
            environment_path, mammoth_root=mammoth_root, device=str(args.device)
        )
        environment_manifest_sha256 = sha256_file(environment_path)
    else:
        environment_path = None
        environment_payload = build_environment_manifest(
            mammoth_root=mammoth_root, device=str(args.device)
        )
        environment_manifest_sha256 = canonical_sha256(environment_payload)
    environment_bundle_sha256 = str(
        environment_payload["environment_bundle_sha256"]
    )
    if not args.smoke:
        validate_required_mammoth_packages(environment_payload)

    units = enumerate_units(lock)
    selected, selection = _select_campaign_units(
        units,
        benchmark=args.benchmark,
        method=args.method,
        unit_id=args.unit_id,
        max_units=args.max_units,
        shard_count=getattr(args, "shard_count", None),
        shard_index=getattr(args, "shard_index", None),
    )
    if args.max_units is not None and not args.smoke:
        raise RuntimeError("--max-units is restricted to paper-ineligible smoke runs")
    effective_hashes = {
        unit.unit_id: canonical_sha256(
            effective_mammoth_args(
                unit, lock, device=str(args.device), smoke=bool(args.smoke)
            )
        )
        for unit in selected
    }
    is_scientific_shard = selection["selection_scheme"] == SHARD_SELECTION_SCHEME
    complete_selection = not is_scientific_shard and [
        unit.unit_id for unit in selected
    ] == [unit.unit_id for unit in units]
    campaign_paper_eligible = not args.smoke and complete_selection
    invocation = {
        "schema_version": "gradient_reuse.mechanism_invocation.v2",
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source_bundle,
        "overlay_manifest_sha256": sha256_file(mammoth_root / MANIFEST_NAME),
        "overlay_bundle_sha256": overlay["overlay_bundle_sha256"],
        "upstream_commit": overlay["upstream_commit"],
        "data_manifest_sha256": data_manifest_sha256,
        "data_bundle_sha256": data_bundle_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "environment_bundle_sha256": environment_bundle_sha256,
        "smoke": bool(args.smoke),
        "paper_eligible": campaign_paper_eligible,
        "device": str(args.device),
        **selection,
        "effective_args_sha256": effective_hashes,
    }
    manifest_path, already_complete = _prepare_campaign_root(
        out, invocation=invocation
    )
    if already_complete:
        return out
    _copy_locked_file(lock_path, out / "LOCK.json")
    _copy_locked_file(source_path, out / "SOURCE_MANIFEST.json")
    _copy_locked_file(
        mammoth_root / MANIFEST_NAME, out / "OVERLAY_MANIFEST.json"
    )
    if data_path is not None:
        _copy_locked_file(data_path, out / "DATA_MANIFEST.json")
    if environment_path is not None:
        _copy_locked_file(environment_path, out / "ENVIRONMENT.json")
    else:
        generated_environment = out / "ENVIRONMENT.json"
        if (
            generated_environment.exists()
            and strict_json(generated_environment) != environment_payload
        ):
            raise RuntimeError("generated environment manifest changed on resume")
        if not generated_environment.exists():
            atomic_json(generated_environment, environment_payload)

    records: list[dict[str, Any]] = []
    try:
        for index, unit in enumerate(selected, 1):
            identity = unit_identity(
                unit,
                lock_sha256=invocation["lock_sha256"],
                source_bundle_sha256=source_bundle,
                overlay_bundle_sha256=overlay["overlay_bundle_sha256"],
                data_manifest_sha256=data_manifest_sha256,
                data_bundle_sha256=data_bundle_sha256,
                environment_manifest_sha256=environment_manifest_sha256,
                environment_bundle_sha256=environment_bundle_sha256,
                effective_args_sha256=effective_hashes[unit.unit_id],
                device=str(args.device),
                smoke=bool(args.smoke),
            )
            print(f"[mechanism] {index}/{len(selected)} {unit.unit_id}", flush=True)
            records.append(
                run_unit(
                    unit,
                    lock,
                    mammoth_root=mammoth_root,
                    data_root=data_root,
                    out=out,
                    identity=identity,
                    device=str(args.device),
                    smoke=bool(args.smoke),
                )
            )
        if complete_selection:
            summary = summarize(
                records,
                lock=lock,
                smoke=bool(args.smoke),
                permutation_replica_ids=lock["runtime"]["permutation_replicas"],
            )
            atomic_json(out / "SUMMARY.json", summary)
        completed_unit_ids = [unit.unit_id for unit in selected]
        completed_unit_digests = {
            unit_id: sha256_file(out / "units" / unit_id / "DONE.json")
            for unit_id in completed_unit_ids
        }
        campaign = strict_json(manifest_path)
        campaign.update(
            {
                "status": "COMPLETED",
                "completed_unix": time.time(),
                "completed_units": len(records),
                "completed_unit_ids": completed_unit_ids,
                "completed_unit_done_sha256": completed_unit_digests,
                "complete_protocol": complete_selection,
            }
        )
        atomic_json(manifest_path, campaign)
        inventory = _artifact_inventory(out, exclude=("DONE.json",))
        atomic_json(
            out / "DONE.json",
            {
                "schema_version": "gradient_reuse.mechanism_done.v2",
                "status": "PASS",
                "paper_eligible": campaign_paper_eligible,
                "complete_protocol": complete_selection,
                "completed_unit_ids": completed_unit_ids,
                "completed_unit_done_sha256": completed_unit_digests,
                "artifact_count": len(inventory),
                "artifact_inventory": inventory,
                "artifact_bundle_sha256": canonical_sha256(inventory),
            },
        )
        return out
    except BaseException as error:
        campaign = strict_json(manifest_path)
        campaign.update(
            {
                "status": "FAILED",
                "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(manifest_path, campaign)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--full",
        action="store_true",
        help="run the paper-eligible protocol (subject to all integrity checks)",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run an explicitly paper-ineligible smoke protocol",
    )
    mode.add_argument(
        "--capture-environment-manifest",
        metavar="PATH",
        help="capture only the validated environment manifest and exit",
    )
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--mammoth-root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--data-manifest")
    parser.add_argument("--environment-manifest")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark")
    parser.add_argument("--method")
    parser.add_argument("--unit-id")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--shard-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture_environment_manifest:
        _validate_device(str(args.device), smoke=False)
        mammoth_root = Path(args.mammoth_root).resolve()
        destination = Path(args.capture_environment_manifest).resolve()
        if _paths_overlap(destination, mammoth_root):
            raise RuntimeError(
                "environment manifest destination must be outside Mammoth root"
            )
        for protected in (Path(args.lock).resolve(), Path(args.source_manifest).resolve()):
            if _paths_overlap(destination, protected):
                raise RuntimeError(
                    "environment manifest destination overlaps a locked input"
                )
        with overlay_checkout_lock(mammoth_root):
            verify_upstream(mammoth_root)
            payload = build_environment_manifest(
                mammoth_root=mammoth_root, device=str(args.device)
            )
        if destination.exists() and strict_json(destination) != payload:
            raise RuntimeError("refusing to overwrite another environment manifest")
        if not destination.exists():
            atomic_json(destination, payload)
        print(destination)
        return 0
    if args.data_root is None or args.out is None:
        raise RuntimeError("campaign execution requires --data-root and --out")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
