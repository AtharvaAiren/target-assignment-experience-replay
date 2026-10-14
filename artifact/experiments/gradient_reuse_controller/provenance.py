"""Fail-closed provenance bindings for the gradient-reuse controllers.

The experiment implementations are intentionally usable as ordinary Python
functions for engineering checks.  A result becomes scientific evidence only
when the CLI supplies opaque bindings produced by the validators in this
module.  The serialized form of every binding contains the complete manifest,
not merely a path to mutable external state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from experiments.source_provenance import (
    SourceProvenanceError,
    get_source_spec,
    strict_json,
    validate_source_manifest,
)


RUNTIME_SCHEMA = "gradient_reuse.controller_runtime.v1"
RUNTIME_BINDING_SCHEMA = "gradient_reuse.controller_runtime_binding.v1"
_VALIDATION_SENTINEL = object()
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ValidatedEvidence:
    """An in-process proof that a serialized manifest passed validation."""

    kind: str
    payload: Mapping[str, Any]
    _sentinel: object


def _validated_evidence(kind: str, payload: Mapping[str, Any]) -> ValidatedEvidence:
    """Construct a binding after a validator succeeds (module-private)."""

    return ValidatedEvidence(str(kind), dict(payload), _VALIDATION_SENTINEL)


def evidence_payload(
    binding: ValidatedEvidence | None, *, kind: str
) -> dict[str, Any] | None:
    if (
        not isinstance(binding, ValidatedEvidence)
        or binding.kind != kind
        or binding._sentinel is not _VALIDATION_SENTINEL
    ):
        return None
    return dict(binding.payload)


def validate_source_binding(
    path: Path,
    *,
    experiment: str,
    config_path: Path,
    repo: Path,
) -> ValidatedEvidence:
    """Revalidate source bytes and return an opaque source binding.

    There is deliberately no public arbitrary-payload binding constructor.
    Scientific CLIs must reach this function through an exact source manifest.
    """

    resolved = path.expanduser().resolve(strict=True)
    resolved_config = config_path.expanduser().resolve(strict=True)
    bundle, files = validate_source_manifest(resolved, experiment, repo=repo)
    spec = get_source_spec(experiment)
    if sha256_file(resolved_config) != files[spec.configuration_path]:
        raise SourceProvenanceError(
            f"source manifest authenticates another {experiment} configuration"
        )
    evidence = {
        "experiment": experiment,
        "manifest_filename": path.name,
        "manifest_sha256": sha256_file(resolved),
        "source_bundle_sha256": bundle,
        "source_file_count": len(files),
        "shared_manifest_validated": True,
        "configuration_matches_manifest": True,
        "validated_for_execution": True,
        "validation_error": None,
        "manifest": strict_json(resolved),
    }
    return _validated_evidence("source", evidence)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def configure_execution_runtime(
    *, experiment: str, expected: Mapping[str, Any] | None = None
) -> None:
    """Put Torch in the state used by the natural scientific computation."""

    if experiment != "controller_natural":
        return
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if expected is None:
        return
    torch_state = expected.get("torch")
    if not isinstance(torch_state, Mapping):
        raise RuntimeError("runtime manifest lacks locked Torch state")
    maximum = max(1, int(os.cpu_count() or 1))
    num_threads = int(torch_state.get("num_threads", 0))
    num_interop_threads = int(torch_state.get("num_interop_threads", 0))
    if not 1 <= num_threads <= maximum:
        raise RuntimeError("runtime manifest has invalid Torch thread count")
    if not 1 <= num_interop_threads <= maximum:
        raise RuntimeError("runtime manifest has invalid Torch interop thread count")
    torch.set_num_threads(num_threads)
    if torch.get_num_interop_threads() != num_interop_threads:
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "cannot configure locked Torch interop thread count in this process"
            ) from error


def observed_runtime(
    *, experiment: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    if experiment not in {"controller", "controller_natural"}:
        raise RuntimeError(f"unsupported controller runtime experiment: {experiment}")
    observed: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__},
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "determinism": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "numpy_random_generator": "PCG64",
        },
    }
    if experiment == "controller":
        observed["optimizer"] = {
            "name": "normalized_sgd_plus_optional_fixed_radius_reuse",
            "base_learning_rate": float(config["model"]["base_learning_rate"]),
            "reuse_radius": float(config["model"]["reuse_radius"]),
        }
    else:
        import torch

        observed["packages"].update(
            {
                "torch": torch.__version__,
                "pandas": _package_version("pandas"),
            }
        )
        observed["torch"] = {
            "device": str(config["model"]["device"]),
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "optimizer": {
                "name": str(config["optimizer"]["name"]),
                "betas": [float(value) for value in config["optimizer"]["betas"]],
                "learning_rate": float(config["optimizer"]["learning_rate"]),
                "weight_decay": float(config["optimizer"]["weight_decay"]),
                "gradient_clip": float(config["optimizer"]["gradient_clip"]),
            },
        }
    return observed


def source_identity(source_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if source_evidence.get("validated_for_execution") is not True:
        raise RuntimeError("runtime capture requires validated source evidence")
    manifest = source_evidence.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("validated source evidence lacks its immutable manifest")
    return {
        "experiment": str(source_evidence["experiment"]),
        "manifest_sha256": str(source_evidence["manifest_sha256"]),
        "source_bundle_sha256": str(source_evidence["source_bundle_sha256"]),
        "configuration_sha256": str(manifest["configuration_sha256"]),
        "manifest": dict(manifest),
    }


def build_runtime_manifest(
    *,
    experiment: str,
    config_path: Path,
    config: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config = config_path.resolve(strict=True)
    if canonical_sha256(strict_json(resolved_config)) != canonical_sha256(config):
        raise RuntimeError("runtime capture config payload differs from --config bytes")
    configure_execution_runtime(experiment=experiment)
    observed = observed_runtime(experiment=experiment, config=config)
    return {
        "schema_version": RUNTIME_SCHEMA,
        "status": "CAPTURED_BEFORE_SCIENTIFIC_RUN",
        "scientific_evidence_eligible": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "configuration_file_sha256": sha256_file(resolved_config),
        "configuration_payload_sha256": canonical_sha256(config),
        "source_identity": source_identity(source_evidence),
        "observed": observed,
        "runtime_fingerprint_sha256": canonical_sha256(observed),
    }


def validate_runtime_manifest(
    path: Path,
    *,
    experiment: str,
    config_path: Path,
    config: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> ValidatedEvidence:
    resolved = path.expanduser().resolve(strict=True)
    resolved_config = config_path.resolve(strict=True)
    try:
        payload = strict_json(resolved)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(f"controller runtime manifest is unreadable: {resolved}") from error
    required = {
        "schema_version",
        "status",
        "scientific_evidence_eligible",
        "created_utc",
        "experiment",
        "configuration_file_sha256",
        "configuration_payload_sha256",
        "source_identity",
        "observed",
        "runtime_fingerprint_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RuntimeError("controller runtime manifest keys differ from schema")
    if payload["schema_version"] != RUNTIME_SCHEMA:
        raise RuntimeError("controller runtime manifest schema mismatch")
    if payload["status"] != "CAPTURED_BEFORE_SCIENTIFIC_RUN":
        raise RuntimeError("controller runtime manifest is not a locked capture")
    if payload["scientific_evidence_eligible"] is not False:
        raise RuntimeError("runtime capture must not claim scientific eligibility")
    if payload["experiment"] != experiment:
        raise RuntimeError("controller runtime manifest authenticates another experiment")
    if payload["configuration_file_sha256"] != sha256_file(
        resolved_config
    ):
        raise RuntimeError("controller runtime manifest authenticates another config")
    if canonical_sha256(strict_json(resolved_config)) != canonical_sha256(config):
        raise RuntimeError("runtime validation config payload differs from --config bytes")
    if payload["configuration_payload_sha256"] != canonical_sha256(config):
        raise RuntimeError("controller runtime manifest authenticates another config payload")
    if payload["source_identity"] != source_identity(source_evidence):
        raise RuntimeError("controller runtime manifest authenticates another source freeze")
    if payload["runtime_fingerprint_sha256"] != canonical_sha256(payload["observed"]):
        raise RuntimeError("controller runtime fingerprint is invalid")
    configure_execution_runtime(experiment=experiment, expected=payload["observed"])
    observed = observed_runtime(experiment=experiment, config=config)
    if payload["observed"] != observed:
        raise RuntimeError("current runtime differs from the controller runtime capture")
    evidence = {
        "schema_version": RUNTIME_BINDING_SCHEMA,
        "validation_status": "PASS",
        "manifest_sha256": sha256_file(resolved),
        "runtime_fingerprint_sha256": str(payload["runtime_fingerprint_sha256"]),
        "manifest": dict(payload),
    }
    return _validated_evidence("runtime", evidence)


def write_fresh_json(path: Path, payload: Mapping[str, Any]) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another process creates
        # the destination after the preflight check.  Unlike ``os.replace``,
        # it can never clobber an existing scientific result.
        os.link(temporary, output)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def require_disjoint_paths(paths: Mapping[str, str | Path | None]) -> None:
    observed: dict[Path, str] = {}
    for role, raw in paths.items():
        if raw is None:
            continue
        resolved = Path(raw).expanduser().resolve()
        previous = observed.get(resolved)
        if previous is not None:
            raise RuntimeError(
                f"path roles must be disjoint: {previous} and {role} both resolve to {resolved}"
            )
        observed[resolved] = role
