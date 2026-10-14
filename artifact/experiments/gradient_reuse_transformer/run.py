"""Command-line runner for the fixed-conditioning Pythia audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import torch

from experiments.source_provenance import (
    SourceProvenanceError,
    strict_json as strict_source_json,
    validate_source_manifest as validate_shared_source_manifest,
)

from .assets import validate_asset_manifest
from .audit import run_next_token_audit, seed_everything, warmup_model
from .data import build_chronological_stream, iter_json_records, packetize
from .model import PythiaNextTokenModel, load_local_pythia


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/transformer.json"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ACTUATOR_NAME = "equal_radius_raw_gradient_actuator"
ENVIRONMENT_SCHEMA = "fmmt.gradient_reuse_transformer.environment.v1"
PREFLIGHT_SCHEMA = "fmmt.gradient_reuse_transformer.cuda_preflight.v1"


def validate_transformer_source_manifest(
    source_manifest: str | Path | None,
    *,
    config_path: str | Path | None = None,
    required: bool,
) -> dict[str, Any]:
    """Validate and return a path-free source identity for output binding.

    Full runs and asset acquisition call this with ``required=True``.  Tiny
    smoke runs may omit the freeze, but the omission remains explicit in their
    output and can never make them paper eligible.
    """

    if source_manifest is None:
        if required:
            raise RuntimeError(
                "a validated transformer --source-manifest is required for "
                "paper-eligible execution"
            )
        return {
            "schema_version": "gradient_reuse.transformer_source_binding.v1",
            "experiment": "transformer",
            "validation_status": "NOT_PROVIDED",
            "source_manifest_sha256": None,
            "source_bundle_sha256": None,
            "configuration_sha256": None,
        }

    try:
        path = Path(source_manifest).expanduser().resolve(strict=True)
        source_bundle, source_hashes = validate_shared_source_manifest(
            path, "transformer", repo=REPO
        )
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(
            f"transformer source manifest failed validation: {error}"
        ) from error

    configuration_sha256 = source_hashes["configs/transformer.json"]
    if config_path is not None:
        resolved_config = Path(config_path).expanduser().resolve(strict=True)
        if _sha256_file(resolved_config) != configuration_sha256:
            raise RuntimeError(
                "transformer source manifest authenticates another configuration"
            )
    return {
        "schema_version": "gradient_reuse.transformer_source_binding.v1",
        "experiment": "transformer",
        "validation_status": "PASS",
        "source_manifest_sha256": _sha256_file(path),
        "source_bundle_sha256": source_bundle,
        "configuration_sha256": configuration_sha256,
    }


def _validate_asset_source_binding(
    asset_manifest: str | Path,
    expected: Mapping[str, Any],
) -> None:
    """Reject assets acquired under a different or absent source freeze."""

    path = Path(asset_manifest).expanduser().resolve(strict=True)
    try:
        payload = strict_source_json(path)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(f"cannot read asset manifest source binding: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("asset manifest must be a JSON object")
    if payload.get("source_provenance") != dict(expected):
        raise RuntimeError(
            "asset manifest source provenance differs from the validated run source"
        )


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = strict_source_json(config_path)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(
            f"Transformer lock is not strict JSON: {config_path}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("Transformer lock must be a JSON object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "fmmt.gradient_reuse_transformer.lock.v1":
        raise RuntimeError("unsupported Transformer lock schema")
    if config.get("status") != "LOCKED_BEFORE_TRANSFORMER_RUN":
        raise RuntimeError("Transformer experiment is not prespecified as locked")
    assets = config.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {"model", "dataset"}:
        raise RuntimeError("lock must specify exactly model and dataset assets")
    for role, value in assets.items():
        if not isinstance(value, Mapping):
            raise RuntimeError(f"invalid {role} asset lock")
        if value.get("kind") != "directory":
            raise RuntimeError(f"{role} asset must be a content-hashed directory")
        if not _REVISION.fullmatch(str(value.get("resolved_revision", ""))):
            raise RuntimeError(f"{role} is not pinned to a full immutable revision")
    stream = config.get("stream", {})
    context = int(stream.get("context_length", 0))
    warmup = int(stream.get("warmup_examples", 0))
    online = int(stream.get("online_examples", 0))
    packet = int(stream.get("packet_size", 0))
    if context < 8 or packet < 2 or warmup < packet or online < 3 * packet:
        raise RuntimeError("invalid locked stream dimensions")
    if warmup % packet or online % packet:
        raise RuntimeError("warmup and online counts must be packet divisible")
    if stream.get("conditioning_policy") != "fixed_context_predicted_token_only":
        raise RuntimeError("the lock does not preserve teacher-forced conditioning")
    audit = config.get("audit", {})
    if int(audit.get("n_permutations", 0)) != 4:
        raise RuntimeError("the locked finite-permutation audit requires four replicas")
    if "seeds" in audit:
        raise RuntimeError("permutation seeds may not be presented as trajectory replicas")
    if not isinstance(audit.get("trajectory_seed"), int):
        raise RuntimeError("lock needs one integer trajectory_seed")
    if audit.get("independent_model_trajectories") != 1:
        raise RuntimeError("the fixed checkpoint defines exactly one model trajectory")
    if audit.get("seed_level_replication") is not False:
        raise RuntimeError("seed-level replication must be explicitly disabled")
    model = config.get("model", {})
    if model.get("use_cache") is not False or model.get("attention_implementation") != "eager":
        raise RuntimeError("model must lock eager attention and disable KV caching")
    if float(model.get("dropout", -1)) != 0.0:
        raise RuntimeError("model dropout must be exactly zero")
    if model.get("logits_to_keep") != 1:
        raise RuntimeError("model must materialize only the final-position logits")
    optimization = config.get("optimization", {})
    if optimization.get("optimizer") != "AdamW":
        raise RuntimeError("the live optimizer must be AdamW")
    if optimization.get("counterfactual_actuator") != ACTUATOR_NAME:
        raise RuntimeError("the counterfactual actuator is not the locked raw-gradient rule")
    if optimization.get("adamw_foreach") is not False:
        raise RuntimeError("AdamW foreach must be explicitly disabled")
    if optimization.get("adamw_fused") is not False:
        raise RuntimeError("fused AdamW must be explicitly disabled")
    runtime = config.get("runtime", {})
    for dependency in ("python", "device", "torch", "transformers", "numpy"):
        if not isinstance(runtime.get(dependency), str) or not runtime[dependency]:
            raise RuntimeError(f"runtime does not pin {dependency}")
    if runtime.get("one_gpu_process") is not True:
        raise RuntimeError("full campaign must use one GPU process")
    preflight = config.get("cuda_preflight", {})
    if preflight.get("required_for_paper_eligible_full_run") is not True:
        raise RuntimeError("paper-eligible full runs must require a CUDA preflight")
    minimum_warmup = int(preflight.get("minimum_warmup_packets", 0))
    minimum_online = int(preflight.get("minimum_online_packets", 0))
    minimum_memory = int(preflight.get("minimum_total_device_memory_bytes", 0))
    maximum_reserved = int(preflight.get("maximum_peak_reserved_bytes", 0))
    if minimum_warmup < 1 or minimum_online < 3:
        raise RuntimeError("CUDA preflight packet minima are too small")
    if minimum_memory < 1 or maximum_reserved < 1 or maximum_reserved >= minimum_memory:
        raise RuntimeError("CUDA preflight memory limits are invalid")
    if audit.get("fail_on_zero_norm_actuator_branch") is not True:
        raise RuntimeError("full audit must fail on zero-norm actuator branches")


def _validate_runtime(config: Mapping[str, Any], device: str) -> None:
    runtime = config["runtime"]
    observed = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    for name, value in observed.items():
        if str(value) != str(runtime[name]):
            raise RuntimeError(
                f"runtime version mismatch for {name}: expected {runtime[name]}, got {value}"
            )
    if str(device) != str(runtime["device"]):
        raise RuntimeError(
            f"paper run requires device={runtime['device']!r}, got {device!r}"
        )
    expected_cublas = str(runtime["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_cublas:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be {expected_cublas!r} before launch"
        )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) if values else "UNAVAILABLE"


def capture_runtime_environment(
    *,
    device: str,
    optimizer_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture path-free runtime facts that can affect the locked computation."""
    torch_device = torch.device(device)
    gpu: dict[str, Any] | None = None
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA environment capture requested but CUDA is unavailable")
        index = torch.cuda.current_device() if torch_device.index is None else torch_device.index
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "logical_index": int(index),
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "visible_device_count": int(torch.cuda.device_count()),
            "driver_version": _nvidia_driver_version(),
        }
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "huggingface_hub",
                "pyarrow",
            )
        },
        "accelerator": {
            "requested_device": str(torch_device),
            "torch_cuda_runtime": str(torch.version.cuda),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu": gpu,
        },
        "numerics": {
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "optimizer": {
            "name": str(optimizer_specification["optimizer"]),
            "foreach": bool(optimizer_specification["adamw_foreach"]),
            "fused": bool(optimizer_specification["adamw_fused"]),
            "counterfactual_actuator": str(
                optimizer_specification["counterfactual_actuator"]
            ),
        },
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_validated_input_bundle(
    files: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Embed complete immutable inputs and inventory both bytes and payloads."""

    payloads: dict[str, dict[str, Any]] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for role, raw_path in sorted(files.items()):
        path = Path(raw_path).expanduser().resolve(strict=True)
        try:
            payload = strict_source_json(path)
        except (OSError, SourceProvenanceError) as error:
            raise RuntimeError(f"cannot embed validated {role}: {path}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"validated {role} must be a JSON object")
        payloads[role] = payload
        inventory[role] = {
            "file_sha256": _sha256_file(path),
            "canonical_payload_sha256": _canonical_sha256(payload),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.input_bundle.v1",
        "payloads": payloads,
        "inventory": inventory,
        "bundle_sha256": _canonical_sha256(payloads),
    }


def validate_embedded_input_bundle(bundle: Mapping[str, Any]) -> None:
    if set(bundle) != {"schema_version", "payloads", "inventory", "bundle_sha256"}:
        raise RuntimeError("embedded input-bundle keys differ from schema")
    if bundle.get("schema_version") != "fmmt.gradient_reuse_transformer.input_bundle.v1":
        raise RuntimeError("unsupported embedded input-bundle schema")
    payloads = bundle.get("payloads")
    inventory = bundle.get("inventory")
    if not isinstance(payloads, dict) or not isinstance(inventory, dict):
        raise RuntimeError("embedded input bundle has invalid payloads or inventory")
    if set(payloads) != set(inventory):
        raise RuntimeError("embedded input-bundle inventory is incomplete")
    if bundle.get("bundle_sha256") != _canonical_sha256(payloads):
        raise RuntimeError("embedded input-bundle hash is invalid")
    for role, payload in payloads.items():
        record = inventory.get(role)
        if not isinstance(payload, dict) or not isinstance(record, dict):
            raise RuntimeError("embedded input-bundle entry is invalid")
        if record.get("canonical_payload_sha256") != _canonical_sha256(payload):
            raise RuntimeError(f"embedded {role} canonical hash is invalid")
        if _SHA256.fullmatch(str(record.get("file_sha256", ""))) is None:
            raise RuntimeError(f"embedded {role} file hash is invalid")
        if int(record.get("size_bytes", -1)) < 2:
            raise RuntimeError(f"embedded {role} byte size is invalid")


def _artifact_inventory(components: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory the self-contained scientific payload without recursive hashes."""

    hashes = {
        name: _canonical_sha256(value)
        for name, value in sorted(components.items())
    }
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.artifact_inventory.v1",
        "component_sha256": hashes,
        "artifact_bundle_sha256": _canonical_sha256(hashes),
    }


def validate_artifact_inventory(payload: Mapping[str, Any]) -> None:
    """Verify the non-recursive inventory embedded in a completed result."""

    inventory = payload.get("artifact_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "schema_version",
        "component_sha256",
        "artifact_bundle_sha256",
    }:
        raise RuntimeError("Transformer result has no valid artifact inventory")
    if (
        inventory.get("schema_version")
        != "fmmt.gradient_reuse_transformer.artifact_inventory.v1"
    ):
        raise RuntimeError("unsupported Transformer artifact-inventory schema")
    without_inventory = copy.deepcopy(dict(payload))
    without_inventory.pop("artifact_inventory", None)
    components: dict[str, Any] = {
        "result_payload_without_inventory": without_inventory,
        "validated_input_bundle": payload.get("validated_input_bundle"),
        "stream": payload.get("stream"),
        "unit": payload.get("unit"),
    }
    if "resources" in payload:
        components["resources"] = payload["resources"]
    if "cuda_preflight_provenance" in payload:
        components["cuda_preflight_provenance"] = payload[
            "cuda_preflight_provenance"
        ]
    expected_hashes = {
        name: _canonical_sha256(value)
        for name, value in sorted(components.items())
    }
    if inventory.get("component_sha256") != expected_hashes:
        raise RuntimeError("Transformer artifact component inventory is invalid")
    if inventory.get("artifact_bundle_sha256") != _canonical_sha256(
        expected_hashes
    ):
        raise RuntimeError("Transformer artifact-bundle hash is invalid")


def build_environment_manifest(
    *,
    config_sha256: str,
    source_provenance: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    if _SHA256.fullmatch(str(config_sha256)) is None:
        raise RuntimeError("environment capture received an invalid configuration hash")
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "status": "CAPTURED_FOR_LOCKED_RUNTIME",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": str(config_sha256),
        "source_provenance": dict(source_provenance),
        "observed": dict(observed),
        "environment_fingerprint_sha256": _canonical_sha256(observed),
    }


def validate_environment_manifest(
    manifest_path: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_provenance: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        payload = strict_source_json(path)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(f"cannot read Transformer environment manifest: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Transformer environment manifest must be a JSON object")
    required = {
        "schema_version",
        "status",
        "created_utc",
        "config_sha256",
        "source_provenance",
        "observed",
        "environment_fingerprint_sha256",
    }
    if set(payload) != required:
        raise RuntimeError("Transformer environment manifest keys differ from schema")
    if payload["schema_version"] != ENVIRONMENT_SCHEMA:
        raise RuntimeError("unsupported Transformer environment-manifest schema")
    if payload["status"] != "CAPTURED_FOR_LOCKED_RUNTIME":
        raise RuntimeError("Transformer environment manifest is not a locked capture")
    if payload["config_sha256"] != expected_config_sha256:
        raise RuntimeError("Transformer environment manifest authenticates another config")
    if payload["source_provenance"] != dict(expected_source_provenance):
        raise RuntimeError("Transformer environment manifest authenticates another source")
    if payload["environment_fingerprint_sha256"] != _canonical_sha256(payload["observed"]):
        raise RuntimeError("Transformer environment-manifest fingerprint is invalid")
    if payload["observed"] != dict(observed):
        raise RuntimeError("current runtime differs from the captured Transformer environment")
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.environment_binding.v1",
        "manifest_sha256": _sha256_file(path),
        "environment_fingerprint_sha256": payload[
            "environment_fingerprint_sha256"
        ],
        "observed": dict(observed),
    }


def _asset_identity(assets: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        role: {
            "repository_id": str(value.repository_id),
            "resolved_revision": str(value.resolved_revision),
            "sha256": str(value.sha256),
        }
        for role, value in assets.items()
    }


def validate_cuda_preflight_report(
    report_path: str | Path,
    *,
    config: Mapping[str, Any],
    expected_config_sha256: str,
    expected_source_provenance: Mapping[str, Any],
    expected_environment_provenance: Mapping[str, Any],
    expected_assets: Mapping[str, Mapping[str, str]],
    expected_input_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a passing full-asset CUDA preflight for a full run."""

    path = Path(report_path).expanduser().resolve(strict=True)
    try:
        payload = strict_source_json(path)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(f"cannot read Transformer CUDA preflight report: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Transformer CUDA preflight report must be a JSON object")
    required_keys = {
        "schema_version",
        "status",
        "paper_evidence_eligible",
        "purpose",
        "config_sha256",
        "source_provenance",
        "environment_provenance",
        "assets",
        "validated_input_bundle",
        "workload",
        "stream",
        "resources",
        "resume_supported",
        "unit",
        "artifact_inventory",
    }
    if set(payload) != required_keys:
        raise RuntimeError("Transformer CUDA preflight report keys differ from schema")
    if payload.get("schema_version") != PREFLIGHT_SCHEMA or payload.get("status") != "PASS":
        raise RuntimeError("Transformer CUDA preflight report is not a PASS")
    if payload.get("paper_evidence_eligible") is not False:
        raise RuntimeError("CUDA preflight must remain paper-ineligible")
    if payload.get("resume_supported") is not False:
        raise RuntimeError("CUDA preflight may not claim resumable execution")
    validate_artifact_inventory(payload)
    if payload.get("config_sha256") != expected_config_sha256:
        raise RuntimeError("CUDA preflight authenticates another configuration")
    if payload.get("source_provenance") != dict(expected_source_provenance):
        raise RuntimeError("CUDA preflight authenticates another source freeze")
    if payload.get("environment_provenance") != dict(expected_environment_provenance):
        raise RuntimeError("CUDA preflight authenticates another runtime environment")
    if payload.get("assets") != {
        role: dict(value) for role, value in expected_assets.items()
    }:
        raise RuntimeError("CUDA preflight authenticates different external assets")
    embedded_input_bundle = payload.get("validated_input_bundle")
    if not isinstance(embedded_input_bundle, dict):
        raise RuntimeError("CUDA preflight omits its complete validated input bundle")
    validate_embedded_input_bundle(embedded_input_bundle)
    if embedded_input_bundle != dict(expected_input_bundle):
        raise RuntimeError("CUDA preflight authenticates different input payloads")

    workload = payload.get("workload")
    if not isinstance(workload, dict) or workload.get("full_model_and_assets") is not True:
        raise RuntimeError("CUDA preflight did not exercise the full model and assets")
    lock = config["cuda_preflight"]
    expected_workload = {
        "packet_size": int(config["stream"]["packet_size"]),
        "context_length": int(config["stream"]["context_length"]),
        "n_permutations": int(config["audit"]["n_permutations"]),
    }
    for name, expected in expected_workload.items():
        if int(workload.get(name, -1)) != expected:
            raise RuntimeError(f"CUDA preflight workload differs for {name}")
    if int(workload.get("warmup_packets", 0)) < int(lock["minimum_warmup_packets"]):
        raise RuntimeError("CUDA preflight warmup was shorter than the lock")
    if int(workload.get("online_packets", 0)) < int(lock["minimum_online_packets"]):
        raise RuntimeError("CUDA preflight online audit was shorter than the lock")

    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise RuntimeError("CUDA preflight has no resource measurements")
    peak_reserved = int(resources.get("peak_reserved_bytes", -1))
    locked_ceiling = int(lock["maximum_peak_reserved_bytes"])
    minimum_total_memory = int(lock["minimum_total_device_memory_bytes"])
    total_memory = int(resources.get("total_device_memory_bytes", -1))
    environment_gpu = expected_environment_provenance.get("observed", {}).get(
        "accelerator", {}
    ).get("gpu")
    environment_total_memory = (
        int(environment_gpu.get("total_memory_bytes", -1))
        if isinstance(environment_gpu, Mapping)
        else -1
    )
    if (
        total_memory < minimum_total_memory
        or environment_total_memory != total_memory
    ):
        raise RuntimeError("CUDA preflight device memory differs from the locked runtime")
    if int(resources.get("maximum_peak_reserved_bytes", -1)) != locked_ceiling:
        raise RuntimeError("CUDA preflight reports another memory ceiling")
    if peak_reserved < 0 or peak_reserved > locked_ceiling:
        raise RuntimeError("CUDA preflight exceeded the locked memory ceiling")
    if float(resources.get("wall_time_seconds", 0.0)) <= 0.0:
        raise RuntimeError("CUDA preflight has no positive wall-time measurement")
    unit = payload.get("unit")
    audit = unit.get("audit") if isinstance(unit, dict) else None
    if (
        not isinstance(audit, dict)
        or audit.get("all_pass") is not True
        or audit.get("zero_norm_actuator_branches") != 0
        or audit.get("zero_norm_actuator_branch_fail_closed") is not True
    ):
        raise RuntimeError("CUDA preflight scientific audit did not pass fail-closed")
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.preflight_binding.v1",
        "report_sha256": _sha256_file(path),
        "report_payload_sha256": _canonical_sha256(payload),
        "validated_input_bundle_sha256": embedded_input_bundle["bundle_sha256"],
        "peak_reserved_bytes": peak_reserved,
        "maximum_peak_reserved_bytes": locked_ceiling,
        "reserved_memory_headroom_bytes": locked_ceiling - peak_reserved,
    }


def _optimizer(model: torch.nn.Module, specification: Mapping[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(specification["learning_rate"]),
        betas=tuple(float(value) for value in specification["betas"]),
        eps=float(specification["epsilon"]),
        weight_decay=float(specification["weight_decay"]),
        foreach=False,
        fused=False,
    )


def _run_unit(
    *,
    model: torch.nn.Module,
    stream: Any,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    stream_config = config["stream"]
    optimization = config["optimization"]
    audit = config["audit"]
    optimizer = _optimizer(model, optimization)
    if optimizer.defaults.get("foreach") is not False:
        raise RuntimeError("AdamW foreach implementation differs from the lock")
    if optimizer.defaults.get("fused") is not False:
        raise RuntimeError("fused AdamW implementation differs from the lock")
    warmup = warmup_model(
        model,
        optimizer,
        packetize(stream.warmup, int(stream_config["packet_size"])),
        device=device,
        epochs=int(optimization["warmup_epochs"]),
        gradient_clip=float(optimization["gradient_clip"]),
        radius_window=int(optimization["radius_window"]),
        progress_callback=progress_callback,
    )
    result = run_next_token_audit(
        model=model,
        optimizer=optimizer,
        packets=packetize(stream.online, int(stream_config["packet_size"])),
        n_permutations=int(audit["n_permutations"]),
        permutation_salt=int(audit["permutation_salt"]),
        seed=int(seed),
        device=device,
        radius_history=warmup["recent_update_radii"],
        radius_window=int(optimization["radius_window"]),
        fallback_radius=float(optimization["fallback_radius"]),
        gradient_clip=float(optimization["gradient_clip"]),
        bootstrap_draws=int(audit["bootstrap_draws"]),
        bootstrap_block_length=int(audit["bootstrap_block_length"]),
        fail_on_zero_norm_actuator_branch=bool(
            audit.get("fail_on_zero_norm_actuator_branch", False)
        ),
        progress_callback=progress_callback,
    )
    result["warmup"] = warmup
    result["optimizer_implementation"] = {
        "name": "AdamW",
        "foreach": False,
        "fused": False,
        "live_update": "true-target AdamW",
        "counterfactual_actuator": ACTUATOR_NAME,
        "counterfactual_is_adamw_replay": False,
    }
    return result


def run_full(
    *,
    config_path: str | Path,
    asset_manifest: str | Path,
    source_manifest: str | Path,
    environment_manifest: str | Path,
    preflight_report: str | Path,
    device: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    source_provenance = validate_transformer_source_manifest(
        source_manifest, config_path=config_path, required=True
    )
    _validate_asset_source_binding(asset_manifest, source_provenance)
    _validate_runtime(config, device)
    assets = validate_asset_manifest(
        asset_manifest,
        config["assets"],
        expected_config_sha256=_sha256_file(config_path),
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # The tokenizer and single prespecified trajectory both start from the
    # authenticated checkpoint.  Permutation draws are nested controls, not
    # independent model/stream replications.
    trajectory_seed = int(config["audit"]["trajectory_seed"])
    seed_everything(trajectory_seed)
    observed_environment = capture_runtime_environment(
        device=device,
        optimizer_specification=config["optimization"],
    )
    environment_binding = validate_environment_manifest(
        environment_manifest,
        expected_config_sha256=_sha256_file(config_path),
        expected_source_provenance=source_provenance,
        observed=observed_environment,
    )
    asset_identity = _asset_identity(assets)
    preflight_input_bundle = build_validated_input_bundle(
        {
            "configuration": config_path,
            "source_manifest": source_manifest,
            "environment_manifest": environment_manifest,
            "asset_manifest": asset_manifest,
        }
    )
    validate_embedded_input_bundle(preflight_input_bundle)
    preflight_binding = validate_cuda_preflight_report(
        preflight_report,
        config=config,
        expected_config_sha256=_sha256_file(config_path),
        expected_source_provenance=source_provenance,
        expected_environment_provenance=environment_binding,
        expected_assets=asset_identity,
        expected_input_bundle=preflight_input_bundle,
    )
    validated_input_bundle = build_validated_input_bundle(
        {
            "configuration": config_path,
            "source_manifest": source_manifest,
            "environment_manifest": environment_manifest,
            "asset_manifest": asset_manifest,
            "cuda_preflight_report": preflight_report,
        }
    )
    validate_embedded_input_bundle(validated_input_bundle)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "validated_inputs",
                "validated_input_identity": {
                    "input_bundle_sha256": validated_input_bundle["bundle_sha256"],
                    "config_sha256": _sha256_file(config_path),
                    "source_provenance": source_provenance,
                    "environment_provenance": environment_binding,
                    "assets": asset_identity,
                    "cuda_preflight_provenance": preflight_binding,
                },
            }
        )
    if progress_callback is not None:
        progress_callback({"phase": "load_model", "completed_packets": 0})
    model, tokenizer = load_local_pythia(
        assets["model"].path, dtype=str(config["model"]["dtype"])
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    lower, upper = [int(value) for value in config["model"]["expected_total_parameter_range"]]
    if not lower <= parameter_count <= upper:
        raise RuntimeError(
            f"model parameter count {parameter_count} is outside locked range [{lower}, {upper}]"
        )
    stream_config = config["stream"]
    if progress_callback is not None:
        progress_callback({"phase": "build_stream", "completed_packets": 0})
    stream = build_chronological_stream(
        iter_json_records(assets["dataset"].path),
        tokenizer,
        context_length=int(stream_config["context_length"]),
        warmup_examples=int(stream_config["warmup_examples"]),
        online_examples=int(stream_config["online_examples"]),
        earliest_utc=str(stream_config["earliest_utc"]),
        offset_salt=str(stream_config["offset_salt"]),
    )
    unit = _run_unit(
        model=model,
        stream=stream,
        config=config,
        seed=trajectory_seed,
        device=device,
        progress_callback=progress_callback,
    )
    unit["source_provenance"] = dict(source_provenance)
    result = {
        "schema_version": "fmmt.gradient_reuse_transformer.campaign.v3",
        "status": "PASS",
        "paper_evidence_eligible": True,
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "permutation_replicates_are_nested_within_packet": True,
        "config_sha256": _sha256_file(config_path),
        "source_provenance": source_provenance,
        "environment_provenance": environment_binding,
        "cuda_preflight_provenance": preflight_binding,
        "assets": asset_identity,
        "validated_input_bundle": validated_input_bundle,
        "stream": stream.manifest,
        "execution": {
            "resume_supported": False,
            "interruption_policy": "restart from the immutable inputs after interruption",
            "heartbeat_is_progress_evidence_only": True,
        },
        "counterfactual_intervention": {
            "name": ACTUATOR_NAME,
            "direction": "stored raw minibatch gradient",
            "radius_calibration": "median recent true-target AdamW update norm",
            "is_adamw_counterfactual": False,
        },
        "unit": unit,
    }
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": validated_input_bundle,
            "stream": stream.manifest,
            "unit": unit,
            "cuda_preflight_provenance": preflight_binding,
        }
    )
    return result


def run_cuda_preflight(
    *,
    config_path: str | Path,
    asset_manifest: str | Path,
    source_manifest: str | Path,
    environment_manifest: str | Path,
    device: str,
    warmup_packets: int = 2,
    online_packets: int = 3,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Exercise the exact real model/assets/runtime on a tiny ineligible stream."""
    if not str(device).startswith("cuda"):
        raise RuntimeError("the full-asset preflight requires a CUDA device")
    if int(warmup_packets) < 1 or int(online_packets) < 3:
        raise RuntimeError("preflight needs >=1 warmup and >=3 online packets")
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    preflight_lock = config["cuda_preflight"]
    if int(warmup_packets) < int(preflight_lock["minimum_warmup_packets"]):
        raise RuntimeError("preflight warmup is shorter than the locked minimum")
    if int(online_packets) < int(preflight_lock["minimum_online_packets"]):
        raise RuntimeError("preflight audit is shorter than the locked minimum")
    source_provenance = validate_transformer_source_manifest(
        source_manifest, config_path=config_path, required=True
    )
    _validate_asset_source_binding(asset_manifest, source_provenance)
    _validate_runtime(config, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    assets = validate_asset_manifest(
        asset_manifest,
        config["assets"],
        expected_config_sha256=_sha256_file(config_path),
    )
    trajectory_seed = int(config["audit"]["trajectory_seed"])
    seed_everything(trajectory_seed)
    observed_environment = capture_runtime_environment(
        device=device,
        optimizer_specification=config["optimization"],
    )
    environment_binding = validate_environment_manifest(
        environment_manifest,
        expected_config_sha256=_sha256_file(config_path),
        expected_source_provenance=source_provenance,
        observed=observed_environment,
    )
    asset_identity = _asset_identity(assets)
    validated_input_bundle = build_validated_input_bundle(
        {
            "configuration": config_path,
            "source_manifest": source_manifest,
            "environment_manifest": environment_manifest,
            "asset_manifest": asset_manifest,
        }
    )
    validate_embedded_input_bundle(validated_input_bundle)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "validated_inputs",
                "validated_input_identity": {
                    "input_bundle_sha256": validated_input_bundle["bundle_sha256"],
                    "config_sha256": _sha256_file(config_path),
                    "source_provenance": source_provenance,
                    "environment_provenance": environment_binding,
                    "assets": asset_identity,
                },
            }
        )
    total_memory = int(
        observed_environment["accelerator"]["gpu"]["total_memory_bytes"]
    )
    if total_memory < int(preflight_lock["minimum_total_device_memory_bytes"]):
        raise RuntimeError("CUDA device is smaller than the locked memory requirement")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    if progress_callback is not None:
        progress_callback({"phase": "preflight_load_model", "completed_packets": 0})
    model, tokenizer = load_local_pythia(
        assets["model"].path, dtype=str(config["model"]["dtype"])
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    lower, upper = [int(value) for value in config["model"]["expected_total_parameter_range"]]
    if not lower <= parameter_count <= upper:
        raise RuntimeError("preflight model parameter count differs from the lock")
    packet_size = int(config["stream"]["packet_size"])
    stream = build_chronological_stream(
        iter_json_records(assets["dataset"].path),
        tokenizer,
        context_length=int(config["stream"]["context_length"]),
        warmup_examples=int(warmup_packets) * packet_size,
        online_examples=int(online_packets) * packet_size,
        earliest_utc=str(config["stream"]["earliest_utc"]),
        offset_salt=str(config["stream"]["offset_salt"]),
    )
    preflight_config = copy.deepcopy(config)
    preflight_config["stream"]["warmup_examples"] = int(warmup_packets) * packet_size
    preflight_config["stream"]["online_examples"] = int(online_packets) * packet_size
    preflight_config["audit"]["bootstrap_draws"] = max(
        64, min(256, int(config["audit"]["bootstrap_draws"]))
    )
    unit = _run_unit(
        model=model,
        stream=stream,
        config=preflight_config,
        seed=trajectory_seed,
        device=device,
        progress_callback=progress_callback,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    resources = {
        "wall_time_seconds": float(elapsed),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_device_memory_bytes": total_memory,
        "maximum_peak_reserved_bytes": int(
            preflight_lock["maximum_peak_reserved_bytes"]
        ),
    }
    if resources["peak_reserved_bytes"] > resources["maximum_peak_reserved_bytes"]:
        raise RuntimeError("CUDA preflight exceeded the locked peak-reserved-memory ceiling")
    result = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "paper_evidence_eligible": False,
        "purpose": "resource and integration validation only",
        "config_sha256": _sha256_file(config_path),
        "source_provenance": source_provenance,
        "environment_provenance": environment_binding,
        "assets": asset_identity,
        "validated_input_bundle": validated_input_bundle,
        "workload": {
            "warmup_packets": int(warmup_packets),
            "online_packets": int(online_packets),
            "packet_size": packet_size,
            "context_length": int(config["stream"]["context_length"]),
            "n_permutations": int(config["audit"]["n_permutations"]),
            "full_model_and_assets": True,
        },
        "stream": stream.manifest,
        "resources": resources,
        "resume_supported": False,
        "unit": unit,
    }
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": validated_input_bundle,
            "stream": stream.manifest,
            "resources": resources,
            "unit": unit,
        }
    )
    return result


class _IntegerTokenizer:
    vocab_size = 23

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": [int(value) % self.vocab_size for value in text.split()]}


def capture_environment_for_config(
    *,
    config_path: str | Path,
    source_manifest: str | Path,
    device: str,
) -> dict[str, Any]:
    """Capture the exact full-run environment before assets or science launch."""
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    source_provenance = validate_transformer_source_manifest(
        source_manifest, config_path=config_path, required=True
    )
    _validate_runtime(config, device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA environment capture requested but CUDA is unavailable")
    seed_everything(int(config["audit"]["trajectory_seed"]))
    observed = capture_runtime_environment(
        device=device,
        optimizer_specification=config["optimization"],
    )
    return build_environment_manifest(
        config_sha256=_sha256_file(config_path),
        source_provenance=source_provenance,
        observed=observed,
    )


class _TinyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int = 23, width: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            is_encoder_decoder=False,
            use_cache=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.projection = torch.nn.Linear(width, vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> Any:
        if (
            use_cache
            or not return_dict
            or logits_to_keep != 1
            or not torch.equal(attention_mask, torch.ones_like(input_ids))
        ):
            raise RuntimeError("invalid fixed-context wrapper call")
        embedded = self.embedding(input_ids)
        denominator = torch.arange(
            1, input_ids.shape[1] + 1, device=input_ids.device, dtype=embedded.dtype
        ).view(1, -1, 1)
        hidden = embedded.cumsum(dim=1) / denominator
        return SimpleNamespace(logits=self.projection(hidden[:, -1:, :]))


def _tiny_records(count: int = 40) -> list[dict[str, Any]]:
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    records = []
    for index in range(count):
        tokens = [str((index * 3 + step * 5 + step**2) % 23) for step in range(28)]
        records.append(
            {
                "id": index,
                "source": "offline-smoke",
                "created": (start + timedelta(days=index)).isoformat(),
                # Trailing whitespace makes every synthetic article byte-unique
                # without changing its token sequence under the smoke tokenizer.
                "text": " ".join(tokens) + (" " * (index + 1)),
                "metadata": {
                    "license": "CC BY 4.0",
                    "url": f"https://invalid.example/{index}",
                },
            }
        )
    return records


def run_tiny_smoke(
    seed: int = 17,
    *,
    source_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Download-free CPU integration smoke; never paper-eligible."""
    source_provenance = validate_transformer_source_manifest(
        source_manifest, required=False
    )
    seed_everything(seed)
    tokenizer = _IntegerTokenizer()
    stream = build_chronological_stream(
        _tiny_records(),
        tokenizer,
        context_length=8,
        warmup_examples=8,
        online_examples=24,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="cpu-tiny-fixed-offset-v1",
    )
    model = PythiaNextTokenModel(_TinyBackbone())
    config = {
        "stream": {"packet_size": 4},
        "optimization": {
            "learning_rate": 0.002,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "warmup_epochs": 1,
            "gradient_clip": 1.0,
            "radius_window": 4,
            "fallback_radius": 0.001,
        },
        "audit": {
            "n_permutations": 2,
            "permutation_salt": 4819,
            "bootstrap_draws": 64,
            "bootstrap_block_length": 2,
        },
    }
    result = _run_unit(
        model=model,
        stream=stream,
        config=config,
        seed=seed,
        device="cpu",
    )
    result["source_provenance"] = dict(source_provenance)
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.smoke.v2",
        "status": "PASS",
        "paper_evidence_eligible": False,
        "network_access": False,
        "source_provenance": source_provenance,
        "stream": stream.manifest,
        "unit": result,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_atomic_new(path: Path, value: Any) -> None:
    """Atomically create a JSON artifact and fail if its name is already owned."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing to overwrite existing artifact: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_disjoint_paths(paths: Mapping[str, str | Path | None]) -> None:
    """Reject role aliasing that could overwrite or mis-authenticate evidence."""

    observed: dict[Path, str] = {}
    for role, raw_path in paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser().resolve()
        previous = observed.get(path)
        if previous is not None:
            raise RuntimeError(
                f"path roles must be disjoint: {previous} and {role} both resolve to {path}"
            )
        observed[path] = role


def _asset_content_roots(asset_manifest: str | Path) -> tuple[Path, ...]:
    """Resolve every content-hashed asset root before creating run artifacts."""

    manifest = Path(asset_manifest).expanduser().resolve(strict=True)
    try:
        payload = strict_source_json(manifest)
    except (OSError, SourceProvenanceError) as error:
        raise RuntimeError(f"cannot read Transformer asset manifest: {manifest}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "fmmt.transformer_assets.v1":
        raise RuntimeError("unsupported Transformer asset-manifest schema")
    records = payload.get("assets")
    if not isinstance(records, dict) or not records:
        raise RuntimeError("Transformer asset manifest has no asset records")
    roots: list[Path] = []
    for name, record in sorted(records.items()):
        if not isinstance(record, dict) or record.get("kind") not in {
            "file",
            "directory",
        }:
            raise RuntimeError(f"invalid asset-root record for {name!r}")
        raw_path = str(record.get("path", ""))
        if not raw_path:
            raise RuntimeError(f"asset-root record has no path for {name!r}")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        roots.append(candidate.resolve())
    return tuple(roots)


def _reject_artifacts_inside_asset_roots(
    artifacts: Mapping[str, str | Path],
    *,
    asset_manifest: str | Path,
) -> None:
    roots = _asset_content_roots(asset_manifest)
    for role, raw_path in artifacts.items():
        path = Path(raw_path).expanduser().resolve()
        for root in roots:
            if path == root or root in path.parents:
                raise RuntimeError(
                    f"{role} must not equal or descend from content-hashed asset root {root}"
                )


def _build_invocation_identity(
    *,
    mode: str,
    device: str,
    output: Path,
    heartbeat: Path,
    input_files: Mapping[str, str | Path],
    preflight_warmup_packets: int | None = None,
    preflight_online_packets: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_bundle = build_validated_input_bundle(input_files)
    validate_embedded_input_bundle(input_bundle)
    identity: dict[str, Any] = {
        "schema_version": "fmmt.gradient_reuse_transformer.invocation.v1",
        "mode": mode,
        "device": str(device),
        "destination_sha256": {
            "output": _canonical_sha256(str(output)),
            "heartbeat": _canonical_sha256(str(heartbeat)),
        },
        "input_bundle_sha256": input_bundle["bundle_sha256"],
        "input_inventory": copy.deepcopy(input_bundle["inventory"]),
    }
    if mode == "cuda_preflight":
        identity["workload"] = {
            "warmup_packets": int(preflight_warmup_packets),
            "online_packets": int(preflight_online_packets),
        }
    identity["invocation_sha256"] = _canonical_sha256(identity)
    return identity, input_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset-manifest")
    parser.add_argument("--source-manifest")
    parser.add_argument("--environment-manifest")
    parser.add_argument("--preflight-report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--full",
        action="store_true",
        help="run the paper-eligible full campaign after an authenticated preflight",
    )
    mode.add_argument("--smoke-tiny", action="store_true")
    mode.add_argument("--cuda-preflight", action="store_true")
    mode.add_argument("--capture-environment", action="store_true")
    parser.add_argument("--preflight-warmup-packets", type=int, default=2)
    parser.add_argument("--preflight-online-packets", type=int, default=3)
    parser.add_argument("--heartbeat")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = Path(arguments.out).resolve()
    _require_disjoint_paths(
        {
            "output": output,
            "config": arguments.config,
            "asset_manifest": arguments.asset_manifest,
            "source_manifest": arguments.source_manifest,
            "environment_manifest": arguments.environment_manifest,
            "preflight_report": arguments.preflight_report,
        }
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    output_written = False
    if arguments.capture_environment:
        if not arguments.source_manifest:
            raise RuntimeError("--source-manifest is required for environment capture")
        result = capture_environment_for_config(
            config_path=arguments.config,
            source_manifest=arguments.source_manifest,
            device=arguments.device,
        )
    elif arguments.smoke_tiny:
        result = run_tiny_smoke(
            seed=arguments.seed if arguments.seed is not None else 17,
            source_manifest=arguments.source_manifest,
        )
    else:
        if arguments.seed is not None:
            raise RuntimeError("--seed is smoke-only; full runs use the locked trajectory_seed")
        if not arguments.asset_manifest:
            raise RuntimeError("--asset-manifest is required for a full offline run")
        if not arguments.source_manifest:
            raise RuntimeError(
                "--source-manifest is required for a full paper-eligible run"
            )
        if not arguments.environment_manifest:
            raise RuntimeError(
                "--environment-manifest is required for a real-asset run"
            )
        if arguments.cuda_preflight and arguments.preflight_report:
            raise RuntimeError("--preflight-report is for the full campaign, not preflight")
        if arguments.full and not arguments.preflight_report:
            raise RuntimeError(
                "--preflight-report is required for a paper-eligible full run"
            )
        heartbeat = (
            Path(arguments.heartbeat).resolve()
            if arguments.heartbeat
            else Path(str(output) + ".heartbeat.json")
        )
        _require_disjoint_paths(
            {
                "output": output,
                "heartbeat": heartbeat,
                "config": arguments.config,
                "asset_manifest": arguments.asset_manifest,
                "source_manifest": arguments.source_manifest,
                "environment_manifest": arguments.environment_manifest,
                "preflight_report": arguments.preflight_report,
            }
        )
        _reject_artifacts_inside_asset_roots(
            {"output": output, "heartbeat": heartbeat},
            asset_manifest=arguments.asset_manifest,
        )
        input_files: dict[str, str | Path] = {
            "configuration": arguments.config,
            "source_manifest": arguments.source_manifest,
            "environment_manifest": arguments.environment_manifest,
            "asset_manifest": arguments.asset_manifest,
        }
        if arguments.full:
            input_files["cuda_preflight_report"] = arguments.preflight_report
        invocation, invocation_input_bundle = _build_invocation_identity(
            mode="cuda_preflight" if arguments.cuda_preflight else "full_campaign",
            device=arguments.device,
            output=output,
            heartbeat=heartbeat,
            input_files=input_files,
            preflight_warmup_packets=arguments.preflight_warmup_packets,
            preflight_online_packets=arguments.preflight_online_packets,
        )
        now = datetime.now(timezone.utc).isoformat()
        heartbeat_state: dict[str, Any] = {
            "schema_version": "fmmt.gradient_reuse_transformer.heartbeat.v1",
            "status": "RUNNING",
            "mode": "cuda_preflight" if arguments.cuda_preflight else "full_campaign",
            "resume_supported": False,
            "warning": (
                "This heartbeat records progress only. Interrupted execution must "
                "restart from immutable inputs."
            ),
            "phase": "initialization",
            "completed_packets": 0,
            "created_utc": now,
            "updated_utc": now,
            "invocation": invocation,
        }
        try:
            _write_atomic_new(heartbeat, heartbeat_state)
        except RuntimeError as error:
            raise RuntimeError(
                f"refusing to overwrite existing heartbeat: {heartbeat}"
            ) from error

        def progress(value: Mapping[str, Any]) -> None:
            heartbeat_state.update(dict(value))
            heartbeat_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            _write_atomic(heartbeat, heartbeat_state)

        try:
            if arguments.cuda_preflight:
                result = run_cuda_preflight(
                    config_path=arguments.config,
                    asset_manifest=arguments.asset_manifest,
                    source_manifest=arguments.source_manifest,
                    environment_manifest=arguments.environment_manifest,
                    device=arguments.device,
                    warmup_packets=arguments.preflight_warmup_packets,
                    online_packets=arguments.preflight_online_packets,
                    progress_callback=progress,
                )
            else:
                result = run_full(
                    config_path=arguments.config,
                    asset_manifest=arguments.asset_manifest,
                    source_manifest=arguments.source_manifest,
                    environment_manifest=arguments.environment_manifest,
                    preflight_report=arguments.preflight_report,
                    device=arguments.device,
                    progress_callback=progress,
                )
            final_input_bundle = build_validated_input_bundle(input_files)
            validate_embedded_input_bundle(final_input_bundle)
            if final_input_bundle != invocation_input_bundle:
                raise RuntimeError("a validated input changed during Transformer execution")
            if result.get("validated_input_bundle") != invocation_input_bundle:
                raise RuntimeError(
                    "Transformer result does not embed the exact invocation input bundle"
                )
            _write_atomic_new(output, result)
            output_written = True
        except BaseException as error:
            heartbeat_state.update(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_atomic(heartbeat, heartbeat_state)
            raise
        heartbeat_state.update(
            {
                "status": "COMPLETE",
                "phase": "complete",
                "result_payload_sha256": _canonical_sha256(result),
                "output_file_sha256": _sha256_file(output),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_atomic(heartbeat, heartbeat_state)
    if not output_written:
        _write_atomic_new(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
