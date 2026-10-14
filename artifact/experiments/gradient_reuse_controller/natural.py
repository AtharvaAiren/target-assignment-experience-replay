"""Prospective gradient-reuse controllers on the frozen Beijing stream.

This module is intentionally separate from :mod:`.core`, the locked synthetic
controller campaign.  It reuses the audited chronological loader, model, and
gradient decomposition, but changes no frozen source or result.

For every policy and packet ``t`` the chronology is:

1. predict with the state entering ``t``;
2. reveal the targets and irrevocably score that prediction;
3. compute the current ordinary and target-permutation gradient channels;
4. decide whether to reuse the stored packet-``t-1`` channel(s);
5. take exactly one common AdamW update using the current true gradient; and
6. optionally take an equal-radius stored-gradient displacement.

Consequently, targets from packet ``t`` can gate an action only after packet
``t`` has been scored, and that action can first affect packet ``t+1``.  The
controller stores parameter-sized gradient channels only; it never stores or
replays raw examples.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chartfilm.fmmt.core import FunctionalUnitRegressor
from experiments.fmmt.common import (
    _chronology_metadata,
    packetize_stream,
    seed_everything,
)
from experiments.fmmt.study_a import (
    _delta,
    _dict_norm,
    _difference_norm,
    _gradient_dict,
    _named_parameters,
    _predict,
    _shift,
)
from experiments.fmmt_reviewer.diagnostics import (
    output_jacobian_matrix,
    packet_strength,
)
from experiments.fmmt_workshop.datasets import (
    dataset_manifest_json,
    load_workshop_stream,
)
from experiments.source_provenance import (
    SourceProvenanceError,
    get_source_spec,
    strict_json,
    validate_source_manifest,
)
from experiments.validation import stage4_regression as stage4

from .provenance import (
    ValidatedEvidence,
    _validated_evidence,
    build_runtime_manifest,
    evidence_payload,
    observed_runtime,
    require_disjoint_paths,
    sha256_file,
    source_identity as shared_source_identity,
    validate_source_binding,
    validate_runtime_manifest,
    write_fresh_json,
)


REPO = Path(__file__).resolve().parents[2]
POLICIES = ("never", "always", "cosine", "pairing_aware")
LOCK_STATUS = "LOCKED_BEFORE_RUN"
SCHEMA_VERSION = "gradient_reuse.controller_natural.v2"
RESULT_SCHEMA_VERSION = "gradient_reuse.controller_natural_results.v2"
DATA_MANIFEST_SCHEMA_VERSION = "gradient_reuse.beijing_data_manifest.v1"
DATA_BINDING_SCHEMA_VERSION = "gradient_reuse.beijing_data_binding.v1"
FULL_SEEDS = tuple(range(100, 108))
SOURCE_EXPERIMENT = "controller_natural"
DEFAULT_SHARED_SOURCE_MANIFEST = (
    REPO / "provenance/source/controller_natural.json"
)

_GATE_TIMING = "score_t_then_gate_action_for_t_plus_1"
_RADIUS_RULE = "median_recent_never_reuse_shadow_adamw_update_norms"
_PAIRING_RULE = "aligned_m_plus_binary_reliability_gated_aligned_c"
_VARIANCE_RULE = "exact_finite_permutation_variance_of_full_mse_gradient"


class NaturalControllerIntegrityError(RuntimeError):
    """Raised when the prespecified natural-stream protocol is violated."""


class NaturalControllerConfigError(ValueError):
    """Raised when a natural-stream campaign is not exactly prespecified."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing or extra:
        raise NaturalControllerConfigError(
            f"{context} keys differ from lock; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NaturalControllerConfigError(f"{context} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NaturalControllerConfigError(
            f"{context} must be a nonnegative integer"
        )
    return int(value)


def _finite_float(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NaturalControllerConfigError(f"{context} must be numeric")
    output = float(value)
    if not math.isfinite(output) or (positive and output <= 0.0):
        raise NaturalControllerConfigError(
            f"{context} must be {'positive and ' if positive else ''}finite"
        )
    return output


def _safe_source_path(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise NaturalControllerConfigError(
            f"source identity path is not safe and repo-relative: {relative!r}"
        )
    resolved = (REPO / candidate).resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise NaturalControllerConfigError(
            f"source identity escapes repository: {relative!r}"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise NaturalControllerConfigError(
            f"source identity file is missing or not regular: {relative!r}"
        )
    return resolved


def _validate_source_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise NaturalControllerConfigError("source_identity must be an object")
    _require_exact_keys(
        identity, {"algorithm", "files", "bundle_sha256"}, "source_identity"
    )
    if identity["algorithm"] != "sha256":
        raise NaturalControllerConfigError("source identity algorithm must be sha256")
    files = identity["files"]
    if not isinstance(files, Mapping) or not files:
        raise NaturalControllerConfigError("source_identity.files must be nonempty")
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise NaturalControllerConfigError("source hashes must be string mappings")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise NaturalControllerConfigError(
                f"invalid SHA-256 for source file {relative!r}"
            )
        observed[relative] = _sha256_file(_safe_source_path(relative))
        if observed[relative] != expected:
            raise NaturalControllerConfigError(
                f"source identity mismatch for {relative}: "
                f"expected={expected}, observed={observed[relative]}"
            )
    bundle = _sha256_bytes(_canonical_json(observed))
    if bundle != identity["bundle_sha256"]:
        raise NaturalControllerConfigError(
            "source bundle identity mismatch: "
            f"expected={identity['bundle_sha256']}, observed={bundle}"
        )
    return {"algorithm": "sha256", "files": observed, "bundle_sha256": bundle}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the full/smoke boundary and the frozen source bundle."""

    if not isinstance(config, Mapping):
        raise NaturalControllerConfigError("configuration must be a JSON object")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "status",
            "policies",
            "dataset",
            "model",
            "optimizer",
            "permutations",
            "controller",
            "full",
            "smoke",
            "source_identity",
        },
        "configuration",
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise NaturalControllerConfigError("unsupported schema_version")
    if config["status"] != LOCK_STATUS:
        raise NaturalControllerConfigError(
            f"status must be {LOCK_STATUS!r}; refusing an unlocked campaign"
        )
    if list(config["policies"]) != list(POLICIES):
        raise NaturalControllerConfigError(
            f"policies must be exactly {list(POLICIES)}"
        )

    dataset = config["dataset"]
    if not isinstance(dataset, Mapping):
        raise NaturalControllerConfigError("dataset must be an object")
    _require_exact_keys(
        dataset,
        {"full_stream", "smoke_stream", "packet_size", "chronology"},
        "dataset",
    )
    if dataset["full_stream"] != "BEIJING":
        raise NaturalControllerConfigError("full stream must be frozen BEIJING")
    if dataset["smoke_stream"] != "deterministic_local_synthetic":
        raise NaturalControllerConfigError("smoke must use the local synthetic loader")
    if dataset["packet_size"] != 32:
        raise NaturalControllerConfigError("natural packet size is locked to 32")
    if dataset["chronology"] != "frozen_event_atomic":
        raise NaturalControllerConfigError("chronology must be frozen_event_atomic")

    model = config["model"]
    if not isinstance(model, Mapping):
        raise NaturalControllerConfigError("model must be an object")
    _require_exact_keys(model, {"name", "functional_units", "device"}, "model")
    if model != {
        "name": "functional_unit_regressor",
        "functional_units": 16,
        "device": "cpu",
    }:
        raise NaturalControllerConfigError(
            "model must match the frozen 16-unit CPU functional regressor"
        )

    optimizer = config["optimizer"]
    if not isinstance(optimizer, Mapping):
        raise NaturalControllerConfigError("optimizer must be an object")
    _require_exact_keys(
        optimizer, {"name", "learning_rate", "betas", "weight_decay", "gradient_clip"}, "optimizer"
    )
    if optimizer["name"] != "adamw":
        raise NaturalControllerConfigError("optimizer must be AdamW")
    if _finite_float(optimizer["learning_rate"], "optimizer.learning_rate", positive=True) != 0.001:
        raise NaturalControllerConfigError("AdamW learning rate is locked to 0.001")
    if list(optimizer["betas"]) != [0.9, 0.999]:
        raise NaturalControllerConfigError("AdamW betas are locked")
    if _finite_float(optimizer["weight_decay"], "optimizer.weight_decay") != 0.0001:
        raise NaturalControllerConfigError("AdamW weight decay is locked")
    if _finite_float(optimizer["gradient_clip"], "optimizer.gradient_clip", positive=True) != 1.0:
        raise NaturalControllerConfigError("gradient clip is locked to 1.0")

    permutations = config["permutations"]
    if not isinstance(permutations, Mapping):
        raise NaturalControllerConfigError("permutations must be an object")
    _require_exact_keys(
        permutations,
        {"replicas", "draws_per_replica", "seed_salt"},
        "permutations",
    )
    replicas = _positive_int(permutations["replicas"], "permutations.replicas")
    if replicas < 2:
        raise NaturalControllerConfigError(
            "permutations.replicas must be at least two for nested MC uncertainty"
        )
    _positive_int(
        permutations["draws_per_replica"], "permutations.draws_per_replica"
    )
    _nonnegative_int(permutations["seed_salt"], "permutations.seed_salt")

    controller = config["controller"]
    if not isinstance(controller, Mapping):
        raise NaturalControllerConfigError("controller must be an object")
    _require_exact_keys(
        controller,
        {
            "radius_history",
            "epsilon",
            "pairing_reliability_threshold",
            "gate_timing",
            "radius_rule",
            "pairing_rule",
            "variance_rule",
        },
        "controller",
    )
    if controller["radius_history"] != 16:
        raise NaturalControllerConfigError("radius history is locked to 16")
    _finite_float(controller["epsilon"], "controller.epsilon", positive=True)
    threshold = _finite_float(
        controller["pairing_reliability_threshold"],
        "controller.pairing_reliability_threshold",
    )
    if not 0.0 < threshold < 1.0:
        raise NaturalControllerConfigError(
            "pairing reliability threshold must lie strictly between zero and one"
        )
    expected_semantics = {
        "gate_timing": _GATE_TIMING,
        "radius_rule": _RADIUS_RULE,
        "pairing_rule": _PAIRING_RULE,
        "variance_rule": _VARIANCE_RULE,
    }
    for key, expected in expected_semantics.items():
        if controller[key] != expected:
            raise NaturalControllerConfigError(f"controller.{key} differs from lock")

    run_keys = {
        "seeds",
        "max_packets",
        "bootstrap_draws",
        "bootstrap_seed",
        "bootstrap_block_lengths",
        "primary_bootstrap_block_length",
        "retain_packet_records",
    }
    run_specs: dict[str, dict[str, Any]] = {}
    for tier in ("full", "smoke"):
        spec = config[tier]
        if not isinstance(spec, Mapping):
            raise NaturalControllerConfigError(f"{tier} must be an object")
        _require_exact_keys(spec, run_keys, tier)
        seeds = spec["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in seeds
            )
        ):
            raise NaturalControllerConfigError(
                f"{tier}.seeds must be distinct nonnegative integers"
            )
        _positive_int(spec["bootstrap_draws"], f"{tier}.bootstrap_draws")
        _nonnegative_int(spec["bootstrap_seed"], f"{tier}.bootstrap_seed")
        block_lengths = spec["bootstrap_block_lengths"]
        if (
            not isinstance(block_lengths, list)
            or not block_lengths
            or len(set(block_lengths)) != len(block_lengths)
        ):
            raise NaturalControllerConfigError(
                f"{tier}.bootstrap_block_lengths must be a nonempty unique list"
            )
        for value in block_lengths:
            _positive_int(value, f"{tier}.bootstrap_block_lengths")
        primary = _positive_int(
            spec["primary_bootstrap_block_length"],
            f"{tier}.primary_bootstrap_block_length",
        )
        if primary not in block_lengths:
            raise NaturalControllerConfigError(
                f"{tier}.primary_bootstrap_block_length must be in sensitivity list"
            )
        if not isinstance(spec["retain_packet_records"], bool):
            raise NaturalControllerConfigError(
                f"{tier}.retain_packet_records must be boolean"
            )
        run_specs[tier] = dict(spec)

    if tuple(run_specs["full"]["seeds"]) != FULL_SEEDS:
        raise NaturalControllerConfigError(
            f"full seeds must be the fixed confirmation seeds {list(FULL_SEEDS)}"
        )
    if run_specs["full"]["max_packets"] is not None:
        raise NaturalControllerConfigError("full run must retain the complete stream")
    if not run_specs["full"]["retain_packet_records"]:
        raise NaturalControllerConfigError(
            "full run must retain compact packet audit records"
        )
    smoke_max = _positive_int(run_specs["smoke"]["max_packets"], "smoke.max_packets")
    if smoke_max >= 60:
        raise NaturalControllerConfigError("smoke must use fewer than 60 packets")
    if not run_specs["smoke"]["retain_packet_records"]:
        raise NaturalControllerConfigError("smoke must retain audit packet records")
    if set(run_specs["full"]["seeds"]).intersection(run_specs["smoke"]["seeds"]):
        raise NaturalControllerConfigError("full and smoke seeds must be disjoint")
    if run_specs["smoke"]["bootstrap_draws"] >= run_specs["full"]["bootstrap_draws"]:
        raise NaturalControllerConfigError(
            "smoke bootstrap resources must be smaller than full"
        )

    source_identity = _validate_source_identity(config["source_identity"])
    detached = json.loads(json.dumps(config))
    detached["source_identity"] = source_identity
    return detached


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(payload)


def _shared_source_evidence(
    path: Path, config_path: Path, *, smoke: bool
) -> dict[str, Any]:
    """Validate the repository-wide source freeze at the CLI boundary."""

    resolved_path = path.expanduser().resolve()
    resolved_config = config_path.expanduser().resolve(strict=True)
    evidence: dict[str, Any] = {
        "experiment": SOURCE_EXPERIMENT,
        "manifest_filename": path.name,
        "manifest_sha256": None,
        "source_bundle_sha256": None,
        "source_file_count": None,
        "shared_manifest_validated": False,
        "configuration_matches_manifest": False,
        "validated_for_execution": False,
        "validation_error": None,
        "manifest": None,
    }
    try:
        bundle, files = validate_source_manifest(
            resolved_path, SOURCE_EXPERIMENT, repo=REPO
        )
        evidence.update(
            {
                "manifest_sha256": _sha256_file(resolved_path),
                "source_bundle_sha256": bundle,
                "source_file_count": len(files),
                "shared_manifest_validated": True,
            }
        )
        spec = get_source_spec(SOURCE_EXPERIMENT)
        config_matches = (
            _sha256_file(resolved_config) == files[spec.configuration_path]
        )
        if not config_matches:
            raise SourceProvenanceError(
                "source manifest authenticates another natural-controller configuration"
            )
        evidence.update(
            {
                "configuration_matches_manifest": True,
                "validated_for_execution": True,
                "manifest": strict_json(resolved_path),
            }
        )
    except (OSError, SourceProvenanceError) as error:
        evidence["validation_error"] = f"{type(error).__name__}: {error}"
        if not smoke:
            raise SourceProvenanceError(
                f"paper-eligible {SOURCE_EXPERIMENT} source validation failed: {error}"
            ) from error
    return evidence


def _dict_clone(values: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().clone()) for name, value in values.items()
    )


def _dict_subtract(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, left[name].detach() - right[name].detach()) for name in left
    )


def _dict_linear_combination(
    left: Mapping[str, torch.Tensor],
    left_weight: float,
    right: Mapping[str, torch.Tensor],
    right_weight: float,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            name,
            float(left_weight) * left[name].detach()
            + float(right_weight) * right[name].detach(),
        )
        for name in left
    )


def _dict_cosine(
    left: Mapping[str, torch.Tensor] | None,
    right: Mapping[str, torch.Tensor] | None,
    epsilon: float,
) -> float:
    if not left or not right or tuple(left) != tuple(right):
        return 0.0
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for name in left:
        a = left[name].detach().cpu().double().reshape(-1)
        b = right[name].detach().cpu().double().reshape(-1)
        dot += torch.dot(a, b)
        left_sq += torch.dot(a, a)
        right_sq += torch.dot(b, b)
    denominator = float(torch.sqrt(left_sq * right_sq).item())
    if denominator <= float(epsilon):
        return 0.0
    return float(np.clip(float(dot.item()) / denominator, -1.0, 1.0))


def _parameter_digest(parameters: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in parameters.items():
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _prediction_digest(prediction: np.ndarray) -> str:
    array = np.ascontiguousarray(prediction, dtype="<f4")
    return _sha256_bytes(array.tobytes())


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _stable_manifest_value(value: Any) -> Any:
    """Remove acquisition-clock fields while retaining content provenance."""

    volatile = {
        "downloaded_at",
        "downloaded_ts",
        "acquired_at",
        "acquisition_timestamp",
        "generated_at",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_manifest_value(item)
            for key, item in sorted(value.items())
            if str(key) not in volatile
        }
    if isinstance(value, list):
        return [_stable_manifest_value(item) for item in value]
    return value


def stream_identity(data: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "name",
        "X",
        "y",
        "month_id",
        "scored_mask",
        "event_id",
        "event_time_seconds",
        "chronology_preserved",
        "all_preprocessed_rows_retained",
    }
    missing = required.difference(data)
    if missing:
        raise NaturalControllerIntegrityError(
            f"stream is missing frozen fields: {sorted(missing)}"
        )
    n = len(data["X"])
    if any(len(data[key]) != n for key in required.intersection({
        "y", "month_id", "scored_mask", "event_id", "event_time_seconds"
    })):
        raise NaturalControllerIntegrityError("stream field lengths differ")
    if not bool(data["chronology_preserved"]):
        raise NaturalControllerIntegrityError("stream chronology is not preserved")
    if not bool(data["all_preprocessed_rows_retained"]):
        raise NaturalControllerIntegrityError("stream preprocessing dropped rows")
    x = np.asarray(data["X"])
    y = np.asarray(data["y"])
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise NaturalControllerIntegrityError("stream arrays have invalid shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise NaturalControllerIntegrityError("stream arrays contain non-finite values")
    manifest_payload = json.loads(dataset_manifest_json(dict(data)))
    components = {
        "dataset_manifest": _stable_manifest_value(manifest_payload),
        "features_sha256": _array_digest(x),
        "targets_sha256": _array_digest(y),
        "month_sha256": _array_digest(data["month_id"]),
        "scored_sha256": _array_digest(data["scored_mask"]),
        "event_sha256": _array_digest(data["event_id"]),
        "event_time_sha256": _array_digest(data["event_time_seconds"]),
    }
    return {
        "name": str(data["name"]),
        "rows": int(len(x)),
        "dimension": int(x.shape[1]),
        "components": components,
        "sha256": _sha256_bytes(_canonical_json(components)),
    }


def _locked_archive(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise NaturalControllerIntegrityError(
            f"data archive must not be a symbolic link: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise NaturalControllerIntegrityError(
            f"locked Beijing archive is unavailable: {candidate}"
        ) from error
    if not resolved.is_file():
        raise NaturalControllerIntegrityError(
            f"locked Beijing archive is not a regular file: {resolved}"
        )
    return resolved


def _raw_archive_identity(archive: Path) -> dict[str, Any]:
    frame, parse_provenance = stage4._parse_archive("BEIJING", str(archive))
    return {
        "selected_members": list(parse_provenance["selected_members"]),
        "row_count": int(len(frame)),
        "schema": [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in frame.dtypes.items()
        ],
    }


def _load_locked_beijing(
    archive: Path, *, seed: int
) -> dict[str, Any]:
    """Load only caller-supplied local bytes; this path cannot acquire data."""

    # Reproduce ``experiments.fmmt.common.load_regression_stream`` exactly,
    # starting from an already validated local archive.  Calling the legacy
    # loader here is forbidden because its cache-miss behavior downloads.
    _ = int(seed)  # the frozen real stream is intentionally seed-independent
    frame, parse_provenance = stage4._parse_archive("BEIJING", str(archive))
    spec = stage4.DS["BEIJING"]
    manifest = {
        "dataset": "BEIJING",
        "uci_id": spec["uci_id"],
        "source_url": spec["url"],
        "dataset_page": spec["page"],
        "license": spec["license"],
        "sha256": sha256_file(archive),
        "zip_bytes": int(archive.stat().st_size),
        "n_rows_raw": int(len(frame)),
        "members": list(parse_provenance["selected_members"]),
        "parse_provenance": parse_provenance,
    }
    raw = stage4._build_arrays(
        "BEIJING", frame, drop_final_short_packet=False
    )
    chronology = _chronology_metadata("BEIJING", frame, int(raw["N"]))
    event_id = chronology["event_id"]
    event_time_seconds = chronology["event_time_seconds"]
    event_audit = chronology["audit"]
    scale_cal_n = max(2, int(raw["cal_n"]))
    ylog = np.asarray(raw["ylog"], dtype=np.float32)
    y_mean = float(np.mean(ylog[:scale_cal_n]))
    y_std = float(np.std(ylog[:scale_cal_n], ddof=1))
    if not np.isfinite(y_std) or y_std < 1.0e-8:
        y_std = 1.0
    calibration_event = int(event_id[min(scale_cal_n - 1, len(event_id) - 1)])
    scored_mask = np.asarray(event_id > calibration_event, dtype=bool)
    cal_n = (
        int(np.flatnonzero(scored_mask)[0]) if scored_mask.any() else len(event_id)
    )
    event_audit = dict(event_audit)
    event_audit.update(
        {
            "row_count_calibration_boundary": scale_cal_n,
            "event_aligned_scoring_boundary": cal_n,
            "calibration_event_id": calibration_event,
            "no_acquisition_event_crosses_scoring_boundary": bool(
                not np.any(event_id[:cal_n] == event_id[cal_n])
                if cal_n < len(event_id)
                else True
            ),
        }
    )
    output = dict(raw)
    output.update(
        name="BEIJING",
        X=np.asarray(raw["X"], dtype=np.float32),
        y=((ylog - y_mean) / y_std).astype(np.float32),
        ylog=ylog,
        y_calibration_mean=y_mean,
        y_calibration_std=y_std,
        y_scale_calibration_rows=scale_cal_n,
        scoring_calibration_rows=cal_n,
        scored_mask=scored_mask,
        data_manifest=manifest,
        event_id=event_id,
        event_time_seconds=event_time_seconds,
        chronology_audit=event_audit,
        chronology_preserved=True,
        all_preprocessed_rows_retained=(
            int(raw.get("N", len(raw["X"])))
            == int(raw.get("n_rows", len(raw["X"])))
        ),
    )
    if not np.isfinite(output["X"]).all() or not np.isfinite(output["y"]).all():
        raise NaturalControllerIntegrityError(
            "BEIJING: non-finite value after frozen preprocessing"
        )
    if not bool(output["all_preprocessed_rows_retained"]):
        raise NaturalControllerIntegrityError(
            "BEIJING: preprocessing dropped a final short packet"
        )
    if not bool(event_audit["no_acquisition_event_crosses_scoring_boundary"]):
        raise NaturalControllerIntegrityError(
            "BEIJING: an acquisition event crosses the scoring boundary"
        )
    return output


def build_beijing_data_manifest(
    *,
    archive_path: Path,
    config_path: Path,
    config: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze local Beijing archive bytes and their preprocessing identity.

    This is an acquisition/freeze artifact, never a scientific result.  The
    scientific CLI subsequently revalidates every field and passes the loaded
    in-memory stream into :func:`run_campaign`.
    """

    locked = validate_config(config)
    resolved_config = config_path.expanduser().resolve(strict=True)
    if _sha256_bytes(_canonical_json(strict_json(resolved_config))) != _sha256_bytes(
        _canonical_json(locked)
    ):
        raise NaturalControllerIntegrityError(
            "data freeze config payload differs from --config bytes"
        )
    archive = _locked_archive(archive_path)
    raw_identity = _raw_archive_identity(archive)
    data = _load_locked_beijing(
        archive, seed=int(locked["full"]["seeds"][0])
    )
    preprocessing = stream_identity(data)
    return {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_SCIENTIFIC_RUN",
        "scientific_evidence_eligible": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIJING",
        "configuration_file_sha256": sha256_file(
            resolved_config
        ),
        "configuration_payload_sha256": _sha256_bytes(_canonical_json(locked)),
        "source_identity": shared_source_identity(source_evidence),
        "archive": {
            "sha256": sha256_file(archive),
            "bytes": int(archive.stat().st_size),
        },
        "raw_archive_identity": raw_identity,
        "preprocessing_identity": preprocessing,
    }


def validate_beijing_data_manifest(
    path: Path,
    *,
    archive_path: Path,
    config_path: Path,
    config: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> tuple[ValidatedEvidence, dict[str, Any]]:
    """Validate archive and preprocessing before any scientific computation."""

    manifest_path = path.expanduser().resolve(strict=True)
    payload = strict_json(manifest_path)
    expected_keys = {
        "schema_version",
        "status",
        "scientific_evidence_eligible",
        "created_utc",
        "dataset",
        "configuration_file_sha256",
        "configuration_payload_sha256",
        "source_identity",
        "archive",
        "raw_archive_identity",
        "preprocessing_identity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise NaturalControllerIntegrityError(
            "Beijing data manifest keys differ from schema"
        )
    if payload["schema_version"] != DATA_MANIFEST_SCHEMA_VERSION:
        raise NaturalControllerIntegrityError("Beijing data manifest schema mismatch")
    if payload["status"] != "FROZEN_BEFORE_SCIENTIFIC_RUN":
        raise NaturalControllerIntegrityError("Beijing data manifest is not frozen")
    if payload["scientific_evidence_eligible"] is not False:
        raise NaturalControllerIntegrityError(
            "data acquisition/freeze artifact cannot claim scientific eligibility"
        )
    if payload["dataset"] != "BEIJING":
        raise NaturalControllerIntegrityError("data manifest authenticates another dataset")
    if payload["configuration_file_sha256"] != sha256_file(
        config_path.expanduser().resolve(strict=True)
    ):
        raise NaturalControllerIntegrityError("data manifest authenticates another config")
    if payload["source_identity"] != shared_source_identity(source_evidence):
        raise NaturalControllerIntegrityError(
            "data manifest authenticates another source freeze"
        )

    locked = validate_config(config)
    if _sha256_bytes(
        _canonical_json(strict_json(config_path.expanduser().resolve(strict=True)))
    ) != _sha256_bytes(_canonical_json(locked)):
        raise NaturalControllerIntegrityError(
            "data validation config payload differs from --config bytes"
        )
    if payload["configuration_payload_sha256"] != _sha256_bytes(
        _canonical_json(locked)
    ):
        raise NaturalControllerIntegrityError(
            "data manifest authenticates another config payload"
        )
    archive = _locked_archive(archive_path)
    archive_identity = {
        "sha256": sha256_file(archive),
        "bytes": int(archive.stat().st_size),
    }
    if payload["archive"] != archive_identity:
        raise NaturalControllerIntegrityError(
            "current Beijing archive bytes differ from data manifest"
        )
    raw_identity = _raw_archive_identity(archive)
    if payload["raw_archive_identity"] != raw_identity:
        raise NaturalControllerIntegrityError(
            "current Beijing archive member/schema/rows differ from data manifest"
        )
    data = _load_locked_beijing(
        archive, seed=int(locked["full"]["seeds"][0])
    )
    preprocessing = stream_identity(data)
    if payload["preprocessing_identity"] != preprocessing:
        raise NaturalControllerIntegrityError(
            "current Beijing preprocessing differs from data manifest"
        )
    evidence = {
        "schema_version": DATA_BINDING_SCHEMA_VERSION,
        "validation_status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "archive_sha256": archive_identity["sha256"],
        "preprocessing_sha256": preprocessing["sha256"],
        "manifest": dict(payload),
    }
    return _validated_evidence("data", evidence), data


def _risk(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: np.ndarray,
    scored: np.ndarray,
) -> float:
    prediction = _predict(model, parameters, inputs).astype(np.float64)
    residual = prediction[scored] - np.asarray(targets, dtype=np.float64)[scored]
    return float(np.mean(residual * residual))


def _apply_reuse(
    model: torch.nn.Module,
    direction: Mapping[str, torch.Tensor] | None,
    radius: float,
    epsilon: float,
) -> float:
    norm = _dict_norm(direction)
    if direction is None or norm <= epsilon or radius <= epsilon:
        return 0.0
    before = _dict_clone(_named_parameters(model))
    scale = -float(radius) / float(norm)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.add_(direction[name].to(parameter), alpha=scale)
    after = _named_parameters(model)
    return _difference_norm(before, after)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _pearson_or_none(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or float(np.std(x)) <= 1.0e-15
        or float(np.std(y)) <= 1.0e-15
    ):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _v_stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "packets": len(rows),
        "mean_v_total": _mean_or_none([float(row["v_total"]) for row in rows]),
        "mean_v_perm": _mean_or_none([float(row["v_perm"]) for row in rows]),
        "mean_v_pair": _mean_or_none([float(row["v_pair"]) for row in rows]),
    }


def _summarize_mean_drift(
    transitions: Sequence[Mapping[str, Any]], epsilon: float
) -> dict[str, Any]:
    if not transitions:
        raise NaturalControllerIntegrityError("mean-drift diagnostic has no transitions")
    absolute_changes = [float(row["absolute_target_mean_change"]) for row in transitions]
    wedge = [
        row
        for row in transitions
        if bool(row["full_cosine_positive_and_c_cosine_nonpositive"])
    ]
    audited = [row for row in transitions if row["v_total"] is not None]
    audited_wedge = [
        row
        for row in audited
        if bool(row["full_cosine_positive_and_c_cosine_nonpositive"])
    ]
    audited_other = [
        row
        for row in audited
        if not bool(row["full_cosine_positive_and_c_cosine_nonpositive"])
    ]
    threshold = float(np.median([row["absolute_target_mean_change"] for row in audited])) if audited else None
    lower = (
        [row for row in audited if row["absolute_target_mean_change"] <= threshold]
        if threshold is not None
        else []
    )
    higher = (
        [row for row in audited if row["absolute_target_mean_change"] > threshold]
        if threshold is not None
        else []
    )
    return {
        "definition": "post_score_packet_target_mean_change_on_standardized_target",
        "labels_used_only_after_prequential_score": True,
        "transitions": len(transitions),
        "mean_signed_packet_target_mean_change": float(
            np.mean([row["target_mean_change"] for row in transitions])
        ),
        "mean_absolute_packet_target_mean_change": float(np.mean(absolute_changes)),
        "rms_packet_target_mean_change": float(
            math.sqrt(np.mean(np.square(absolute_changes)))
        ),
        "fraction_full_cosine_positive": float(
            np.mean([row["full_cosine"] > 0.0 for row in transitions])
        ),
        "fraction_c_cosine_nonpositive": float(
            np.mean([row["c_cosine"] <= 0.0 for row in transitions])
        ),
        "fraction_full_cosine_positive_and_c_cosine_nonpositive": len(wedge)
        / len(transitions),
        "v_by_alignment_stratum": {
            "full_positive_c_nonpositive": _v_stratum(audited_wedge),
            "all_other_transitions": _v_stratum(audited_other),
        },
        "v_by_absolute_target_mean_change_stratum": {
            "threshold_rule": "within_seed_median_over_scored_v_audited_transitions",
            "threshold": threshold,
            "at_or_below_median": _v_stratum(lower),
            "above_median": _v_stratum(higher),
        },
        "association_with_absolute_target_mean_change": {
            "audited_packets": len(audited),
            "pearson_v_total": _pearson_or_none(
                [row["absolute_target_mean_change"] for row in audited],
                [row["v_total"] for row in audited],
            ),
            "pearson_v_perm": _pearson_or_none(
                [row["absolute_target_mean_change"] for row in audited],
                [row["v_perm"] for row in audited],
            ),
            "pearson_v_pair": _pearson_or_none(
                [row["absolute_target_mean_change"] for row in audited],
                [row["v_pair"] for row in audited],
            ),
        },
        "epsilon": float(epsilon),
    }


def _unflatten_gradient(
    vector: torch.Tensor, parameters: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    cursor = 0
    for name, parameter in parameters.items():
        count = int(parameter.numel())
        output[name] = vector[cursor : cursor + count].reshape_as(parameter).to(
            dtype=parameter.dtype, device=parameter.device
        )
        cursor += count
    if cursor != int(vector.numel()):
        raise NaturalControllerIntegrityError(
            "permutation gradient does not match the parameter layout"
        )
    return output


def _full_mse_permutation_gradients(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: np.ndarray,
    *,
    seed: int,
    packet_number: int,
    permutation_config: Mapping[str, Any],
) -> list[list[OrderedDict[str, torch.Tensor]]]:
    """Materialize locked full-gradient target permutations in nested replicas."""

    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    rows = int(target.size)
    if rows < 2:
        raise NaturalControllerIntegrityError("permutation audit needs two rows")
    jacobian = output_jacobian_matrix(model, parameters, inputs).double()
    prediction = torch.as_tensor(
        _predict(model, parameters, inputs),
        dtype=torch.float64,
        device=jacobian.device,
    ).reshape(-1)
    replicas = int(permutation_config["replicas"])
    draws = int(permutation_config["draws_per_replica"])
    salt = int(permutation_config["seed_salt"])
    output: list[list[OrderedDict[str, torch.Tensor]]] = []
    for replica in range(replicas):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), salt, int(packet_number), replica, 0xA11D17]
            )
        )
        replica_output: list[OrderedDict[str, torch.Tensor]] = []
        for _ in range(draws):
            order = rng.permutation(rows)
            permuted = torch.as_tensor(
                target[order], dtype=torch.float64, device=jacobian.device
            )
            flat = 2.0 * jacobian.transpose(0, 1).matmul(
                prediction - permuted
            ) / float(rows)
            replica_output.append(_unflatten_gradient(flat, parameters))
        output.append(replica_output)
    return output


def _make_optimizer(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )


def _adamw_update(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    gradient: Mapping[str, torch.Tensor],
    gradient_clip: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    before = _dict_clone(_named_parameters(model))
    for name, parameter in model.named_parameters():
        parameter.grad = gradient[name].detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = _dict_clone(_named_parameters(model))
    norm = _difference_norm(before, after)
    if not math.isfinite(norm):
        raise NaturalControllerIntegrityError("non-finite AdamW update norm")
    return norm


def build_common_radius_schedule(
    data: dict[str, Any],
    *,
    seed: int,
    config: Mapping[str, Any],
    max_packets: int | None,
) -> dict[str, Any]:
    """Build a causal radius schedule on one never-reuse shadow trajectory."""

    packets = packetize_stream(
        data, packet_size=int(config["dataset"]["packet_size"])
    )
    if max_packets is not None:
        packets = packets[: int(max_packets)]
    seed_everything(int(seed))
    device = torch.device(config["model"]["device"])
    model = FunctionalUnitRegressor(
        input_dim=int(np.asarray(data["X"]).shape[1]),
        n_atoms=int(config["model"]["functional_units"]),
    ).to(device)
    model.train()
    optimizer = _make_optimizer(model, config)
    history: deque[float] = deque(
        maxlen=int(config["controller"]["radius_history"])
    )
    values: list[float] = []
    update_norms: list[float] = []
    for packet in packets:
        # This value depends only on shadow updates from strictly earlier
        # packets, even though the whole schedule is materialized up front.
        values.append(float(np.median(history)) if history else 0.0)
        x = torch.as_tensor(packet["x"], dtype=torch.float32, device=device)
        y = torch.as_tensor(packet["y"], dtype=torch.float32, device=device)
        parameters = _named_parameters(model)
        gradient, _ = _gradient_dict(model, parameters, x, y)
        update_norm = _adamw_update(
            model,
            optimizer,
            gradient,
            float(config["optimizer"]["gradient_clip"]),
        )
        history.append(update_norm)
        update_norms.append(update_norm)
    payload = {
        "rule": _RADIUS_RULE,
        "seed": int(seed),
        "history_length": int(config["controller"]["radius_history"]),
        "values": values,
        "never_reuse_shadow_update_norms": update_norms,
    }
    payload["sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _build_v_shadow(
    baseline: Mapping[str, torch.Tensor],
    true_gradient: Mapping[str, torch.Tensor],
    permutation_gradients: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    *,
    radius: float,
    action_packet: int,
    gradient_source_packet: int,
) -> dict[str, Any]:
    """Construct separately normalized equal-radius prospective actions."""

    return {
        "baseline": _dict_clone(baseline),
        "true": _shift(baseline, _delta(true_gradient, radius)),
        "permuted": [
            [_shift(baseline, _delta(draw, radius)) for draw in replica]
            for replica in permutation_gradients
        ],
        "action_packet": int(action_packet),
        "gradient_source_packet": int(gradient_source_packet),
    }


def _score_v_shadow(
    model: torch.nn.Module,
    shadow: Mapping[str, Any],
    inputs: torch.Tensor,
    targets: np.ndarray,
    scored: np.ndarray,
) -> dict[str, Any]:
    """Score expected prospective risk across normalized permutation actions."""

    baseline_risk = _risk(model, shadow["baseline"], inputs, targets, scored)
    true_risk = _risk(model, shadow["true"], inputs, targets, scored)
    replica_values = np.asarray(
        [
            baseline_risk
            - float(
                np.mean(
                    [
                        _risk(model, draw, inputs, targets, scored)
                        for draw in replica
                    ]
                )
            )
            for replica in shadow["permuted"]
        ],
        dtype=np.float64,
    )
    if replica_values.size < 2:
        raise NaturalControllerIntegrityError(
            "nested MC V audit requires at least two replicas"
        )
    v_total = baseline_risk - true_risk
    v_perm = float(replica_values.mean())
    v_pair = v_total - v_perm
    return {
        "v_total": v_total,
        "v_perm": v_perm,
        "v_pair": v_pair,
        "replica_values": replica_values,
        "mc_se": float(
            np.std(replica_values, ddof=1) / math.sqrt(replica_values.size)
        ),
        "identity_residual": abs(v_total - (v_perm + v_pair)),
    }


def run_policy(
    data: dict[str, Any],
    policy: str,
    *,
    seed: int,
    config: Mapping[str, Any],
    max_packets: int | None,
    radius_schedule: Mapping[str, Any],
    retain_packet_records: bool,
) -> dict[str, Any]:
    """Run one live policy with causal common radii and nested-MC V shadows."""

    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    identity = stream_identity(data)
    packet_size = int(config["dataset"]["packet_size"])
    packets = packetize_stream(data, packet_size=packet_size)
    if max_packets is not None:
        packets = packets[: int(max_packets)]
    if len(packets) < 2:
        raise NaturalControllerIntegrityError("controller needs at least two packets")
    radii = [float(value) for value in radius_schedule.get("values", [])]
    if (
        radius_schedule.get("rule") != _RADIUS_RULE
        or len(radii) != len(packets)
        or not np.isfinite(radii).all()
        or any(value < 0.0 for value in radii)
    ):
        raise NaturalControllerIntegrityError("invalid common radius schedule")
    schedule_without_hash = dict(radius_schedule)
    expected_schedule_sha = str(schedule_without_hash.pop("sha256", ""))
    if _sha256_bytes(_canonical_json(schedule_without_hash)) != expected_schedule_sha:
        raise NaturalControllerIntegrityError("common radius schedule hash mismatch")

    seed_everything(int(seed))
    device = torch.device(config["model"]["device"])
    model = FunctionalUnitRegressor(
        input_dim=int(np.asarray(data["X"]).shape[1]),
        n_atoms=int(config["model"]["functional_units"]),
    ).to(device)
    model.train()
    optimizer = _make_optimizer(model, config)
    epsilon = float(config["controller"]["epsilon"])
    reliability_threshold = float(
        config["controller"]["pairing_reliability_threshold"]
    )
    permutation_replicas = int(config["permutations"]["replicas"])
    draws_per_replica = int(config["permutations"]["draws_per_replica"])

    previous_g: OrderedDict[str, torch.Tensor] | None = None
    previous_m: OrderedDict[str, torch.Tensor] | None = None
    previous_c: OrderedDict[str, torch.Tensor] | None = None
    previous_permutations: list[list[OrderedDict[str, torch.Tensor]]] | None = None
    previous_q: float | None = None
    previous_v: float | None = None
    pending_v_shadow: dict[str, Any] | None = None

    month_stats: dict[int, dict[str, float | int]] = {}
    packet_records: list[dict[str, Any]] = []
    prediction_digests: list[str] = []
    state_digests: list[str] = []
    action_count = 0
    eligible_count = 0
    scored_rows = 0
    cumulative_squared_error = 0.0
    covariance_norms: list[float] = []
    finite_permutation_variances: list[float] = []
    s_j_values: list[float] = []
    s_y_values: list[float] = []
    full_cosines: list[float] = []
    m_cosines: list[float] = []
    c_cosines: list[float] = []
    reliabilities: list[float] = []
    m_weights: list[float] = []
    c_weights: list[float] = []
    scored_v_total: list[float] = []
    scored_v_perm: list[float] = []
    scored_v_pair: list[float] = []
    scored_v_perm_replicas: list[np.ndarray] = []
    v_residuals: list[float] = []
    calibrated_reuse_radii: list[float] = []
    true_update_norms: list[float] = []
    realized_reuse_displacements: list[float] = []
    mean_drift_transitions: list[dict[str, Any]] = []
    previous_target_mean: float | None = None
    true_update_count = 0
    pairing_channel_open_count = 0

    for packet_number, packet in enumerate(packets):
        indices = np.asarray(packet["index"], dtype=np.int64)
        x = torch.as_tensor(packet["x"], dtype=torch.float32, device=device)
        parameters = _named_parameters(model)
        radius = radii[packet_number]
        calibrated_reuse_radii.append(radius)

        # Strict prequential boundary: predictions are fixed before this
        # function materializes the current packet's target array.
        prediction = _predict(model, parameters, x)
        prediction_sha = _prediction_digest(prediction)
        prediction_digests.append(prediction_sha)

        targets_np = np.asarray(packet["y"], dtype=np.float32)
        scored = np.asarray(packet["scored"], dtype=bool)
        if scored.shape != targets_np.shape:
            raise NaturalControllerIntegrityError("packet scoring mask shape differs")
        squared_error = (prediction.astype(np.float64) - targets_np.astype(np.float64)) ** 2
        packet_scored_sse = float(np.sum(squared_error[scored]))
        packet_scored_n = int(np.sum(scored))
        cumulative_squared_error += packet_scored_sse
        scored_rows += packet_scored_n
        months = np.asarray(packet["month"])
        for month in np.unique(months[scored]):
            keep = scored & (months == month)
            key = int(month)
            entry = month_stats.setdefault(key, {"sse": 0.0, "n": 0})
            entry["sse"] = float(entry["sse"]) + float(np.sum(squared_error[keep]))
            entry["n"] = int(entry["n"]) + int(np.sum(keep))

        # Evaluate shadows made after packet t-1's ordinary update, at exactly
        # the state where its live reuse action was applied.  Each permuted
        # target gradient is normalized separately before risk is averaged.
        v_total = v_perm = v_pair = v_residual = None
        v_perm_mc_se = v_pair_mc_se = None
        v_perm_replica_values: list[float] | None = None
        v_source_action_packet = v_source_gradient_packet = None
        if pending_v_shadow is not None and packet_scored_n > 0:
            scored_shadow = _score_v_shadow(
                model, pending_v_shadow, x, targets_np, scored
            )
            v_total = float(scored_shadow["v_total"])
            v_perm = float(scored_shadow["v_perm"])
            v_pair = float(scored_shadow["v_pair"])
            v_residual = float(scored_shadow["identity_residual"])
            replica_values = np.asarray(
                scored_shadow["replica_values"], dtype=np.float64
            )
            v_perm_mc_se = float(scored_shadow["mc_se"])
            v_pair_mc_se = v_perm_mc_se
            v_perm_replica_values = replica_values.astype(float).tolist()
            v_source_action_packet = int(pending_v_shadow["action_packet"])
            v_source_gradient_packet = int(
                pending_v_shadow["gradient_source_packet"]
            )
            scored_v_total.append(v_total)
            scored_v_perm.append(v_perm)
            scored_v_pair.append(v_pair)
            scored_v_perm_replicas.append(replica_values)
            v_residuals.append(v_residual)

        # Label revelation point.  Current labels can only influence the
        # action after prediction and score have been fixed above.
        y = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
        current_target_mean = float(np.mean(targets_np, dtype=np.float64))
        current_g_raw, current_loss = _gradient_dict(model, parameters, x, y)
        current_g = _dict_clone(current_g_raw)
        strength, expected_gradient = packet_strength(
            model, parameters, x, y, current_g
        )
        current_m = _dict_clone(expected_gradient)
        current_c = _dict_subtract(current_g, current_m)
        current_q = float(_dict_norm(current_c) ** 2)
        # packet_strength reports the half-gradient RMS variation.  The live
        # MSE gradient is exactly twice that half-gradient.
        current_v = float(4.0 * float(strength["permutation_gradient_l2_sd"]) ** 2)
        covariance_norms.append(float(strength["covariance_norm"]))
        finite_permutation_variances.append(current_v)
        s_j_values.append(float(strength["S_J"]))
        s_y_values.append(float(strength["S_y"]))
        if float(strength["decomposition_relative_error"]) > 2.0e-5:
            raise NaturalControllerIntegrityError(
                "target-permutation gradient decomposition failed"
            )
        current_permutations = _full_mse_permutation_gradients(
            model,
            parameters,
            x,
            targets_np,
            seed=int(seed),
            packet_number=packet_number,
            permutation_config=config["permutations"],
        )

        full_cosine = _dict_cosine(previous_g, current_g, epsilon)
        m_cosine = _dict_cosine(previous_m, current_m, epsilon)
        c_cosine = _dict_cosine(previous_c, current_c, epsilon)
        full_cosines.append(full_cosine)
        m_cosines.append(m_cosine)
        c_cosines.append(c_cosine)

        if previous_target_mean is not None:
            target_mean_change = current_target_mean - previous_target_mean
            mean_drift_transitions.append(
                {
                    "target_mean_change": target_mean_change,
                    "absolute_target_mean_change": abs(target_mean_change),
                    "full_cosine": full_cosine,
                    "c_cosine": c_cosine,
                    "full_cosine_positive_and_c_cosine_nonpositive": (
                        full_cosine > 0.0 and c_cosine <= 0.0
                    ),
                    "v_total": v_total,
                    "v_perm": v_perm,
                    "v_pair": v_pair,
                }
            )

        direction: Mapping[str, torch.Tensor] | None = None
        reliability = 0.0
        m_weight = 0.0
        c_weight = 0.0
        pairing_channel_open = False
        eligible = previous_g is not None and radius > epsilon
        if eligible:
            eligible_count += 1
            if policy == "always":
                direction = previous_g
            elif policy == "cosine" and full_cosine > 0.0:
                direction = previous_g
            elif policy == "pairing_aware":
                if (
                    previous_m is None
                    or previous_c is None
                    or previous_q is None
                    or previous_v is None
                ):
                    raise NaturalControllerIntegrityError(
                        "eligible pairing controller has no stored channels"
                    )
                m_weight = max(0.0, m_cosine)
                reliability = previous_q / (previous_q + previous_v + epsilon)
                pairing_channel_open = bool(
                    c_cosine > 0.0
                    and reliability >= reliability_threshold
                )
                c_weight = max(0.0, c_cosine) if pairing_channel_open else 0.0
                direction = _dict_linear_combination(
                    previous_m, m_weight, previous_c, c_weight
                )
        if pairing_channel_open:
            pairing_channel_open_count += 1
        reliabilities.append(reliability)
        m_weights.append(m_weight)
        c_weights.append(c_weight)

        # Exactly one ordinary AdamW update is common to all controller rules.
        true_update_norm = _adamw_update(
            model,
            optimizer,
            current_g,
            float(config["optimizer"]["gradient_clip"]),
        )
        true_update_norms.append(true_update_norm)
        true_update_count += 1

        next_v_shadow: dict[str, Any] | None = None
        post_update = _dict_clone(_named_parameters(model))
        if eligible and previous_permutations is not None:
            next_v_shadow = _build_v_shadow(
                post_update,
                previous_g,
                previous_permutations,
                radius=radius,
                action_packet=packet_number,
                gradient_source_packet=packet_number - 1,
            )

        reuse_displacement = _apply_reuse(model, direction, radius, epsilon)
        acted = reuse_displacement > epsilon
        if acted:
            action_count += 1
            realized_reuse_displacements.append(reuse_displacement)
            if not math.isclose(
                reuse_displacement, radius, rel_tol=2.0e-5, abs_tol=2.0e-7
            ):
                raise NaturalControllerIntegrityError(
                    "reuse displacement does not match the calibrated radius"
                )
        state_sha = _parameter_digest(_named_parameters(model))
        state_digests.append(state_sha)

        record = {
            "packet": int(packet_number),
            "index_start": int(indices[0]),
            "index_stop_exclusive": int(indices[-1]) + 1,
            "rows": int(len(indices)),
            "scored_rows": packet_scored_n,
            "prediction_sha256": prediction_sha,
            "state_after_update_sha256": state_sha,
            "prequential_squared_error": packet_scored_sse,
            "current_mse_loss": float(current_loss),
            "gradient_norm": _dict_norm(current_g),
            "m_norm": _dict_norm(current_m),
            "c_norm": _dict_norm(current_c),
            "q_pairing_strength_full_gradient": current_q,
            "v_finite_permutation_variance_full_gradient": current_v,
            "covariance_j_y_norm_half_gradient": float(strength["covariance_norm"]),
            "S_J": float(strength["S_J"]),
            "S_y": float(strength["S_y"]),
            "full_cosine": full_cosine,
            "m_cosine": m_cosine,
            "c_cosine": c_cosine,
            "stored_reliability_q_over_q_plus_v": reliability,
            "pairing_reliability_threshold": reliability_threshold,
            "pairing_channel_open": pairing_channel_open,
            "m_weight": m_weight,
            "c_weight": c_weight,
            "radius_before_current_update": radius,
            "true_adamw_update_norm": true_update_norm,
            "reuse_displacement_norm": reuse_displacement,
            "eligible": eligible,
            "acted": acted,
            "reuse_source_packet": packet_number - 1 if eligible else None,
            "v_total": v_total,
            "v_perm": v_perm,
            "v_pair": v_pair,
            "v_identity_residual": v_residual,
            "v_perm_mc_replica_values": v_perm_replica_values,
            "v_perm_mc_se": v_perm_mc_se,
            "v_pair_mc_se": v_pair_mc_se,
            "v_source_action_packet": v_source_action_packet,
            "v_source_gradient_packet": v_source_gradient_packet,
        }
        packet_records.append(record)

        # Only parameter-sized channels cross the packet boundary.  No input,
        # target, prediction, or example identifier is retained for reuse.
        previous_g = _dict_clone(current_g)
        previous_m = _dict_clone(current_m)
        previous_c = _dict_clone(current_c)
        previous_permutations = current_permutations
        previous_q = current_q
        previous_v = current_v
        previous_target_mean = current_target_mean
        pending_v_shadow = next_v_shadow

    if scored_rows <= 0:
        raise NaturalControllerIntegrityError("run contains no scored examples")
    if true_update_count != len(packets):
        raise NaturalControllerIntegrityError("not exactly one true update per packet")
    max_v_residual = max(v_residuals, default=0.0)
    if max_v_residual > 1.0e-12:
        raise NaturalControllerIntegrityError(
            f"V identity failed with residual {max_v_residual}"
        )
    if not scored_v_perm_replicas:
        raise NaturalControllerIntegrityError(
            "no action-aligned nested permutation shadows were scored"
        )
    replica_matrix = np.stack(scored_v_perm_replicas, axis=0)
    aggregate_replica_values = replica_matrix.mean(axis=0)
    nested_mc_se = float(
        np.std(aggregate_replica_values, ddof=1)
        / math.sqrt(permutation_replicas)
    )
    per_month = {
        str(month): {
            "sse": float(entry["sse"]),
            "n": int(entry["n"]),
            "mse": float(entry["sse"]) / int(entry["n"]),
        }
        for month, entry in sorted(month_stats.items())
    }
    if sum(entry["n"] for entry in per_month.values()) != scored_rows:
        raise NaturalControllerIntegrityError("month accounting differs from score")

    return {
        "dataset": str(data["name"]),
        "stream_identity": identity,
        "policy": policy,
        "seed": int(seed),
        "packets": int(len(packets)),
        "scored_rows": scored_rows,
        "cumulative_prequential_squared_error": cumulative_squared_error,
        "prequential_mse": cumulative_squared_error / scored_rows,
        "prequential_rmse": math.sqrt(cumulative_squared_error / scored_rows),
        "per_month": per_month,
        "month_ids": [int(month) for month in sorted(month_stats)],
        "actions": action_count,
        "eligible_actions": eligible_count,
        "action_rate": action_count / max(1, eligible_count),
        "pairing_channel_open_rate": pairing_channel_open_count
        / max(1, eligible_count),
        "mean_full_cosine": _mean_or_none(full_cosines[1:]),
        "mean_m_cosine": _mean_or_none(m_cosines[1:]),
        "mean_c_cosine": _mean_or_none(c_cosines[1:]),
        "mean_stored_reliability": _mean_or_none(reliabilities[1:]),
        "mean_m_weight": _mean_or_none(m_weights[1:]),
        "mean_c_weight": _mean_or_none(c_weights[1:]),
        "radius_diagnostics": {
            "calibration_rule": _RADIUS_RULE,
            "schedule_sha256": expected_schedule_sha,
            "schedule_shared_across_policies": True,
            "history_length": int(config["controller"]["radius_history"]),
            "mean_calibrated_reuse_radius": float(np.mean(calibrated_reuse_radii)),
            "mean_true_adamw_update_norm": float(np.mean(true_update_norms)),
            "mean_realized_reuse_displacement": _mean_or_none(
                realized_reuse_displacements
            ),
        },
        "strength_diagnostics": {
            "mean_covariance_j_y_norm_half_gradient": float(np.mean(covariance_norms)),
            "mean_finite_permutation_variance_full_gradient": float(
                np.mean(finite_permutation_variances)
            ),
            "mean_S_J": float(np.mean(s_j_values)),
            "mean_S_y": float(np.mean(s_y_values)),
        },
        "natural_mean_drift_diagnostics": _summarize_mean_drift(
            mean_drift_transitions, epsilon
        ),
        "v_decomposition": {
            "definition": "post_current_update_equal_radius_action_evaluated_on_next_packet",
            "permutation_action": "mean_risk_over_normalized_full_gradient_target_permutations",
            "permutation_replicas": permutation_replicas,
            "draws_per_replica": draws_per_replica,
            "audited_scored_packets": len(scored_v_total),
            "mean_v_total": _mean_or_none(scored_v_total),
            "mean_v_perm": _mean_or_none(scored_v_perm),
            "mean_v_pair": _mean_or_none(scored_v_pair),
            "sum_v_total": float(np.sum(scored_v_total)),
            "sum_v_perm": float(np.sum(scored_v_perm)),
            "sum_v_pair": float(np.sum(scored_v_pair)),
            "max_identity_residual": max_v_residual,
            "v_perm_nested_mc_se": nested_mc_se,
            "v_pair_nested_mc_se": nested_mc_se,
        },
        "prediction_before_target_revelation": True,
        "gate_after_scoring": True,
        "reuse_lag_packets": 1,
        "true_optimizer_updates": true_update_count,
        "raw_examples_replayed": 0,
        "stored_parameter_sized_channels": ["g", "m", "c", "permuted_g_nested_mc"],
        "v_shadow_matches_post_current_update_action_timing": True,
        "prediction_sha256_sequence": prediction_digests,
        "state_sha256_sequence": state_digests,
        "packet_records": packet_records,
        "detailed_packet_payload_retained": bool(retain_packet_records),
    }


def paired_non_circular_month_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int,
    block_length: int,
) -> dict[str, Any]:
    """Paired seed-by-month moving-block bootstrap without circular wrap."""

    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 2 or candidate.shape != baseline.shape:
        raise ValueError("bootstrap arrays must share [seeds, months] shape")
    if (
        baseline.size == 0
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("bootstrap arrays must be finite and nonempty")
    seeds, months = baseline.shape
    if draws <= 0 or block_length <= 0 or block_length > months:
        raise ValueError("invalid bootstrap draws or noncircular block length")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB00757]))
    effects = np.empty(draws, dtype=np.float64)
    relative = np.empty(draws, dtype=np.float64)
    blocks_needed = math.ceil(months / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    for draw in range(draws):
        seed_indices = rng.integers(0, seeds, size=seeds)
        starts = rng.integers(
            0, months - block_length + 1, size=blocks_needed
        )
        month_indices = np.concatenate(
            [start + offsets for start in starts]
        )[:months]
        baseline_mean = float(baseline[seed_indices][:, month_indices].mean())
        candidate_mean = float(candidate[seed_indices][:, month_indices].mean())
        effects[draw] = baseline_mean - candidate_mean
        relative[draw] = effects[draw] / baseline_mean
    baseline_mean = float(baseline.mean())
    candidate_mean = float(candidate.mean())
    effect = baseline_mean - candidate_mean
    return {
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "improvement": effect,
        "ci95": np.quantile(effects, [0.025, 0.975]).astype(float).tolist(),
        "relative_improvement": effect / baseline_mean,
        "relative_ci95": np.quantile(relative, [0.025, 0.975]).astype(float).tolist(),
        "bootstrap": {
            "method": "paired_seed_and_non_circular_moving_month_blocks",
            "draws": int(draws),
            "seed": int(seed),
            "block_length": int(block_length),
            "ordered_calendar_month_clusters": True,
            "circular_wrap": False,
            "initialization_rows_resampled": True,
        },
    }


def _aggregate(
    units: Sequence[Mapping[str, Any]], run_spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    by_key = {(int(unit["seed"]), str(unit["policy"])): unit for unit in units}
    seeds = [int(seed) for seed in run_spec["seeds"]]
    first = by_key[(seeds[0], POLICIES[0])]
    month_ids = [int(value) for value in first["month_ids"]]
    if not month_ids:
        raise NaturalControllerIntegrityError("aggregate has no scored month clusters")
    for unit in units:
        if [int(value) for value in unit["month_ids"]] != month_ids:
            raise NaturalControllerIntegrityError("policies do not share month support")
    if run_spec["max_packets"] is None and any(
        int(block) > len(month_ids)
        for block in run_spec["bootstrap_block_lengths"]
    ):
        raise NaturalControllerIntegrityError(
            "a full-run month-block sensitivity exceeds scored month support"
        )

    def arrays(policy: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    by_key[(seed, policy)]["per_month"][str(month)][metric]
                    for month in month_ids
                ]
                for seed in seeds
            ],
            dtype=np.float64,
        )

    def contrast(baseline: str, candidate: str) -> dict[str, Any]:
        baseline_mse = arrays(baseline, "mse")
        candidate_mse = arrays(candidate, "mse")
        baseline_sse = arrays(baseline, "sse")
        candidate_sse = arrays(candidate, "sse")
        block_lengths = [int(value) for value in run_spec["bootstrap_block_lengths"]]
        primary = int(run_spec["primary_bootstrap_block_length"])
        month_sensitivity = {
            str(block): paired_non_circular_month_bootstrap(
                baseline_mse,
                candidate_mse,
                seed=int(run_spec["bootstrap_seed"]) + 37 * block,
                draws=int(run_spec["bootstrap_draws"]),
                block_length=min(block, len(month_ids)),
            )
            for block in block_lengths
        }
        sse_sensitivity = {
            str(block): paired_non_circular_month_bootstrap(
                baseline_sse,
                candidate_sse,
                seed=int(run_spec["bootstrap_seed"]) + 1009 + 37 * block,
                draws=int(run_spec["bootstrap_draws"]),
                block_length=min(block, len(month_ids)),
            )
            for block in block_lengths
        }
        for requested, summary in month_sensitivity.items():
            summary["bootstrap"]["requested_block_length"] = int(requested)
        for requested, summary in sse_sensitivity.items():
            summary["bootstrap"]["requested_block_length"] = int(requested)
        month_contrast = dict(month_sensitivity[str(primary)])
        month_contrast["sensitivity_by_block_length"] = month_sensitivity
        month_contrast["primary_block_length"] = primary
        sse_contrast = sse_sensitivity[str(primary)]
        month_count = len(month_ids)
        return {
            "baseline": baseline,
            "candidate": candidate,
            "positive_improvement_favors": candidate,
            "macro_month_prequential_mse": month_contrast,
            "cumulative_prequential_squared_error": {
                "baseline_mean": float(np.mean(np.sum(baseline_sse, axis=1))),
                "candidate_mean": float(np.mean(np.sum(candidate_sse, axis=1))),
                "improvement": float(sse_contrast["improvement"]) * month_count,
                "ci95": [float(value) * month_count for value in sse_contrast["ci95"]],
                "relative_improvement": float(sse_contrast["relative_improvement"]),
                "seed_effects": (
                    np.sum(baseline_sse - candidate_sse, axis=1).astype(float).tolist()
                ),
                "bootstrap": sse_contrast["bootstrap"],
                "primary_block_length": primary,
                "sensitivity_by_block_length": {
                    block: {
                        "improvement": float(summary["improvement"]) * month_count,
                        "ci95": [
                            float(value) * month_count for value in summary["ci95"]
                        ],
                        "relative_improvement": summary["relative_improvement"],
                        "bootstrap": summary["bootstrap"],
                    }
                    for block, summary in sse_sensitivity.items()
                },
            },
        }

    comparisons_vs_never = {
        policy: contrast("never", policy) for policy in POLICIES[1:]
    }
    controller_wedge = {
        "pairing_aware_vs_cosine": contrast("cosine", "pairing_aware"),
        "pairing_aware_vs_always": contrast("always", "pairing_aware"),
    }
    return comparisons_vs_never, controller_wedge, month_ids


def _aggregate_mean_drift(
    units: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    def mean_present(values: Sequence[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return _mean_or_none(present)

    def pooled_stratum(
        summaries: Sequence[Mapping[str, Any]], group: str, stratum: str
    ) -> dict[str, Any]:
        rows = [summary[group][stratum] for summary in summaries]
        total = sum(int(row["packets"]) for row in rows)
        result: dict[str, Any] = {"packets": total}
        for metric in ("mean_v_total", "mean_v_perm", "mean_v_pair"):
            weighted = [
                (int(row["packets"]), row[metric])
                for row in rows
                if int(row["packets"]) > 0 and row[metric] is not None
            ]
            result[metric] = (
                sum(count * float(value) for count, value in weighted)
                / sum(count for count, _ in weighted)
                if weighted
                else None
            )
        return result

    by_key = {(int(unit["seed"]), str(unit["policy"])): unit for unit in units}
    output: dict[str, Any] = {}
    for policy in POLICIES:
        summaries = [
            by_key[(int(seed), policy)]["natural_mean_drift_diagnostics"]
            for seed in seeds
        ]
        per_seed = [
            {
                "seed": int(seed),
                "mean_absolute_packet_target_mean_change": summary[
                    "mean_absolute_packet_target_mean_change"
                ],
                "fraction_full_cosine_positive_and_c_cosine_nonpositive": summary[
                    "fraction_full_cosine_positive_and_c_cosine_nonpositive"
                ],
                "pearson_absolute_mean_change_v_perm": summary[
                    "association_with_absolute_target_mean_change"
                ]["pearson_v_perm"],
                "pearson_absolute_mean_change_v_pair": summary[
                    "association_with_absolute_target_mean_change"
                ]["pearson_v_pair"],
            }
            for seed, summary in zip(seeds, summaries)
        ]
        output[policy] = {
            "labels_used_only_after_prequential_score": True,
            "seeds": [int(seed) for seed in seeds],
            "mean_absolute_packet_target_mean_change": float(
                np.mean(
                    [
                        summary["mean_absolute_packet_target_mean_change"]
                        for summary in summaries
                    ]
                )
            ),
            "mean_fraction_full_cosine_positive_and_c_cosine_nonpositive": float(
                np.mean(
                    [
                        summary[
                            "fraction_full_cosine_positive_and_c_cosine_nonpositive"
                        ]
                        for summary in summaries
                    ]
                )
            ),
            "mean_seed_pearson_absolute_mean_change_v_perm": mean_present(
                [
                    summary["association_with_absolute_target_mean_change"][
                        "pearson_v_perm"
                    ]
                    for summary in summaries
                ]
            ),
            "mean_seed_pearson_absolute_mean_change_v_pair": mean_present(
                [
                    summary["association_with_absolute_target_mean_change"][
                        "pearson_v_pair"
                    ]
                    for summary in summaries
                ]
            ),
            "v_by_alignment_stratum": {
                name: pooled_stratum(summaries, "v_by_alignment_stratum", name)
                for name in (
                    "full_positive_c_nonpositive",
                    "all_other_transitions",
                )
            },
            "v_by_absolute_target_mean_change_stratum": {
                "threshold_rule": "within_seed_median_over_scored_v_audited_transitions",
                **{
                    name: pooled_stratum(
                        summaries, "v_by_absolute_target_mean_change_stratum", name
                    )
                    for name in ("at_or_below_median", "above_median")
                },
            },
            "per_seed": per_seed,
        }
    return output


def run_campaign(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    source_binding: ValidatedEvidence | None = None,
    runtime_binding: ValidatedEvidence | None = None,
    data_binding: ValidatedEvidence | None = None,
    full_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    locked = validate_config(config)
    tier = "smoke" if smoke else "full"
    spec = locked[tier]
    source_evidence = evidence_payload(source_binding, kind="source")
    runtime_evidence = evidence_payload(runtime_binding, kind="runtime")
    data_evidence = evidence_payload(data_binding, kind="data")
    configuration_sha256 = _sha256_bytes(_canonical_json(locked))
    bindings_consistent = False
    if (
        source_evidence is not None
        and runtime_evidence is not None
        and data_evidence is not None
    ):
        expected_source = shared_source_identity(source_evidence)
        runtime_manifest = runtime_evidence.get("manifest", {})
        data_manifest = data_evidence.get("manifest", {})
        bindings_consistent = bool(
            runtime_manifest.get("source_identity") == expected_source
            and data_manifest.get("source_identity") == expected_source
            and runtime_manifest.get("configuration_payload_sha256")
            == configuration_sha256
            and data_manifest.get("configuration_payload_sha256")
            == configuration_sha256
            and runtime_manifest.get("observed")
            == observed_runtime(experiment=SOURCE_EXPERIMENT, config=locked)
        )
    evidence_eligible = bool(
        not smoke
        and source_evidence is not None
        and runtime_evidence is not None
        and data_evidence is not None
        and bindings_consistent
    )
    units: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    radius_schedules: dict[str, dict[str, Any]] = {}
    # The real loader is seed-independent.  Load and verify it once so every
    # initialization sees literally the same arrays and acquisition manifest.
    if not smoke and full_data is None:
        raise NaturalControllerIntegrityError(
            "full data must be supplied by the fail-closed data-manifest validator"
        )
    shared_full_data = None if smoke else full_data
    if shared_full_data is not None and data_evidence is not None:
        bound_preprocessing = data_evidence["manifest"]["preprocessing_identity"]
        if stream_identity(shared_full_data) != bound_preprocessing:
            raise NaturalControllerIntegrityError(
                "in-memory full stream differs from validated data binding"
            )
    for seed in spec["seeds"]:
        data = (
            load_workshop_stream("BEIJING", smoke=True, seed=int(seed))
            if smoke
            else shared_full_data
        )
        if data is None:  # defensive narrowing for static and runtime checks
            raise NaturalControllerIntegrityError("full data was not initialized")
        manifest = data.get("data_manifest")
        if smoke:
            if (
                not str(data.get("name", "")).startswith("SMOKE_")
                or not isinstance(manifest, Mapping)
                or manifest.get("source") != "deterministic_smoke"
            ):
                raise NaturalControllerIntegrityError(
                    "smoke loader did not return deterministic local data"
                )
        elif str(data.get("name")) != "BEIJING":
            raise NaturalControllerIntegrityError("full loader did not return BEIJING")
        expected_identity = stream_identity(data)["sha256"]
        identities[str(seed)] = expected_identity
        radius_schedule = build_common_radius_schedule(
            data,
            seed=int(seed),
            config=locked,
            max_packets=spec["max_packets"],
        )
        radius_schedules[str(seed)] = radius_schedule
        for policy in POLICIES:
            unit = run_policy(
                data,
                policy,
                seed=int(seed),
                config=locked,
                max_packets=spec["max_packets"],
                radius_schedule=radius_schedule,
                retain_packet_records=bool(spec["retain_packet_records"]),
            )
            if unit["stream_identity"]["sha256"] != expected_identity:
                raise NaturalControllerIntegrityError(
                    "policy data identity differs within a seed"
                )
            units.append(unit)

    comparisons, controller_wedge, month_ids = _aggregate(units, spec)
    mean_drift_evidence = _aggregate_mean_drift(units, spec["seeds"])
    max_v_residual = max(
        float(unit["v_decomposition"]["max_identity_residual"]) for unit in units
    )
    if any(unit["true_optimizer_updates"] != unit["packets"] for unit in units):
        raise NaturalControllerIntegrityError("optimizer update accounting failed")
    if any(unit["raw_examples_replayed"] != 0 for unit in units):
        raise NaturalControllerIntegrityError("raw replay detected")
    for seed in spec["seeds"]:
        observed = {
            unit["radius_diagnostics"]["schedule_sha256"]
            for unit in units
            if int(unit["seed"]) == int(seed)
        }
        if observed != {radius_schedules[str(seed)]["sha256"]}:
            raise NaturalControllerIntegrityError(
                "policies did not use the same locked radius schedule"
            )
    status = "SMOKE_PASS" if smoke else "PASS"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "mode": tier,
        "scientific_evidence_eligible": evidence_eligible,
        "warning": (
            "SMOKE OUTPUT: deterministic local engineering check; not scientific evidence."
            if smoke
            else (
                None
                if evidence_eligible
                else "INELIGIBLE FULL OUTPUT: validated source, runtime, and data bindings were not supplied."
            )
        ),
        "source_provenance": source_evidence,
        "runtime_provenance": runtime_evidence,
        "data_provenance": data_evidence,
        "configuration_sha256": configuration_sha256,
        "source_identity": locked["source_identity"],
        "run_spec": spec,
        "dataset": "deterministic_local_synthetic" if smoke else "BEIJING",
        "stream_identities_by_seed": identities,
        "common_radius_schedules_by_seed": radius_schedules,
        "policies": list(POLICIES),
        "month_ids": month_ids,
        "units": units,
        "comparisons_vs_never": comparisons,
        "controller_wedge_comparisons": controller_wedge,
        "natural_mean_drift_evidence_by_policy": mean_drift_evidence,
        "integrity": {
            "strictly_prequential": True,
            "gate_after_scoring": True,
            "action_first_affects_next_packet": True,
            "one_true_adamw_update_per_packet": True,
            "equal_radius_for_open_reuse_actions": True,
            "common_never_reuse_radius_schedule_across_policies": True,
            "no_raw_replay": True,
            "same_seed_and_data_order_across_policies": True,
            "v_total_equals_v_perm_plus_v_pair": max_v_residual <= 1.0e-12,
            "v_shadows_match_post_current_update_action_timing": all(
                bool(unit["v_shadow_matches_post_current_update_action_timing"])
                for unit in units
            ),
            "max_v_identity_residual": max_v_residual,
            "locked_source_identity_verified": True,
            "validated_source_binding": source_evidence is not None,
            "validated_runtime_binding": runtime_evidence is not None,
            "validated_data_binding": data_evidence is not None,
            "provenance_bindings_consistent": bindings_consistent,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked natural-stream pairing-aware controller"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SHARED_SOURCE_MANIFEST,
        help="shared source manifest; mandatory and fail-closed for --full",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help="captured runtime manifest; mandatory and fail-closed for --full",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="frozen Beijing data manifest; mandatory and fail-closed for --full",
    )
    parser.add_argument(
        "--data-archive",
        type=Path,
        help="exact local Beijing archive authenticated by --data-manifest",
    )
    tier = parser.add_mutually_exclusive_group(required=True)
    tier.add_argument("--smoke", action="store_true")
    tier.add_argument("--full", action="store_true")
    tier.add_argument(
        "--capture-runtime",
        action="store_true",
        help="freeze the current runtime to --output without running the campaign",
    )
    tier.add_argument(
        "--freeze-data",
        action="store_true",
        help="freeze a local Beijing archive to --output; never a scientific run",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.overwrite and not arguments.smoke:
        raise RuntimeError("--overwrite is permitted only for ineligible smoke output")
    require_disjoint_paths(
        {
            "output": arguments.output,
            "configuration": arguments.config,
            "source_manifest": arguments.source_manifest,
            "runtime_manifest": arguments.runtime_manifest,
            "data_manifest": arguments.data_manifest,
            "data_archive": arguments.data_archive,
        }
    )
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    config = load_config(arguments.config)
    source_evidence = _shared_source_evidence(
        arguments.source_manifest,
        arguments.config,
        smoke=bool(arguments.smoke),
    )
    if arguments.capture_runtime:
        if any(
            value is not None
            for value in (
                arguments.runtime_manifest,
                arguments.data_manifest,
                arguments.data_archive,
            )
        ):
            raise RuntimeError(
                "--capture-runtime writes to --output and accepts no data/runtime input"
            )
        result = build_runtime_manifest(
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
        write_fresh_json(arguments.output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "scientific_evidence_eligible": False,
                    "output": str(arguments.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.freeze_data:
        if arguments.data_archive is None:
            raise RuntimeError("--freeze-data requires --data-archive")
        if arguments.runtime_manifest is not None or arguments.data_manifest is not None:
            raise RuntimeError(
                "--freeze-data writes to --output; omit --runtime-manifest/--data-manifest"
            )
        result = build_beijing_data_manifest(
            archive_path=arguments.data_archive,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
        write_fresh_json(arguments.output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "scientific_evidence_eligible": False,
                    "output": str(arguments.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0

    source_binding = (
        validate_source_binding(
            arguments.source_manifest,
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            repo=REPO,
        )
        if source_evidence["validated_for_execution"]
        else None
    )
    runtime_binding = None
    data_binding = None
    full_data = None
    if arguments.full:
        if arguments.runtime_manifest is None:
            raise RuntimeError("--full requires --runtime-manifest")
        if arguments.data_manifest is None or arguments.data_archive is None:
            raise RuntimeError("--full requires --data-manifest and --data-archive")
        runtime_binding = validate_runtime_manifest(
            arguments.runtime_manifest,
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
        data_binding, full_data = validate_beijing_data_manifest(
            arguments.data_manifest,
            archive_path=arguments.data_archive,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
    elif any(
        value is not None
        for value in (
            arguments.runtime_manifest,
            arguments.data_manifest,
            arguments.data_archive,
        )
    ):
        raise RuntimeError("runtime and data bindings are only accepted with --full")

    result = run_campaign(
        config,
        smoke=bool(arguments.smoke),
        source_binding=source_binding,
        runtime_binding=runtime_binding,
        data_binding=data_binding,
        full_data=full_data,
    )
    result["source_validation_attempt"] = source_evidence
    result.setdefault("integrity", {})[
        "shared_source_manifest_validated"
    ] = bool(source_evidence["validated_for_execution"])
    if arguments.full and not result["scientific_evidence_eligible"]:
        raise RuntimeError("validated full natural-controller result was marked ineligible")
    if arguments.overwrite:
        arguments.output.unlink(missing_ok=True)
    write_fresh_json(arguments.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "scientific_evidence_eligible": result[
                    "scientific_evidence_eligible"
                ],
                "shared_source_manifest_validated": source_evidence["validated_for_execution"],
                "runtime_manifest_validated": runtime_binding is not None,
                "data_manifest_validated": data_binding is not None,
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
