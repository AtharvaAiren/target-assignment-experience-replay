"""E7: one authenticated Pythia audit on a disjoint historical news stream.

The completed Transformer campaign is immutable.  This extension rebuilds the
exact post-2021 parent stream from the same pinned corpus, authenticates it
against the completed result, and separately selects the latest 12,288
tokenizable eligible articles strictly before 2021-01-01 after excluding every
parent ID and article-text hash.  The first 4,096 historical examples are
warm-up and the remaining 8,192 are audited online.  Parent and historical
streams are temporal-generalization evaluations, never corpus replications.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from experiments.gradient_reuse_reviewer_extension.protocol import (
    DEFAULT_LOCK,
    load_lock,
    sha256_file,
    strict_json,
    validate_parent_completion,
)
from experiments.gradient_reuse_reviewer_extension.source import (
    validate_source_manifest,
)
from experiments.gradient_reuse_transformer.assets import validate_asset_manifest
from experiments.gradient_reuse_transformer.audit import seed_everything
from experiments.gradient_reuse_transformer.data import (
    StreamBuild,
    StreamExample,
    _deterministic_offset,
    _is_allowed_license,
    _parse_timestamp,
    _stable_identity,
    _token_ids,
    build_chronological_stream,
    iter_json_records,
)
from experiments.gradient_reuse_transformer.model import load_local_pythia
from experiments.gradient_reuse_transformer.run import (
    ACTUATOR_NAME,
    _artifact_inventory,
    _canonical_sha256,
    _reject_artifacts_inside_asset_roots,
    _run_unit,
    _validate_runtime,
    capture_runtime_environment,
    read_config,
    validate_artifact_inventory,
    validate_embedded_input_bundle,
)


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "gradient_reuse.reviewer_extension.transformer_stream2.v1"
STREAM_SCHEMA = "gradient_reuse.reviewer_extension.transformer_stream.v1"
HISTORICAL_BUILD_SCHEMA = "gradient_reuse.reviewer_extension.historical_stream_build.v1"
HISTORICAL_SELECTION = (
    "latest_required_eligible_articles_strictly_before_parent_cutoff_"
    "after_parent_id_and_text_exclusion"
)
SEPARATE_STREAM_REPORTING = "separate_historical_stream_not_a_replication"
HEARTBEAT_SCHEMA = "gradient_reuse.reviewer_extension.transformer_stream2_heartbeat.v1"
SOURCE_BINDING_SCHEMA = (
    "gradient_reuse.reviewer_extension.transformer_source_binding.v1"
)
ENVIRONMENT_BINDING_SCHEMA = (
    "gradient_reuse.reviewer_extension.transformer_environment_reuse.v1"
)
PARENT_BINDING_SCHEMA = (
    "gradient_reuse.reviewer_extension.transformer_parent_binding.v1"
)
INPUT_BUNDLE_SCHEMA = "gradient_reuse.reviewer_extension.transformer_input_bundle.v1"
CORRECTION_PROVENANCE_SCHEMA = (
    "gradient_reuse.reviewer_extension.transformer_correction_provenance.v1"
)
PORTABLE_CORRECTION_PROVENANCE_SCHEMA = (
    "gradient_reuse.reviewer_extension.transformer_correction_provenance.v2"
)
PORTABLE_HISTORICAL_RELATION = (
    "documented_rules_reimplemented; historical_patch_bytes_absent; "
    "semantic_equivalence_not_byte_verified"
)
EXPECTED_CORRECTION_MANIFEST_SHA256 = (
    "a5911ac2a37642a4a6b16f9373ec290ff1e85a75ac05bdb3f2aa24422f99afee"
)
EXPECTED_CORRECTION_PATCH_SHA256 = (
    "e0ca3cb07f89e276090675053553cb6d947ccb75212494e9f9a1c9426d834ec1"
)
EXPECTED_PADDED_VOCAB_MANIFEST_SHA256 = (
    "223c8af59a04b636ced07da012b20fca8a1128e8bc07db35018099496b140bd4"
)
EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256 = (
    "677999e4f2fc02554ff89ee9c6de7f3659ee3ef8618a634111d84c1d35a26510"
)
PORTABLE_CORRECTION_MANIFEST = (
    REPO / "provenance/transformer_correction_portable.json"
)
PORTABLE_CORRECTION_PATCH = (
    REPO
    / "experiments/gradient_reuse_reviewer_extension/transformer_correction.py"
)
PORTABLE_CORRECTION_HISTORY = (
    REPO / "provenance/transformer_correction_history.json"
)
EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256 = (
    "2507ab7393194ff521bdfc13e869b2a5079ead25cc06ba743422c2bed97e0bc7"
)
EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256 = (
    "0d1b13ee2f96bab5933d443d49b60235a9808f4aaf20cbb6d2534ccbc024c810"
)
EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256 = (
    "1785f8a388e7e083a6d539795bf6773bba3d73cfb188aacbd9dafde7947f6416"
)
E7_INPUT_ROLES = frozenset(
    {
        "reviewer_extension_lock",
        "reviewer_extension_source_manifest",
        "transformer_configuration",
        "transformer_asset_manifest",
        "parent_completion",
        "parent_transformer_result",
        "transformer_correction_manifest",
        "transformer_correction_patch",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_CORRECTION_PROVENANCE: dict[str, Any] | None = None


def _require_regular_file(path: Path, name: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"{name} must be a regular nonsymbolic file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{name} must be a regular nonsymbolic file: {resolved}")
    return resolved


def build_e7_input_bundle(files: Mapping[str, Path]) -> dict[str, Any]:
    """Bind exact invocation bytes without embedding path-bearing payloads."""

    if set(files) != E7_INPUT_ROLES:
        raise RuntimeError("E7 invocation input roles differ from the locked set")
    inventory: dict[str, dict[str, Any]] = {}
    for role, raw_path in sorted(files.items()):
        path = _require_regular_file(raw_path, role)
        size = int(path.stat().st_size)
        if size < 1:
            raise RuntimeError(f"E7 invocation input is empty: {role}")
        inventory[role] = {
            "file_sha256": sha256_file(path),
            "size_bytes": size,
        }
    return {
        "schema_version": INPUT_BUNDLE_SCHEMA,
        "input_roles": sorted(E7_INPUT_ROLES),
        "inventory": inventory,
        "bundle_sha256": _canonical_sha256(inventory),
        "absolute_paths_embedded": False,
    }


def validate_e7_input_bundle(
    bundle: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    if set(bundle) != {
        "schema_version",
        "input_roles",
        "inventory",
        "bundle_sha256",
        "absolute_paths_embedded",
    }:
        raise RuntimeError("E7 input-bundle keys differ from schema")
    inventory = bundle.get("inventory")
    if not (
        bundle.get("schema_version") == INPUT_BUNDLE_SCHEMA
        and bundle.get("input_roles") == sorted(E7_INPUT_ROLES)
        and bundle.get("absolute_paths_embedded") is False
        and isinstance(inventory, dict)
        and set(inventory) == E7_INPUT_ROLES
        and bundle.get("bundle_sha256") == _canonical_sha256(inventory)
    ):
        raise RuntimeError("E7 input bundle failed authentication")
    for role, record in inventory.items():
        if not isinstance(record, dict) or set(record) != {
            "file_sha256",
            "size_bytes",
        }:
            raise RuntimeError(f"E7 input inventory changed: {role}")
        if _SHA256.fullmatch(str(record.get("file_sha256", ""))) is None:
            raise RuntimeError(f"E7 input hash is invalid: {role}")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise RuntimeError(f"E7 input size is invalid: {role}")
    if expected is not None and dict(bundle) != dict(expected):
        raise RuntimeError("E7 result does not embed the live invocation inputs")


def authenticate_correction_files(
    manifest_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    """Authenticate legacy bytes or the bundled portable reimplementation.

    Legacy bytes retain the execution-time v1 record used by the frozen E7
    result. Portable bytes receive a v2 record that separates what executed
    from the author-recorded historical digest chain. The latter is not treated
    as a proof of byte or semantic equivalence.
    """

    manifest_path = _require_regular_file(
        manifest_path, "Transformer correction manifest"
    )
    patch_path = _require_regular_file(patch_path, "Transformer correction patch")
    manifest_sha256 = sha256_file(manifest_path)
    patch_sha256 = sha256_file(patch_path)
    manifest = strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("Transformer correction manifest is not an object")
    correction_files = manifest.get("correction_files")
    prior = manifest.get("prior_correction_manifest")
    legacy = (
        manifest_sha256 == EXPECTED_CORRECTION_MANIFEST_SHA256
        and patch_sha256 == EXPECTED_CORRECTION_PATCH_SHA256
        and manifest.get("schema_version")
        == "gradient_reuse.external_transformer_effective_vocab_correction.v1"
        and manifest.get("status") == "DOCUMENTED_POST_FAILURE_BEFORE_RETRY"
        and isinstance(correction_files, dict)
        and correction_files.get("apply_transformer_effective_vocab_correction.py")
        == patch_sha256
        and isinstance(prior, dict)
        and prior.get("sha256") == EXPECTED_PADDED_VOCAB_MANIFEST_SHA256
    )

    portable = False
    if (
        manifest_sha256 == EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256
        and patch_sha256 == EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256
    ):
        implementation = manifest.get("implementation")
        bridge_record = manifest.get("historical_bridge")
        bridge_path = _require_regular_file(
            PORTABLE_CORRECTION_HISTORY, "portable Transformer history bridge"
        )
        bridge = strict_json(bridge_path)
        expected_history = [
            {
                "manifest_sha256": EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256,
                "patch_sha256": (
                    "6f750c55e020651034d0e8fe9b080f59225abb69f1fdf02144a3239a24132637"
                ),
                "role": "classifier_dropout_compatibility",
                "schema_version": (
                    "gradient_reuse.external_transformer_"
                    "classifier_dropout_correction.v1"
                ),
            },
            {
                "manifest_sha256": EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
                "patch_sha256": (
                    "9c74a3140f06a5e2252466eeeb15f41ae4c50ffc2ed02f775b989afc491e209a"
                ),
                "role": "padded_vocabulary_compatibility",
                "schema_version": (
                    "gradient_reuse.external_transformer_"
                    "padded_vocab_correction.v1"
                ),
            },
            {
                "manifest_sha256": EXPECTED_CORRECTION_MANIFEST_SHA256,
                "patch_sha256": EXPECTED_CORRECTION_PATCH_SHA256,
                "role": "effective_vocabulary_compatibility",
                "schema_version": (
                    "gradient_reuse.external_transformer_"
                    "effective_vocab_correction.v1"
                ),
            },
        ]
        portable = (
            manifest.get("schema_version")
            == "gradient_reuse.portable_transformer_correction.v1"
            and manifest.get("status")
            == "PORTABLE_DOCUMENTED_RULE_REIMPLEMENTATION"
            and manifest.get("no_private_paths") is True
            and manifest.get("historical_patch_bytes_distributed") is False
            and manifest.get("semantic_equivalence_verified") is False
            and implementation
            == {
                "path": (
                    "experiments/gradient_reuse_reviewer_extension/"
                    "transformer_correction.py"
                ),
                "sha256": EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256,
            }
            and bridge_record
            == {
                "path": "provenance/transformer_correction_history.json",
                "sha256": EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256,
            }
            and sha256_file(bridge_path)
            == EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256
            and bridge.get("schema_version")
            == "gradient_reuse.transformer_correction_history_bridge.v1"
            and bridge.get("status") == "AUTHOR_RECORDED_HISTORICAL_DIGESTS"
            and bridge.get("historical_chain") == expected_history
            and bridge.get("historical_digest_provenance")
            == "author_recorded_from_private_execution_artifacts"
            and bridge.get("historical_patch_bytes_distributed") is False
            and bridge.get("portable_source_is_byte_identical_to_historical_patch")
            is False
            and bridge.get("private_execution_paths_distributed") is False
            and bridge.get("semantic_equivalence_verified") is False
        )

    if not (legacy or portable):
        raise RuntimeError("Transformer correction chain failed authentication")
    historical_manifest_chain = [
        EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256,
        EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
        EXPECTED_CORRECTION_MANIFEST_SHA256,
    ]
    if legacy:
        return {
            "schema_version": CORRECTION_PROVENANCE_SCHEMA,
            "status": "AUTHENTICATED",
            "effective_vocab_manifest_sha256": EXPECTED_CORRECTION_MANIFEST_SHA256,
            "effective_vocab_patch_sha256": EXPECTED_CORRECTION_PATCH_SHA256,
            "padded_vocab_manifest_sha256": EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
            "classifier_dropout_manifest_sha256": (
                EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256
            ),
            "manifest_chain_sha256": _canonical_sha256(
                historical_manifest_chain
            ),
            "effective_patch_commits_composed_chain": True,
        }
    return {
        "schema_version": PORTABLE_CORRECTION_PROVENANCE_SCHEMA,
        "status": "PORTABLE_BYTES_AUTHENTICATED",
        "execution_kind": "portable_documented_rule_reimplementation",
        "executed_manifest_sha256": manifest_sha256,
        "executed_patch_sha256": patch_sha256,
        "historical_effective_vocab_manifest_sha256": (
            EXPECTED_CORRECTION_MANIFEST_SHA256
        ),
        "historical_effective_vocab_patch_sha256": EXPECTED_CORRECTION_PATCH_SHA256,
        "historical_padded_vocab_manifest_sha256": (
            EXPECTED_PADDED_VOCAB_MANIFEST_SHA256
        ),
        "historical_classifier_dropout_manifest_sha256": (
            EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256
        ),
        "historical_manifest_chain_sha256": _canonical_sha256(
            historical_manifest_chain
        ),
        "historical_relation": PORTABLE_HISTORICAL_RELATION,
        "historical_patch_bytes_distributed": False,
        "semantic_equivalence_verified": False,
    }


def activated_correction_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact provenance record installed by the bootstrap."""

    legacy_keys = {
        "schema_version",
        "status",
        "effective_vocab_manifest_sha256",
        "effective_vocab_patch_sha256",
        "padded_vocab_manifest_sha256",
        "classifier_dropout_manifest_sha256",
        "manifest_chain_sha256",
        "effective_patch_commits_composed_chain",
    }
    legacy = set(provenance) == legacy_keys and (
        provenance.get("schema_version") == CORRECTION_PROVENANCE_SCHEMA
        and provenance.get("status") == "AUTHENTICATED"
        and provenance.get("effective_vocab_manifest_sha256")
        == EXPECTED_CORRECTION_MANIFEST_SHA256
        and provenance.get("effective_vocab_patch_sha256")
        == EXPECTED_CORRECTION_PATCH_SHA256
        and provenance.get("padded_vocab_manifest_sha256")
        == EXPECTED_PADDED_VOCAB_MANIFEST_SHA256
        and provenance.get("classifier_dropout_manifest_sha256")
        == EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256
        and provenance.get("manifest_chain_sha256")
        == _canonical_sha256(
            [
                EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256,
                EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
                EXPECTED_CORRECTION_MANIFEST_SHA256,
            ]
        )
        and provenance.get("effective_patch_commits_composed_chain") is True
    )
    portable_keys = {
        "schema_version",
        "status",
        "execution_kind",
        "executed_manifest_sha256",
        "executed_patch_sha256",
        "historical_effective_vocab_manifest_sha256",
        "historical_effective_vocab_patch_sha256",
        "historical_padded_vocab_manifest_sha256",
        "historical_classifier_dropout_manifest_sha256",
        "historical_manifest_chain_sha256",
        "historical_relation",
        "historical_patch_bytes_distributed",
        "semantic_equivalence_verified",
    }
    portable = set(provenance) == portable_keys and (
        provenance.get("schema_version") == PORTABLE_CORRECTION_PROVENANCE_SCHEMA
        and provenance.get("status") == "PORTABLE_BYTES_AUTHENTICATED"
        and provenance.get("execution_kind")
        == "portable_documented_rule_reimplementation"
        and provenance.get("executed_manifest_sha256")
        == EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256
        and provenance.get("executed_patch_sha256")
        == EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256
        and provenance.get("historical_effective_vocab_manifest_sha256")
        == EXPECTED_CORRECTION_MANIFEST_SHA256
        and provenance.get("historical_effective_vocab_patch_sha256")
        == EXPECTED_CORRECTION_PATCH_SHA256
        and provenance.get("historical_padded_vocab_manifest_sha256")
        == EXPECTED_PADDED_VOCAB_MANIFEST_SHA256
        and provenance.get("historical_classifier_dropout_manifest_sha256")
        == EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256
        and provenance.get("historical_manifest_chain_sha256")
        == _canonical_sha256(
            [
                EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256,
                EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
                EXPECTED_CORRECTION_MANIFEST_SHA256,
            ]
        )
        and provenance.get("historical_relation") == PORTABLE_HISTORICAL_RELATION
        and provenance.get("historical_patch_bytes_distributed") is False
        and provenance.get("semantic_equivalence_verified") is False
    )
    if not (legacy or portable):
        raise RuntimeError("invalid Transformer correction activation provenance")
    activated = dict(provenance)
    activated["runtime_patch_installed_before_project_execution"] = True
    activated["activation_sha256"] = _canonical_sha256(provenance)
    return activated


def activate_correction_sentinel(provenance: Mapping[str, Any]) -> None:
    """Called by the authenticated bootstrap only after executing the patch."""

    global _ACTIVE_CORRECTION_PROVENANCE
    _ACTIVE_CORRECTION_PROVENANCE = activated_correction_provenance(provenance)


def require_active_correction(
    manifest_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    authenticated = authenticate_correction_files(manifest_path, patch_path)
    expected = activated_correction_provenance(authenticated)
    if _ACTIVE_CORRECTION_PROVENANCE != expected:
        raise RuntimeError(
            "authenticated Transformer correction patch is not active in this process"
        )
    return expected


def validate_activated_correction_provenance(
    correction: Mapping[str, Any],
) -> None:
    """Validate either frozen legacy-v1 or fresh portable-v2 activation."""

    if not isinstance(correction, dict):
        raise RuntimeError("E7 Transformer correction provenance is invalid")
    runtime_key = "runtime_patch_installed_before_project_execution"
    activation_key = "activation_sha256"
    if correction.get(runtime_key) is not True or activation_key not in correction:
        raise RuntimeError("E7 Transformer correction provenance is invalid")
    unsigned = {
        key: value
        for key, value in correction.items()
        if key not in {runtime_key, activation_key}
    }
    expected = activated_correction_provenance(unsigned)
    if dict(correction) != expected:
        raise RuntimeError("E7 Transformer correction provenance is invalid")


def _stream_identity_payload(
    examples: Sequence[StreamExample],
) -> list[dict[str, Any]]:
    """Use the exact identity projection used by ``build_chronological_stream``."""

    return [
        {
            "id": item.example_id,
            "timestamp": item.timestamp_utc,
            "context": list(item.input_ids),
            "target": item.target_id,
            "start": item.window_start,
            "target_position": item.target_position,
        }
        for item in examples
    ]


def stream_identity_sha256(examples: Sequence[StreamExample]) -> str:
    return _canonical_sha256(_stream_identity_payload(examples))


def _set_sha256(values: Sequence[str] | set[str]) -> str:
    return _canonical_sha256(sorted(values))


def post_parent_candidate_upper_bound(parent_stream: Mapping[str, Any]) -> int:
    """Upper-bound usable records after the completed parent stream.

    The parent builder filters the complete asset before tokenizing candidates
    in chronological order.  Consequently, filtered candidates not yet
    tokenized equal the candidate count minus accepted and observed-too-short
    records.  They are only an upper bound on a second stream because those
    unexamined records can themselves be too short.
    """

    integer_fields = (
        "records_seen",
        "rejected_timestamp",
        "rejected_license",
        "rejected_duplicate",
        "rejected_empty",
        "rejected_too_short_after_tokenization",
        "warmup_examples",
        "online_examples",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = parent_stream.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"parent stream count is invalid: {field}")
        values[field] = value
    filtered_candidates = values["records_seen"] - sum(
        values[field]
        for field in (
            "rejected_timestamp",
            "rejected_license",
            "rejected_duplicate",
            "rejected_empty",
        )
    )
    examined = (
        values["warmup_examples"]
        + values["online_examples"]
        + values["rejected_too_short_after_tokenization"]
    )
    if filtered_candidates < examined:
        raise RuntimeError("parent stream filtering counts are internally inconsistent")
    return filtered_candidates - examined


def authenticate_rebuilt_parent_stream(
    rebuilt: StreamBuild,
    *,
    parent_stream: Mapping[str, Any],
    historical_cutoff_utc: str,
) -> tuple[StreamExample, ...]:
    """Prove that a fresh asset scan exactly reproduces the parent stream."""

    examples = tuple((*rebuilt.warmup, *rebuilt.online))
    if not examples:
        raise RuntimeError("rebuilt parent stream is empty")
    if dict(rebuilt.manifest) != dict(parent_stream):
        raise RuntimeError(
            "rebuilt parent stream manifest differs from the completed parent"
        )
    rebuilt_hash = stream_identity_sha256(examples)
    if rebuilt_hash != rebuilt.manifest.get("stream_sha256"):
        raise RuntimeError("rebuilt parent stream does not authenticate its examples")
    cutoff = _parse_timestamp(historical_cutoff_utc)
    if any(_parse_timestamp(item.timestamp_utc) < cutoff for item in examples):
        raise RuntimeError("rebuilt parent stream contains a pre-cutoff article")
    ids = {item.example_id for item in examples}
    texts = {item.text_sha256 for item in examples}
    if len(ids) != len(examples) or len(texts) != len(examples):
        raise RuntimeError("rebuilt parent stream is not ID/text unique")
    return examples


def build_historical_stream(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    context_length: int,
    warmup_examples: int,
    online_examples: int,
    historical_cutoff_utc: str,
    offset_salt: str,
    excluded_example_ids: set[str],
    excluded_text_sha256: set[str],
) -> StreamBuild:
    """Select the latest eligible tokenizable articles before the cutoff.

    This deliberately reuses the parent data module's private timestamp,
    license, stable-identity, tokenization, and deterministic-window
    primitives.  Selection observes no model state, seed, loss, gradient, or
    downstream outcome.
    """

    context = int(context_length)
    warmup = int(warmup_examples)
    online = int(online_examples)
    required = warmup + online
    if context < 2 or min(warmup, online) < 1:
        raise ValueError("invalid historical stream sizes")
    cutoff = _parse_timestamp(historical_cutoff_utc)
    cutoff_utc = cutoff.isoformat()
    parent_ids = set(excluded_example_ids)
    parent_text = set(excluded_text_sha256)
    if not parent_ids or not parent_text:
        raise RuntimeError("historical stream requires nonempty parent exclusions")
    if len(parent_ids) != len(excluded_example_ids) or len(parent_text) != len(
        excluded_text_sha256
    ):
        raise RuntimeError("parent exclusion identities are not unique")

    # Keep only the latest ``required`` tokenizable examples.  Holding every
    # eligible article body until a final sort can consume multiple GiB on the
    # pinned news corpus; the bounded min-heap implements the identical
    # outcome-independent selection rule in O(required) retained payloads.
    latest: list[tuple[datetime, str, str, StreamExample]] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    counts = {
        "records_seen": 0,
        "rejected_timestamp": 0,
        "rejected_unparseable_timestamp": 0,
        "rejected_at_or_after_cutoff": 0,
        "rejected_license": 0,
        "rejected_duplicate": 0,
        "rejected_empty": 0,
        "rejected_parent_example_id": 0,
        "rejected_parent_article_text": 0,
        "eligible_candidates_before_tokenization": 0,
        "rejected_too_short_after_tokenization": 0,
    }
    for record in records:
        counts["records_seen"] += 1
        try:
            timestamp = _parse_timestamp(record.get("created"))
        except ValueError:
            counts["rejected_timestamp"] += 1
            counts["rejected_unparseable_timestamp"] += 1
            continue
        if timestamp >= cutoff:
            counts["rejected_timestamp"] += 1
            counts["rejected_at_or_after_cutoff"] += 1
            continue
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if not _is_allowed_license(metadata.get("license")):
            counts["rejected_license"] += 1
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            counts["rejected_empty"] += 1
            continue
        stable_id = _stable_identity(record)
        example_id = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if stable_id in seen_ids or text_hash in seen_text:
            counts["rejected_duplicate"] += 1
            continue
        seen_ids.add(stable_id)
        seen_text.add(text_hash)
        if example_id in parent_ids:
            counts["rejected_parent_example_id"] += 1
            continue
        if text_hash in parent_text:
            counts["rejected_parent_article_text"] += 1
            continue
        counts["eligible_candidates_before_tokenization"] += 1
        tokens = _token_ids(tokenizer, text)
        choices = len(tokens) - context
        if choices <= 0:
            counts["rejected_too_short_after_tokenization"] += 1
            continue
        start = _deterministic_offset(
            stable_id=stable_id,
            choices=choices,
            salt=str(offset_salt),
        )
        stop = start + context
        conditioning = tuple(tokens[start:stop])
        if len(conditioning) != context or stop >= len(tokens):
            raise RuntimeError("historical next-token window construction failed")
        example = StreamExample(
            example_id=example_id,
            timestamp_utc=timestamp.isoformat(),
            source=str(record.get("source", "")),
            input_ids=conditioning,
            target_id=int(tokens[stop]),
            window_start=start,
            target_position=stop,
            text_sha256=text_hash,
        )
        entry = (timestamp, example.source, stable_id, example)
        if len(latest) < required:
            heapq.heappush(latest, entry)
        elif entry[:3] > latest[0][:3]:
            heapq.heapreplace(latest, entry)
    if len(latest) != required:
        raise RuntimeError(
            f"only {len(latest)} eligible tokenized historical "
            f"articles; {required} required"
        )
    examples = tuple(entry[3] for entry in sorted(latest, key=lambda row: row[:3]))
    stream_hash = stream_identity_sha256(examples)
    manifest = {
        "schema_version": HISTORICAL_BUILD_SCHEMA,
        "selection": HISTORICAL_SELECTION,
        **counts,
        "tokenized_candidates_examined": counts[
            "eligible_candidates_before_tokenization"
        ],
        "historical_cutoff_utc": cutoff_utc,
        "context_length": context,
        "warmup_examples": warmup,
        "online_examples": online,
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "selection_depends_on_model_or_outcomes": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": all(
            item.target_position == item.window_start + context for item in examples
        ),
        "excluded_parent_example_id_set_sha256": _set_sha256(parent_ids),
        "excluded_parent_text_sha256_set_sha256": _set_sha256(parent_text),
        "stream_sha256": stream_hash,
        "first_timestamp_utc": examples[0].timestamp_utc,
        "last_timestamp_utc": examples[-1].timestamp_utc,
    }
    return StreamBuild(
        warmup=examples[:warmup],
        online=examples[warmup:],
        manifest=manifest,
    )


def select_historical_stream(
    historical: StreamBuild,
    rebuilt_parent: StreamBuild,
    *,
    parent_stream: Mapping[str, Any],
    historical_cutoff_utc: str,
    required_examples: int,
    warmup_examples: int,
    online_examples: int,
    packet_size: int,
) -> StreamBuild:
    """Authenticate and bind separately built parent and historical streams."""

    required = int(required_examples)
    warmup = int(warmup_examples)
    online = int(online_examples)
    packet = int(packet_size)
    if min(required, warmup, online, packet) < 1:
        raise RuntimeError("historical-stream dimensions must be positive")
    if warmup + online != required:
        raise RuntimeError("historical warm-up and online counts changed")
    if warmup % packet or online % packet:
        raise RuntimeError("historical partitions must be packet divisible")

    parent = authenticate_rebuilt_parent_stream(
        rebuilt_parent,
        parent_stream=parent_stream,
        historical_cutoff_utc=historical_cutoff_utc,
    )
    selected = tuple((*historical.warmup, *historical.online))
    if len(historical.warmup) != warmup or len(historical.online) != online:
        raise RuntimeError("historical stream was not built with the locked partition")
    if len(selected) != required:
        raise RuntimeError("historical stream has the wrong number of examples")
    if stream_identity_sha256(selected) != historical.manifest.get("stream_sha256"):
        raise RuntimeError("historical manifest does not authenticate its examples")
    if historical.manifest.get("selection") != HISTORICAL_SELECTION:
        raise RuntimeError("historical selection rule changed")

    parent_ids = {item.example_id for item in parent}
    selected_ids = {item.example_id for item in selected}
    parent_text = {item.text_sha256 for item in parent}
    selected_text = {item.text_sha256 for item in selected}
    parent_timestamps = {item.timestamp_utc for item in parent}
    selected_timestamps = {item.timestamp_utc for item in selected}
    if len(parent_ids) != len(parent) or len(selected_ids) != required:
        raise RuntimeError("stream example IDs are not unique within each stream")
    if len(parent_text) != len(parent) or len(selected_text) != required:
        raise RuntimeError("stream article contents are not unique within each stream")
    id_overlap = parent_ids & selected_ids
    text_overlap = parent_text & selected_text
    timestamp_overlap = parent_timestamps & selected_timestamps
    if id_overlap or text_overlap or timestamp_overlap:
        raise RuntimeError("historical Pythia stream overlaps its parent stream")
    cutoff = _parse_timestamp(historical_cutoff_utc)
    if any(_parse_timestamp(item.timestamp_utc) >= cutoff for item in selected):
        raise RuntimeError("historical stream is not strictly before the cutoff")
    if any(_parse_timestamp(item.timestamp_utc) < cutoff for item in parent):
        raise RuntimeError("rebuilt parent stream is not at or after the cutoff")
    if _parse_timestamp(selected[-1].timestamp_utc) >= _parse_timestamp(
        parent[0].timestamp_utc
    ):
        raise RuntimeError("historical and parent streams are not temporally separated")
    parent_hash = stream_identity_sha256(parent)
    stream_hash = stream_identity_sha256(selected)
    if stream_hash == parent_hash:
        raise RuntimeError("parent and historical streams have the same identity hash")
    if historical.manifest.get("excluded_parent_example_id_set_sha256") != _set_sha256(
        parent_ids
    ) or historical.manifest.get(
        "excluded_parent_text_sha256_set_sha256"
    ) != _set_sha256(
        parent_text
    ):
        raise RuntimeError("historical scan used different parent exclusions")

    remaining_upper_bound = post_parent_candidate_upper_bound(parent_stream)
    stream_manifest = {
        "schema_version": STREAM_SCHEMA,
        "selection": HISTORICAL_SELECTION,
        "comparison_reporting": SEPARATE_STREAM_REPORTING,
        "historical_cutoff_utc": cutoff.isoformat(),
        "required_examples": required,
        "warmup_examples": warmup,
        "online_examples": online,
        "packet_size": packet,
        "context_length": int(historical.manifest["context_length"]),
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "selection_depends_on_model_or_outcomes": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": all(
            item.target_position
            == item.window_start + int(historical.manifest["context_length"])
            for item in selected
        ),
        "parent_stream_rebuilt_from_exact_asset": True,
        "parent_stream_manifest_matches_completed_result": True,
        "parent_stream_sha256": parent_hash,
        "stream_sha256": stream_hash,
        "parent_example_id_set_sha256": _set_sha256(parent_ids),
        "stream_example_id_set_sha256": _set_sha256(selected_ids),
        "parent_text_sha256_set_sha256": _set_sha256(parent_text),
        "stream_text_sha256_set_sha256": _set_sha256(selected_text),
        "parent_timestamp_set_sha256": _set_sha256(parent_timestamps),
        "stream_timestamp_set_sha256": _set_sha256(selected_timestamps),
        "example_id_overlap_count": 0,
        "article_text_overlap_count": 0,
        "timestamp_overlap_count": 0,
        "parent_first_timestamp_utc": parent[0].timestamp_utc,
        "parent_last_timestamp_utc": parent[-1].timestamp_utc,
        "first_timestamp_utc": selected[0].timestamp_utc,
        "last_timestamp_utc": selected[-1].timestamp_utc,
        "strict_temporal_separation": True,
        "post_parent_candidate_upper_bound": remaining_upper_bound,
        "historical_selection_scan": dict(historical.manifest),
        "rebuilt_parent_stream": dict(rebuilt_parent.manifest),
    }
    return StreamBuild(
        warmup=selected[:warmup],
        online=selected[warmup:],
        manifest=stream_manifest,
    )


def _validate_parent_result(
    parent_completion: Path,
    parent_result: Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_path = _require_regular_file(parent_completion, "parent completion")
    result_path = _require_regular_file(parent_result, "parent Transformer result")
    completion = validate_parent_completion(completion_path)
    artifacts = completion.get("artifacts")
    record = artifacts.get("transformer_audit") if isinstance(artifacts, dict) else None
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise RuntimeError("parent completion has no Transformer artifact binding")
    result_file_sha256 = sha256_file(result_path)
    if result_file_sha256 != record.get("sha256"):
        raise RuntimeError("parent Transformer result bytes differ from completion")
    attempt = completion.get("transformer_attempt_index")
    expected_suffix = f"/transformer/attempt_{attempt}/RESULT.json"
    if not str(record.get("path", "")).endswith(expected_suffix):
        raise RuntimeError("parent Transformer artifact path identity changed")

    result = strict_json(result_path)
    required = {
        "schema_version",
        "status",
        "paper_evidence_eligible",
        "independent_model_trajectories",
        "seed_level_replication",
        "permutation_replicates_are_nested_within_packet",
        "config_sha256",
        "source_provenance",
        "environment_provenance",
        "cuda_preflight_provenance",
        "assets",
        "validated_input_bundle",
        "stream",
        "execution",
        "counterfactual_intervention",
        "unit",
        "artifact_inventory",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise RuntimeError("parent Transformer result schema changed")
    if not (
        result.get("schema_version") == "fmmt.gradient_reuse_transformer.campaign.v3"
        and result.get("status") == "PASS"
        and result.get("paper_evidence_eligible") is True
        and result.get("independent_model_trajectories") == 1
        and result.get("seed_level_replication") is False
        and result.get("permutation_replicates_are_nested_within_packet") is True
    ):
        raise RuntimeError("parent Transformer result is not an eligible PASS")
    validate_artifact_inventory(result)
    input_bundle = result.get("validated_input_bundle")
    if not isinstance(input_bundle, dict):
        raise RuntimeError("parent Transformer result omits its validated inputs")
    validate_embedded_input_bundle(input_bundle)
    if result.get("config_sha256") != config_sha256:
        raise RuntimeError("parent Transformer result authenticates another config")
    if (
        input_bundle["inventory"].get("configuration", {}).get("file_sha256")
        != config_sha256
    ):
        raise RuntimeError("parent validated inputs authenticate another config")

    parent_stream = result.get("stream")
    stream_config = config["stream"]
    if not isinstance(parent_stream, dict) or not (
        parent_stream.get("schema_version") == "fmmt.fixed_conditioning_stream.v1"
        and parent_stream.get("warmup_examples")
        == int(stream_config["warmup_examples"])
        and parent_stream.get("online_examples")
        == int(stream_config["online_examples"])
        and parent_stream.get("context_length") == int(stream_config["context_length"])
        and _SHA256.fullmatch(str(parent_stream.get("stream_sha256", "")))
    ):
        raise RuntimeError("parent Transformer stream identity changed")

    source = result.get("source_provenance")
    unit = result.get("unit")
    if not isinstance(source, dict) or not (
        source.get("validation_status") == "PASS"
        and source.get("configuration_sha256") == config_sha256
        and isinstance(unit, dict)
        and unit.get("source_provenance") == source
        and unit.get("audit", {}).get("all_pass") is True
        and unit.get("audit", {}).get("zero_norm_actuator_branches") == 0
    ):
        raise RuntimeError("parent Transformer scientific or source audit changed")

    environment = result.get("environment_provenance")
    if not isinstance(environment, dict) or set(environment) != {
        "schema_version",
        "manifest_sha256",
        "environment_fingerprint_sha256",
        "observed",
    }:
        raise RuntimeError("parent Transformer environment binding changed")
    if not (
        environment.get("schema_version")
        == "fmmt.gradient_reuse_transformer.environment_binding.v1"
        and environment.get("environment_fingerprint_sha256")
        == _canonical_sha256(environment.get("observed"))
    ):
        raise RuntimeError("parent Transformer environment fingerprint is invalid")

    assets = result.get("assets")
    if not isinstance(assets, dict) or set(assets) != {"model", "dataset"}:
        raise RuntimeError("parent Transformer asset identity changed")
    for role, value in assets.items():
        if not isinstance(value, dict) or set(value) != {
            "repository_id",
            "resolved_revision",
            "sha256",
        }:
            raise RuntimeError(f"parent {role} asset binding changed")
        if _SHA256.fullmatch(str(value.get("sha256", ""))) is None:
            raise RuntimeError(f"parent {role} asset hash is invalid")

    counterfactual = result.get("counterfactual_intervention")
    if not isinstance(counterfactual, dict) or not (
        counterfactual.get("name") == ACTUATOR_NAME
        and counterfactual.get("is_adamw_counterfactual") is False
    ):
        raise RuntimeError("parent Transformer intervention changed")
    parent_binding = {
        "schema_version": PARENT_BINDING_SCHEMA,
        "completion_file_sha256": sha256_file(completion_path),
        "completion_manifest_sha256": str(completion["manifest_sha256"]),
        "profile_id": str(completion["profile_id"]),
        "transformer_attempt_index": int(attempt),
        "transformer_result_file_sha256": result_file_sha256,
        "transformer_artifact_bundle_sha256": str(
            result["artifact_inventory"]["artifact_bundle_sha256"]
        ),
        "transformer_source_provenance": dict(source),
        "config_sha256": config_sha256,
        "stream_sha256": str(parent_stream["stream_sha256"]),
    }
    return result, parent_binding


def _asset_identity(assets: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        role: {
            "repository_id": str(value.repository_id),
            "resolved_revision": str(value.resolved_revision),
            "sha256": str(value.sha256),
        }
        for role, value in assets.items()
    }


def _source_binding(source_manifest: Path, *, lock_path: Path) -> dict[str, Any]:
    manifest = validate_source_manifest(source_manifest, lock_path=lock_path, repo=REPO)
    return {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "status": "PASS",
        "profile_id": str(manifest["profile_id"]),
        "source_manifest_file_sha256": sha256_file(source_manifest),
        "source_manifest_sha256": str(manifest["manifest_sha256"]),
        "source_bundle_sha256": str(manifest["source_bundle_sha256"]),
        "lock_sha256": str(manifest["configuration_sha256"]),
    }


def validate_stream2_result(
    payload: Mapping[str, Any],
    *,
    expected_input_bundle: Mapping[str, Any] | None = None,
) -> None:
    required = {
        "schema_version",
        "status",
        "paper_evidence_eligible",
        "experiment",
        "independent_model_trajectories",
        "seed_level_replication",
        "packet_blocks_are_not_corpus_replications",
        "parent_and_historical_streams_are_replications",
        "config_sha256",
        "source_provenance",
        "parent",
        "environment_provenance",
        "correction_provenance",
        "assets",
        "validated_input_bundle",
        "stream",
        "execution",
        "counterfactual_intervention",
        "unit",
        "artifact_inventory",
    }
    if set(payload) != required:
        raise RuntimeError("E7 Transformer result keys differ from schema")
    if not (
        payload.get("schema_version") == SCHEMA
        and payload.get("status") == "PASS"
        and payload.get("paper_evidence_eligible") is True
        and payload.get("experiment") == "E7_second_pythia_stream"
        and payload.get("independent_model_trajectories") == 1
        and payload.get("seed_level_replication") is False
        and payload.get("packet_blocks_are_not_corpus_replications") is True
        and payload.get("parent_and_historical_streams_are_replications") is False
        and _SHA256.fullmatch(str(payload.get("config_sha256", "")))
    ):
        raise RuntimeError("E7 Transformer result is not an eligible PASS")
    validate_artifact_inventory(payload)
    input_bundle = payload.get("validated_input_bundle")
    if not isinstance(input_bundle, dict):
        raise RuntimeError("E7 Transformer result has no validated input bundle")
    validate_e7_input_bundle(input_bundle, expected=expected_input_bundle)

    correction = payload.get("correction_provenance")
    validate_activated_correction_provenance(correction)

    stream = payload.get("stream")
    stream_keys = {
        "schema_version",
        "selection",
        "comparison_reporting",
        "historical_cutoff_utc",
        "required_examples",
        "warmup_examples",
        "online_examples",
        "packet_size",
        "context_length",
        "one_window_per_article",
        "selection_depends_on_model_seed",
        "selection_depends_on_model_or_outcomes",
        "conditioning_tokens_are_never_permuted",
        "target_position_strictly_after_context",
        "parent_stream_rebuilt_from_exact_asset",
        "parent_stream_manifest_matches_completed_result",
        "parent_stream_sha256",
        "stream_sha256",
        "parent_example_id_set_sha256",
        "stream_example_id_set_sha256",
        "parent_text_sha256_set_sha256",
        "stream_text_sha256_set_sha256",
        "parent_timestamp_set_sha256",
        "stream_timestamp_set_sha256",
        "example_id_overlap_count",
        "article_text_overlap_count",
        "timestamp_overlap_count",
        "parent_first_timestamp_utc",
        "parent_last_timestamp_utc",
        "first_timestamp_utc",
        "last_timestamp_utc",
        "strict_temporal_separation",
        "post_parent_candidate_upper_bound",
        "historical_selection_scan",
        "rebuilt_parent_stream",
    }
    if (
        not isinstance(stream, dict)
        or set(stream) != stream_keys
        or not (
            stream.get("schema_version") == STREAM_SCHEMA
            and stream.get("selection") == HISTORICAL_SELECTION
            and stream.get("comparison_reporting") == SEPARATE_STREAM_REPORTING
            and stream.get("historical_cutoff_utc") == "2021-01-01T00:00:00+00:00"
            and stream.get("required_examples") == 12288
            and stream.get("warmup_examples") == 4096
            and stream.get("online_examples") == 8192
            and stream.get("packet_size") == 16
            and stream.get("context_length") == 128
            and stream.get("one_window_per_article") is True
            and stream.get("example_id_overlap_count") == 0
            and stream.get("article_text_overlap_count") == 0
            and stream.get("timestamp_overlap_count") == 0
            and stream.get("strict_temporal_separation") is True
            and stream.get("selection_depends_on_model_seed") is False
            and stream.get("selection_depends_on_model_or_outcomes") is False
            and stream.get("conditioning_tokens_are_never_permuted") is True
            and stream.get("target_position_strictly_after_context") is True
            and stream.get("parent_stream_rebuilt_from_exact_asset") is True
            and stream.get("parent_stream_manifest_matches_completed_result") is True
            and stream.get("post_parent_candidate_upper_bound") == 291
            and _SHA256.fullmatch(str(stream.get("stream_sha256", "")))
            and _SHA256.fullmatch(str(stream.get("parent_stream_sha256", "")))
            and stream.get("stream_sha256") != stream.get("parent_stream_sha256")
        )
    ):
        raise RuntimeError("E7 second-stream integrity fields failed")
    for field in (
        "parent_example_id_set_sha256",
        "stream_example_id_set_sha256",
        "parent_text_sha256_set_sha256",
        "stream_text_sha256_set_sha256",
        "parent_timestamp_set_sha256",
        "stream_timestamp_set_sha256",
    ):
        if _SHA256.fullmatch(str(stream.get(field, ""))) is None:
            raise RuntimeError(f"E7 historical-stream hash is invalid: {field}")
    historical_scan = stream.get("historical_selection_scan")
    historical_scan_keys = {
        "schema_version",
        "selection",
        "records_seen",
        "rejected_timestamp",
        "rejected_unparseable_timestamp",
        "rejected_at_or_after_cutoff",
        "rejected_license",
        "rejected_duplicate",
        "rejected_empty",
        "rejected_parent_example_id",
        "rejected_parent_article_text",
        "rejected_too_short_after_tokenization",
        "eligible_candidates_before_tokenization",
        "tokenized_candidates_examined",
        "historical_cutoff_utc",
        "context_length",
        "warmup_examples",
        "online_examples",
        "one_window_per_article",
        "selection_depends_on_model_seed",
        "selection_depends_on_model_or_outcomes",
        "conditioning_tokens_are_never_permuted",
        "target_position_strictly_after_context",
        "excluded_parent_example_id_set_sha256",
        "excluded_parent_text_sha256_set_sha256",
        "stream_sha256",
        "first_timestamp_utc",
        "last_timestamp_utc",
    }
    if not isinstance(historical_scan, dict) or not (
        set(historical_scan) == historical_scan_keys
        and historical_scan.get("schema_version") == HISTORICAL_BUILD_SCHEMA
        and historical_scan.get("selection") == HISTORICAL_SELECTION
        and historical_scan.get("historical_cutoff_utc")
        == stream.get("historical_cutoff_utc")
        and historical_scan.get("warmup_examples") == 4096
        and historical_scan.get("online_examples") == 8192
        and historical_scan.get("context_length") == 128
        and historical_scan.get("one_window_per_article") is True
        and historical_scan.get("selection_depends_on_model_seed") is False
        and historical_scan.get("selection_depends_on_model_or_outcomes") is False
        and historical_scan.get("conditioning_tokens_are_never_permuted") is True
        and historical_scan.get("target_position_strictly_after_context") is True
        and historical_scan.get("stream_sha256") == stream.get("stream_sha256")
        and historical_scan.get("excluded_parent_example_id_set_sha256")
        == stream.get("parent_example_id_set_sha256")
        and historical_scan.get("excluded_parent_text_sha256_set_sha256")
        == stream.get("parent_text_sha256_set_sha256")
    ):
        raise RuntimeError("E7 historical selection scan identity is invalid")
    for field in (
        "records_seen",
        "rejected_timestamp",
        "rejected_unparseable_timestamp",
        "rejected_at_or_after_cutoff",
        "rejected_license",
        "rejected_duplicate",
        "rejected_empty",
        "rejected_parent_example_id",
        "rejected_parent_article_text",
        "rejected_too_short_after_tokenization",
        "eligible_candidates_before_tokenization",
        "tokenized_candidates_examined",
    ):
        value = historical_scan.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"E7 historical scan count is invalid: {field}")
    if not (
        historical_scan["rejected_timestamp"]
        == historical_scan["rejected_unparseable_timestamp"]
        + historical_scan["rejected_at_or_after_cutoff"]
        and historical_scan["tokenized_candidates_examined"]
        == historical_scan["eligible_candidates_before_tokenization"]
        and historical_scan["tokenized_candidates_examined"]
        - historical_scan["rejected_too_short_after_tokenization"]
        >= 12288
    ):
        raise RuntimeError("E7 historical scan counts are inconsistent")

    rebuilt_parent = stream.get("rebuilt_parent_stream")
    parent_stream_keys = {
        "schema_version",
        "records_seen",
        "rejected_timestamp",
        "rejected_license",
        "rejected_duplicate",
        "rejected_empty",
        "rejected_too_short_after_tokenization",
        "context_length",
        "warmup_examples",
        "online_examples",
        "one_window_per_article",
        "selection_depends_on_model_seed",
        "conditioning_tokens_are_never_permuted",
        "target_position_strictly_after_context",
        "stream_sha256",
        "first_timestamp_utc",
        "last_timestamp_utc",
    }
    if not isinstance(rebuilt_parent, dict) or not (
        set(rebuilt_parent) == parent_stream_keys
        and rebuilt_parent.get("schema_version") == "fmmt.fixed_conditioning_stream.v1"
        and rebuilt_parent.get("warmup_examples") == 4096
        and rebuilt_parent.get("online_examples") == 8192
        and rebuilt_parent.get("context_length") == 128
        and rebuilt_parent.get("stream_sha256") == stream.get("parent_stream_sha256")
        and post_parent_candidate_upper_bound(rebuilt_parent) == 291
    ):
        raise RuntimeError("E7 rebuilt parent stream identity is invalid")
    cutoff = _parse_timestamp(stream["historical_cutoff_utc"])
    if not (
        _parse_timestamp(stream["first_timestamp_utc"])
        <= _parse_timestamp(stream["last_timestamp_utc"])
        < cutoff
        <= _parse_timestamp(stream["parent_first_timestamp_utc"])
        <= _parse_timestamp(stream["parent_last_timestamp_utc"])
        and rebuilt_parent["first_timestamp_utc"]
        == stream["parent_first_timestamp_utc"]
        and rebuilt_parent["last_timestamp_utc"] == stream["parent_last_timestamp_utc"]
        and historical_scan["first_timestamp_utc"] == stream["first_timestamp_utc"]
        and historical_scan["last_timestamp_utc"] == stream["last_timestamp_utc"]
    ):
        raise RuntimeError("E7 historical/parent chronology is invalid")

    source = payload.get("source_provenance")
    parent = payload.get("parent")
    environment = payload.get("environment_provenance")
    assets = payload.get("assets")
    unit = payload.get("unit")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "schema_version",
            "status",
            "profile_id",
            "source_manifest_file_sha256",
            "source_manifest_sha256",
            "source_bundle_sha256",
            "lock_sha256",
        }
        or not (
            source.get("schema_version") == SOURCE_BINDING_SCHEMA
            and source.get("status") == "PASS"
            and source.get("profile_id") == "iclr_reviewer_extension_v1"
        )
    ):
        raise RuntimeError("E7 source provenance is invalid")
    for field in (
        "source_manifest_file_sha256",
        "source_manifest_sha256",
        "source_bundle_sha256",
        "lock_sha256",
    ):
        if _SHA256.fullmatch(str(source.get(field, ""))) is None:
            raise RuntimeError(f"E7 source hash is invalid: {field}")
    parent_keys = {
        "schema_version",
        "completion_file_sha256",
        "completion_manifest_sha256",
        "profile_id",
        "transformer_attempt_index",
        "transformer_result_file_sha256",
        "transformer_artifact_bundle_sha256",
        "transformer_source_provenance",
        "config_sha256",
        "stream_sha256",
    }
    if (
        not isinstance(parent, dict)
        or set(parent) != parent_keys
        or not (
            parent.get("schema_version") == PARENT_BINDING_SCHEMA
            and parent.get("profile_id") == "rtx_7day_v1"
            and isinstance(parent.get("transformer_attempt_index"), int)
            and parent.get("transformer_attempt_index") >= 0
            and parent.get("config_sha256") == payload.get("config_sha256")
            and parent.get("stream_sha256") == stream.get("parent_stream_sha256")
            and isinstance(parent.get("transformer_source_provenance"), dict)
            and parent["transformer_source_provenance"].get("validation_status")
            == "PASS"
        )
    ):
        raise RuntimeError("E7 parent binding is invalid")
    for field in (
        "completion_file_sha256",
        "completion_manifest_sha256",
        "transformer_result_file_sha256",
        "transformer_artifact_bundle_sha256",
        "config_sha256",
        "stream_sha256",
    ):
        if _SHA256.fullmatch(str(parent.get(field, ""))) is None:
            raise RuntimeError(f"E7 parent hash is invalid: {field}")
    environment_keys = {
        "schema_version",
        "status",
        "exact_parent_runtime_reused",
        "parent_environment_manifest_sha256",
        "environment_fingerprint_sha256",
        "trajectory_seed",
        "observed",
    }
    if (
        not isinstance(environment, dict)
        or set(environment) != environment_keys
        or not (
            environment.get("schema_version") == ENVIRONMENT_BINDING_SCHEMA
            and environment.get("status") == "PASS"
            and environment.get("exact_parent_runtime_reused") is True
            and environment.get("trajectory_seed") == 300
            and _SHA256.fullmatch(
                str(environment.get("parent_environment_manifest_sha256", ""))
            )
            and environment.get("environment_fingerprint_sha256")
            == _canonical_sha256(environment.get("observed"))
        )
    ):
        raise RuntimeError("E7 environment binding is invalid")
    if not isinstance(assets, dict) or set(assets) != {"model", "dataset"}:
        raise RuntimeError("E7 asset identity is invalid")
    for role, value in assets.items():
        if not isinstance(value, dict) or set(value) != {
            "repository_id",
            "resolved_revision",
            "sha256",
        }:
            raise RuntimeError(f"E7 {role} asset identity is invalid")
        if _SHA256.fullmatch(str(value.get("sha256", ""))) is None:
            raise RuntimeError(f"E7 {role} asset hash is invalid")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or not (
        execution.get("resume_supported") is False
        and execution.get("heartbeat_is_progress_evidence_only") is True
        and execution.get("interruption_policy")
        == "restart in a fresh attempt from the immutable inputs"
    ):
        raise RuntimeError("E7 execution policy is invalid")
    counterfactual = payload.get("counterfactual_intervention")
    if not isinstance(counterfactual, dict) or not (
        counterfactual.get("name") == ACTUATOR_NAME
        and counterfactual.get("is_adamw_counterfactual") is False
    ):
        raise RuntimeError("E7 counterfactual intervention is invalid")
    if not isinstance(unit, dict) or not (
        unit.get("source_provenance") == source
        and unit.get("audit", {}).get("all_pass") is True
        and unit.get("audit", {}).get("zero_norm_actuator_branches") == 0
        and unit.get("packet_count") == 512
        and unit.get("scored_token_count") == 8176
        and unit.get("permutation_replicates_nested_within_packet") is True
        and unit.get("seed_level_replication") is False
    ):
        raise RuntimeError("E7 scientific unit did not pass its audit")
    warmup = unit.get("warmup")
    if not isinstance(warmup, dict) or not (
        warmup.get("packets") == 256
        and warmup.get("epochs") == 1
        and warmup.get("true_targets_only") is True
    ):
        raise RuntimeError("E7 warm-up workload differs from the parent protocol")
    task_metrics = unit.get("task_metrics")
    if not isinstance(task_metrics, dict) or set(task_metrics) != {"K0", "K1", "K4"}:
        raise RuntimeError("E7 task metrics changed")
    for branch, metrics in task_metrics.items():
        if not isinstance(metrics, dict) or set(metrics) != {
            "count",
            "nll",
            "perplexity",
            "top1_accuracy",
            "top5_accuracy",
        }:
            raise RuntimeError(f"E7 {branch} metric schema changed")
        values = {
            field: float(metrics[field])
            for field in (
                "count",
                "nll",
                "perplexity",
                "top1_accuracy",
                "top5_accuracy",
            )
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"E7 {branch} contains a non-finite metric")
        if not (
            values["count"] == 8176.0
            and values["nll"] >= 0.0
            and values["perplexity"] > 0.0
            and math.isclose(
                values["perplexity"],
                math.exp(values["nll"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and 0.0 <= values["top1_accuracy"] <= values["top5_accuracy"] <= 1.0
        ):
            raise RuntimeError(f"E7 {branch} metric identities failed")
    optimizer = unit.get("optimizer_implementation")
    if not isinstance(optimizer, dict) or not (
        optimizer.get("name") == "AdamW"
        and optimizer.get("foreach") is False
        and optimizer.get("fused") is False
        and optimizer.get("counterfactual_actuator") == ACTUATOR_NAME
        and optimizer.get("counterfactual_is_adamw_replay") is False
    ):
        raise RuntimeError("E7 optimizer protocol differs from the parent")


def run_stream2(
    *,
    lock_path: Path,
    source_manifest: Path,
    config_path: Path,
    asset_manifest: Path,
    parent_completion: Path,
    parent_transformer_result: Path,
    correction_manifest: Path,
    correction_patch: Path,
    device: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    lock_path = _require_regular_file(lock_path, "reviewer-extension lock")
    source_manifest = _require_regular_file(
        source_manifest, "reviewer-extension source manifest"
    )
    config_path = _require_regular_file(config_path, "parent Transformer config")
    asset_manifest = _require_regular_file(asset_manifest, "Transformer asset manifest")
    parent_completion = _require_regular_file(parent_completion, "parent completion")
    parent_transformer_result = _require_regular_file(
        parent_transformer_result, "parent Transformer result"
    )
    correction_manifest = _require_regular_file(
        correction_manifest, "Transformer correction manifest"
    )
    correction_patch = _require_regular_file(
        correction_patch, "Transformer correction patch"
    )
    correction_provenance = require_active_correction(
        correction_manifest, correction_patch
    )
    lock = load_lock(lock_path)
    e7 = lock["e7_second_pythia_stream"]
    expected_config_path = (REPO / str(e7["parent_config"])).resolve(strict=True)
    if config_path != expected_config_path:
        raise RuntimeError("E7 config path differs from the locked repository config")
    source_binding = _source_binding(source_manifest, lock_path=lock_path)
    config = read_config(config_path)
    config_sha256 = sha256_file(config_path)
    parent_result, parent_binding = _validate_parent_result(
        parent_completion,
        parent_transformer_result,
        config=config,
        config_sha256=config_sha256,
    )

    input_paths = {
        "reviewer_extension_lock": lock_path,
        "reviewer_extension_source_manifest": source_manifest,
        "transformer_configuration": config_path,
        "transformer_asset_manifest": asset_manifest,
        "parent_completion": parent_completion,
        "parent_transformer_result": parent_transformer_result,
        "transformer_correction_manifest": correction_manifest,
        "transformer_correction_patch": correction_patch,
    }
    current_input_bundle = build_e7_input_bundle(input_paths)
    validate_e7_input_bundle(current_input_bundle)
    parent_asset_manifest_hash = parent_result["validated_input_bundle"]["inventory"][
        "asset_manifest"
    ]["file_sha256"]
    if sha256_file(asset_manifest) != parent_asset_manifest_hash:
        raise RuntimeError("E7 asset manifest bytes differ from the parent invocation")

    _validate_runtime(config, device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    trajectory_seed = int(config["audit"]["trajectory_seed"])
    seed_everything(trajectory_seed)
    observed_environment = capture_runtime_environment(
        device=device,
        optimizer_specification=config["optimization"],
    )
    parent_environment = parent_result["environment_provenance"]
    if observed_environment != parent_environment["observed"]:
        raise RuntimeError("E7 runtime differs from the authenticated parent runtime")
    environment_binding = {
        "schema_version": ENVIRONMENT_BINDING_SCHEMA,
        "status": "PASS",
        "exact_parent_runtime_reused": True,
        "parent_environment_manifest_sha256": str(
            parent_environment["manifest_sha256"]
        ),
        "environment_fingerprint_sha256": _canonical_sha256(observed_environment),
        "trajectory_seed": trajectory_seed,
        "observed": observed_environment,
    }

    assets = validate_asset_manifest(
        asset_manifest,
        config["assets"],
        expected_config_sha256=config_sha256,
    )
    asset_identity = _asset_identity(assets)
    if asset_identity != parent_result["assets"]:
        raise RuntimeError("E7 did not resolve the exact parent checkpoint and corpus")

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "validated_inputs",
                "completed_packets": 0,
                "input_bundle_sha256": current_input_bundle["bundle_sha256"],
            }
        )
        progress_callback({"phase": "load_model", "completed_packets": 0})
    model, tokenizer = load_local_pythia(
        assets["model"].path, dtype=str(config["model"]["dtype"])
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lower, upper = [
        int(value) for value in config["model"]["expected_total_parameter_range"]
    ]
    if not lower <= parameter_count <= upper:
        raise RuntimeError("E7 model parameter count differs from the parent lock")

    required = int(e7["required_examples"])
    historical_warmup = int(e7["warmup_examples"])
    historical_online = int(e7["online_examples"])
    historical_cutoff = str(e7["historical_cutoff_utc"])
    stream_config = config["stream"]
    parent_total = int(stream_config["warmup_examples"]) + int(
        stream_config["online_examples"]
    )
    if not (
        required == parent_total
        and historical_warmup == int(stream_config["warmup_examples"])
        and historical_online == int(stream_config["online_examples"])
        and _parse_timestamp(historical_cutoff)
        == _parse_timestamp(str(stream_config["earliest_utc"]))
    ):
        raise RuntimeError("E7 stream sizes no longer match the parent protocol")
    if progress_callback is not None:
        progress_callback({"phase": "rebuild_parent_stream", "completed_packets": 0})
    rebuilt_parent = build_chronological_stream(
        iter_json_records(assets["dataset"].path),
        tokenizer,
        context_length=int(stream_config["context_length"]),
        warmup_examples=int(stream_config["warmup_examples"]),
        online_examples=int(stream_config["online_examples"]),
        earliest_utc=str(stream_config["earliest_utc"]),
        offset_salt=str(stream_config["offset_salt"]),
    )
    rebuilt_parent_examples = authenticate_rebuilt_parent_stream(
        rebuilt_parent,
        parent_stream=parent_result["stream"],
        historical_cutoff_utc=historical_cutoff,
    )
    parent_ids = {item.example_id for item in rebuilt_parent_examples}
    parent_text = {item.text_sha256 for item in rebuilt_parent_examples}
    if progress_callback is not None:
        progress_callback({"phase": "build_historical_stream", "completed_packets": 0})
    historical = build_historical_stream(
        iter_json_records(assets["dataset"].path),
        tokenizer,
        context_length=int(stream_config["context_length"]),
        warmup_examples=historical_warmup,
        online_examples=historical_online,
        historical_cutoff_utc=historical_cutoff,
        offset_salt=str(stream_config["offset_salt"]),
        excluded_example_ids=parent_ids,
        excluded_text_sha256=parent_text,
    )
    stream = select_historical_stream(
        historical,
        rebuilt_parent,
        parent_stream=parent_result["stream"],
        historical_cutoff_utc=historical_cutoff,
        required_examples=required,
        warmup_examples=historical_warmup,
        online_examples=historical_online,
        packet_size=int(stream_config["packet_size"]),
    )

    unit = _run_unit(
        model=model,
        stream=stream,
        config=config,
        seed=trajectory_seed,
        device=device,
        progress_callback=progress_callback,
    )
    unit["source_provenance"] = dict(source_binding)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "paper_evidence_eligible": True,
        "experiment": "E7_second_pythia_stream",
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "packet_blocks_are_not_corpus_replications": True,
        "parent_and_historical_streams_are_replications": False,
        "config_sha256": config_sha256,
        "source_provenance": source_binding,
        "parent": parent_binding,
        "environment_provenance": environment_binding,
        "correction_provenance": correction_provenance,
        "assets": asset_identity,
        "validated_input_bundle": current_input_bundle,
        "stream": stream.manifest,
        "execution": {
            "resume_supported": False,
            "interruption_policy": (
                "restart in a fresh attempt from the immutable inputs"
            ),
            "heartbeat_is_progress_evidence_only": True,
        },
        "counterfactual_intervention": dict(
            parent_result["counterfactual_intervention"]
        ),
        "unit": unit,
    }
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": current_input_bundle,
            "stream": stream.manifest,
            "unit": unit,
        }
    )
    validate_stream2_result(result, expected_input_bundle=current_input_bundle)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "final_input_authentication",
                "completed_packets": int(stream_config["online_examples"])
                // int(stream_config["packet_size"]),
            }
        )
    final_source_binding = _source_binding(source_manifest, lock_path=lock_path)
    if final_source_binding != source_binding:
        raise RuntimeError("reviewer-extension source changed during E7 execution")
    final_assets = validate_asset_manifest(
        asset_manifest,
        config["assets"],
        expected_config_sha256=config_sha256,
    )
    if _asset_identity(final_assets) != asset_identity:
        raise RuntimeError("a content-addressed E7 asset changed during execution")
    final_input_bundle = build_e7_input_bundle(input_paths)
    if final_input_bundle != current_input_bundle:
        raise RuntimeError("an authenticated E7 input changed during execution")
    if require_active_correction(correction_manifest, correction_patch) != (
        correction_provenance
    ):
        raise RuntimeError("Transformer correction activation changed during E7")
    return result


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_atomic_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing to overwrite existing artifact: {path}"
            ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    observed: dict[Path, str] = {}
    for role, path in paths.items():
        resolved = Path(path).expanduser().resolve()
        if resolved in observed:
            raise RuntimeError(
                f"path roles must be disjoint: {observed[resolved]} and {role}"
            )
        observed[resolved] = role


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--parent-completion", required=True)
    parser.add_argument("--parent-transformer-result", required=True)
    parser.add_argument("--correction-manifest", required=True)
    parser.add_argument("--correction-patch", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--heartbeat")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = Path(arguments.out).expanduser().resolve()
    heartbeat = (
        Path(arguments.heartbeat).expanduser().resolve()
        if arguments.heartbeat
        else Path(str(output) + ".heartbeat.json")
    )
    inputs = {
        "lock": Path(arguments.lock),
        "source_manifest": Path(arguments.source_manifest),
        "config": Path(arguments.config),
        "asset_manifest": Path(arguments.asset_manifest),
        "parent_completion": Path(arguments.parent_completion),
        "parent_transformer_result": Path(arguments.parent_transformer_result),
        "correction_manifest": Path(arguments.correction_manifest),
        "correction_patch": Path(arguments.correction_patch),
    }
    _require_disjoint_paths({"output": output, "heartbeat": heartbeat, **inputs})
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    if heartbeat.exists() or heartbeat.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing heartbeat: {heartbeat}")
    _reject_artifacts_inside_asset_roots(
        {"output": output, "heartbeat": heartbeat},
        asset_manifest=inputs["asset_manifest"],
    )

    now = datetime.now(timezone.utc).isoformat()
    heartbeat_state: dict[str, Any] = {
        "schema_version": HEARTBEAT_SCHEMA,
        "status": "RUNNING",
        "resume_supported": False,
        "warning": (
            "Progress evidence only; an interrupted E7 trajectory must restart "
            "in a fresh attempt."
        ),
        "phase": "initialization",
        "completed_packets": 0,
        "created_utc": now,
        "updated_utc": now,
        "destination_sha256": {
            "output": _canonical_sha256(str(output)),
            "heartbeat": _canonical_sha256(str(heartbeat)),
        },
    }
    _write_atomic_new(heartbeat, heartbeat_state)

    def progress(value: Mapping[str, Any]) -> None:
        heartbeat_state.update(dict(value))
        heartbeat_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_atomic(heartbeat, heartbeat_state)

    try:
        result = run_stream2(
            lock_path=inputs["lock"],
            source_manifest=inputs["source_manifest"],
            config_path=inputs["config"],
            asset_manifest=inputs["asset_manifest"],
            parent_completion=inputs["parent_completion"],
            parent_transformer_result=inputs["parent_transformer_result"],
            correction_manifest=inputs["correction_manifest"],
            correction_patch=inputs["correction_patch"],
            device=str(arguments.device),
            progress_callback=progress,
        )
        _write_atomic_new(output, result)
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
            "output_file_sha256": sha256_file(output),
            "result_payload_sha256": _canonical_sha256(result),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_atomic(heartbeat, heartbeat_state)
    print(json.dumps({"status": "PASS", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
