"""Focused contract tests for reviewer-extension Pythia stream E7."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.gradient_reuse_reviewer_extension import transformer_stream2 as e7
from experiments.gradient_reuse_transformer.data import StreamBuild, StreamExample
from experiments.gradient_reuse_transformer.run import (
    _artifact_inventory,
    _canonical_sha256,
)


def _example(
    index: int,
    *,
    historical: bool = False,
    text_index: int | None = None,
) -> StreamExample:
    text_value = index if text_index is None else text_index
    start = datetime(2020 if historical else 2021, 1, 1, tzinfo=timezone.utc)
    return StreamExample(
        example_id=f"{index:064x}",
        timestamp_utc=(start + timedelta(days=index)).isoformat(),
        source="news",
        input_ids=(index, index + 1, index + 2, index + 3),
        target_id=index + 4,
        window_start=index,
        target_position=index + 4,
        text_sha256=f"{text_value + 1000:064x}",
    )


def _parent_build(examples: list[StreamExample], split: int = 2) -> StreamBuild:
    manifest = {
        "schema_version": "fmmt.fixed_conditioning_stream.v1",
        "records_seen": len(examples) + 39,
        "rejected_timestamp": 10,
        "rejected_license": 20,
        "rejected_duplicate": 3,
        "rejected_empty": 2,
        "rejected_too_short_after_tokenization": 4,
        "context_length": 4,
        "warmup_examples": split,
        "online_examples": len(examples) - split,
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": True,
        "stream_sha256": e7.stream_identity_sha256(examples),
        "first_timestamp_utc": examples[0].timestamp_utc,
        "last_timestamp_utc": examples[-1].timestamp_utc,
    }
    return StreamBuild(
        warmup=tuple(examples[:split]),
        online=tuple(examples[split:]),
        manifest=manifest,
    )


def _historical_build(
    examples: list[StreamExample],
    parent: list[StreamExample],
    split: int = 2,
) -> StreamBuild:
    manifest = {
        "schema_version": e7.HISTORICAL_BUILD_SCHEMA,
        "selection": e7.HISTORICAL_SELECTION,
        "records_seen": 30,
        "rejected_timestamp": 8,
        "rejected_unparseable_timestamp": 1,
        "rejected_at_or_after_cutoff": 7,
        "rejected_license": 2,
        "rejected_duplicate": 1,
        "rejected_empty": 1,
        "rejected_parent_example_id": 1,
        "rejected_parent_article_text": 1,
        "rejected_too_short_after_tokenization": 2,
        "eligible_candidates_before_tokenization": len(examples) + 2,
        "tokenized_candidates_examined": len(examples) + 2,
        "historical_cutoff_utc": "2021-01-01T00:00:00+00:00",
        "context_length": 4,
        "warmup_examples": split,
        "online_examples": len(examples) - split,
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "selection_depends_on_model_or_outcomes": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": True,
        "excluded_parent_example_id_set_sha256": e7._set_sha256(
            {item.example_id for item in parent}
        ),
        "excluded_parent_text_sha256_set_sha256": e7._set_sha256(
            {item.text_sha256 for item in parent}
        ),
        "stream_sha256": e7.stream_identity_sha256(examples),
        "first_timestamp_utc": examples[0].timestamp_utc,
        "last_timestamp_utc": examples[-1].timestamp_utc,
    }
    return StreamBuild(
        warmup=tuple(examples[:split]),
        online=tuple(examples[split:]),
        manifest=manifest,
    )


def _parent_stream(prefix: list[StreamExample]) -> dict[str, Any]:
    return _parent_build(prefix).manifest


def _parent_embedded_bundle(config_sha256: str = "1" * 64) -> dict[str, Any]:
    payloads = {"configuration": {"locked": True}}
    inventory = {
        "configuration": {
            "file_sha256": config_sha256,
            "canonical_payload_sha256": _canonical_sha256(payloads["configuration"]),
            "size_bytes": 16,
        }
    }
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.input_bundle.v1",
        "payloads": payloads,
        "inventory": inventory,
        "bundle_sha256": _canonical_sha256(payloads),
    }


def _e7_embedded_bundle() -> dict[str, Any]:
    inventory = {
        role: {"file_sha256": f"{index + 1:x}" * 64, "size_bytes": index + 2}
        for index, role in enumerate(sorted(e7.E7_INPUT_ROLES))
    }
    return {
        "schema_version": e7.INPUT_BUNDLE_SCHEMA,
        "input_roles": sorted(e7.E7_INPUT_ROLES),
        "inventory": inventory,
        "bundle_sha256": _canonical_sha256(inventory),
        "absolute_paths_embedded": False,
    }


def _parent_result(config_sha256: str = "1" * 64) -> dict[str, Any]:
    source = {
        "schema_version": "gradient_reuse.transformer_source_binding.v1",
        "experiment": "transformer",
        "validation_status": "PASS",
        "source_manifest_sha256": "2" * 64,
        "source_bundle_sha256": "3" * 64,
        "configuration_sha256": config_sha256,
    }
    observed = {"numerics": {"deterministic_algorithms": True}}
    stream = {
        "schema_version": "fmmt.fixed_conditioning_stream.v1",
        "warmup_examples": 2,
        "online_examples": 4,
        "context_length": 4,
        "stream_sha256": "4" * 64,
    }
    unit = {
        "source_provenance": source,
        "audit": {"all_pass": True, "zero_norm_actuator_branches": 0},
    }
    result: dict[str, Any] = {
        "schema_version": "fmmt.gradient_reuse_transformer.campaign.v3",
        "status": "PASS",
        "paper_evidence_eligible": True,
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "permutation_replicates_are_nested_within_packet": True,
        "config_sha256": config_sha256,
        "source_provenance": source,
        "environment_provenance": {
            "schema_version": (
                "fmmt.gradient_reuse_transformer.environment_binding.v1"
            ),
            "manifest_sha256": "5" * 64,
            "environment_fingerprint_sha256": _canonical_sha256(observed),
            "observed": observed,
        },
        "cuda_preflight_provenance": {"report_sha256": "6" * 64},
        "assets": {
            "model": {
                "repository_id": "model",
                "resolved_revision": "a" * 40,
                "sha256": "7" * 64,
            },
            "dataset": {
                "repository_id": "dataset",
                "resolved_revision": "b" * 40,
                "sha256": "8" * 64,
            },
        },
        "validated_input_bundle": _parent_embedded_bundle(config_sha256),
        "stream": stream,
        "execution": {
            "resume_supported": False,
            "interruption_policy": "restart",
            "heartbeat_is_progress_evidence_only": True,
        },
        "counterfactual_intervention": {
            "name": "equal_radius_raw_gradient_actuator",
            "direction": "stored raw minibatch gradient",
            "radius_calibration": "median",
            "is_adamw_counterfactual": False,
        },
        "unit": unit,
    }
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": result["validated_input_bundle"],
            "stream": stream,
            "unit": unit,
            "cuda_preflight_provenance": result["cuda_preflight_provenance"],
        }
    )
    return result


def _valid_extension_result() -> dict[str, Any]:
    source = {
        "schema_version": e7.SOURCE_BINDING_SCHEMA,
        "status": "PASS",
        "profile_id": "iclr_reviewer_extension_v1",
        "source_manifest_file_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "source_bundle_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }
    observed = {"numerics": {"deterministic_algorithms": True}}
    bundle = _e7_embedded_bundle()
    parent_scan = {
        "schema_version": "fmmt.fixed_conditioning_stream.v1",
        "records_seen": 172308,
        "rejected_timestamp": 150022,
        "rejected_license": 5191,
        "rejected_duplicate": 830,
        "rejected_empty": 0,
        "rejected_too_short_after_tokenization": 3686,
        "context_length": 128,
        "warmup_examples": 4096,
        "online_examples": 8192,
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": True,
        "stream_sha256": "4" * 64,
        "first_timestamp_utc": "2021-01-01T00:00:00+00:00",
        "last_timestamp_utc": "2024-05-07T00:00:00+00:00",
    }
    historical_scan = {
        "schema_version": e7.HISTORICAL_BUILD_SCHEMA,
        "selection": e7.HISTORICAL_SELECTION,
        "records_seen": 172308,
        "rejected_timestamp": 20000,
        "rejected_unparseable_timestamp": 10,
        "rejected_at_or_after_cutoff": 19990,
        "rejected_license": 1000,
        "rejected_duplicate": 50,
        "rejected_empty": 0,
        "rejected_parent_example_id": 0,
        "rejected_parent_article_text": 0,
        "rejected_too_short_after_tokenization": 100,
        "eligible_candidates_before_tokenization": 13000,
        "tokenized_candidates_examined": 13000,
        "historical_cutoff_utc": "2021-01-01T00:00:00+00:00",
        "context_length": 128,
        "warmup_examples": 4096,
        "online_examples": 8192,
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "selection_depends_on_model_or_outcomes": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": True,
        "excluded_parent_example_id_set_sha256": "1" * 64,
        "excluded_parent_text_sha256_set_sha256": "3" * 64,
        "stream_sha256": "9" * 64,
        "first_timestamp_utc": "2018-01-01T00:00:00+00:00",
        "last_timestamp_utc": "2020-12-31T23:59:59+00:00",
    }
    stream = {
        "schema_version": e7.STREAM_SCHEMA,
        "required_examples": 12288,
        "warmup_examples": 4096,
        "online_examples": 8192,
        "selection": e7.HISTORICAL_SELECTION,
        "comparison_reporting": e7.SEPARATE_STREAM_REPORTING,
        "historical_cutoff_utc": "2021-01-01T00:00:00+00:00",
        "packet_size": 16,
        "context_length": 128,
        "one_window_per_article": True,
        "target_position_strictly_after_context": True,
        "example_id_overlap_count": 0,
        "article_text_overlap_count": 0,
        "timestamp_overlap_count": 0,
        "strict_temporal_separation": True,
        "selection_depends_on_model_seed": False,
        "selection_depends_on_model_or_outcomes": False,
        "conditioning_tokens_are_never_permuted": True,
        "parent_stream_rebuilt_from_exact_asset": True,
        "parent_stream_manifest_matches_completed_result": True,
        "stream_sha256": "9" * 64,
        "parent_stream_sha256": "4" * 64,
        "parent_example_id_set_sha256": "1" * 64,
        "stream_example_id_set_sha256": "2" * 64,
        "parent_text_sha256_set_sha256": "3" * 64,
        "stream_text_sha256_set_sha256": "5" * 64,
        "parent_timestamp_set_sha256": "6" * 64,
        "stream_timestamp_set_sha256": "7" * 64,
        "parent_first_timestamp_utc": "2021-01-01T00:00:00+00:00",
        "parent_last_timestamp_utc": "2024-05-07T00:00:00+00:00",
        "first_timestamp_utc": "2018-01-01T00:00:00+00:00",
        "last_timestamp_utc": "2020-12-31T23:59:59+00:00",
        "post_parent_candidate_upper_bound": 291,
        "historical_selection_scan": historical_scan,
        "rebuilt_parent_stream": parent_scan,
    }
    correction = e7.activated_correction_provenance(
        {
            "schema_version": e7.CORRECTION_PROVENANCE_SCHEMA,
            "status": "AUTHENTICATED",
            "effective_vocab_manifest_sha256": (e7.EXPECTED_CORRECTION_MANIFEST_SHA256),
            "effective_vocab_patch_sha256": e7.EXPECTED_CORRECTION_PATCH_SHA256,
            "padded_vocab_manifest_sha256": (e7.EXPECTED_PADDED_VOCAB_MANIFEST_SHA256),
            "classifier_dropout_manifest_sha256": (
                e7.EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256
            ),
            "manifest_chain_sha256": _canonical_sha256(
                [
                    e7.EXPECTED_CLASSIFIER_DROPOUT_MANIFEST_SHA256,
                    e7.EXPECTED_PADDED_VOCAB_MANIFEST_SHA256,
                    e7.EXPECTED_CORRECTION_MANIFEST_SHA256,
                ]
            ),
            "effective_patch_commits_composed_chain": True,
        }
    )
    metrics = {
        "K0": {
            "count": 8176,
            "nll": 1.0,
            "perplexity": math.exp(1.0),
            "top1_accuracy": 0.10,
            "top5_accuracy": 0.20,
        },
        "K1": {
            "count": 8176,
            "nll": 0.9,
            "perplexity": math.exp(0.9),
            "top1_accuracy": 0.12,
            "top5_accuracy": 0.24,
        },
        "K4": {
            "count": 8176,
            "nll": 0.95,
            "perplexity": math.exp(0.95),
            "top1_accuracy": 0.11,
            "top5_accuracy": 0.22,
        },
    }
    unit = {
        "schema_version": "fmmt.gradient_reuse_transformer.unit.v2",
        "source_provenance": source,
        "audit": {"all_pass": True, "zero_norm_actuator_branches": 0},
        "trajectory_seed": 300,
        "packet_count": 512,
        "scored_token_count": 8176,
        "permutation_replicates_nested_within_packet": True,
        "seed_level_replication": False,
        "warmup": {"packets": 256, "epochs": 1, "true_targets_only": True},
        "task_metrics": metrics,
        "effects": {
            "gross_nll_reduction": 0.1,
            "permuted_nll_reduction": 0.05,
            "pairing_nll_reduction": 0.05,
            "gross_top1_accuracy": 0.02,
            "permuted_top1_accuracy": 0.01,
            "pairing_top1_accuracy": 0.01,
            "gross_top5_accuracy": 0.04,
            "permuted_top5_accuracy": 0.02,
            "pairing_top5_accuracy": 0.02,
        },
        "optimizer_implementation": {
            "name": "AdamW",
            "foreach": False,
            "fused": False,
            "counterfactual_actuator": "equal_radius_raw_gradient_actuator",
            "counterfactual_is_adamw_replay": False,
        },
    }
    result: dict[str, Any] = {
        "schema_version": e7.SCHEMA,
        "status": "PASS",
        "paper_evidence_eligible": True,
        "experiment": "E7_second_pythia_stream",
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "packet_blocks_are_not_corpus_replications": True,
        "parent_and_historical_streams_are_replications": False,
        "config_sha256": "1" * 64,
        "source_provenance": source,
        "parent": {
            "schema_version": e7.PARENT_BINDING_SCHEMA,
            "completion_file_sha256": "6" * 64,
            "completion_manifest_sha256": "7" * 64,
            "profile_id": "rtx_7day_v1",
            "transformer_attempt_index": 0,
            "transformer_result_file_sha256": "8" * 64,
            "transformer_artifact_bundle_sha256": "a" * 64,
            "transformer_source_provenance": {"validation_status": "PASS"},
            "config_sha256": "1" * 64,
            "stream_sha256": "4" * 64,
        },
        "environment_provenance": {
            "schema_version": e7.ENVIRONMENT_BINDING_SCHEMA,
            "status": "PASS",
            "exact_parent_runtime_reused": True,
            "parent_environment_manifest_sha256": "b" * 64,
            "trajectory_seed": 300,
            "environment_fingerprint_sha256": _canonical_sha256(observed),
            "observed": observed,
        },
        "correction_provenance": correction,
        "assets": {
            "model": {
                "repository_id": "model",
                "resolved_revision": "a" * 40,
                "sha256": "c" * 64,
            },
            "dataset": {
                "repository_id": "dataset",
                "resolved_revision": "b" * 40,
                "sha256": "d" * 64,
            },
        },
        "validated_input_bundle": bundle,
        "stream": stream,
        "execution": {
            "resume_supported": False,
            "interruption_policy": (
                "restart in a fresh attempt from the immutable inputs"
            ),
            "heartbeat_is_progress_evidence_only": True,
        },
        "counterfactual_intervention": {
            "name": "equal_radius_raw_gradient_actuator",
            "is_adamw_counterfactual": False,
        },
        "unit": unit,
    }
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": bundle,
            "stream": stream,
            "unit": result["unit"],
        }
    )
    return result


def test_selects_historical_stream_and_proves_three_kinds_of_disjointness() -> None:
    historical = [_example(index, historical=True) for index in range(6)]
    parent = [_example(index + 20) for index in range(6)]
    parent_build = _parent_build(parent)
    stream = e7.select_historical_stream(
        _historical_build(historical, parent),
        parent_build,
        parent_stream=parent_build.manifest,
        historical_cutoff_utc="2021-01-01T00:00:00+00:00",
        required_examples=6,
        warmup_examples=2,
        online_examples=4,
        packet_size=2,
    )
    assert stream.warmup == tuple(historical[:2])
    assert stream.online == tuple(historical[2:])
    assert stream.manifest["parent_stream_sha256"] == e7.stream_identity_sha256(parent)
    assert stream.manifest["stream_sha256"] == e7.stream_identity_sha256(historical)
    assert stream.manifest["example_id_overlap_count"] == 0
    assert stream.manifest["article_text_overlap_count"] == 0
    assert stream.manifest["timestamp_overlap_count"] == 0
    assert stream.manifest["strict_temporal_separation"] is True
    assert stream.manifest["comparison_reporting"] == e7.SEPARATE_STREAM_REPORTING


def test_historical_stream_rejects_parent_drift_and_hidden_text_overlap() -> None:
    historical = [_example(index, historical=True) for index in range(6)]
    parent = [_example(index + 20) for index in range(6)]
    parent_build = _parent_build(parent)
    changed_parent = copy.deepcopy(parent_build.manifest)
    changed_parent["stream_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest differs"):
        e7.select_historical_stream(
            _historical_build(historical, parent),
            parent_build,
            parent_stream=changed_parent,
            historical_cutoff_utc="2021-01-01T00:00:00+00:00",
            required_examples=6,
            warmup_examples=2,
            online_examples=4,
            packet_size=2,
        )

    overlap = list(historical)
    overlap[0] = replace(overlap[0], text_sha256=parent[0].text_sha256)
    with pytest.raises(RuntimeError, match="overlaps"):
        e7.select_historical_stream(
            _historical_build(overlap, parent),
            parent_build,
            parent_stream=parent_build.manifest,
            historical_cutoff_utc="2021-01-01T00:00:00+00:00",
            required_examples=6,
            warmup_examples=2,
            online_examples=4,
            packet_size=2,
        )


def test_historical_builder_uses_latest_tokenizable_pre_cutoff_records() -> None:
    class Tokenizer:
        vocab_size = 100

        def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
            return {"input_ids": [1, 2] if text == "short" else [1, 2, 3, 4]}

    def record(index: int, text: str, *, year: int = 2020) -> dict[str, Any]:
        return {
            "id": f"article-{index}",
            "source": "news",
            "created": datetime(year, 1, 1, tzinfo=timezone.utc)
            .replace(day=index + 1)
            .isoformat(),
            "text": text,
            "metadata": {
                "url": f"https://example.invalid/{index}",
                "license": "CC BY 4.0",
            },
        }

    records = [record(index, f"text-{index}") for index in range(10)]
    records[7]["text"] = "parent-shared-text"
    records[9]["text"] = "short"
    records.append(record(10, "post-cutoff", year=2021))
    excluded_id = hashlib.sha256(
        e7._stable_identity(records[8]).encode("utf-8")
    ).hexdigest()
    excluded_text = hashlib.sha256(b"parent-shared-text").hexdigest()

    stream = e7.build_historical_stream(
        records,
        Tokenizer(),
        context_length=2,
        warmup_examples=2,
        online_examples=4,
        historical_cutoff_utc="2021-01-01T00:00:00+00:00",
        offset_salt="fixed",
        excluded_example_ids={excluded_id},
        excluded_text_sha256={excluded_text},
    )
    selected = (*stream.warmup, *stream.online)
    expected_ids = [
        hashlib.sha256(e7._stable_identity(records[index]).encode("utf-8")).hexdigest()
        for index in range(1, 7)
    ]
    assert [item.example_id for item in selected] == expected_ids
    assert all(item.timestamp_utc < "2021-01-01T00:00:00+00:00" for item in selected)
    assert stream.manifest["rejected_parent_example_id"] == 1
    assert stream.manifest["rejected_parent_article_text"] == 1
    assert stream.manifest["rejected_too_short_after_tokenization"] == 1
    assert stream.manifest["eligible_candidates_before_tokenization"] == 8
    assert stream.manifest["tokenized_candidates_examined"] == 8
    assert stream.manifest["selection_depends_on_model_or_outcomes"] is False


def test_exact_parent_counts_prove_post_parent_second_stream_is_infeasible() -> None:
    parent = {
        "records_seen": 172308,
        "rejected_timestamp": 150022,
        "rejected_license": 5191,
        "rejected_duplicate": 830,
        "rejected_empty": 0,
        "rejected_too_short_after_tokenization": 3686,
        "warmup_examples": 4096,
        "online_examples": 8192,
    }
    remaining = e7.post_parent_candidate_upper_bound(parent)
    assert remaining == 291
    assert remaining < 4096
    assert remaining < 12288


def test_bootstrap_installs_patch_sentinel_and_forwards_authenticated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.gradient_reuse_reviewer_extension import transformer_bootstrap

    manifest = tmp_path / "correction.json"
    patch = tmp_path / "correction.py"
    manifest.write_text("{}\n", encoding="utf-8")
    patch.write_text("# correction\n", encoding="utf-8")
    authenticated = {
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": "a" * 64,
        "patch_path": str(patch.resolve()),
        "patch_sha256": "b" * 64,
    }
    provenance = {"authenticated": True}
    activated: list[dict[str, Any]] = []
    forwarded: list[list[str]] = []
    monkeypatch.setattr(transformer_bootstrap, "isolate_import_path", lambda: tmp_path)
    monkeypatch.setattr(
        transformer_bootstrap,
        "validate_correction",
        lambda *args: authenticated,
    )
    monkeypatch.setattr(
        transformer_bootstrap.runpy, "run_path", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        e7,
        "authenticate_correction_files",
        lambda *args: provenance,
    )
    monkeypatch.setattr(
        e7,
        "activate_correction_sentinel",
        lambda value: activated.append(dict(value)),
    )
    monkeypatch.setattr(
        e7,
        "main",
        lambda argv: forwarded.append(list(argv)) or 0,
    )

    assert (
        transformer_bootstrap.main(
            [
                "--correction-manifest",
                str(manifest),
                "--correction-patch",
                str(patch),
                "--",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
        == 0
    )
    assert activated == [provenance]
    assert forwarded == [
        [
            "--correction-manifest",
            str(manifest.resolve()),
            "--correction-patch",
            str(patch.resolve()),
            "--out",
            str(tmp_path / "result.json"),
        ]
    ]


def test_parent_result_is_bound_by_completion_hash_and_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_sha256 = "1" * 64
    result = _parent_result(config_sha256)
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion_path = tmp_path / "DONE.json"
    completion_path.write_text("{}\n", encoding="utf-8")
    completion = {
        "manifest_sha256": "a" * 64,
        "profile_id": "rtx_7day_v1",
        "transformer_attempt_index": 0,
        "artifacts": {
            "transformer_audit": {
                "path": "/sealed/transformer/attempt_0/RESULT.json",
                "sha256": e7.sha256_file(result_path),
            }
        },
    }
    monkeypatch.setattr(e7, "validate_parent_completion", lambda _: completion)
    observed, binding = e7._validate_parent_result(
        completion_path,
        result_path,
        config={
            "stream": {
                "warmup_examples": 2,
                "online_examples": 4,
                "context_length": 4,
            }
        },
        config_sha256=config_sha256,
    )
    assert observed == result
    assert binding["transformer_result_file_sha256"] == e7.sha256_file(result_path)
    assert binding["stream_sha256"] == "4" * 64

    completion["artifacts"]["transformer_audit"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bytes differ"):
        e7._validate_parent_result(
            completion_path,
            result_path,
            config={
                "stream": {
                    "warmup_examples": 2,
                    "online_examples": 4,
                    "context_length": 4,
                }
            },
            config_sha256=config_sha256,
        )


def test_result_validator_rejects_overlap_even_with_resealed_inventory() -> None:
    result = _valid_extension_result()
    e7.validate_stream2_result(
        result, expected_input_bundle=result["validated_input_bundle"]
    )
    result["stream"]["article_text_overlap_count"] = 1
    result["artifact_inventory"] = _artifact_inventory(
        {
            "result_payload_without_inventory": {
                key: value
                for key, value in result.items()
                if key != "artifact_inventory"
            },
            "validated_input_bundle": result["validated_input_bundle"],
            "stream": result["stream"],
            "unit": result["unit"],
        }
    )
    with pytest.raises(RuntimeError, match="second-stream integrity"):
        e7.validate_stream2_result(result)


def test_run_reuses_seed_runtime_assets_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    config_path = repo / "configs" / "transformer.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("lock", "source", "assets", "completion", "parent", "correction")
    }
    paths["patch"] = tmp_path / "correction.py"
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    model_path = tmp_path / "model"
    dataset_path = tmp_path / "dataset"
    model_path.mkdir()
    dataset_path.mkdir()

    config = {
        "runtime": {"device": "cpu"},
        "audit": {"trajectory_seed": 300},
        "optimization": {},
        "assets": {"model": {}, "dataset": {}},
        "model": {
            "dtype": "float32",
            "expected_total_parameter_range": [1, 100],
        },
        "stream": {
            "context_length": 4,
            "warmup_examples": 2,
            "online_examples": 4,
            "packet_size": 2,
            "earliest_utc": "2021-01-01T00:00:00+00:00",
            "offset_salt": "fixed",
        },
    }
    parent_examples = [_example(index + 20) for index in range(6)]
    historical_examples = [_example(index, historical=True) for index in range(6)]
    parent_build = _parent_build(parent_examples)
    historical_build = _historical_build(historical_examples, parent_examples)
    parent_stream = parent_build.manifest
    assets_identity = {
        "model": {
            "repository_id": "model",
            "resolved_revision": "a" * 40,
            "sha256": "7" * 64,
        },
        "dataset": {
            "repository_id": "dataset",
            "resolved_revision": "b" * 40,
            "sha256": "8" * 64,
        },
    }
    observed_environment = {"deterministic": True}
    asset_manifest_hash = e7.sha256_file(paths["assets"])
    parent_result = {
        "stream": parent_stream,
        "assets": assets_identity,
        "environment_provenance": {
            "manifest_sha256": "5" * 64,
            "observed": observed_environment,
        },
        "counterfactual_intervention": {
            "name": "equal_radius_raw_gradient_actuator",
            "is_adamw_counterfactual": False,
        },
        "validated_input_bundle": {
            "inventory": {"asset_manifest": {"file_sha256": asset_manifest_hash}}
        },
    }
    seeds: list[int] = []
    loaded: list[tuple[Path, str]] = []
    run_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(e7, "REPO", repo)
    monkeypatch.setattr(
        e7,
        "load_lock",
        lambda _: {
            "e7_second_pythia_stream": {
                "parent_config": "configs/transformer.json",
                "historical_cutoff_utc": "2021-01-01T00:00:00+00:00",
                "required_examples": 6,
                "warmup_examples": 2,
                "online_examples": 4,
            }
        },
    )
    monkeypatch.setattr(
        e7,
        "_source_binding",
        lambda *args, **kwargs: {
            "schema_version": e7.SOURCE_BINDING_SCHEMA,
            "status": "PASS",
        },
    )
    monkeypatch.setattr(e7, "read_config", lambda _: config)
    monkeypatch.setattr(
        e7,
        "_validate_parent_result",
        lambda *args, **kwargs: (
            parent_result,
            {
                "schema_version": e7.PARENT_BINDING_SCHEMA,
                "config_sha256": e7.sha256_file(config_path),
                "stream_sha256": parent_stream["stream_sha256"],
            },
        ),
    )
    monkeypatch.setattr(e7, "_validate_runtime", lambda *args, **kwargs: None)
    correction = {"status": "AUTHENTICATED", "sentinel": True}
    monkeypatch.setattr(
        e7,
        "require_active_correction",
        lambda *args, **kwargs: correction,
    )
    monkeypatch.setattr(e7, "seed_everything", lambda seed: seeds.append(seed))
    monkeypatch.setattr(
        e7,
        "capture_runtime_environment",
        lambda **kwargs: observed_environment,
    )
    monkeypatch.setattr(
        e7,
        "validate_asset_manifest",
        lambda *args, **kwargs: {
            "model": SimpleNamespace(path=model_path, **assets_identity["model"]),
            "dataset": SimpleNamespace(path=dataset_path, **assets_identity["dataset"]),
        },
    )
    model = torch.nn.Linear(2, 2)

    def fake_load(path: Path, *, dtype: str) -> tuple[torch.nn.Module, object]:
        loaded.append((Path(path), dtype))
        return model, object()

    monkeypatch.setattr(e7, "load_local_pythia", fake_load)
    monkeypatch.setattr(e7, "iter_json_records", lambda _: iter(()))
    monkeypatch.setattr(
        e7,
        "build_chronological_stream",
        lambda *args, **kwargs: parent_build,
    )
    monkeypatch.setattr(
        e7,
        "build_historical_stream",
        lambda *args, **kwargs: historical_build,
    )

    def fake_run_unit(**kwargs: Any) -> dict[str, Any]:
        run_calls.append(kwargs)
        return {"audit": {"all_pass": True, "zero_norm_actuator_branches": 0}}

    monkeypatch.setattr(e7, "_run_unit", fake_run_unit)
    monkeypatch.setattr(e7, "validate_stream2_result", lambda *args, **kwargs: None)
    result = e7.run_stream2(
        lock_path=paths["lock"],
        source_manifest=paths["source"],
        config_path=config_path,
        asset_manifest=paths["assets"],
        parent_completion=paths["completion"],
        parent_transformer_result=paths["parent"],
        correction_manifest=paths["correction"],
        correction_patch=paths["patch"],
        device="cpu",
    )
    assert seeds == [300]
    assert loaded == [(model_path, "float32")]
    assert run_calls[0]["model"] is model
    assert run_calls[0]["config"] is config
    assert run_calls[0]["seed"] == 300
    assert run_calls[0]["stream"].warmup == tuple(historical_examples[:2])
    assert result["assets"] == assets_identity
    assert result["environment_provenance"]["exact_parent_runtime_reused"] is True


def test_cli_writes_atomic_result_and_terminal_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = []
    for name in (
        "lock",
        "source",
        "config",
        "assets",
        "completion",
        "parent",
        "correction",
        "patch",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "RESULT.json"
    heartbeat = tmp_path / "RESULT.heartbeat.json"
    monkeypatch.setattr(e7, "run_stream2", lambda **kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        e7, "_reject_artifacts_inside_asset_roots", lambda *args, **kwargs: None
    )
    arguments = [
        "--lock",
        str(inputs[0]),
        "--source-manifest",
        str(inputs[1]),
        "--config",
        str(inputs[2]),
        "--asset-manifest",
        str(inputs[3]),
        "--parent-completion",
        str(inputs[4]),
        "--parent-transformer-result",
        str(inputs[5]),
        "--correction-manifest",
        str(inputs[6]),
        "--correction-patch",
        str(inputs[7]),
        "--device",
        "cpu",
        "--out",
        str(output),
        "--heartbeat",
        str(heartbeat),
    ]
    assert e7.main(arguments) == 0
    assert json.loads(output.read_text()) == {"status": "PASS"}
    state = json.loads(heartbeat.read_text())
    assert state["status"] == "COMPLETE"
    assert state["output_file_sha256"] == e7.sha256_file(output)
    assert not list(tmp_path.glob(".*.tmp"))
    completed_output = output.read_bytes()
    with pytest.raises(RuntimeError, match="existing output"):
        e7.main(arguments)
    assert output.read_bytes() == completed_output


def test_cli_records_failure_without_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for index in range(8):
        path = tmp_path / f"input-{index}.json"
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)
    output = tmp_path / "failed.json"
    heartbeat = tmp_path / "failed.heartbeat.json"

    def fail(**_: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(e7, "run_stream2", fail)
    monkeypatch.setattr(
        e7, "_reject_artifacts_inside_asset_roots", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        e7.main(
            [
                "--lock",
                str(paths[0]),
                "--source-manifest",
                str(paths[1]),
                "--config",
                str(paths[2]),
                "--asset-manifest",
                str(paths[3]),
                "--parent-completion",
                str(paths[4]),
                "--parent-transformer-result",
                str(paths[5]),
                "--correction-manifest",
                str(paths[6]),
                "--correction-patch",
                str(paths[7]),
                "--device",
                "cpu",
                "--out",
                str(output),
                "--heartbeat",
                str(heartbeat),
            ]
        )
    assert not output.exists()
    state = json.loads(heartbeat.read_text())
    assert state["status"] == "FAILED"
    assert state["error_type"] == "RuntimeError"
    assert state["error"] == "synthetic failure"
