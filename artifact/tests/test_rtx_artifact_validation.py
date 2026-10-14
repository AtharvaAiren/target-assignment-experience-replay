"""Fail-closed resume checks for the combined RTX campaign."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_controller import core as controller_core
from experiments.gradient_reuse_controller import natural as natural_controller
from experiments.gradient_reuse_controller.provenance import canonical_sha256
from experiments.gradient_reuse_mechanisms import rtx_validation
from experiments.gradient_reuse_transformer import run as transformer_run


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _synthetic_result(config: dict, source: dict, runtime: dict) -> dict:
    return {
        "schema_version": "gradient_reuse.controller.results.v3",
        "status": "PASS",
        "mode": "full",
        "scientific_evidence_eligible": True,
        "warning": None,
        "source_provenance": source,
        "runtime_provenance": runtime,
        "configuration_sha256": canonical_sha256(config),
        "policies": list(controller_core.CONTROLLER_POLICIES),
        "prespecified_ablations": {},
        "streams": ["mean_drift"],
        "run_spec": config["full"],
        "comparisons": {},
        "integrity": {
            key: True
            for key in (
                "locked_configuration",
                "all_predictions_prequential",
                "all_v_shadows_action_aligned",
                "persistent_reversal_regimes",
                "seed_level_inference",
                "v_identity_exact",
                "all_outputs_finite",
                "validated_source_binding",
                "validated_runtime_binding",
                "provenance_bindings_consistent",
                "shared_source_manifest_validated",
            )
        },
        "units": [{"unit": 1}],
        "source_validation_attempt": source,
    }


def test_controller_resume_validator_rejects_tamper_and_stale_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"full": {"seeds": [1]}, "ablations": {}}
    source = {"source": "locked"}
    runtime = {"runtime": "locked"}
    monkeypatch.setattr(controller_core, "load_config", lambda _: config)
    monkeypatch.setattr(
        rtx_validation,
        "_source_and_runtime",
        lambda **_: (source, runtime),
    )
    common = {
        "config_path": tmp_path / "config.json",
        "source_manifest": tmp_path / "source.json",
        "runtime_manifest": tmp_path / "runtime.json",
        "repo": tmp_path,
    }
    for path in common.values():
        if isinstance(path, Path) and path.suffix:
            path.write_text("{}\n")
    result = _write(tmp_path / "result.json", _synthetic_result(config, source, runtime))
    assert rtx_validation.validate_controller_result(result, **common)["status"] == "PASS"

    tampered = _synthetic_result(config, source, runtime)
    tampered["source_provenance"] = {"source": "changed"}
    _write(result, tampered)
    with pytest.raises(rtx_validation.RTXArtifactValidationError, match="provenance"):
        rtx_validation.validate_controller_result(result, **common)

    monkeypatch.setattr(
        rtx_validation,
        "_source_and_runtime",
        lambda **_: (_ for _ in ()).throw(RuntimeError("stale source freeze")),
    )
    with pytest.raises(RuntimeError, match="stale source freeze"):
        rtx_validation.validate_controller_result(result, **common)


def _natural_result(config: dict, source: dict, runtime: dict, data: dict) -> dict:
    required_integrity = (
        "strictly_prequential",
        "gate_after_scoring",
        "action_first_affects_next_packet",
        "one_true_adamw_update_per_packet",
        "equal_radius_for_open_reuse_actions",
        "common_never_reuse_radius_schedule_across_policies",
        "no_raw_replay",
        "same_seed_and_data_order_across_policies",
        "v_total_equals_v_perm_plus_v_pair",
        "v_shadows_match_post_current_update_action_timing",
        "locked_source_identity_verified",
        "validated_source_binding",
        "validated_runtime_binding",
        "validated_data_binding",
        "provenance_bindings_consistent",
        "shared_source_manifest_validated",
    )
    return {
        "schema_version": natural_controller.RESULT_SCHEMA_VERSION,
        "status": "PASS",
        "mode": "full",
        "scientific_evidence_eligible": True,
        "warning": None,
        "source_provenance": source,
        "runtime_provenance": runtime,
        "data_provenance": data,
        "configuration_sha256": canonical_sha256(config),
        "source_identity": config["source_identity"],
        "run_spec": config["full"],
        "dataset": "BEIJING",
        "stream_identities_by_seed": {},
        "common_radius_schedules_by_seed": {},
        "policies": list(natural_controller.POLICIES),
        "month_ids": [],
        "units": [{"unit": 1}],
        "comparisons_vs_never": {},
        "controller_wedge_comparisons": {},
        "natural_mean_drift_evidence_by_policy": {},
        "integrity": {key: True for key in required_integrity},
        "source_validation_attempt": source,
    }


def test_natural_controller_resume_validator_rejects_data_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"full": {"seeds": [1]}, "source_identity": "beijing"}
    source = {"source": "locked"}
    runtime = {"runtime": "locked"}
    data = {"data": "locked"}
    monkeypatch.setattr(natural_controller, "load_config", lambda _: config)
    monkeypatch.setattr(
        rtx_validation,
        "_source_and_runtime",
        lambda **_: (source, runtime),
    )
    monkeypatch.setattr(
        natural_controller,
        "validate_beijing_data_manifest",
        lambda *_, **__: (object(), {}),
    )
    original_evidence_payload = rtx_validation.evidence_payload
    monkeypatch.setattr(
        rtx_validation,
        "evidence_payload",
        lambda binding, *, kind: data
        if kind == "data"
        else original_evidence_payload(binding, kind=kind),
    )
    paths = {
        name: _write(tmp_path / f"{name}.json", {})
        for name in ("config", "source", "runtime", "data", "archive")
    }
    result = _write(
        tmp_path / "result.json",
        _natural_result(config, source, runtime, data),
    )
    kwargs = {
        "config_path": paths["config"],
        "source_manifest": paths["source"],
        "runtime_manifest": paths["runtime"],
        "data_manifest": paths["data"],
        "data_archive": paths["archive"],
        "repo": tmp_path,
    }
    assert rtx_validation.validate_natural_controller_result(result, **kwargs)[
        "dataset"
    ] == "BEIJING"
    changed = _natural_result(config, source, runtime, data)
    changed["data_provenance"] = {"data": "changed"}
    _write(result, changed)
    with pytest.raises(rtx_validation.RTXArtifactValidationError, match="provenance"):
        rtx_validation.validate_natural_controller_result(result, **kwargs)


def _input_bundle() -> dict:
    payloads = {"configuration": {"schema_version": "test"}}
    inventory = {
        "configuration": {
            "file_sha256": "a" * 64,
            "canonical_payload_sha256": transformer_run._canonical_sha256(
                payloads["configuration"]
            ),
            "size_bytes": 2,
        }
    }
    return {
        "schema_version": "fmmt.gradient_reuse_transformer.input_bundle.v1",
        "payloads": payloads,
        "inventory": inventory,
        "bundle_sha256": transformer_run._canonical_sha256(payloads),
    }


def _transformer_result(context: tuple) -> dict:
    config_sha, source, environment, assets, preflight, inputs = context
    result = {
        "schema_version": "fmmt.gradient_reuse_transformer.campaign.v3",
        "status": "PASS",
        "paper_evidence_eligible": True,
        "independent_model_trajectories": 1,
        "seed_level_replication": False,
        "permutation_replicates_are_nested_within_packet": True,
        "config_sha256": config_sha,
        "source_provenance": source,
        "environment_provenance": environment,
        "cuda_preflight_provenance": preflight,
        "assets": assets,
        "validated_input_bundle": inputs,
        "stream": {"stream": "locked"},
        "execution": {
            "resume_supported": False,
            "interruption_policy": "restart from the immutable inputs after interruption",
            "heartbeat_is_progress_evidence_only": True,
        },
        "counterfactual_intervention": {"name": "locked"},
        "unit": {
            "audit": {
                "all_pass": True,
                "zero_norm_actuator_branches": 0,
                "zero_norm_actuator_branch_fail_closed": True,
            }
        },
    }
    result["artifact_inventory"] = transformer_run._artifact_inventory(
        {
            "result_payload_without_inventory": result,
            "validated_input_bundle": inputs,
            "stream": result["stream"],
            "unit": result["unit"],
            "cuda_preflight_provenance": preflight,
        }
    )
    return result


def test_transformer_resume_validator_rejects_inventory_tamper_and_stale_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _input_bundle()
    context = (
        "a" * 64,
        {"source": "locked"},
        {"environment": "locked"},
        {"model": {"sha256": "b" * 64}},
        {"preflight": "locked"},
        inputs,
    )
    monkeypatch.setattr(
        rtx_validation,
        "_validated_transformer_context",
        lambda **_: context,
    )
    result = _write(tmp_path / "result.json", _transformer_result(context))
    kwargs = {
        "config_path": tmp_path / "config.json",
        "source_manifest": tmp_path / "source.json",
        "environment_manifest": tmp_path / "environment.json",
        "asset_manifest": tmp_path / "assets.json",
        "preflight_report": tmp_path / "preflight.json",
        "device": "cuda",
    }
    assert rtx_validation.validate_transformer_result(result, **kwargs)["status"] == "PASS"

    tampered = deepcopy(_transformer_result(context))
    tampered["unit"]["audit"]["all_pass"] = False
    _write(result, tampered)
    with pytest.raises(RuntimeError, match="component inventory"):
        rtx_validation.validate_transformer_result(result, **kwargs)

    monkeypatch.setattr(
        rtx_validation,
        "_validated_transformer_context",
        lambda **_: (_ for _ in ()).throw(RuntimeError("stale source freeze")),
    )
    with pytest.raises(RuntimeError, match="stale source freeze"):
        rtx_validation.validate_transformer_result(result, **kwargs)
