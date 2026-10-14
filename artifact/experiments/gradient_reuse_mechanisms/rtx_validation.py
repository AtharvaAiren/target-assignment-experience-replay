"""Deep validators for resumable artifacts in the combined RTX profile.

The combined launcher may skip an expensive stage only after revalidating the
same configuration, source freeze, runtime capture, and external-data binding
used to create the result.  These checks deliberately live outside the
scientific runners: they authenticate persisted evidence without recomputing a
scientific campaign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_controller import core as controller_core
from experiments.gradient_reuse_controller import natural as natural_controller
from experiments.gradient_reuse_controller.provenance import (
    canonical_sha256,
    evidence_payload,
    validate_runtime_manifest,
    validate_source_binding,
)
from experiments.gradient_reuse_transformer.assets import validate_asset_manifest
from experiments.gradient_reuse_transformer import run as transformer_run
from experiments.source_provenance import SourceProvenanceError, strict_json


class RTXArtifactValidationError(RuntimeError):
    """A resumable artifact does not match the current locked inputs."""


def _object(path: Path, *, name: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        value = strict_json(resolved)
    except (OSError, SourceProvenanceError) as error:
        raise RTXArtifactValidationError(f"cannot read {name}: {resolved}") from error
    if not isinstance(value, dict):
        raise RTXArtifactValidationError(f"{name} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RTXArtifactValidationError(message)


def _source_and_runtime(
    *,
    experiment: str,
    config_path: Path,
    config: Mapping[str, Any],
    source_manifest: Path,
    runtime_manifest: Path,
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = validate_source_binding(
        source_manifest,
        experiment=experiment,
        config_path=config_path,
        repo=repo,
    )
    source_payload = evidence_payload(source, kind="source")
    _require(source_payload is not None, "source binding did not validate")
    runtime = validate_runtime_manifest(
        runtime_manifest,
        experiment=experiment,
        config_path=config_path,
        config=config,
        source_evidence=source_payload,
    )
    runtime_payload = evidence_payload(runtime, kind="runtime")
    _require(runtime_payload is not None, "runtime binding did not validate")
    return source_payload, runtime_payload


def validate_controller_result(
    result_path: Path,
    *,
    config_path: Path,
    source_manifest: Path,
    runtime_manifest: Path,
    repo: Path,
) -> dict[str, Any]:
    """Authenticate one full synthetic-controller result against live inputs."""

    config_path = config_path.expanduser().resolve(strict=True)
    config = controller_core.load_config(config_path)
    source, runtime = _source_and_runtime(
        experiment="controller",
        config_path=config_path,
        config=config,
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        repo=repo,
    )
    result = _object(result_path, name="synthetic controller result")
    required = {
        "schema_version",
        "status",
        "mode",
        "scientific_evidence_eligible",
        "warning",
        "source_provenance",
        "runtime_provenance",
        "configuration_sha256",
        "policies",
        "prespecified_ablations",
        "streams",
        "run_spec",
        "comparisons",
        "integrity",
        "units",
        "source_validation_attempt",
    }
    _require(set(result) == required, "synthetic controller result schema changed")
    _require(
        result["schema_version"] == "gradient_reuse.controller.results.v3"
        and result["status"] == "PASS"
        and result["mode"] == "full"
        and result["scientific_evidence_eligible"] is True
        and result["warning"] is None,
        "synthetic controller result is not a full evidence-eligible PASS",
    )
    _require(
        result["configuration_sha256"] == canonical_sha256(config),
        "synthetic controller result authenticates another configuration",
    )
    _require(
        result["source_provenance"] == source
        and result["source_validation_attempt"] == source
        and result["runtime_provenance"] == runtime,
        "synthetic controller provenance differs from current validated inputs",
    )
    _require(
        result["run_spec"] == config["full"]
        and result["policies"] == list(controller_core.CONTROLLER_POLICIES),
        "synthetic controller full tier or policies changed",
    )
    integrity = result["integrity"]
    _require(isinstance(integrity, Mapping), "synthetic controller integrity is absent")
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
    ):
        _require(integrity.get(key) is True, f"synthetic controller integrity failed: {key}")
    _require(isinstance(result["units"], list) and result["units"], (
        "synthetic controller result has no scientific units"
    ))
    return result


def validate_natural_controller_result(
    result_path: Path,
    *,
    config_path: Path,
    source_manifest: Path,
    runtime_manifest: Path,
    data_manifest: Path,
    data_archive: Path,
    repo: Path,
) -> dict[str, Any]:
    """Authenticate one full Beijing-controller result against live inputs."""

    config_path = config_path.expanduser().resolve(strict=True)
    config = natural_controller.load_config(config_path)
    source, runtime = _source_and_runtime(
        experiment="controller_natural",
        config_path=config_path,
        config=config,
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        repo=repo,
    )
    data_binding, _ = natural_controller.validate_beijing_data_manifest(
        data_manifest,
        archive_path=data_archive,
        config_path=config_path,
        config=config,
        source_evidence=source,
    )
    data = evidence_payload(data_binding, kind="data")
    _require(data is not None, "Beijing data binding did not validate")
    result = _object(result_path, name="Beijing controller result")
    required = {
        "schema_version",
        "status",
        "mode",
        "scientific_evidence_eligible",
        "warning",
        "source_provenance",
        "runtime_provenance",
        "data_provenance",
        "configuration_sha256",
        "source_identity",
        "run_spec",
        "dataset",
        "stream_identities_by_seed",
        "common_radius_schedules_by_seed",
        "policies",
        "month_ids",
        "units",
        "comparisons_vs_never",
        "controller_wedge_comparisons",
        "natural_mean_drift_evidence_by_policy",
        "integrity",
        "source_validation_attempt",
    }
    _require(set(result) == required, "Beijing controller result schema changed")
    _require(
        result["schema_version"]
            == natural_controller.RESULT_SCHEMA_VERSION
        and result["status"] == "PASS"
        and result["mode"] == "full"
        and result["dataset"] == "BEIJING"
        and result["scientific_evidence_eligible"] is True
        and result["warning"] is None,
        "Beijing controller result is not a full evidence-eligible PASS",
    )
    _require(
        result["configuration_sha256"] == canonical_sha256(config)
        and result["source_identity"] == config["source_identity"],
        "Beijing controller result authenticates another configuration",
    )
    _require(
        result["source_provenance"] == source
        and result["source_validation_attempt"] == source
        and result["runtime_provenance"] == runtime
        and result["data_provenance"] == data,
        "Beijing controller provenance differs from current validated inputs",
    )
    _require(
        result["run_spec"] == config["full"]
        and result["policies"] == list(natural_controller.POLICIES),
        "Beijing controller full tier or policies changed",
    )
    integrity = result["integrity"]
    _require(isinstance(integrity, Mapping), "Beijing controller integrity is absent")
    for key in (
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
    ):
        _require(integrity.get(key) is True, f"Beijing controller integrity failed: {key}")
    _require(isinstance(result["units"], list) and result["units"], (
        "Beijing controller result has no scientific units"
    ))
    return result


def _validated_transformer_context(
    *,
    config_path: Path,
    source_manifest: Path,
    environment_manifest: Path,
    asset_manifest: Path,
    preflight_report: Path,
    device: str,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild every path-free identity needed by a Transformer result."""

    config_path = config_path.expanduser().resolve(strict=True)
    config = transformer_run.read_config(config_path)
    config_sha256 = transformer_run._sha256_file(config_path)
    source = transformer_run.validate_transformer_source_manifest(
        source_manifest, config_path=config_path, required=True
    )
    transformer_run._validate_asset_source_binding(asset_manifest, source)
    transformer_run._validate_runtime(config, device)
    assets = validate_asset_manifest(
        asset_manifest,
        config["assets"],
        expected_config_sha256=config_sha256,
    )
    observed = transformer_run.capture_runtime_environment(
        device=device,
        optimizer_specification=config["optimization"],
    )
    environment = transformer_run.validate_environment_manifest(
        environment_manifest,
        expected_config_sha256=config_sha256,
        expected_source_provenance=source,
        observed=observed,
    )
    asset_identity = transformer_run._asset_identity(assets)
    preflight_inputs = transformer_run.build_validated_input_bundle(
        {
            "configuration": config_path,
            "source_manifest": source_manifest,
            "environment_manifest": environment_manifest,
            "asset_manifest": asset_manifest,
        }
    )
    transformer_run.validate_embedded_input_bundle(preflight_inputs)
    preflight = transformer_run.validate_cuda_preflight_report(
        preflight_report,
        config=config,
        expected_config_sha256=config_sha256,
        expected_source_provenance=source,
        expected_environment_provenance=environment,
        expected_assets=asset_identity,
        expected_input_bundle=preflight_inputs,
    )
    final_inputs = transformer_run.build_validated_input_bundle(
        {
            "configuration": config_path,
            "source_manifest": source_manifest,
            "environment_manifest": environment_manifest,
            "asset_manifest": asset_manifest,
            "cuda_preflight_report": preflight_report,
        }
    )
    transformer_run.validate_embedded_input_bundle(final_inputs)
    return config_sha256, source, environment, asset_identity, preflight, final_inputs


def validate_transformer_preflight(
    preflight_report: Path,
    *,
    config_path: Path,
    source_manifest: Path,
    environment_manifest: Path,
    asset_manifest: Path,
    device: str,
) -> dict[str, Any]:
    """Authenticate a selected CUDA preflight against current locked inputs."""

    context = _validated_transformer_context(
        config_path=config_path,
        source_manifest=source_manifest,
        environment_manifest=environment_manifest,
        asset_manifest=asset_manifest,
        preflight_report=preflight_report,
        device=device,
    )
    return context[4]


def validate_transformer_result(
    result_path: Path,
    *,
    config_path: Path,
    source_manifest: Path,
    environment_manifest: Path,
    asset_manifest: Path,
    preflight_report: Path,
    device: str,
) -> dict[str, Any]:
    """Authenticate a Pythia result without rerunning the model trajectory."""

    (
        config_sha256,
        source,
        environment,
        asset_identity,
        preflight,
        final_inputs,
    ) = _validated_transformer_context(
        config_path=config_path,
        source_manifest=source_manifest,
        environment_manifest=environment_manifest,
        asset_manifest=asset_manifest,
        preflight_report=preflight_report,
        device=device,
    )

    result = _object(result_path, name="Transformer result")
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
    _require(set(result) == required, "Transformer result schema changed")
    _require(
        result["schema_version"] == "fmmt.gradient_reuse_transformer.campaign.v3"
        and result["status"] == "PASS"
        and result["paper_evidence_eligible"] is True
        and result["independent_model_trajectories"] == 1
        and result["seed_level_replication"] is False
        and result["permutation_replicates_are_nested_within_packet"] is True,
        "Transformer result is not the full paper-eligible campaign",
    )
    transformer_run.validate_artifact_inventory(result)
    embedded = result["validated_input_bundle"]
    _require(isinstance(embedded, Mapping), "Transformer result has no input bundle")
    transformer_run.validate_embedded_input_bundle(embedded)
    _require(
        result["config_sha256"] == config_sha256
        and result["source_provenance"] == source
        and result["environment_provenance"] == environment
        and result["cuda_preflight_provenance"] == preflight
        and result["assets"] == asset_identity
        and embedded == final_inputs,
        "Transformer result provenance differs from current validated inputs",
    )
    _require(
        result["execution"] == {
            "resume_supported": False,
            "interruption_policy": (
                "restart from the immutable inputs after interruption"
            ),
            "heartbeat_is_progress_evidence_only": True,
        },
        "Transformer result execution contract changed",
    )
    audit = result.get("unit", {}).get("audit")
    _require(
        isinstance(audit, Mapping)
        and audit.get("all_pass") is True
        and audit.get("zero_norm_actuator_branches") == 0
        and audit.get("zero_norm_actuator_branch_fail_closed") is True,
        "Transformer result scientific audit did not pass fail-closed",
    )
    return result
