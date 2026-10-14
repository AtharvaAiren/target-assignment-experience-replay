from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_crossover_analysis.analysis import (
    CROSSOVER_STRENGTHS_BP,
    FULL_GRID_BP,
    AnalysisError,
    _parser,
    _raw_crossing_audit,
    analyze_campaign_bundle,
    adapt_assignment_strength_history,
    adapt_corrected_pmnist_history,
    adapt_followup_aggregate,
    analyze_crossover,
    estimate_crossing,
    induced_strength,
    isotonic_nonincreasing,
    load_authenticated_analysis_inputs,
    support_size,
)
from experiments.gradient_reuse_crossover.protocol import (
    canonical_sha256,
    enumerate_units,
    load_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def test_support_grid_fills_every_attainable_level_between_75_and_100() -> None:
    strengths = (7500, *CROSSOVER_STRENGTHS_BP, 10000)
    assert [support_size(32, strength) for strength in strengths] == list(range(24, 33))
    assert [induced_strength(32, strength) for strength in strengths] == pytest.approx(
        [value / 31 for value in range(23, 32)]
    )


def test_nonincreasing_pava_and_crossing_censoring() -> None:
    fitted = isotonic_nonincreasing([4.0, 3.0, 3.5, 1.0, -1.0])
    assert fitted == pytest.approx([4.0, 3.25, 3.25, 1.0, -1.0])
    crossing = estimate_crossing([0.0, 0.25, 0.5, 0.75, 1.0], fitted)
    assert crossing.kind == "finite"
    assert crossing.value == pytest.approx(0.875)
    assert estimate_crossing([0.0, 1.0], [-1.0, -2.0]).kind == "left_censored"
    assert estimate_crossing([0.0, 1.0], [2.0, 1.0]).kind == "right_censored"


def _synthetic_inputs():
    references = [
        {"seed": seed, "final_average_accuracy": 40.0 + 0.1 * index}
        for index, seed in enumerate(range(9001, 9013))
    ]
    all_strengths = (7500, *CROSSOVER_STRENGTHS_BP, 10000)
    historical = []
    followup = []
    for index, seed in enumerate(range(9001, 9013)):
        reference = references[index]["final_average_accuracy"]
        seed_shift = (index - 5.5) * 0.03
        for strength in all_strengths:
            x = induced_strength(32, strength)
            contrast = 26.0 * (0.87 - x) + seed_shift
            target = historical if strength in {7500, 10000} else followup
            if target is historical:
                target.append(
                    {
                        "seed": seed,
                        "requested_strength_bp": strength,
                        "induced_strength": x,
                        "selected_rows": support_size(32, strength),
                        "replay_minibatch_size": 32,
                        "final_average_accuracy": reference + contrast,
                        "assignment_replica": None,
                        "replica_averaged": True,
                        "nested_replica_count": 4,
                        "actual_changed_label_fraction": None,
                        "source": "history",
                    }
                )
            else:
                for replica in range(4):
                    target.append(
                        {
                            "seed": seed,
                            "requested_strength_bp": strength,
                            "induced_strength": x,
                            "selected_rows": support_size(32, strength),
                            "replay_minibatch_size": 32,
                            "final_average_accuracy": reference
                            + contrast
                            + (replica - 1.5) * 0.02,
                            "assignment_replica": replica,
                            "replica_averaged": False,
                            "nested_replica_count": 1,
                            "actual_changed_label_fraction": 0.8 * x,
                            "source": "followup",
                        }
                    )
    return references, historical, followup


def test_analysis_is_seed_matched_deterministic_and_plot_ready() -> None:
    references, historical, followup = _synthetic_inputs()
    first = analyze_crossover(
        campaign="pmnist",
        reference_rows=references,
        historical_records=historical,
        followup_records=followup,
        bootstrap_replicates=500,
        bootstrap_seed=12345,
    )
    second = analyze_crossover(
        campaign="pmnist",
        reference_rows=references,
        historical_records=historical,
        followup_records=followup,
        bootstrap_replicates=500,
        bootstrap_seed=12345,
    )
    assert first == second
    assert first["crossing"]["kind"] == "finite"
    assert first["crossing"]["value"] == pytest.approx(0.87, abs=0.002)
    assert first["crossing"]["secondary_interpolation_available"] is True
    assert first["crossing"]["fitted_bracket_agrees_with_primary"] is True
    assert first["primary_crossing"]["kind"] == "unique_discrete_bracket"
    assert first["primary_crossing"]["unique_threshold_justified"] is True
    assert first["raw_curve_audit"]["material_reversal_count"] == 0
    assert len(first["plot_rows"]) == 9
    assert all(len(row["seed_contrasts"]) == 12 for row in first["plot_rows"])
    assert [row["selected_rows"] for row in first["plot_rows"]] == list(
        range(24, 33)
    )
    assert [row["induced_support_numerator"] for row in first["plot_rows"]] == list(
        range(23, 32)
    )
    assert first["bootstrap"]["censoring_counts"]["finite"] == 500
    assert first["all_required_dose_descriptors_complete"] is False
    assert first["dose_descriptor_availability"][
        "actual_changed_label_fraction"
    ]["missing_requested_strength_bp"] == [7500, 10000]
    assert (
        first["bootstrap"]["simultaneous_band"]["half_width_accuracy_points"]
        > 0.0
    )


def test_material_reversal_demotes_isotonic_root_to_sensitivity() -> None:
    references, historical, followup = _synthetic_inputs()
    desired = [2.0, -1.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    ordered_strengths = (7500, *CROSSOVER_STRENGTHS_BP, 10000)
    desired_by_strength = dict(zip(ordered_strengths, desired))
    reference_by_seed = {
        row["seed"]: row["final_average_accuracy"] for row in references
    }
    for row in [*historical, *followup]:
        seed_offset = (row["seed"] - 9001 - 5.5) * 0.03
        replica_offset = (
            0.0
            if row["replica_averaged"]
            else (row["assignment_replica"] - 1.5) * 0.02
        )
        row["final_average_accuracy"] = (
            reference_by_seed[row["seed"]]
            + desired_by_strength[row["requested_strength_bp"]]
            + seed_offset
            + replica_offset
        )
    result = analyze_crossover(
        campaign="pmnist",
        reference_rows=references,
        historical_records=historical,
        followup_records=followup,
        bootstrap_replicates=100,
        bootstrap_seed=20261008,
    )
    assert result["raw_curve_audit"]["material_reversal_count"] >= 1
    assert result["raw_curve_audit"]["raw_positive_to_nonpositive_bracket_count"] == 2
    assert result["primary_crossing"]["kind"] == "crossing_set"
    assert result["primary_crossing"]["unique_threshold_justified"] is False
    assert result["crossing"]["reporting_role"].startswith("sensitivity_only")


def test_isotonic_interpolation_must_match_the_unique_raw_bracket() -> None:
    references, historical, followup = _synthetic_inputs()
    strengths = (7500, *CROSSOVER_STRENGTHS_BP, 10000)
    means = [1.0, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    amplitudes = [0.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
    centered = [-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    mean_by_strength = dict(zip(strengths, means))
    amplitude_by_strength = dict(zip(strengths, amplitudes))
    reference_by_seed = {
        row["seed"]: row["final_average_accuracy"] for row in references
    }
    centered_by_seed = dict(zip(sorted(reference_by_seed), centered))
    for row in [*historical, *followup]:
        replica_offset = (
            0.0
            if row["replica_averaged"]
            else (row["assignment_replica"] - 1.5) * 0.02
        )
        strength = row["requested_strength_bp"]
        row["final_average_accuracy"] = (
            reference_by_seed[row["seed"]]
            + mean_by_strength[strength]
            + amplitude_by_strength[strength] * centered_by_seed[row["seed"]]
            + replica_offset
        )
    result = analyze_crossover(
        campaign="pmnist",
        reference_rows=references,
        historical_records=historical,
        followup_records=followup,
        bootstrap_replicates=50,
        bootstrap_seed=20261008,
    )
    assert result["raw_curve_audit"]["material_reversal_count"] == 0
    assert result["primary_crossing"]["kind"] == "unique_discrete_bracket"
    assert result["crossing"]["kind"] == "right_censored"
    assert result["crossing"]["secondary_interpolation_available"] is False
    assert result["crossing"]["fitted_bracket_agrees_with_primary"] is False
    assert (
        result["crossing"]["reporting_role"]
        == "unavailable_discordant_isotonic_sensitivity"
    )
    assert result["crossing"]["requested_strength_interpolation"] is None


def test_raw_crossing_uses_positive_versus_nonpositive_state() -> None:
    grid = [induced_strength(32, strength) for strength in FULL_GRID_BP]
    means = [2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
    audit = _raw_crossing_audit(
        strengths=FULL_GRID_BP,
        grid=grid,
        means=means,
        profiles=[means[:] for _ in range(12)],
    )
    assert audit["raw_positive_to_nonpositive_bracket_count"] == 1
    assert len(audit["raw_sign_change_brackets"]) == 1
    assert audit["primary_crossing"]["kind"] == "unique_discrete_bracket"


def test_multiple_raw_downward_crossings_yield_crossing_set_without_material_reversal() -> None:
    grid = [induced_strength(32, strength) for strength in FULL_GRID_BP]
    means = [2.0, -1.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    spread = [-11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    profiles = []
    for perturbation in spread:
        profile = means[:]
        # Widen the uncertain upward step while retaining its raw mean.
        profile[2] += perturbation
        profiles.append(profile)
    audit = _raw_crossing_audit(
        strengths=FULL_GRID_BP,
        grid=grid,
        means=means,
        profiles=profiles,
    )
    assert audit["material_reversal_count"] == 0
    assert audit["raw_positive_to_nonpositive_bracket_count"] == 2
    assert audit["primary_crossing"]["kind"] == "crossing_set"


def test_real_historical_adapters_recover_locked_seed_panels() -> None:
    pmnist = json.loads(
        (ROOT / "results/corrected_reference_order/pmnist/AGGREGATE.json").read_text()
    )
    pmnist_adapted = adapt_corrected_pmnist_history(pmnist)
    assert pmnist_adapted["campaign"] == "pmnist"
    assert len(pmnist_adapted["reference_rows"]) == 12
    assert len(pmnist_adapted["sensitivity_reference_rows"]) == 12
    assert len(pmnist_adapted["records"]) == 24

    assignment = json.loads((ROOT / "results/assignment_strength/aggregate.json").read_text())
    cifar_adapted = adapt_assignment_strength_history(assignment)
    assert cifar_adapted["campaign"] == "cifar"
    assert [row["seed"] for row in cifar_adapted["reference_rows"]] == list(
        range(3501, 3513)
    )
    assert len(cifar_adapted["records"]) == 24
    assert len(cifar_adapted["secondary_reference_rows"]) == 12
    assert len(cifar_adapted["secondary_records"]) == 24
    assert cifar_adapted["secondary_endpoint"] == "class_il"
    assert all(
        row["actual_changed_label_fraction"] is not None
        for row in cifar_adapted["records"]
    )


def test_followup_adapter_requires_all_cells_and_digest_bindings() -> None:
    _, _, records = _synthetic_inputs()
    raw = []
    for index, row in enumerate(records):
        raw.append(
            {
                "seed": row["seed"],
                "assignment_replica": row["assignment_replica"],
                "requested_strength_bp": row["requested_strength_bp"],
                "selected_rows": row["selected_rows"],
                "replay_minibatch_size": row["replay_minibatch_size"],
                "induced_strength": row["induced_strength"],
                "actual_changed_label_fraction": row["actual_changed_label_fraction"],
                "final_average_accuracy": row["final_average_accuracy"],
                "endpoint_values": {
                    "domain_il": row["final_average_accuracy"],
                },
                "unit_id": f"unit-{index}",
                "digests": {
                    "unit_json_sha256": f"{index:064x}",
                    "unit_done_sha256": f"{index + 1:064x}",
                    "artifact_bundle_sha256": f"{index + 2:064x}",
                },
            }
        )
    aggregate = {
        "schema_version": "gradient_reuse.crossover_aggregate.v1",
        "status": "PASS",
        "campaign": "pmnist",
        "complete_grid_required": True,
        "unit_count": 336,
        "unit_ids_sha256": canonical_sha256(
            sorted(row["unit_id"] for row in raw)
        ),
        "replicas_are_nested_monte_carlo": True,
        "records": raw,
    }
    assert len(adapt_followup_aggregate(aggregate)["records"]) == 336
    cifar = copy.deepcopy(aggregate)
    cifar["campaign"] = "cifar"
    for row in cifar["records"]:
        row["endpoint_values"] = {
            "task_il": row["final_average_accuracy"],
            "class_il": row["final_average_accuracy"] - 10.0,
        }
    cifar_adapted = adapt_followup_aggregate(cifar)
    assert len(cifar_adapted["secondary_records"]) == 336
    assert cifar_adapted["secondary_endpoint"] == "class_il"
    bad = copy.deepcopy(aggregate)
    bad["records"][0]["digests"] = {}
    with pytest.raises(AnalysisError, match="digest binding"):
        adapt_followup_aggregate(bad)


def test_campaign_bundle_keeps_pmnist_control_sensitivity_separate() -> None:
    references, historical_records, followup_records = _synthetic_inputs()
    historical = {
        "campaign": "pmnist",
        "reference_rows": references,
        "sensitivity_reference_rows": [
            {
                "seed": row["seed"],
                "final_average_accuracy": row["final_average_accuracy"] + 1.0,
            }
            for row in references
        ],
        "records": historical_records,
    }
    followup = {"campaign": "pmnist", "records": followup_records}
    result = analyze_campaign_bundle(
        historical=historical,
        followup=followup,
        bootstrap_replicates=50,
        bootstrap_seed=20261008,
    )
    assert result["primary"]["reference_label"] == "standard_no_replay"
    assert (
        result["pmnist_fixed_current_sensitivity"]["reference_label"]
        == "fixed_current_no_replay"
    )
    assert result["thresholds_combined_across_references"] is False


def test_campaign_bundle_reports_cifar_class_il_as_mandatory_secondary() -> None:
    references, historical_records, followup_records = _synthetic_inputs()
    secondary_references = copy.deepcopy(references)
    secondary_historical = copy.deepcopy(historical_records)
    secondary_followup = copy.deepcopy(followup_records)
    for row in secondary_references:
        row["final_average_accuracy"] -= 10.0
    for row in [*secondary_historical, *secondary_followup]:
        row["final_average_accuracy"] -= 10.0
        row["source"] += "_class_il"
    result = analyze_campaign_bundle(
        historical={
            "campaign": "cifar",
            "reference_rows": references,
            "records": historical_records,
            "secondary_endpoint": "class_il",
            "secondary_reference_rows": secondary_references,
            "secondary_records": secondary_historical,
        },
        followup={
            "campaign": "cifar",
            "records": followup_records,
            "secondary_endpoint": "class_il",
            "secondary_records": secondary_followup,
        },
        bootstrap_replicates=50,
        bootstrap_seed=20261008,
    )
    secondary = result["cifar_class_il_mandatory_secondary"]
    assert secondary["analysis_role"] == "mandatory_secondary"
    assert secondary["reference_label"] == "class_il_no_replay_reference"
    assert result["thresholds_combined_across_endpoints"] is False


def test_authenticated_inputs_bind_lock_historical_and_followup_seals(
    tmp_path: Path,
) -> None:
    lock_path = ROOT / "configs/crossover_threshold_followup.json"
    historical_path = ROOT / "results/assignment_strength/aggregate.json"
    lock = load_lock(lock_path)
    units = enumerate_units(lock, "cifar")
    records = [
        {
            "unit_id": unit.unit_id,
            "campaign": "cifar",
            "benchmark": unit.benchmark,
            "seed": unit.seed,
            "assignment_replica": unit.assignment_replica,
            "requested_strength_bp": unit.lambda_bp,
            "requested_strength": unit.lambda_bp / 10000.0,
            "selected_rows": unit.selected_rows,
            "replay_minibatch_size": 32,
            "selected_row_fraction": unit.selected_rows / 32,
            "induced_strength": unit.induced_strength,
            "actual_changed_label_fraction": unit.induced_strength * 0.9,
            "final_average_accuracy": 50.0,
            "endpoint_values": {"task_il": 50.0, "class_il": 20.0},
            "digests": {
                "unit_json_sha256": "a" * 64,
                "unit_done_sha256": "b" * 64,
                "artifact_bundle_sha256": "c" * 64,
            },
            "sampled_event_fingerprints_sha256": "d" * 64,
        }
        for unit in units
    ]
    unit_ids_sha = canonical_sha256(sorted(row["unit_id"] for row in records))
    aggregate = {
        "schema_version": "gradient_reuse.crossover_aggregate.v1",
        "status": "PASS",
        "campaign": "cifar",
        "complete_grid_required": True,
        "unit_count": 336,
        "unit_ids_sha256": unit_ids_sha,
        "replicas_are_nested_monte_carlo": True,
        "records": records,
    }
    aggregate_path = tmp_path / "AGGREGATE.json"
    aggregate_path.write_text(
        json.dumps(aggregate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    locked_anchor = lock["historical_anchors"]["cifar"]
    binding = {
        "schema_version": "gradient_reuse.crossover_campaign_binding.v1",
        "status": "LOCKED",
        "campaign": "cifar",
        "expected_units": 336,
        "lock_sha256": file_sha(lock_path),
        "unit_ids_sha256": unit_ids_sha,
        "mammoth_commit": lock["upstream"]["commit"],
        "source_manifest_sha256": "1" * 64,
        "source_bundle_sha256": "2" * 64,
        "data_manifest_sha256": "3" * 64,
        "data_bundle_sha256": "4" * 64,
        "environment_manifest_sha256": "5" * 64,
        "environment_bundle_sha256": "6" * 64,
        "python_sha256": "7" * 64,
        "historical_anchors": {
            name: {
                "path": locked_anchor[f"{name}_path"],
                "sha256": locked_anchor[f"{name}_sha256"],
            }
            for name in ("parent_lock", "aggregate", "done")
        },
    }
    binding_path = tmp_path / "CAMPAIGN.json"
    binding_path.write_text(
        json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    done = {
        "schema_version": "gradient_reuse.crossover_queue_done.v1",
        "status": "PASS",
        "campaign": "cifar",
        "completed_units": 336,
        "required_units": 336,
        "aggregate_sha256": file_sha(aggregate_path),
        "unit_ids_sha256": unit_ids_sha,
        "campaign_binding_sha256": file_sha(binding_path),
    }
    done_path = tmp_path / "DONE.json"
    done_path.write_text(
        json.dumps(done, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    historical, followup, bindings = load_authenticated_analysis_inputs(
        campaign="cifar",
        lock_path=lock_path,
        historical_aggregate_path=historical_path,
        followup_aggregate_path=aggregate_path,
        followup_done_path=done_path,
        followup_campaign_binding_path=binding_path,
    )
    assert historical["schema_version"] == "gradient_reuse.assignment_strength_aggregate.v1"
    assert followup["unit_count"] == 336
    assert bindings["followup_done"]["sha256"] == file_sha(done_path)

    changed = copy.deepcopy(done)
    changed["aggregate_sha256"] = "0" * 64
    done_path.write_text(
        json.dumps(changed, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(AnalysisError, match="completion seal"):
        load_authenticated_analysis_inputs(
            campaign="cifar",
            lock_path=lock_path,
            historical_aggregate_path=historical_path,
            followup_aggregate_path=aggregate_path,
            followup_done_path=done_path,
            followup_campaign_binding_path=binding_path,
        )


def test_analysis_cli_exposes_no_bootstrap_override() -> None:
    arguments = [
        "--campaign",
        "cifar",
        "--lock",
        "lock.json",
        "--historical-aggregate",
        "history.json",
        "--followup-aggregate",
        "aggregate.json",
        "--followup-done",
        "done.json",
        "--followup-campaign-binding",
        "campaign.json",
        "--output",
        "analysis.json",
        "--bootstrap-replicates",
        "1",
    ]
    with pytest.raises(SystemExit):
        _parser().parse_args(arguments)
