from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fmmt_workshop import datasets as D
from experiments.fmmt_workshop import run as R
from experiments.fmmt_workshop.ot_positive import run_ot_positive_unit
from experiments.fmmt_workshop.specificity import ROWS, run_specificity_unit
from experiments.fmmt_workshop.synthetic_control import run_synthetic_suite


REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "configs/scalar.json").read_text())


def _runtime() -> dict:
    model = dict(LOCK["runtime"]["model"])
    model.update(branch_radius_history=8, torch_threads_per_worker=1)
    return model


def test_extra_dataset_preprocessing_is_chronological_and_calibration_only():
    n = 720
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h")
    appliances = pd.DataFrame(
        {
            "date": timestamps.astype(str),
            "Appliances": 100.0 + np.arange(n) % 25,
            "T1": np.sin(np.arange(n) / 20.0),
            "RH_1": 40.0 + np.cos(np.arange(n) / 13.0),
            "rv1": np.arange(n),
            "rv2": np.arange(n)[::-1],
        }
    )
    stream = D._from_frame("APPLIANCES", appliances, {"sha256": "fixture"})
    assert stream["X"].shape[0] == n
    assert stream["chronology_audit"]["timestamps_nondecreasing"] is True
    assert stream["chronology_audit"]["no_event_crosses_calibration"] is True
    assert stream["scored_mask"].sum() == n - stream["cal_n"]
    assert all("rv" not in name for name in stream["feature_names"])
    assert len(np.unique(stream["month_id"][stream["scored_mask"]])) >= 3

    tetouan = pd.DataFrame(
        {
            "DateTime": timestamps.strftime("%m/%d/%Y %H:%M"),
            "Temperature": 15.0 + np.sin(np.arange(n) / 20.0),
            "Humidity": 50.0 + np.cos(np.arange(n) / 17.0),
            "Wind Speed": 1.0 + np.arange(n) % 4,
            "Zone 1 Power Consumption": 20000.0 + np.arange(n) % 500,
            "Zone 2 Power Consumption": 15000.0 + np.arange(n) % 300,
            "Zone 3 Power Consumption": 10000.0 + np.arange(n) % 200,
        }
    )
    stream = D._from_frame("TETOUAN", tetouan, {"sha256": "fixture"})
    assert stream["X"].shape[0] == n
    assert all("Zone 2" not in name and "Zone 3" not in name for name in stream["feature_names"])
    assert stream["chronology_preserved"] is True


def test_specificity_rows_are_state_isolated_and_causally_tagged():
    data = D.load_workshop_stream("METRO", smoke=True, seed=9)
    result = run_specificity_unit(
        data,
        _runtime(),
        seed=9,
        model_type="linear",
        optimizer_type="sgd",
        packet_size=128,
    )
    assert tuple(result["metrics"]) == ROWS
    assert result["audit"]["all_pass"] is True
    assert result["audit"]["all_causal_rows_snapshotted_before_current_labels"] is True
    assert result["resources"]["center_updates"] == result["n_packets"]
    assert result["resources"]["raw_examples_persistently_stored"] == 0


def test_corrected_synthetic_directly_controls_observed_gradient():
    local = json.loads(json.dumps(LOCK))
    local["runtime"]["cluster_bootstrap_draws"] = 300
    result = run_synthetic_suite(local, [10, 11, 12, 13], seed_offset=0)
    summary = result["summary"]
    assert summary["all_pass"] is True
    assert abs(summary["regimes"]["rho_+0.9"]["realized_mean_cosine"] - 0.9) < 1e-10
    assert abs(summary["regimes"]["rho_+0.0"]["realized_mean_cosine"]) < 1e-10
    assert abs(summary["regimes"]["rho_-0.9"]["realized_mean_cosine"] + 0.9) < 1e-10


def test_engineered_ot_positive_control_prefers_correct_geometry():
    specification = dict(LOCK["ot_positive_control"], steps=16, burn_in=2)
    unit = run_ot_positive_unit(20, specification)
    assert unit["audit"]["all_pass"] is True
    assert unit["metrics"]["functional_cost"] < unit["metrics"]["uniform"]
    assert unit["metrics"]["functional_cost"] < unit["metrics"]["shuffled_cost"]


def _summary(verdict: str, improvement_ci_lower: float = 0.0) -> dict:
    return {
        "verdict": verdict,
        "gates": {"all_integrity_audits": True},
        "contrasts_vs_K0": {
            "K1_PREVIOUS_GRADIENT": {"ci95_relative": [improvement_ci_lower, 0.1]}
        },
    }


def test_combined_verdict_distinguishes_signal_from_boundary_evidence():
    primary = {
        "primary_development/BEIJING/x": _summary("TARGET_SPECIFIC_TEMPORAL_SIGNAL"),
        "primary_confirmation/BEIJING/x": _summary("TARGET_SPECIFIC_TEMPORAL_SIGNAL"),
        "primary_development/METRO/x": _summary("TRIVIAL_CORRECTION_EXPLAINS_GAIN", -0.005),
        "primary_confirmation/METRO/x": _summary("TRIVIAL_CORRECTION_EXPLAINS_GAIN", -0.005),
    }
    good = R._combined_verdict(
        LOCK, primary, {}, {"all_pass": True}, {"all_pass": True}, smoke=False
    )
    assert good["verdict"] == "WORKSHOP_TARGET_SPECIFIC_SIGNAL_CONFIRMED"
    primary["primary_confirmation/BEIJING/x"]["verdict"] = "TRIVIAL_CORRECTION_EXPLAINS_GAIN"
    boundary = R._combined_verdict(
        LOCK, primary, {}, {"all_pass": True}, {"all_pass": True}, smoke=False
    )
    assert boundary["verdict"] == "WORKSHOP_BOUNDARY_EVIDENCE_COMPLETE"


def test_workshop_source_freeze_verifies_exact_files():
    bundle, files = R._verify_source_freeze()
    manifest = json.loads((REPO / "provenance/source/scalar.json").read_text())
    assert bundle == manifest["source_bundle_sha256"]
    assert files == manifest["files"]
    assert files["configs/scalar.json"] == manifest["lock_sha256"]
