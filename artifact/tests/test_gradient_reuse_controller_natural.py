"""Focused integrity tests for the locked natural-stream reuse controller."""

from __future__ import annotations

import copy
from collections import OrderedDict
import itertools
import json
from pathlib import Path
import urllib.request

import numpy as np
import pytest
import torch

from experiments.gradient_reuse_controller import natural
from experiments.gradient_reuse_controller.provenance import (
    _validated_evidence,
    evidence_payload,
)
from experiments.source_provenance import (
    SourceProvenanceError,
    build_source_manifest,
    sha256_file,
)
from experiments.validation import stage4_regression as stage4


REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs/controller_natural.json"


def _config() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def _tiny_beijing_stream() -> dict:
    rows = 6
    return {
        "name": "BEIJING",
        "X": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        "y": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        "month_id": np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64),
        "scored_mask": np.asarray([False, False, True, True, True, True]),
        "event_id": np.arange(rows, dtype=np.int64),
        "event_time_seconds": np.arange(rows, dtype=np.float64),
        "chronology_preserved": True,
        "all_preprocessed_rows_retained": True,
        "chronology_audit": {"timestamps_nondecreasing": True},
        "data_manifest": {"source": "test-locked-archive"},
    }


@pytest.fixture(scope="module")
def smoke_result() -> dict:
    return natural.run_campaign(natural.load_config(LOCK), smoke=True)


def test_lock_is_source_bound_and_rejects_protocol_drift() -> None:
    config = natural.validate_config(_config())
    assert config["source_identity"]["bundle_sha256"]
    assert config["full"]["bootstrap_block_lengths"] == [1, 3, 6, 12]

    unlocked = copy.deepcopy(config)
    unlocked["status"] = "DRAFT"
    with pytest.raises(natural.NaturalControllerConfigError, match="unlocked"):
        natural.validate_config(unlocked)

    wrong_full_seeds = copy.deepcopy(config)
    wrong_full_seeds["full"]["seeds"][-1] = 999
    with pytest.raises(natural.NaturalControllerConfigError, match="full seeds"):
        natural.validate_config(wrong_full_seeds)

    lookahead = copy.deepcopy(config)
    lookahead["controller"]["gate_timing"] = "gate_before_score"
    with pytest.raises(natural.NaturalControllerConfigError, match="gate_timing"):
        natural.validate_config(lookahead)

    unknown = copy.deepcopy(config)
    unknown["controller"]["replay_buffer"] = 32
    with pytest.raises(natural.NaturalControllerConfigError, match="keys differ"):
        natural.validate_config(unknown)

    drifted_hash = copy.deepcopy(config)
    path = "src/chartfilm/fmmt/core.py"
    drifted_hash["source_identity"]["files"][path] = "0" * 64
    with pytest.raises(natural.NaturalControllerConfigError, match="identity mismatch"):
        natural.validate_config(drifted_hash)


def test_smoke_loader_is_deterministic_local_and_never_downloads(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("smoke attempted network or frozen real-data acquisition")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(stage4, "_download", forbidden)
    first = natural.load_workshop_stream("BEIJING", smoke=True, seed=7)
    second = natural.load_workshop_stream("BEIJING", smoke=True, seed=7)
    assert first["data_manifest"] == {"source": "deterministic_smoke", "seed": 7}
    assert natural.stream_identity(first) == natural.stream_identity(second)
    timestamp_only = copy.deepcopy(first)
    timestamp_only["data_manifest"] = dict(timestamp_only["data_manifest"])
    timestamp_only["data_manifest"]["downloaded_ts"] = "2099-01-01T00:00:00Z"
    assert natural.stream_identity(first) == natural.stream_identity(timestamp_only)


def test_all_policies_share_initial_prediction_and_identical_data(smoke_result: dict) -> None:
    assert smoke_result["status"] == "SMOKE_PASS"
    assert smoke_result["scientific_evidence_eligible"] is False
    assert smoke_result["warning"].startswith("SMOKE OUTPUT")
    units = smoke_result["units"]
    assert [unit["policy"] for unit in units] == list(natural.POLICIES)
    assert len({unit["stream_identity"]["sha256"] for unit in units}) == 1
    assert len({unit["prediction_sha256_sequence"][0] for unit in units}) == 1


def test_predictions_are_prefix_invariant_to_unseen_current_and_future_targets() -> None:
    config = natural.load_config(LOCK)
    data = natural.load_workshop_stream("BEIJING", smoke=True, seed=23)
    packets = natural.packetize_stream(data, packet_size=32)
    cut = 5
    changed = copy.deepcopy(data)
    changed_y = np.asarray(changed["y"]).copy()
    changed_y[np.asarray(packets[cut]["index"], dtype=np.int64)[0] :] = (
        -17.0 * changed_y[np.asarray(packets[cut]["index"], dtype=np.int64)[0] :] + 41.0
    )
    changed["y"] = changed_y

    original = natural.run_policy(
        data,
        "pairing_aware",
        seed=23,
        config=config,
        max_packets=8,
        radius_schedule=natural.build_common_radius_schedule(
            data, seed=23, config=config, max_packets=8
        ),
        retain_packet_records=True,
    )
    perturbed = natural.run_policy(
        changed,
        "pairing_aware",
        seed=23,
        config=config,
        max_packets=8,
        radius_schedule=natural.build_common_radius_schedule(
            changed, seed=23, config=config, max_packets=8
        ),
        retain_packet_records=True,
    )
    # Prediction at cut is made before y_cut is revealed.  The state after cut
    # may differ because the post-score update and gate may use y_cut.
    assert original["prediction_sha256_sequence"][: cut + 1] == (
        perturbed["prediction_sha256_sequence"][: cut + 1]
    )
    assert original["state_sha256_sequence"][:cut] == (
        perturbed["state_sha256_sequence"][:cut]
    )
    assert (
        original["packet_records"][cut]["prequential_squared_error"]
        != perturbed["packet_records"][cut]["prequential_squared_error"]
    )


def test_exactly_one_true_update_equal_radius_and_no_raw_replay(smoke_result: dict) -> None:
    for unit in smoke_result["units"]:
        assert unit["true_optimizer_updates"] == unit["packets"]
        assert unit["raw_examples_replayed"] == 0
        assert unit["stored_parameter_sized_channels"] == [
            "g",
            "m",
            "c",
            "permuted_g_nested_mc",
        ]
        for record in unit["packet_records"]:
            if record["acted"]:
                assert record["reuse_displacement_norm"] == pytest.approx(
                    record["radius_before_current_update"], rel=2.0e-5, abs=2.0e-7
                )
    serialized = json.dumps(smoke_result, sort_keys=True)
    assert '"inputs"' not in serialized
    assert '"targets"' not in serialized


def test_pairing_gate_uses_stored_q_over_q_plus_v_as_binary_gate(smoke_result: dict) -> None:
    unit = next(
        row for row in smoke_result["units"] if row["policy"] == "pairing_aware"
    )
    records = unit["packet_records"]
    checked = 0
    epsilon = _config()["controller"]["epsilon"]
    for index, record in enumerate(records[1:], start=1):
        if not record["eligible"]:
            continue
        stored = records[index - 1]
        q = stored["q_pairing_strength_full_gradient"]
        v = stored["v_finite_permutation_variance_full_gradient"]
        expected = q / (q + v + epsilon)
        assert record["stored_reliability_q_over_q_plus_v"] == pytest.approx(
            expected, rel=1.0e-12, abs=1.0e-15
        )
        should_open = (
            record["c_cosine"] > 0.0
            and expected >= _config()["controller"]["pairing_reliability_threshold"]
        )
        assert record["pairing_channel_open"] is should_open
        assert record["c_weight"] == pytest.approx(
            max(0.0, record["c_cosine"]) if should_open else 0.0,
            rel=1.0e-12,
            abs=1.0e-15,
        )
        checked += 1
    assert checked > 0


def test_v_identity_and_cluster_paired_cis_are_reported(smoke_result: dict) -> None:
    for unit in smoke_result["units"]:
        decomposition = unit["v_decomposition"]
        assert decomposition["max_identity_residual"] <= 1.0e-12
        assert decomposition["sum_v_total"] == pytest.approx(
            decomposition["sum_v_perm"] + decomposition["sum_v_pair"],
            abs=1.0e-12,
            rel=1.0e-12,
        )
        assert decomposition["permutation_action"].startswith(
            "mean_risk_over_normalized"
        )
        assert decomposition["v_perm_nested_mc_se"] >= 0.0
        assert unit["strength_diagnostics"][
            "mean_covariance_j_y_norm_half_gradient"
        ] >= 0.0
        assert unit["strength_diagnostics"][
            "mean_finite_permutation_variance_full_gradient"
        ] >= 0.0
        first_v = next(
            record
            for record in unit["packet_records"]
            if record["v_total"] is not None
        )
        assert first_v["v_source_action_packet"] == first_v["packet"] - 1
        assert first_v["v_source_gradient_packet"] == first_v["packet"] - 2
    for policy, comparison in smoke_result["comparisons_vs_never"].items():
        assert policy in natural.POLICIES[1:]
        macro = comparison["macro_month_prequential_mse"]
        cumulative = comparison["cumulative_prequential_squared_error"]
        assert len(macro["ci95"]) == 2
        assert macro["bootstrap"]["ordered_calendar_month_clusters"] is True
        assert macro["bootstrap"]["initialization_rows_resampled"] is True
        assert macro["bootstrap"]["circular_wrap"] is False
        assert set(macro["sensitivity_by_block_length"]) == {"1", "2"}
        assert len(cumulative["ci95"]) == 2

    assert set(smoke_result["controller_wedge_comparisons"]) == {
        "pairing_aware_vs_cosine",
        "pairing_aware_vs_always",
    }
    for name, comparison in smoke_result["controller_wedge_comparisons"].items():
        assert comparison["candidate"] == "pairing_aware"
        assert comparison["baseline"] in {"cosine", "always"}
        assert comparison["positive_improvement_favors"] == "pairing_aware"
        assert len(comparison["macro_month_prequential_mse"]["ci95"]) == 2
        assert len(comparison["cumulative_prequential_squared_error"]["ci95"]) == 2


def test_common_radius_schedule_is_causal_and_shared(smoke_result: dict) -> None:
    config = natural.load_config(LOCK)
    data = natural.load_workshop_stream("BEIJING", smoke=True, seed=31)
    packets = natural.packetize_stream(data, packet_size=32)
    cut = 5
    changed = copy.deepcopy(data)
    changed_y = np.asarray(changed["y"]).copy()
    start = int(np.asarray(packets[cut]["index"], dtype=np.int64)[0])
    changed_y[start:] = changed_y[start:] * -11.0 + 19.0
    changed["y"] = changed_y
    first = natural.build_common_radius_schedule(
        data, seed=31, config=config, max_packets=8
    )
    second = natural.build_common_radius_schedule(
        changed, seed=31, config=config, max_packets=8
    )
    # radius[cut] only uses updates through cut-1.
    assert first["values"][: cut + 1] == second["values"][: cut + 1]
    hashes = {
        unit["radius_diagnostics"]["schedule_sha256"]
        for unit in smoke_result["units"]
    }
    assert hashes == {
        smoke_result["common_radius_schedules_by_seed"]["7"]["sha256"]
    }
    assert smoke_result["integrity"][
        "common_never_reuse_radius_schedule_across_policies"
    ] is True


def test_v_perm_is_exhaustive_expected_risk_after_individual_normalization() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    parameters = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_parameters()
    )
    source_x = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    source_y = np.asarray([3.0, -1.0, 2.0])
    gradients = []
    for order in itertools.permutations(range(3)):
        permuted = source_y[np.asarray(order)]
        flat = 2.0 * source_x.T @ (-permuted) / 3.0
        gradients.append(
            OrderedDict(weight=torch.as_tensor(flat.reshape(1, 2), dtype=torch.float32))
        )
    true_gradient = gradients[0]
    shadow = natural._build_v_shadow(
        parameters,
        true_gradient,
        [gradients, gradients],
        radius=0.2,
        action_packet=1,
        gradient_source_packet=0,
    )
    next_x_np = np.asarray([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.0]])
    next_y = np.asarray([0.4, -0.3, 1.2])
    next_x = torch.as_tensor(next_x_np, dtype=torch.float32)
    observed = natural._score_v_shadow(
        model, shadow, next_x, next_y, np.ones(3, dtype=bool)
    )

    baseline_risk = float(np.mean(next_y**2))
    permutation_risks = []
    for gradient in gradients:
        vector = gradient["weight"].numpy().reshape(-1)
        theta = -0.2 * vector / np.linalg.norm(vector)
        permutation_risks.append(float(np.mean((next_x_np @ theta - next_y) ** 2)))
    expected = baseline_risk - float(np.mean(permutation_risks))
    assert observed["v_perm"] == pytest.approx(expected, abs=2.0e-7)
    assert observed["mc_se"] == pytest.approx(0.0, abs=1.0e-15)

    # Normalizing the expected gradient is a different estimand; guard against
    # silently reverting to that shortcut.
    mean_vector = np.mean(
        [gradient["weight"].numpy().reshape(-1) for gradient in gradients], axis=0
    )
    mean_theta = -0.2 * mean_vector / np.linalg.norm(mean_vector)
    shortcut = baseline_risk - float(
        np.mean((next_x_np @ mean_theta - next_y) ** 2)
    )
    assert observed["v_perm"] != pytest.approx(shortcut, abs=1.0e-4)


def test_post_score_mean_drift_evidence_is_aggregated_without_packet_records_dependency(
    smoke_result: dict,
) -> None:
    evidence = smoke_result["natural_mean_drift_evidence_by_policy"]
    assert set(evidence) == set(natural.POLICIES)
    for policy, summary in evidence.items():
        assert summary["labels_used_only_after_prequential_score"] is True
        assert summary["mean_absolute_packet_target_mean_change"] >= 0.0
        fraction = summary[
            "mean_fraction_full_cosine_positive_and_c_cosine_nonpositive"
        ]
        assert 0.0 <= fraction <= 1.0
        assert set(summary["v_by_alignment_stratum"]) == {
            "full_positive_c_nonpositive",
            "all_other_transitions",
        }
        assert summary["per_seed"][0]["seed"] == 7

    unit = next(
        row for row in smoke_result["units"] if row["policy"] == "pairing_aware"
    )
    diagnostic = unit["natural_mean_drift_diagnostics"]
    assert diagnostic["labels_used_only_after_prequential_score"] is True
    assert diagnostic["transitions"] == unit["packets"] - 1
    assert diagnostic["v_by_absolute_target_mean_change_stratum"][
        "threshold_rule"
    ].startswith("within_seed_median")


def test_cli_requires_an_explicit_exclusive_tier() -> None:
    parser = natural.build_parser()
    common = ["--config", str(LOCK), "--output", "/tmp/natural.json"]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--smoke", "--full"])
    assert parser.parse_args([*common, "--smoke"]).smoke is True
    assert parser.parse_args([*common, "--full"]).full is True


def test_natural_cli_full_requires_and_binds_shared_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "controller-natural-source.json"
    source.write_text(
        json.dumps(build_source_manifest("controller_natural", repo=REPO)),
        encoding="utf-8",
    )
    output = tmp_path / "full.json"
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    data_manifest = tmp_path / "data.json"
    data_manifest.write_text("{}", encoding="utf-8")
    archive = tmp_path / "beijing.zip"
    archive.write_bytes(b"locked archive")

    def fake_runtime(*args, **kwargs):
        return _validated_evidence(
            "runtime", {"validation_status": "PASS", "manifest": {}}
        )

    def fake_data(*args, **kwargs):
        return (
            _validated_evidence(
                "data", {"validation_status": "PASS", "manifest": {}}
            ),
            {"name": "BEIJING"},
        )

    def fake_campaign(
        config,
        *,
        smoke,
        source_binding,
        runtime_binding,
        data_binding,
        full_data,
    ):
        assert smoke is False
        bound_source = evidence_payload(source_binding, kind="source")
        bound_runtime = evidence_payload(runtime_binding, kind="runtime")
        bound_data = evidence_payload(data_binding, kind="data")
        assert bound_source is not None
        assert bound_runtime is not None
        assert bound_data is not None
        assert full_data == {"name": "BEIJING"}
        return {
            "status": "PASS",
            "mode": "full",
            "scientific_evidence_eligible": True,
            "integrity": {},
            "source_provenance": bound_source,
            "runtime_provenance": bound_runtime,
            "data_provenance": bound_data,
        }

    monkeypatch.setattr(natural, "validate_runtime_manifest", fake_runtime)
    monkeypatch.setattr(natural, "validate_beijing_data_manifest", fake_data)
    monkeypatch.setattr(natural, "run_campaign", fake_campaign)
    assert (
        natural.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(output),
                "--runtime-manifest",
                str(runtime),
                "--data-manifest",
                str(data_manifest),
                "--data-archive",
                str(archive),
                "--full",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    evidence = payload["source_provenance"]
    assert evidence["experiment"] == "controller_natural"
    assert evidence["validated_for_execution"] is True
    assert evidence["manifest_sha256"] == sha256_file(source)
    assert evidence["source_bundle_sha256"]
    assert payload["integrity"]["shared_source_manifest_validated"] is True


def test_natural_cli_full_fails_closed_and_smoke_records_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def fake_campaign(
        config,
        *,
        smoke,
        source_binding,
        runtime_binding,
        data_binding,
        full_data,
    ):
        calls.append(bool(smoke))
        assert source_binding is None
        assert runtime_binding is None
        assert data_binding is None
        assert full_data is None
        return {
            "status": "SMOKE_PASS" if smoke else "PASS",
            "mode": "smoke" if smoke else "full",
            "scientific_evidence_eligible": not smoke,
            "integrity": {},
        }

    monkeypatch.setattr(natural, "run_campaign", fake_campaign)
    missing = tmp_path / "missing.json"
    full_output = tmp_path / "full-missing.json"
    with pytest.raises(SourceProvenanceError, match="source validation failed"):
        natural.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(missing),
                "--output",
                str(full_output),
                "--full",
            ]
        )
    assert calls == []
    assert not full_output.exists()

    smoke_output = tmp_path / "smoke-missing.json"
    assert (
        natural.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(missing),
                "--output",
                str(smoke_output),
                "--smoke",
            ]
        )
        == 0
    )
    payload = json.loads(smoke_output.read_text(encoding="utf-8"))
    assert calls == [True]
    assert payload["scientific_evidence_eligible"] is False
    assert payload["source_validation_attempt"]["validated_for_execution"] is False
    assert payload["source_validation_attempt"]["validation_error"]


def test_full_api_cannot_load_or_claim_evidence_without_validated_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("direct full API attempted implicit data acquisition")

    monkeypatch.setattr(natural, "load_workshop_stream", forbidden)
    with pytest.raises(
        natural.NaturalControllerIntegrityError,
        match="full data must be supplied",
    ):
        natural.run_campaign(natural.load_config(LOCK), smoke=False)


def test_beijing_manifest_freezes_bytes_schema_rows_and_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(build_source_manifest("controller_natural", repo=REPO)),
        encoding="utf-8",
    )
    source_evidence = natural._shared_source_evidence(source, LOCK, smoke=False)
    archive = tmp_path / "beijing.zip"
    archive.write_bytes(b"exact public archive bytes")
    raw_identity = {
        "selected_members": ["PRSA_Data_20130301-20170228/data.csv"],
        "row_count": 6,
        "schema": [
            {"name": "year", "dtype": "int64"},
            {"name": "pm2.5", "dtype": "float64"},
        ],
    }
    stream = _tiny_beijing_stream()
    monkeypatch.setattr(natural, "_raw_archive_identity", lambda path: raw_identity)
    monkeypatch.setattr(
        natural,
        "_load_locked_beijing",
        lambda path, *, seed: copy.deepcopy(stream),
    )
    manifest = natural.build_beijing_data_manifest(
        archive_path=archive,
        config_path=LOCK,
        config=natural.load_config(LOCK),
        source_evidence=source_evidence,
    )
    assert manifest["scientific_evidence_eligible"] is False
    assert manifest["archive"]["bytes"] == len(b"exact public archive bytes")
    assert manifest["raw_archive_identity"] == raw_identity
    assert manifest["preprocessing_identity"] == natural.stream_identity(stream)
    manifest_path = tmp_path / "data-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    binding, loaded = natural.validate_beijing_data_manifest(
        manifest_path,
        archive_path=archive,
        config_path=LOCK,
        config=natural.load_config(LOCK),
        source_evidence=source_evidence,
    )
    assert evidence_payload(binding, kind="data")["manifest"] == manifest
    assert natural.stream_identity(loaded) == manifest["preprocessing_identity"]

    archive.write_bytes(b"changed bytes")
    with pytest.raises(
        natural.NaturalControllerIntegrityError, match="archive bytes differ"
    ):
        natural.validate_beijing_data_manifest(
            manifest_path,
            archive_path=archive,
            config_path=LOCK,
            config=natural.load_config(LOCK),
            source_evidence=source_evidence,
        )


def test_fail_closed_stage4_loader_never_downloads_missing_locked_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("locked loader attempted network acquisition")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(stage4, "_download", forbidden)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(build_source_manifest("controller_natural", repo=REPO)),
        encoding="utf-8",
    )
    source_evidence = natural._shared_source_evidence(source, LOCK, smoke=False)
    with pytest.raises(
        natural.NaturalControllerIntegrityError, match="archive is unavailable"
    ):
        natural.build_beijing_data_manifest(
            archive_path=tmp_path / "missing.zip",
            config_path=LOCK,
            config=natural.load_config(LOCK),
            source_evidence=source_evidence,
        )


def test_natural_runtime_capture_binds_torch_optimizer_and_full_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source_payload = build_source_manifest("controller_natural", repo=REPO)
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    torch.use_deterministic_algorithms(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert (
        natural.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(runtime),
                "--capture-runtime",
            ]
        )
        == 0
    )
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["scientific_evidence_eligible"] is False
    assert payload["source_identity"]["manifest"] == source_payload
    assert payload["observed"]["packages"]["torch"] == torch.__version__
    assert payload["observed"]["torch"]["optimizer"]["name"] == "adamw"
    assert payload["observed"]["torch"]["deterministic_algorithms"] is True
    assert payload["observed"]["torch"]["cuda_matmul_allow_tf32"] is False
    assert payload["observed"]["torch"]["cudnn_allow_tf32"] is False
    assert payload["observed"]["torch"]["cudnn_benchmark"] is False
    assert payload["observed"]["torch"]["num_threads"] == torch.get_num_threads()
    assert payload["observed"]["torch"][
        "num_interop_threads"
    ] == torch.get_num_interop_threads()
    assert torch.are_deterministic_algorithms_enabled()
    source_evidence = natural._shared_source_evidence(source, LOCK, smoke=False)
    binding = natural.validate_runtime_manifest(
        runtime,
        experiment="controller_natural",
        config_path=LOCK,
        config=natural.load_config(LOCK),
        source_evidence=source_evidence,
    )
    assert evidence_payload(binding, kind="runtime")["manifest"] == payload
