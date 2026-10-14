"""Focused tests for the prospective two-channel reuse controller."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.gradient_reuse_controller import run as controller_cli
from experiments.gradient_reuse_controller import provenance as controller_provenance
from experiments.gradient_reuse_controller.core import (
    CONTROLLER_POLICIES,
    ControllerConfigError,
    Stream,
    generate_stream,
    gradient_components,
    paired_seed_bootstrap,
    run_campaign,
    run_policy,
    validate_config,
)
from experiments.gradient_reuse_controller.run import build_parser
from experiments.gradient_reuse_controller.provenance import (
    ValidatedEvidence,
    build_runtime_manifest,
    evidence_payload,
    validate_runtime_manifest,
)
from experiments.source_provenance import (
    SourceProvenanceError,
    build_source_manifest,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs/controller.json"


def _config() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def _compact_config() -> dict:
    """A valid but small lock used only by unit tests."""

    config = _config()
    config["permutations"].update(replicas=2, draws_per_replica=4)
    config["full"].update(
        seeds=[101, 211],
        packets=20,
        scored_after_packets=3,
        bootstrap_draws=32,
        inference_unit="independent_stream_seed",
        retain_packet_records=True,
    )
    config["smoke"].update(
        seeds=[17, 29],
        packets=12,
        scored_after_packets=2,
        bootstrap_draws=8,
        inference_unit="independent_stream_seed",
        retain_packet_records=True,
    )
    return config


def _run(
    stream: Stream,
    policy: str,
    config: dict,
    *,
    seed: int,
    scored_after_packets: int = 4,
) -> dict:
    return run_policy(
        stream,
        policy,
        seed=seed,
        model_config=config["model"],
        permutation_config=config["permutations"],
        controller_config=config["controller"],
        scored_after_packets=scored_after_packets,
        retain_packet_records=True,
    )


def test_gradient_decomposition_and_finite_permutation_variance_are_exact() -> None:
    features = np.asarray(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 2.0], [1.0, 4.0]],
        dtype=np.float64,
    )
    targets = np.asarray([-2.0, 0.5, 3.0, 1.5], dtype=np.float64)
    theta = np.asarray([0.3, -0.4], dtype=np.float64)
    result = gradient_components(theta, features, targets)

    realized = []
    prediction = features @ theta
    for permutation in itertools.permutations(range(targets.size)):
        permuted = targets[np.asarray(permutation)]
        realized.append(features.T @ (prediction - permuted) / targets.size)
    realized = np.asarray(realized)

    assert np.allclose(realized.mean(axis=0), result["preserved"], atol=1e-14)
    observed_variance = np.mean(
        np.sum((realized - result["preserved"]) ** 2, axis=1)
    )
    assert observed_variance == pytest.approx(
        result["permutation_variance"], rel=1e-13, abs=1e-13
    )
    assert np.allclose(
        result["ordinary"], result["preserved"] + result["pairing"], atol=1e-14
    )
    assert np.allclose(result["pairing"], -result["covariance_j_y"], atol=1e-14)


def test_mean_drift_filters_null_pairing_noise_but_keeps_the_preserved_channel() -> None:
    config = _config()
    seed = 17
    stream = generate_stream(
        "mean_drift",
        config["streams"]["mean_drift"],
        seed=seed,
        packets=32,
        packet_size=config["model"]["packet_size"],
        conditional_dimension=config["model"]["conditional_dimension"],
    )
    always = _run(stream, "always", config, seed=seed)
    cosine = _run(stream, "cosine", config, seed=seed)
    pairing_aware = _run(stream, "pairing_aware", config, seed=seed)

    assert stream.metadata["population_pairing_is_zero"] is True
    assert stream.metadata["nuisance_features_are_target_independent"] is True
    assert stream.features.shape[-1] == config["model"]["conditional_dimension"] + 1
    nuisance = stream.features[:, :, 1:]
    assert np.allclose(nuisance.mean(axis=1), 0.0, atol=1e-14)
    nuisance_covariances = np.einsum(
        "pni,pnj->pij", nuisance, nuisance
    ) / nuisance.shape[1]
    assert np.allclose(
        nuisance_covariances,
        np.eye(config["model"]["conditional_dimension"])[None, :, :],
        atol=1e-13,
    )
    # Population pairing is null by construction, but finite-packet sample
    # covariance and its exact permutation variance are deliberately nonzero.
    assert np.mean(
        [record["pairing_strength"] for record in pairing_aware["packet_records"]]
    ) > 0.0
    assert np.mean(
        [record["permutation_variance"] for record in pairing_aware["packet_records"]]
    ) > 0.0
    assert pairing_aware["mean_pairing_weight"] == 0.0
    assert pairing_aware["pairing_channel_open_rate"] == 0.0
    assert pairing_aware["mean_preserved_weight"] > 0.9
    assert pairing_aware["actions"] > 0
    assert cosine["fraction_positive_full_cosine"] > 0.9
    assert pairing_aware["stream_construction"] == {
        "population_pairing_is_zero": True,
        "nuisance_features_are_target_independent": True,
        "realized_pairing_source": "finite_packet_sampling_noise",
    }
    # This is a directional engineering invariant for the locked construction,
    # not a substituted paper result: the full-gradient gates carry nuisance
    # sample covariance, whereas the two-channel controller removes it.
    assert pairing_aware["prequential_mse"] < cosine["prequential_mse"]
    assert pairing_aware["prequential_mse"] < always["prequential_mse"]


def test_recurrent_reversal_fools_full_gradient_cosine_but_not_pairing_gate() -> None:
    config = _config()
    seed = 29
    stream = generate_stream(
        "recurrent_reversal",
        config["streams"]["recurrent_reversal"],
        seed=seed,
        packets=32,
        packet_size=config["model"]["packet_size"],
        conditional_dimension=config["model"]["conditional_dimension"],
    )
    results = {
        policy: _run(stream, policy, config, seed=seed)
        for policy in CONTROLLER_POLICIES
    }

    pairing = results["pairing_aware"]
    signs = stream.metadata["relationship_signs"]
    boundaries = stream.metadata["reversal_boundaries"]
    assert boundaries
    for boundary in boundaries:
        assert signs[boundary - 1] != signs[boundary]
        assert signs[boundary + 1] == signs[boundary]
    assert stream.metadata["reversal_action_horizon_persistent"] is True
    # The boundary comparison is prospective: the stale source relationship
    # differs from t, while t+1 remains in t's newly entered regime.
    assert pairing["fraction_positive_full_cosine"] > 0.5
    assert pairing["fraction_negative_pairing_cosine"] > 0.05
    assert pairing["prequential_mse"] < results["cosine"]["prequential_mse"]
    assert pairing["prequential_mse"] < results["always"]["prequential_mse"]


def test_reuse_direction_is_the_stored_previous_gradient_not_the_current_gradient() -> None:
    """The end-of-t action must reuse g_(t-1), never apply g_t twice."""

    config = _config()
    model = dict(config["model"])
    model.update(base_learning_rate=0.1, reuse_radius=0.2, warmup_packets=1)
    features = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]], dtype=np.float64
    )
    stream = Stream(
        name="lag_semantics", features=features, targets=targets, metadata={}
    )
    result = run_policy(
        stream,
        "always",
        seed=13,
        model_config=model,
        permutation_config=config["permutations"],
        controller_config=config["controller"],
        scored_after_packets=0,
        retain_packet_records=True,
    )

    # g0=(-1,0), so theta after packet 0 is (0.1,0).  At packet 1,
    # g1=(0.05,-1.5); the common step gives (0.095,0.15), then reusing
    # unit(g0)=(-1,0) at radius 0.2 gives (0.295,0.15).
    np.testing.assert_allclose(
        result["packet_records"][1]["theta_after_update"],
        [0.295, 0.15],
        rtol=0.0,
        atol=1.0e-14,
    )
    assert result["packet_records"][1]["reuse_source_packet"] == 0
    assert result["packet_records"][1]["reuse_gradient_lag_packets"] == 1
    assert result["reuse_gradient_lag_packets"] == 1
    assert result["gate_observes_current_target_only_after_scoring"] is True


def test_predictions_are_prefix_invariant_to_unseen_future_targets() -> None:
    config = _config()
    seed = 101
    stream = generate_stream(
        "recurrent_reversal",
        config["streams"]["recurrent_reversal"],
        seed=seed,
        packets=16,
        packet_size=config["model"]["packet_size"],
        conditional_dimension=config["model"]["conditional_dimension"],
    )
    cut = 7
    changed_targets = stream.targets.copy()
    changed_targets[cut:] = changed_targets[cut:] * -13.0 + 97.0
    changed = Stream(
        name=stream.name,
        features=stream.features.copy(),
        targets=changed_targets,
        metadata=stream.metadata,
    )
    original_result = _run(stream, "pairing_aware", config, seed=seed, scored_after_packets=2)
    changed_result = _run(changed, "pairing_aware", config, seed=seed, scored_after_packets=2)

    original_records = original_result["packet_records"]
    changed_records = changed_result["packet_records"]
    # The prediction at ``cut`` is also fixed before y_cut is revealed.
    assert [row["prediction_sha256"] for row in original_records[: cut + 1]] == [
        row["prediction_sha256"] for row in changed_records[: cut + 1]
    ]
    # States after all strictly earlier packets are identical as well.
    assert [row["theta_after_update"] for row in original_records[:cut]] == [
        row["theta_after_update"] for row in changed_records[:cut]
    ]
    assert original_records[cut]["mse"] != changed_records[cut]["mse"]


def test_packetwise_v_decomposition_is_an_exact_finite_identity() -> None:
    config = _config()
    stream = generate_stream(
        "recurrent_reversal",
        config["streams"]["recurrent_reversal"],
        seed=307,
        packets=16,
        packet_size=config["model"]["packet_size"],
        conditional_dimension=config["model"]["conditional_dimension"],
    )
    result = _run(stream, "pairing_aware", config, seed=307, scored_after_packets=2)
    audited = result["packet_records"][2:]
    assert audited
    assert audited[0]["v_source_action_packet"] == 1
    assert audited[0]["v_source_gradient_packet"] == 0
    for record in audited:
        assert record["v_total"] == pytest.approx(
            record["v_perm"] + record["v_pair"], abs=1e-14, rel=1e-14
        )
    assert result["max_v_identity_residual"] <= 1e-14
    decomposition = result["v_decomposition"]
    assert decomposition["audited_scored_packets"] == 14
    assert decomposition["sum_v_total"] == pytest.approx(
        decomposition["sum_v_perm"] + decomposition["sum_v_pair"],
        abs=1e-13,
        rel=1e-13,
    )


def test_paired_seed_bootstrap_is_deterministic_and_fail_closed() -> None:
    baseline = np.asarray(
        [[2.0, 2.2, 1.8, 2.1, 2.3], [1.9, 2.0, 2.1, 2.2, 1.8]]
    )
    candidate = baseline - 0.2
    first = paired_seed_bootstrap(baseline, candidate, seed=55, draws=128)
    second = paired_seed_bootstrap(baseline, candidate, seed=55, draws=128)
    assert first == second
    assert first["mse_improvement"] == pytest.approx(0.2)
    assert first["mse_improvement_ci95"] == pytest.approx([0.2, 0.2])
    assert first["bootstrap_unit"] == "independent_stream_seed"

    with pytest.raises(ValueError, match="equal"):
        paired_seed_bootstrap(baseline, candidate[:, :-1], seed=1, draws=10)
    poisoned = baseline.copy()
    poisoned[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        paired_seed_bootstrap(poisoned, candidate, seed=1, draws=10)


def test_pairing_reliability_is_an_effective_binary_channel_gate() -> None:
    from experiments.gradient_reuse_controller.core import _pairing_gate_weights

    low = _pairing_gate_weights(
        preserved_cosine=0.0,
        pairing_cosine=1.0,
        pairing_strength=1.0,
        permutation_variance=9.0,
        reliability_threshold=0.5,
        epsilon=1.0e-12,
    )
    high = _pairing_gate_weights(
        preserved_cosine=0.0,
        pairing_cosine=1.0,
        pairing_strength=9.0,
        permutation_variance=1.0,
        reliability_threshold=0.5,
        epsilon=1.0e-12,
    )
    assert low[1] == 0.0 and low[3] is False
    assert high[1] == pytest.approx(1.0) and high[3] is True


def test_weak_noisy_stream_prespecifies_an_effective_reliability_ablation() -> None:
    result = run_campaign(_compact_config(), smoke=True)
    weak = result["comparisons"]["weak_noisy_pairing"]
    ablation = weak["reliability_ablation"]
    assert ablation["baseline"] == "pairing_aware_no_reliability"
    assert ablation["candidate"] == "pairing_aware"
    assert ablation["positive_improvement_favors"] == "pairing_aware"
    assert ablation["mean_candidate_pairing_channel_open_rate"] < (
        ablation["mean_ablation_pairing_channel_open_rate"]
    )
    assert ablation["mse_improvement"] > 0.0
    assert result["prespecified_ablations"]["pairing_aware_no_reliability"][
        "enabled_streams"
    ] == ["weak_noisy_pairing"]


def test_smoke_and_full_outputs_cannot_be_confused() -> None:
    config = _compact_config()
    validate_config(config)
    smoke = run_campaign(config, smoke=True)
    full = run_campaign(config, smoke=False)

    assert smoke["status"] == "SMOKE_PASS"
    assert smoke["mode"] == "smoke"
    assert smoke["scientific_evidence_eligible"] is False
    assert smoke["warning"].startswith("SMOKE OUTPUT")
    assert smoke["run_spec"] == config["smoke"]

    assert full["status"] == "PASS"
    assert full["mode"] == "full"
    assert full["scientific_evidence_eligible"] is False
    assert full["warning"].startswith("INELIGIBLE FULL OUTPUT")
    assert full["integrity"]["validated_source_binding"] is False
    assert full["integrity"]["validated_runtime_binding"] is False
    assert full["run_spec"] == config["full"]
    assert full["integrity"]["v_identity_exact"] is True
    for stream in full["streams"]:
        for policy in CONTROLLER_POLICIES:
            assert (
                full["comparisons"][stream]["v_decomposition_by_policy"][policy][
                    "max_identity_residual"
                ]
                <= 1e-12
            )
        assert full["comparisons"][stream]["pairing_aware_vs_cosine"][
            "positive_improvement_favors"
        ] == "pairing_aware"
        assert full["comparisons"][stream]["pairing_aware_vs_always"][
            "bootstrap_unit"
        ] == "independent_stream_seed"


def test_handcrafted_bindings_cannot_make_direct_api_evidence_eligible() -> None:
    assert not hasattr(controller_provenance, "validated_evidence")
    forged_source = ValidatedEvidence("source", {"forged": True}, object())
    forged_runtime = ValidatedEvidence("runtime", {"forged": True}, object())
    result = run_campaign(
        _compact_config(),
        smoke=False,
        source_binding=forged_source,
        runtime_binding=forged_runtime,
    )
    assert result["scientific_evidence_eligible"] is False
    assert result["source_provenance"] is None
    assert result["runtime_provenance"] is None


def test_cli_requires_exactly_one_explicit_execution_tier() -> None:
    parser = build_parser()
    common = ["--config", str(LOCK), "--output", "/tmp/controller-result.json"]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--smoke", "--full"])
    assert parser.parse_args([*common, "--smoke"]).smoke is True
    full = parser.parse_args([*common, "--full"])
    assert full.full is True and full.smoke is False


def test_lock_rejects_unprespecified_or_ambiguous_campaigns() -> None:
    unlocked = _compact_config()
    unlocked["status"] = "DRAFT"
    with pytest.raises(ControllerConfigError, match="unlocked"):
        validate_config(unlocked)

    overlapping = _compact_config()
    overlapping["smoke"]["seeds"] = [overlapping["full"]["seeds"][0]]
    with pytest.raises(ControllerConfigError, match="disjoint"):
        validate_config(overlapping)

    unknown_policy = _compact_config()
    unknown_policy["policies"][-1] = "oracle"
    with pytest.raises(ControllerConfigError, match="prespecified"):
        validate_config(unknown_policy)

    unknown_option = copy.deepcopy(_compact_config())
    unknown_option["controller"]["lookahead"] = 1
    with pytest.raises(ControllerConfigError, match="keys differ"):
        validate_config(unknown_option)


def test_cli_full_requires_and_binds_the_shared_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "controller-source.json"
    source.write_text(
        json.dumps(build_source_manifest("controller", repo=REPO)),
        encoding="utf-8",
    )
    output = tmp_path / "full.json"

    source_evidence = controller_cli._source_evidence(source, LOCK, smoke=False)
    runtime = tmp_path / "controller-runtime.json"
    runtime.write_text(
        json.dumps(
            build_runtime_manifest(
                experiment="controller",
                config_path=LOCK,
                config=_config(),
                source_evidence=source_evidence,
            )
        ),
        encoding="utf-8",
    )

    def fake_campaign(config, *, smoke, source_binding, runtime_binding):
        assert smoke is False
        bound_source = evidence_payload(source_binding, kind="source")
        bound_runtime = evidence_payload(runtime_binding, kind="runtime")
        assert bound_source is not None
        assert bound_runtime is not None
        return {
            "status": "PASS",
            "mode": "full",
            "scientific_evidence_eligible": True,
            "integrity": {},
            "source_provenance": bound_source,
            "runtime_provenance": bound_runtime,
        }

    monkeypatch.setattr(controller_cli, "run_campaign", fake_campaign)
    assert (
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(output),
                "--runtime-manifest",
                str(runtime),
                "--full",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    evidence = payload["source_provenance"]
    assert evidence["experiment"] == "controller"
    assert evidence["validated_for_execution"] is True
    assert evidence["manifest_sha256"] == sha256_file(source)
    assert evidence["source_bundle_sha256"]
    assert payload["integrity"]["shared_source_manifest_validated"] is True


def test_cli_full_fails_before_execution_when_source_is_missing_or_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("scientific run began before provenance validation")

    monkeypatch.setattr(controller_cli, "run_campaign", forbidden)
    output = tmp_path / "should-not-exist.json"
    with pytest.raises(SourceProvenanceError, match="source validation failed"):
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
                "--full",
            ]
        )
    assert not output.exists()

    stale = build_source_manifest("controller", repo=REPO)
    stale["source_bundle_sha256"] = "0" * 64
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SourceProvenanceError, match="source validation failed"):
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(stale_path),
                "--output",
                str(output),
                "--full",
            ]
        )
    assert not output.exists()


def test_cli_smoke_records_unvalidated_source_without_becoming_paper_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_campaign(config, *, smoke, source_binding, runtime_binding):
        assert smoke is True
        assert source_binding is None
        assert runtime_binding is None
        return {
            "status": "SMOKE_PASS",
            "mode": "smoke",
            "scientific_evidence_eligible": False,
            "integrity": {},
        }

    monkeypatch.setattr(controller_cli, "run_campaign", fake_campaign)
    output = tmp_path / "smoke.json"
    assert (
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
                "--smoke",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scientific_evidence_eligible"] is False
    evidence = payload["source_validation_attempt"]
    assert evidence["shared_manifest_validated"] is False
    assert evidence["validated_for_execution"] is False
    assert evidence["validation_error"]


def test_runtime_capture_is_ineligible_complete_and_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "controller-source.json"
    source_payload = build_source_manifest("controller", repo=REPO)
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    runtime = tmp_path / "controller-runtime.json"
    assert (
        controller_cli.main(
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
    source_evidence = controller_cli._source_evidence(source, LOCK, smoke=False)
    binding = validate_runtime_manifest(
        runtime,
        experiment="controller",
        config_path=LOCK,
        config=_config(),
        source_evidence=source_evidence,
    )
    assert evidence_payload(binding, kind="runtime")["manifest"] == payload
    with pytest.raises(FileExistsError, match="overwrite"):
        controller_cli.main(
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


def test_full_cli_rejects_missing_runtime_overwrite_and_path_aliases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "controller-source.json"
    source.write_text(
        json.dumps(build_source_manifest("controller", repo=REPO)),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requires --runtime-manifest"):
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(tmp_path / "full.json"),
                "--full",
            ]
        )
    with pytest.raises(RuntimeError, match="overwrite"):
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(tmp_path / "full.json"),
                "--full",
                "--overwrite",
            ]
        )
    with pytest.raises(RuntimeError, match="path roles must be disjoint"):
        controller_cli.main(
            [
                "--config",
                str(LOCK),
                "--source-manifest",
                str(source),
                "--output",
                str(source),
                "--capture-runtime",
            ]
        )
