from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.fmmt import study_a as A
from experiments.fmmt_reviewer.diagnostics import (
    _flatten,
    output_jacobian_matrix,
    packet_strength,
    run_diagnostic_unit,
    summarize_units,
)
from experiments.fmmt_reviewer.run import (
    ForeignInvocationError,
    _initialize_manifest,
    _identity_sha256,
    _load_resumable,
    _strict_json,
    _unit_path,
    reviewer_process_context,
    verify_source_freeze,
)
from experiments.fmmt_reviewer import run as reviewer_run
from experiments.fmmt_workshop.datasets import load_workshop_stream
from experiments.fmmt_workshop.specificity import _model
from experiments.fmmt.common import atomic_json


REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs/diagnostics.json"


def _config() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


@pytest.mark.parametrize("model_type", ["linear", "functional_units"])
def test_analytic_output_jacobian_matches_autograd(model_type: str) -> None:
    torch.manual_seed(7)
    model = _model(3, model_type, 4)
    parameters = A._named_parameters(model)
    x = torch.randn(5, 3)
    analytic = output_jacobian_matrix(model, parameters, x).double()
    rows = []
    for example in x:
        output = A._call(model, parameters, example.reshape(1, -1)).sum()
        gradient = torch.autograd.grad(output, tuple(parameters.values()))
        rows.append(torch.cat([value.reshape(-1) for value in gradient]).double())
    expected = torch.stack(rows)
    assert torch.allclose(analytic, expected, atol=1e-7, rtol=1e-6)


def test_packet_strength_exact_permutation_moments() -> None:
    torch.manual_seed(11)
    model = _model(2, "linear", 1)
    parameters = A._named_parameters(model)
    x = torch.tensor([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 2.0]])
    y = torch.tensor([-2.0, 0.5, 3.0, 1.5])
    gradient, _ = A._gradient_dict(model, parameters, x, y)
    strength, expectation = packet_strength(model, parameters, x, y, gradient)
    realized = []
    for permutation in itertools.permutations(range(len(y))):
        permuted, _ = A._gradient_dict(
            model, parameters, x, y[torch.tensor(permutation)]
        )
        realized.append(_flatten(permuted).double() / 2.0)
    realized = torch.stack(realized)
    expected = _flatten(expectation).double() / 2.0
    assert torch.allclose(realized.mean(dim=0), expected, atol=1e-7, rtol=1e-7)
    observed_mse = float((realized - expected).square().sum(dim=1).mean().item())
    theory = float(strength["permutation_gradient_l2_sd"]) ** 2
    assert observed_mse == pytest.approx(theory, abs=1e-9, rel=1e-7)
    assert strength["decomposition_relative_error"] < 1e-10
    assert strength["autograd_relative_error"] < 1e-6


@pytest.fixture(scope="module")
def smoke_units() -> list[dict]:
    config = _config()
    runtime = dict(config["model_hyperparameters"])
    runtime.update(
        functional_units=4,
        branch_radius_history=4,
        torch_threads_per_worker=1,
    )
    units = []
    for seed in (100, 101):
        data = load_workshop_stream("BEIJING", smoke=True, seed=seed)
        units.append(
            run_diagnostic_unit(
                data,
                runtime,
                seed=seed,
                model_type="functional_units",
                optimizer_type="adamw",
                packet_size=16,
                n_permutations=4,
                permutation_salt=int(config["runtime"]["permutation_salt"]),
                radius_multipliers=(0.5, 1.0, 2.0),
                multistep_counts=(1, 2),
                device="cpu",
            )
        )
    return units


def test_smoke_unit_is_fixed_trajectory_and_diagnostic_complete(
    smoke_units: list[dict],
) -> None:
    for unit in smoke_units:
        assert unit["audit"]["all_pass"] is True
        assert unit["audit"]["branches_state_isolated"] is True
        assert unit["audit"]["one_true_center_update_per_packet"] is True
        assert len(unit["packet_diagnostics"]) == unit["n_packets"]
        scored = [
            row
            for row in unit["packet_diagnostics"]
            if row["destination_packet_scored"]
        ]
        assert scored
        assert all(row["used_by_next_packet"] for row in scored)
        assert all(row["strength"]["S_J"] >= 0 for row in scored)
        assert all(row["strength"]["S_y"] >= 0 for row in scored)
        assert any(
            entry["activation_segment"]["applicable"]
            for row in scored
            for entry in row["locality"].values()
        )
        assert unit["numerical_diagnostics"][
            "normalized_raw_true_max_prediction_difference"
        ] <= 1e-5


def test_cpu_replay_never_touches_cuda_rng_when_cuda_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for CUDA re-initialization from a forked CPU worker."""
    config = _config()
    runtime = dict(config["model_hyperparameters"])
    runtime.update(
        functional_units=4,
        branch_radius_history=4,
        torch_threads_per_worker=1,
    )
    data = load_workshop_stream("BEIJING", smoke=True, seed=909)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def forbidden_cuda_rng(*args, **kwargs):
        raise AssertionError("a CPU reviewer worker touched the CUDA RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda_rng)
    # A CPU-only local test host has CUDA stubs, whereas the server has a real
    # visible driver. Keep Adam's unrelated graph-capture health probe inert so
    # this regression isolates forbidden RNG/runtime state access.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    unit = run_diagnostic_unit(
        data,
        runtime,
        seed=909,
        model_type="functional_units",
        optimizer_type="adamw",
        packet_size=32,
        n_permutations=2,
        permutation_salt=int(config["runtime"]["permutation_salt"]),
        radius_multipliers=(1.0,),
        multistep_counts=(1,),
        device="cpu",
    )
    assert unit["audit"]["all_pass"] is True
    assert unit["device"] == "cpu"


def test_parallel_reviewer_workers_use_spawn_not_fork() -> None:
    assert reviewer_process_context().get_start_method() == "spawn"


def test_summary_includes_permutation_and_block_uncertainty(
    smoke_units: list[dict],
) -> None:
    config = copy.deepcopy(_config())
    config["inference"]["bootstrap_draws"] = 32
    config["inference"]["block_length_sensitivity"] = [1, 2, 3]
    summary = summarize_units(smoke_units, config)
    assert summary["integrity"]["all_pass"] is True
    assert set(summary["circular_block_length_sensitivity"]) == {"1", "2", "3"}
    primary = summary["primary_normalized_r1_pairing"]
    assert primary["bootstrap"]["permutation_replicas_resampled_within_seed"] is True
    assert "normalized/r0.5" in summary["actuator_pairing_results"]
    assert "raw/r2" in summary["actuator_pairing_results"]
    assert "replay_sgd/steps2" in summary["actuator_pairing_results"]
    assert summary["packet_strength"]["population"]["n_source_packets"] > 0
    assert "t+1" in summary["packet_strength"]["population"]["inclusion_rule"]
    assert summary["locality_diagnostics"]["does_not_prove_smoothness"] is True
    assert "chord" in summary["locality_diagnostics"]["multistep_geometry_limit"]


def test_resume_rejects_identity_or_integrity_mismatch(
    tmp_path: Path, smoke_units: list[dict]
) -> None:
    identity = {
        "dataset": "BEIJING",
        "seed": 100,
        "model": "functional_units",
        "optimizer": "adamw",
        "packet_size": 16,
        "n_permutations": 4,
        "smoke": True,
        "device": "cpu",
        "config_sha256": "c" * 64,
        "source_bundle_sha256": "s" * 64,
    }
    path = _unit_path(tmp_path, identity).with_suffix("")
    unit = copy.deepcopy(smoke_units[0])
    unit["task_identity"] = identity
    unit["task_identity_sha256"] = _identity_sha256(identity)
    unit["source_bundle_sha256"] = identity["source_bundle_sha256"]
    atomic_json(path, unit)
    assert _load_resumable(path, identity) is not None
    bad_identity = dict(identity, seed=101)
    with pytest.raises(RuntimeError, match="identity"):
        _load_resumable(path, bad_identity)
    payload = _strict_json(path)
    payload["audit"]["all_pass"] = False
    atomic_json(path, payload)
    with pytest.raises(RuntimeError, match="failed-integrity"):
        _load_resumable(path, identity)


@pytest.mark.parametrize(
    ("foreign_status", "foreign_field", "foreign_value"),
    [
        ("RUNNING", "source_bundle_sha256", "f" * 64),
        ("COMPLETED", "device", "cuda"),
    ],
)
def test_cli_never_mutates_a_foreign_invocation_directory(
    tmp_path: Path,
    foreign_status: str,
    foreign_field: str,
    foreign_value: str,
) -> None:
    config = _config()
    source_bundle, _ = verify_source_freeze()
    config_sha = reviewer_run.sha256_file(LOCK)
    out = tmp_path / "foreign"
    out.mkdir()
    invocation = {
        "config_sha256": config_sha,
        "source_bundle_sha256": source_bundle,
        "smoke": True,
        "device": "cpu",
    }
    invocation[foreign_field] = foreign_value
    manifest = {
        "schema_version": "fmmt.reviewer_diagnostics.run_manifest.v1",
        "status": foreign_status,
        "started_unix": 1.0,
        "invocation_identity": invocation,
    }
    atomic_json(out / "RUN_MANIFEST.json", manifest)
    (out / "FOREIGN_PAYLOAD.bin").write_bytes(b"immutable foreign evidence\x00\xff")
    if foreign_status == "COMPLETED":
        atomic_json(out / "DONE.json", {"status": "PASS", "foreign": True})

    def snapshot() -> dict[str, bytes]:
        return {
            str(path.relative_to(out)): path.read_bytes()
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    exit_code = reviewer_run.main(
        [
            "--config",
            str(LOCK),
            "--out-dir",
            str(out),
            "--workers",
            "1",
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert exit_code == 2
    assert snapshot() == before
    assert not (out / "FAILURE.json").exists()
    assert json.loads((out / "RUN_MANIFEST.json").read_text())["status"] == foreign_status


def test_exact_completed_invocation_is_byte_exact_idempotent(tmp_path: Path) -> None:
    source_bundle, source_hashes = verify_source_freeze()
    config_sha = reviewer_run.sha256_file(LOCK)
    out = tmp_path / "completed_exact"
    out.mkdir()
    invocation = {
        "config_sha256": config_sha,
        "source_bundle_sha256": source_bundle,
        "smoke": True,
        "device": "cpu",
    }
    manifest = {
        "schema_version": "fmmt.reviewer_diagnostics.run_manifest.v1",
        "status": "COMPLETED",
        "started_unix": 1.0,
        "updated_unix": 2.0,
        "resume_count": 0,
        "invocation_identity": invocation,
        "config_path": str(LOCK),
        "source_hashes": source_hashes,
        "workers": 2,
        "provenance": {},
    }
    atomic_json(out / "RUN_MANIFEST.json", manifest)
    atomic_json(out / "DONE.json", {"status": "PASS", "exact": True})
    (out / "COMPLETED_PAYLOAD.bin").write_bytes(b"completed evidence\x00\xff")

    def snapshot() -> dict[str, bytes]:
        return {
            str(path.relative_to(out)): path.read_bytes()
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    observed = _initialize_manifest(
        out,
        config_path=LOCK,
        config_sha256=config_sha,
        source_bundle=source_bundle,
        source_hashes=source_hashes,
        smoke=True,
        device="cpu",
        workers=2,
    )
    assert observed["status"] == "COMPLETED"
    assert snapshot() == before


def test_source_freeze_matches() -> None:
    bundle, files = verify_source_freeze()
    assert len(bundle) == 64
    assert files
