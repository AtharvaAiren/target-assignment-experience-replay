"""Regression tests for the separate FMMT ICLR breadth campaign."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import experiments.fmmt_iclr_breadth.run as breadth_run
from experiments.fmmt_iclr_breadth.classification import (
    DeepResidualClassifier,
    run_classification_unit,
)
from experiments.fmmt_iclr_breadth.core import (
    AuditPacket,
    _gradient,
    _named_buffers,
    _named_parameters,
    _validate_mode_stable_audit_model,
    run_fixed_trajectory_audit,
    sha256_file,
    sha256_json,
)
from experiments.fmmt_iclr_breadth.run import (
    DEFAULT_LOCK,
    DEFAULT_SOURCE_MANIFEST,
    OutputContractError,
    _identity,
    run_campaign,
    validate_lock,
    validate_source_manifest,
)
from experiments.fmmt_iclr_breadth.sequence import SequentialGRURegressor


@pytest.fixture(scope="module")
def locked_config() -> dict:
    return json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))


def _smoke_args(output: Path, *, device: str = "cpu") -> argparse.Namespace:
    return argparse.Namespace(
        lock=str(DEFAULT_LOCK),
        source_manifest=str(DEFAULT_SOURCE_MANIFEST),
        out=str(output),
        setting="all",
        device=device,
        smoke=True,
        smoke_seed=11,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_incomplete_copy(source: Path, target: Path) -> tuple[float, int]:
    shutil.copytree(source, target)
    (target / "DONE.json").unlink()
    manifest_path = target / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "FAILED"
    manifest.pop("completed_unix", None)
    manifest.pop("completed_units", None)
    started = float(manifest["started_unix"])
    resume_count = int(manifest["resume_count"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return started, resume_count


@pytest.fixture(scope="module")
def smoke_campaign(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("fmmt_iclr_breadth") / "campaign"
    args = _smoke_args(output)
    completed = run_campaign(args)
    assert completed == output.resolve()
    return completed


def test_lock_commits_real_breadth_axes(locked_config: dict) -> None:
    validate_lock(locked_config, lock_path=DEFAULT_LOCK)
    classification = locked_config["tasks"]["classification"]
    sequence = locked_config["tasks"]["sequence"]
    assert classification["dataset"] == "CIFAR100"
    assert classification["loss"] == "cross_entropy"
    assert sum(classification["model"]["blocks_per_stage"]) >= 6
    assert sequence["model"]["layers"] >= 2
    assert sequence["model"]["hidden_size"] >= 64
    assert {value["kind"] for value in sequence["actuators"]} == {
        "stored_gradient",
        "replay_projected",
    }
    assert max(value["steps"] for value in sequence["actuators"]) >= 3


def test_lock_rejects_a_missing_multistep_axis(locked_config: dict) -> None:
    changed = copy.deepcopy(locked_config)
    changed["tasks"]["sequence"]["actuators"] = changed["tasks"]["sequence"][
        "actuators"
    ][:1]
    with pytest.raises(RuntimeError, match="actuator comparison is incomplete"):
        validate_lock(changed, lock_path=DEFAULT_LOCK)


def test_full_classifier_is_genuinely_deep_and_has_no_running_statistics(
    locked_config: dict,
) -> None:
    specification = locked_config["tasks"]["classification"]["model"]
    model = DeepResidualClassifier(
        classes=100,
        widths=list(specification["widths"]),
        blocks_per_stage=list(specification["blocks_per_stage"]),
    )
    convolutions = sum(isinstance(value, torch.nn.Conv2d) for value in model.modules())
    parameters = sum(value.numel() for value in model.parameters())
    assert convolutions >= 13
    assert parameters >= 2_000_000
    assert not any(isinstance(value, torch.nn.modules.batchnorm._BatchNorm) for value in model.modules())


def test_cross_entropy_permutation_expectation_preserves_class_marginal() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 3, bias=False).double()
    inputs = torch.tensor(
        [[1.0, 0.0, 0.2], [0.0, 1.0, -0.3], [0.5, 0.4, 1.0], [-0.2, 0.3, 0.7]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    gradients = []
    for permutation in itertools.permutations(range(len(labels))):
        permuted = labels[torch.tensor(permutation)]
        gradients.append(
            _gradient(
                model,
                parameters,
                buffers,
                inputs,
                permuted,
                "classification",
            )[0]["weight"]
        )
    empirical = torch.stack(gradients).mean(dim=0)
    logits = model(inputs)
    class_marginal = F.one_hot(labels, num_classes=3).double().mean(dim=0)
    marginal_loss = -(class_marginal[None, :] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    expected = torch.autograd.grad(marginal_loss, model.weight)[0]
    torch.testing.assert_close(empirical, expected, rtol=1e-12, atol=1e-12)


class _MockCuDNNBackward(torch.autograd.Function):
    """CPU-only stand-in for CuDNN's RNN backward-mode requirement."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, training: bool) -> torch.Tensor:
        ctx.forward_was_training = bool(training)
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        if not ctx.forward_was_training:
            raise RuntimeError(
                "cudnn RNN backward can only be called in training mode"
            )
        return gradient, None


class _MockCuDNNRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.3]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs @ self.weight
        return _MockCuDNNBackward.apply(value, self.training)


def test_gradient_uses_training_forward_for_cudnn_and_restores_eval() -> None:
    model = _MockCuDNNRegressor().eval()
    inputs = torch.tensor([[1.0, 2.0], [-0.5, 0.3], [0.2, -0.7]])
    targets = torch.tensor([0.0, 1.0, -0.5])

    # This reproduces the failure contract of an eval-mode CuDNN RNN forward.
    direct = F.mse_loss(model(inputs), targets)
    with pytest.raises(RuntimeError, match="backward can only be called in training mode"):
        torch.autograd.grad(direct, tuple(model.parameters()))

    gradients, loss = _gradient(
        model,
        _named_parameters(model),
        _named_buffers(model),
        inputs,
        targets,
        "regression",
    )
    assert np.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert model.training is False
    assert all(module.training is False for module in model.modules())


def test_cpu_gru_gradient_preserves_eval_prediction_semantics() -> None:
    torch.manual_seed(31)
    model = SequentialGRURegressor(
        3, hidden_size=7, layers=2, mlp_width=5
    ).eval()
    inputs = torch.randn(6, 4, 3)
    targets = torch.randn(6)
    with torch.no_grad():
        evaluation_output = model(inputs).clone()

    gradients, loss = _gradient(
        model,
        _named_parameters(model),
        _named_buffers(model),
        inputs,
        targets,
        "regression",
    )
    assert np.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert all(module.training is False for module in model.modules())

    # With locked zero dropout and deterministic LayerNorm, the temporary
    # gradient-forward mode has the same function as evaluation mode.
    model.train()
    with torch.no_grad():
        training_output = model(inputs)
    torch.testing.assert_close(evaluation_output, training_output)
    model.eval()


def test_audit_rejects_a_mode_sensitive_predictor() -> None:
    unsafe = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(4, 1),
    )
    with pytest.raises(RuntimeError, match="active dropout"):
        _validate_mode_stable_audit_model(unsafe)

    safe = SequentialGRURegressor(
        3, hidden_size=7, layers=2, mlp_width=5
    )
    _validate_mode_stable_audit_model(safe)


def test_source_manifest_authenticates_every_breadth_input() -> None:
    bundle, files = validate_source_manifest(DEFAULT_SOURCE_MANIFEST)
    assert len(bundle) == 64
    assert "configs/breadth.json" in files
    assert "experiments/fmmt_iclr_breadth/core.py" in files
    assert "tests/test_fmmt_iclr_breadth.py" in files
    manifest = json.loads(DEFAULT_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["lock_sha256"] == sha256_file(DEFAULT_LOCK)


def test_smoke_unit_is_independent_of_inherited_rng_state(locked_config: dict) -> None:
    first = run_classification_unit(
        locked_config["tasks"]["classification"],
        seed=19,
        device="cpu",
        smoke=True,
    )
    # Deliberately perturb both RNGs that affect model/data construction.
    _ = torch.randn(1000)
    _ = np.random.random(1000)
    second = run_classification_unit(
        locked_config["tasks"]["classification"],
        seed=19,
        device="cpu",
        smoke=True,
    )
    assert sha256_json(first) == sha256_json(second)


def test_end_to_end_smoke_is_fixed_trajectory_and_paper_ineligible(
    smoke_campaign: Path,
) -> None:
    done = json.loads((smoke_campaign / "DONE.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (smoke_campaign / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (smoke_campaign / "PAPER_SUMMARY.json").read_text(encoding="utf-8")
    )
    assert done["status"] == "PASS"
    assert done["paper_evidence_eligible"] is False
    assert summary["paper_evidence_eligible"] is False
    assert manifest["status"] == "COMPLETED"
    assert manifest["resume_count"] == 0
    assert manifest["invocation_identity"]["device"] == "cpu"
    assert len(summary["rows"]) == 3  # classification s1; sequence s1 and s3.
    assert all(row["units"] == 1 for row in summary["rows"])
    assert {row["setting"] for row in summary["seed_rows"]} == {
        "classification_cross_entropy",
        "sequence_regression",
    }
    assert (smoke_campaign / "PAPER_SEED_SUMMARY.csv").is_file()
    assert (smoke_campaign / "PAPER_DOMAIN_SUMMARY.csv").is_file()
    for artifact_path in sorted((smoke_campaign / "units").glob("*.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        result = artifact["result"]
        assert result["audit"]["all_pass"] is True
        assert result["audit"]["live_trajectory_uses_true_labels_only"] is True
        assert result["audit"]["common_center_and_estimator_matched"] is True
        assert result["audit"]["scored_predictions_use_evaluation_mode"] is True
        assert result["audit"]["gradient_forward_mode_restored"] is True
        assert result["audit"]["mode_sensitive_layers_absent"] is True
        assert result["distribution_alignment"][
            "target_multiset_preserved_all_packets"
        ] is True


def test_completed_directory_is_rejected_without_mutation(
    smoke_campaign: Path,
) -> None:
    before = _tree_bytes(smoke_campaign)
    with pytest.raises(OutputContractError, match="already complete"):
        run_campaign(_smoke_args(smoke_campaign))
    assert _tree_bytes(smoke_campaign) == before


def test_exact_incomplete_resume_preserves_start_and_units(
    smoke_campaign: Path, tmp_path: Path
) -> None:
    resumed = tmp_path / "resume"
    started, resume_count = _make_incomplete_copy(smoke_campaign, resumed)
    units_before = _tree_bytes(resumed / "units")
    run_campaign(_smoke_args(resumed))
    units_after = _tree_bytes(resumed / "units")
    manifest = json.loads((resumed / "RUN_MANIFEST.json").read_text())
    assert units_after == units_before
    assert manifest["started_unix"] == started
    assert manifest["resume_count"] == resume_count + 1
    assert manifest["status"] == "COMPLETED"


def test_changed_source_identity_rejected_byte_exact(
    smoke_campaign: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source_mismatch"
    _make_incomplete_copy(smoke_campaign, target)
    before = _tree_bytes(target)
    _, source_files = validate_source_manifest(DEFAULT_SOURCE_MANIFEST)
    monkeypatch.setattr(
        breadth_run,
        "validate_source_manifest",
        lambda _path: ("f" * 64, source_files),
    )
    with pytest.raises(OutputContractError, match="invocation identity mismatch"):
        run_campaign(_smoke_args(target))
    assert _tree_bytes(target) == before


def test_cpu_cuda_smoke_identity_mismatch_is_read_only(
    smoke_campaign: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "device_mismatch"
    _make_incomplete_copy(smoke_campaign, target)
    before = _tree_bytes(target)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(OutputContractError, match="invocation identity mismatch"):
        run_campaign(_smoke_args(target, device="cuda"))
    assert _tree_bytes(target) == before

    common = {
        "setting": "sequence",
        "seed": 11,
        "smoke": True,
        "lock_sha256": "a" * 64,
        "source_bundle": "b" * 64,
    }
    assert _identity(**common, device="cpu") != _identity(**common, device="cuda")


def test_duplicate_stream_identity_fails_closed() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    packets = [
        AuditPacket(
            inputs=torch.randn(4, 2),
            targets=np.arange(4, dtype=np.float32),
            ids=np.arange(4),
            cluster="0",
            domain="smoke",
        ),
        AuditPacket(
            inputs=torch.randn(4, 2),
            targets=np.arange(4, dtype=np.float32),
            ids=np.asarray([3, 4, 5, 6]),
            cluster="1",
            domain="smoke",
        ),
    ]
    with pytest.raises(RuntimeError, match="reuses an example ID"):
        run_fixed_trajectory_audit(
            model=model,
            optimizer=optimizer,
            packets=packets,
            task="regression",
            actuators=[{"name": "s1", "kind": "stored_gradient", "steps": 1}],
            n_permutations=2,
            permutation_salt=7,
            seed=3,
            device="cpu",
            radius_history=[0.01],
            radius_window=2,
            fallback_radius=0.01,
            gradient_clip=1.0,
            bootstrap_draws=20,
            bootstrap_block_length=1,
        )
