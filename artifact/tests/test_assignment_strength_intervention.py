"""Focused tests for the isolated assignment-strength ER follow-up."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "assignment_strength_overlay" / "models"
LEGACY_OVERLAY = ROOT / "mammoth_overlay" / "models"


def _load(path: Path, module_name: str):
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


STRENGTH = _load(
    OVERLAY / "utils/assignment_strength_intervention.py",
    "gradient_reuse_assignment_strength_intervention_test",
)
LEGACY = _load(
    LEGACY_OVERLAY / "utils/pairing_intervention.py",
    "gradient_reuse_legacy_pairing_intervention_test",
)


def _args(lambda_bp: int = 5000, **overrides):
    values = {
        "assignment_benchmark": "split_cifar100",
        "assignment_run_seed": 3404,
        "assignment_lambda_bp": lambda_bp,
        "assignment_permutation_seed": 103407,
        "assignment_replica": 1,
        "assignment_selection_seed_salt": 77123,
        "assignment_audit_path": None,
        "assignment_audit_interval": 1,
        "assignment_audit_permutations": 4,
        "dataset": "seq-cifar100",
        "model": "er_assignment_strength",
        "seed": 3404,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _legacy_args(**overrides):
    values = {
        "pairing_intervention": "permuted",
        "pairing_permutation_seed": 103407,
        "pairing_replica": 1,
        "pairing_audit_path": None,
        "pairing_audit_interval": 1,
        "pairing_audit_permutations": 4,
        "pairing_benchmark": "split_cifar100",
        "pairing_run_seed": 3404,
        "dataset": "seq-cifar100",
        "model": "er_pairing",
        "seed": 3404,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "size,lambda_bp,expected",
    [
        (1, 0, 0),
        (1, 10000, 1),
        (10, 0, 0),
        (10, 2500, 3),
        (10, 5000, 6),
        (10, 7500, 8),
        (10, 10000, 10),
    ],
)
def test_locked_round_half_up_support_sizes(size, lambda_bp, expected) -> None:
    assert STRENGTH.support_size(size, lambda_bp) == expected


def test_zero_is_identity_and_full_strength_is_legacy_byte_exact() -> None:
    targets = torch.arange(32)

    zero = STRENGTH.AssignmentStrengthIntervention(_args(0), "er_assignment_strength")
    zero.begin_observation()
    zero_targets, zero_plan = zero.intervene(targets, channel="replay_labels")
    assert zero_targets is targets
    assert torch.equal(zero_targets, targets)
    assert torch.equal(zero_plan.permutation, torch.arange(len(targets)))
    assert zero_plan.selected_rows == 0

    full = STRENGTH.AssignmentStrengthIntervention(
        _args(10000), "er_assignment_strength"
    )
    legacy = LEGACY.PairingIntervention(_legacy_args(), "er_pairing")
    full.begin_observation()
    legacy.begin_observation()
    full_targets, full_plan = full.intervene(targets, channel="replay_labels")
    legacy_targets, legacy_permutation = legacy.intervene(
        targets, channel="replay_labels"
    )
    assert full.counter_method == "er_pairing"
    assert torch.equal(full_plan.permutation, legacy_permutation.cpu())
    assert torch.equal(full_targets, legacy_targets)


def test_supports_are_nested_and_counter_key_does_not_include_strength() -> None:
    supports = []
    orders = []
    legacy_permutations = []
    for strength in STRENGTH.ASSIGNMENT_STRENGTHS_BP:
        intervention = STRENGTH.AssignmentStrengthIntervention(
            _args(strength), "er_assignment_strength"
        )
        intervention.begin_observation()
        plan = intervention.assignment_plan(32, channel="replay_labels")
        supports.append(set(plan.selected_support.tolist()))
        orders.append(plan.support_order)
        legacy_permutations.append(plan.legacy_full_permutation)

    assert all(left <= right for left, right in zip(supports, supports[1:]))
    assert all(torch.equal(orders[0], value) for value in orders[1:])
    assert all(
        torch.equal(legacy_permutations[0], value) for value in legacy_permutations[1:]
    )


def test_partial_plan_preserves_multiset_and_is_identity_off_support() -> None:
    intervention = STRENGTH.AssignmentStrengthIntervention(
        _args(5000), "er_assignment_strength"
    )
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    intervention.begin_observation()
    transformed, plan = intervention.intervene(targets, channel="replay_labels")

    support = set(plan.selected_support.tolist())
    assert sorted(transformed.tolist()) == sorted(targets.tolist())
    assert sorted(plan.permutation.tolist()) == list(range(len(targets)))
    for position in set(range(len(targets))) - support:
        assert plan.permutation[position].item() == position
    assert set(plan.permutation[plan.selected_support].tolist()) == support


def test_counter_keyed_plan_is_global_rng_independent() -> None:
    def plan_after_noise(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        _ = np.random.random(100)
        _ = torch.randn(100)
        intervention = STRENGTH.AssignmentStrengthIntervention(
            _args(7500), "er_assignment_strength"
        )
        intervention.begin_observation()
        return intervention.assignment_plan(32, channel="replay_labels")

    first = plan_after_noise(3)
    second = plan_after_noise(999)
    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first.selected_support, second.selected_support)
    assert torch.equal(first.support_order, second.support_order)


def test_strength_has_exact_centered_covariance_erasure_interpretation() -> None:
    """Exhaustively verify the finite-n identity used to label the x-axis."""

    x = np.asarray([-3.0, -1.0, 1.0, 3.0])
    y = np.asarray([-2.0, 0.0, 1.0, 1.0])
    n = len(x)
    m = STRENGTH.support_size(n, 5000)
    scores = []
    for support in itertools.combinations(range(n), m):
        for order in itertools.permutations(support):
            mapping = np.arange(n)
            mapping[np.asarray(support)] = np.asarray(order)
            scores.append(float(np.dot(x, y[mapping])))
    paired = float(np.dot(x, y))
    realized_lambda = (m - 1) / (n - 1)
    assert float(np.mean(scores)) == pytest.approx(
        (1.0 - realized_lambda) * paired,
        abs=1.0e-12,
    )


def test_intervention_audit_records_strength_displacement_values_and_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    intervention = STRENGTH.AssignmentStrengthIntervention(
        _args(5000, assignment_audit_path=str(path)),
        "er_assignment_strength",
    )
    targets = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    indexes = torch.tensor([7, 3, 2, 8, 11, 4, 6, 5])
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    intervention.begin_observation()
    transformed, plan = intervention.intervene(targets, channel="replay_labels")
    intervention.record_intervention(
        targets,
        transformed,
        plan,
        channel="replay_labels",
        sampled_indexes=indexes,
        sampled_inputs=inputs,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    record = next(
        row
        for row in records
        if row["schema_version"]
        == "gradient_reuse.assignment_strength.mammoth_intervention.v1"
    )
    identity = torch.arange(len(targets))
    expected_changed = (targets != transformed).sum().item()
    assert record["lambda_bp"] == 5000
    assert record["selected_rows"] == STRENGTH.support_size(len(targets), 5000)
    assert record["selected_fraction"] == pytest.approx(
        record["selected_rows"] / len(targets)
    )
    assert record["index_displaced_rows"] == int(
        (plan.permutation != identity).sum().item()
    )
    assert record["target_value_changed_rows"] == expected_changed
    assert record["target_multiset_sha256"]
    assert record["permutation_sha256"]
    assert record["selected_support_sha256"]
    assert record["support_order_sha256"]
    assert record["legacy_full_permutation_sha256"]
    assert record["sampled_indexes_sha256"]
    assert record["input_sha256"]


def test_effective_argument_audit_records_explicit_drop_last(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    STRENGTH.AssignmentStrengthIntervention(
        _args(
            5000,
            assignment_audit_path=str(path),
            drop_last=False,
        ),
        "er_assignment_strength",
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    effective = next(
        row["effective_args"]
        for row in records
        if row["schema_version"]
        == "gradient_reuse.assignment_strength.mammoth_effective_args.v1"
    )
    assert effective["drop_last"] is False


def test_gradient_diagnostic_uses_configured_strength_and_restores_state() -> None:
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 4).train()
    inputs = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    rng_before = torch.get_rng_state().clone()
    intervention = STRENGTH.AssignmentStrengthIntervention(
        _args(5000), "er_assignment_strength"
    )
    intervention.begin_observation()
    record = intervention.gradient_diagnostic(
        model,
        inputs,
        targets,
        STRENGTH.cross_entropy_objective,
        channel="replay_labels",
    )
    assert record is not None
    assert record["requested_lambda"] == pytest.approx(0.5)
    assert record["mean_selected_fraction"] == pytest.approx(5 / 8)
    assert record["mean_realized_lambda"] == pytest.approx(4 / 7)
    assert record["assignment_draws"] == 4
    assert record["finite_assignment_variance"] >= 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is True
    for name, value in model.named_parameters():
        assert torch.equal(value.detach(), parameters_before[name])
        assert value.grad is None


def test_er_overlay_never_intervenes_on_current_labels_or_buffer_targets() -> None:
    source = (OVERLAY / "er_assignment_strength.py").read_text(encoding="utf-8")
    assert 'NAME = "er_assignment_strength"' in source
    assert source.count("self.assignment_strength.intervene(") == 1
    assert 'replay_labels, channel="replay_labels"' in source
    assert "current_labels = labels" in source
    assert "labels=current_labels" in source


@pytest.mark.parametrize("strength", [-1, 1, 2499, 10001])
def test_unlocked_strengths_are_rejected(strength: int) -> None:
    with pytest.raises(ValueError, match="unsupported assignment strength"):
        STRENGTH.AssignmentStrengthIntervention(
            _args(strength), "er_assignment_strength"
        )
