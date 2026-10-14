from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "mean_matched_overlay/models/utils/mean_matched_intervention.py"
SPEC = importlib.util.spec_from_file_location("mean_matched_intervention_test", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _all_parent_targets(labels: torch.Tensor, classes: int, lambda_bp: int) -> list[torch.Tensor]:
    n = len(labels)
    m = MODULE.support_size(n, lambda_bp)
    one_hot = torch.nn.functional.one_hot(labels, classes).double()
    if m == 0:
        return [one_hot]
    rows = []
    for support in itertools.combinations(range(n), m):
        for source_order in itertools.permutations(support):
            permutation = list(range(n))
            for target_row, source_row in zip(support, source_order):
                permutation[target_row] = source_row
            rows.append(one_hot[torch.tensor(permutation)])
    return rows


@pytest.mark.parametrize("n,lambda_bp", [(3, 2500), (3, 5000), (4, 7500), (4, 10000)])
def test_mean_target_is_exact_parent_expectation(n: int, lambda_bp: int) -> None:
    labels = torch.tensor([0, 1, 2, 1][:n])
    expected = torch.stack(_all_parent_targets(labels, 3, lambda_bp)).mean(dim=0)
    observed, strength, support = MODULE.mean_matched_targets(
        labels, num_classes=3, lambda_bp=lambda_bp
    )
    assert support == MODULE.support_size(n, lambda_bp)
    assert strength == (support - 1) / (n - 1)
    assert torch.allclose(observed.double(), expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("lambda_bp", [2500, 5000, 7500, 10000])
def test_gradient_equals_exact_monte_carlo_expectation(lambda_bp: int) -> None:
    torch.manual_seed(91)
    inputs = torch.randn(4, 5, dtype=torch.double)
    labels = torch.tensor([0, 1, 2, 1])
    weight = torch.randn(3, 5, dtype=torch.double, requires_grad=True)

    def gradient(targets: torch.Tensor) -> torch.Tensor:
        logits = inputs @ weight.t()
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return torch.autograd.grad(loss, weight, retain_graph=False)[0]

    stochastic = torch.stack(
        [gradient(target) for target in _all_parent_targets(labels, 3, lambda_bp)]
    ).mean(dim=0)
    mean_target, _, _ = MODULE.mean_matched_targets(
        labels, num_classes=3, lambda_bp=lambda_bp, dtype=torch.double
    )
    deterministic = gradient(mean_target.double())
    assert torch.allclose(deterministic, stochastic, rtol=0, atol=2e-12)


def test_endpoints_are_identity_and_batch_mean() -> None:
    labels = torch.tensor([0, 0, 1, 2])
    identity, zero, m0 = MODULE.mean_matched_targets(labels, num_classes=3, lambda_bp=0)
    full, one, m1 = MODULE.mean_matched_targets(labels, num_classes=3, lambda_bp=10000)
    one_hot = torch.nn.functional.one_hot(labels, 3).float()
    assert zero == 0 and m0 == 0 and torch.equal(identity, one_hot)
    assert one == 1 and m1 == len(labels)
    assert torch.allclose(full, one_hot.mean(dim=0, keepdim=True).expand_as(one_hot))


def test_requested_grid_uses_parent_round_half_up_law() -> None:
    assert [MODULE.support_size(32, value) for value in (2500, 5000, 7500, 10000)] == [9, 17, 24, 32]
    assert MODULE.realized_lambda(32, 2500) == 8 / 31


def test_soft_cross_entropy_matches_hard_labels() -> None:
    logits = torch.tensor([[1.0, -0.2, 0.4], [-0.3, 0.8, 0.1]])
    labels = torch.tensor([2, 1])
    targets = torch.nn.functional.one_hot(labels, 3).float()
    assert torch.allclose(
        MODULE.soft_cross_entropy(logits, targets),
        torch.nn.functional.cross_entropy(logits, labels),
    )
