from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = _load(
    "test_shared_output_structure_intervention",
    "explanatory_followup_overlay/models/utils/shared_output_structure_intervention.py",
)
variance = _load(
    "test_variance_ladder_intervention",
    "explanatory_followup_overlay/models/utils/variance_ladder_intervention.py",
)


def test_four_row_atoms_match_rows_and_preserve_target_multiset() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    domains = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    plan = shared.build_shared_output_plan(
        labels, domains, current_domain=4, counter_seed=303604,
        run_seed=3601, selection_seed_salt=3919349337, observation=12,
    )
    assert plan.selected_atoms == 2
    assert plan.selected_positions.numel() == 8
    within = labels[plan.within_permutation]
    across = labels[plan.across_permutation]
    assert sorted(within.tolist()) == sorted(labels.tolist())
    assert sorted(across.tolist()) == sorted(labels.tolist())
    assert torch.all(within != labels)
    assert torch.all(across != labels)
    assert torch.equal(domains[plan.within_permutation], domains)
    assert torch.all(domains[plan.across_permutation] != domains)


def test_ineligible_shared_batch_is_a_matched_noop() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 0, 1, 1])
    plan = shared.build_shared_output_plan(
        labels, domains, current_domain=2, counter_seed=303604,
        run_seed=3601, selection_seed_salt=3919349337, observation=0,
    )
    assert plan.selected_atoms == 0
    assert torch.equal(plan.within_permutation, torch.arange(4))
    assert torch.equal(plan.across_permutation, torch.arange(4))


def test_no_replay_and_final_domain_il_evidence(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    args = SimpleNamespace(
        shared_structure_arm="no_replay", shared_structure_run_seed=3601,
        shared_structure_counter_seed=303604,
        shared_structure_selection_seed_salt=3919349337,
        shared_structure_audit_path=str(path), shared_structure_gradient_interval=500,
        seed=3601, model="er_shared_output_structure", dataset="perm-mnist",
        dataset_config=None, backbone="mnistmlp", lr=0.1, n_epochs=1,
        device="cpu", num_workers=0, buffer_size=500,
        minibatch_size=8, batch_size=8,
    )
    intervention = shared.SharedOutputStructureIntervention(args, "er_shared_output_structure")
    intervention.begin_observation()
    intervention.no_replay_update(rows=8, current_domain=0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 10))
    dataset = SimpleNamespace(test_loaders=[])
    for domain in range(20):
        inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16 + domain / 100
        labels = torch.tensor([0, 1, 2, 3])
        dataset.test_loaders.append([(inputs, labels)])
    intervention.final_evaluation(model, dataset, completed_domains=20)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-2]["schema_version"].endswith("no_replay_update.v1")
    assert rows[-1]["schema_version"].endswith("final_domain_il.v1")
    assert rows[-1]["examples"] == 80
    assert rows[-1]["historical_domain_supplied_at_inference"] is False
    assert rows[-1]["model_state_before_sha256"] == rows[-1]["model_state_after_sha256"]


def test_variance_draws_are_deterministic_and_domain_separated() -> None:
    first = variance.independent_assignment_draw(
        32, counter_seed=403704, run_seed=3701, observation=5, draw=2,
        support_seed_salt=2654435761, permutation_seed_salt=2246822519,
    )
    second = variance.independent_assignment_draw(
        32, counter_seed=403704, run_seed=3701, observation=5, draw=2,
        support_seed_salt=2654435761, permutation_seed_salt=2246822519,
    )
    assert torch.equal(first.support, second.support)
    assert torch.equal(first.permutation, second.permutation)
    assert first.support_key_sha256 != first.permutation_key_sha256
    assert len(first.support) == variance.support_size(32) == 24


def test_finite_k_and_analytic_mean_preserve_class_mass() -> None:
    labels = torch.arange(32) % 10
    result = variance.finite_k_targets(
        labels, num_classes=10, dtype=torch.float32, k=16,
        counter_seed=403704, run_seed=3701, observation=7,
        support_seed_salt=2654435761, permutation_seed_salt=2246822519,
    )
    mass = torch.nn.functional.one_hot(labels, num_classes=10).sum(0).float()
    assert len(result.draws) == 16
    assert torch.allclose(result.used_targets.sum(0), mass)
    assert torch.allclose(result.exact_mean.sum(0), mass)


def test_finite_k_arms_share_the_locked_common_random_prefix() -> None:
    labels = torch.arange(32) % 10
    common = {
        "num_classes": 10, "dtype": torch.float32,
        "counter_seed": 403704, "run_seed": 3701, "observation": 19,
        "support_seed_salt": 2654435761,
        "permutation_seed_salt": 2246822519,
    }
    one = variance.finite_k_targets(labels, k=1, **common)
    four = variance.finite_k_targets(labels, k=4, **common)
    sixteen = variance.finite_k_targets(labels, k=16, **common)
    for index, draw in enumerate(one.draws):
        assert torch.equal(draw.support, four.draws[index].support)
        assert torch.equal(draw.permutation, four.draws[index].permutation)
        assert draw.support_key_sha256 == four.draws[index].support_key_sha256
        assert draw.permutation_key_sha256 == four.draws[index].permutation_key_sha256
    for index, draw in enumerate(four.draws):
        assert torch.equal(draw.support, sixteen.draws[index].support)
        assert torch.equal(draw.permutation, sixteen.draws[index].permutation)
        assert draw.support_key_sha256 == sixteen.draws[index].support_key_sha256
        assert draw.permutation_key_sha256 == sixteen.draws[index].permutation_key_sha256


def test_analytic_mean_equals_exhaustive_support_permutation_average() -> None:
    labels = torch.tensor([0, 1, 1, 2])
    rows = len(labels)
    support_rows = variance.support_size(rows)
    targets = []
    for support in itertools.combinations(range(rows), support_rows):
        for sources in itertools.permutations(support):
            permutation = torch.arange(rows)
            permutation[list(support)] = torch.tensor(sources)
            targets.append(torch.nn.functional.one_hot(labels[permutation], num_classes=3).float())
    exhaustive = torch.stack(targets).mean(0)
    analytic = variance.exact_mean_targets(labels, num_classes=3, dtype=torch.float32)
    assert torch.allclose(exhaustive, analytic, rtol=0, atol=1e-7)


def test_exact_common_state_gradient_linearity_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    args = SimpleNamespace(
        variance_benchmark="permuted_mnist", variance_run_seed=3701,
        variance_k="4", variance_lambda_bp=7500,
        variance_counter_seed=403704,
        variance_support_seed_salt=2654435761,
        variance_permutation_seed_salt=2246822519,
        variance_audit_path=str(path), variance_audit_interval=1,
        seed=3701, model="er_variance_ladder", dataset="perm-mnist",
        dataset_config=None, backbone="mnistmlp", lr=0.1, n_epochs=1,
        device="cpu", num_workers=0, buffer_size=500,
        minibatch_size=8, batch_size=8,
    )
    intervention = variance.VarianceLadderIntervention(args, "er_variance_ladder")
    intervention.begin_observation()
    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Tanh(), torch.nn.Linear(7, 4))
    inputs = torch.randn(8, 3)
    labels = torch.arange(8) % 4
    targets = intervention.transform(labels, num_classes=4, dtype=torch.float32)
    intervention.exposure(
        labels, targets, sampled_indexes=torch.arange(8), sampled_inputs=inputs,
    )
    intervention.audit(model, inputs, labels, targets, sampled_indexes=torch.arange(8))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[-1]
    assert row["finite_k_gradient_equals_mean_hard_gradient"] is True
    assert row["exact_mean_gradient_equals_analytic_mixture"] is True
    assert row["gradient_linearity_max_abs_residual"] <= 2e-5
    exposure = rows[-2]
    assert len(exposure["support_key_sha256"]) == 4
    assert set(exposure["support_key_sha256"]).isdisjoint(exposure["permutation_key_sha256"])
