"""Tests for equal-severity historical task-structure swaps."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "task_structure_overlay/models/utils/task_structure_intervention.py"
SPEC = importlib.util.spec_from_file_location("task_structure_intervention_test", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _plan(labels: list[int], *, current_task: int = 4, observation: int = 7):
    target = torch.tensor(labels, dtype=torch.long)
    return target, MODULE.build_task_structure_plan(
        target,
        target // 5,
        current_task=current_task,
        counter_seed=203504,
        run_seed=3501,
        selection_seed_salt=3284386755,
        observation=observation,
    )


def _args(tmp_path: Path, arm: str = "paired_probe") -> SimpleNamespace:
    return SimpleNamespace(
        task_structure_arm=arm,
        task_structure_run_seed=3501,
        task_structure_counter_seed=203504,
        task_structure_selection_seed_salt=3284386755,
        task_structure_audit_path=str(tmp_path / "audit.jsonl"),
        task_structure_gradient_interval=1,
        seed=3501,
        dataset="seq-cifar100",
        model="er_task_structure_swap",
        dataset_config="20tasks",
        backbone="resnet18",
        lr=0.03,
        n_epochs=1,
        device="cuda:0",
        num_workers=4,
        buffer_size=500,
        minibatch_size=32,
        batch_size=32,
        permute_classes=False,
        custom_class_order=None,
        custom_task_order=None,
        noise_rate=0.0,
    )


def test_pinned_split_cifar_mapping_is_exact() -> None:
    labels = torch.arange(100)
    observed = MODULE.task_ids_from_split_cifar100_labels(labels)
    assert torch.equal(observed, torch.arange(20).repeat_interleave(5))
    with pytest.raises(RuntimeError):
        MODULE.task_ids_from_split_cifar100_labels(torch.tensor([-1, 0]))
    with pytest.raises(RuntimeError):
        MODULE.task_ids_from_split_cifar100_labels(torch.tensor([99, 100]))


def test_same_quadruples_have_equal_severity_and_opposite_task_structure() -> None:
    labels, plan = _plan([0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17])
    tasks = labels // 5
    within = labels[plan.within_permutation]
    across = labels[plan.across_permutation]
    selected = plan.selected_positions
    assert plan.selected_quadruples == 2
    assert selected.numel() == 8
    assert int((within != labels).sum()) == int(selected.numel())
    assert int((across != labels).sum()) == int(selected.numel())
    assert Counter(within.tolist()) == Counter(labels.tolist())
    assert Counter(across.tolist()) == Counter(labels.tolist())
    assert torch.equal(tasks[plan.within_permutation[selected]], tasks[selected])
    assert bool((tasks[plan.across_permutation[selected]] != tasks[selected]).all())
    outside = sorted(set(range(len(labels))) - set(selected.tolist()))
    assert torch.equal(within[outside], labels[outside])
    assert torch.equal(across[outside], labels[outside])


def test_greedy_pairing_reaches_maximum_disjoint_quadruple_count() -> None:
    rng = np.random.default_rng(19)
    for trial in range(100):
        labels: list[int] = []
        capacities = []
        for task in range(4):
            count = int(rng.integers(0, 10))
            task_labels = [5 * task + int(rng.integers(0, 5)) for _ in range(count)]
            labels.extend(task_labels)
            counts = Counter(task_labels)
            capacities.append(
                min(count // 2, count - max(counts.values(), default=0))
            )
        if not labels:
            labels = [0]
        target, plan = _plan(labels, current_task=4, observation=trial)
        total = sum(capacities)
        expected = min(total // 2, total - max(capacities, default=0))
        assert plan.selected_quadruples == expected
        MODULE.validate_plan(target, target // 5, current_task=4, plan=plan)


def test_plan_is_counter_keyed_and_global_rng_independent() -> None:
    labels = [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]

    def after_noise(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        _ = np.random.random(100)
        _ = torch.randn(100)
        return _plan(labels)[1]

    first, second = after_noise(3), after_noise(999)
    assert first.quadruples == second.quadruples
    assert torch.equal(first.within_permutation, second.within_permutation)
    assert torch.equal(first.across_permutation, second.across_permutation)


@pytest.mark.parametrize(
    "labels,reason",
    [
        ([20, 21], "no_historical_replay_rows"),
        ([0, 1, 2], "fewer_than_two_historical_tasks"),
        ([0, 0, 5, 5], "fewer_than_two_tasks_with_distinct_class_pairs"),
    ],
)
def test_ineligible_replay_batches_are_classified(labels: list[int], reason: str) -> None:
    _, plan = _plan(labels, current_task=4)
    assert plan.selected_quadruples == 0
    assert plan.ineligible_reason == reason


def test_explicit_buffer_tasks_must_match_pinned_mapping() -> None:
    labels = torch.tensor([0, 1, 5, 6])
    with pytest.raises(RuntimeError, match="disagree"):
        MODULE.build_task_structure_plan(
            labels,
            torch.tensor([0, 0, 0, 0]),
            current_task=2,
            counter_seed=1,
            run_seed=3501,
            selection_seed_salt=2,
            observation=0,
        )


def test_probe_common_state_gradients_restore_model_rng_and_grads(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 20)).train()
    model[0].weight.grad = torch.ones_like(model[0].weight)
    inputs = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 5, 6, 10, 11, 15, 16])
    intervention = MODULE.TaskStructureIntervention(
        _args(tmp_path), "er_task_structure_swap"
    )
    intervention.begin_observation()
    plan = intervention.plan(labels, labels // 5, current_task=4)
    rng_before = torch.get_rng_state().clone()
    parameters_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    original_grad = model[0].weight.grad.detach().clone()
    record = intervention.gradient_diagnostic(
        model,
        inputs,
        labels,
        plan,
        MODULE.cross_entropy_objective,
        sampled_indexes=torch.arange(8),
    )
    assert record is not None
    assert record["common_parameter_state"] is True
    assert record["true_across_distance"] >= 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(model[0].weight.grad, original_grad)
    for name, value in model.named_parameters():
        assert torch.equal(value.detach(), parameters_before[name])
    assert model.training is True


def test_exposure_logs_every_arm_and_exact_changed_rows(tmp_path: Path) -> None:
    labels, plan = _plan([0, 1, 5, 6, 10, 11, 15, 16], current_task=4)
    for arm in MODULE.ARMS:
        arm_root = tmp_path / arm
        arm_root.mkdir()
        intervention = MODULE.TaskStructureIntervention(
            _args(arm_root, arm), "er_task_structure_swap"
        )
        intervention.begin_observation()
        intervention.exposure(
            labels,
            labels // 5,
            plan,
            current_task=4,
            sampled_indexes=torch.arange(len(labels)),
            sampled_inputs=torch.arange(len(labels) * 2).reshape(len(labels), 2),
        )
        rows = [json.loads(line) for line in (arm_root / "audit.jsonl").read_text().splitlines()]
        exposure = next(row for row in rows if row["schema_version"].endswith("exposure.v1"))
        expected = 0 if arm == "paired_probe" else plan.selected_positions.numel()
        assert exposure["changed_rows"] == expected
        assert exposure["target_multiset_preserved_within"] is True
        assert exposure["target_multiset_preserved_across"] is True


def test_overlay_never_changes_current_labels_or_buffer_task_metadata() -> None:
    source = (ROOT / "task_structure_overlay/models/er_task_structure_swap.py").read_text()
    assert 'NAME = "er_task_structure_swap"' in source
    assert source.count("self.task_structure.apply(") == 1
    assert "current_labels = labels" in source
    assert "labels=current_labels" in source
    assert "task_labels=current_tasks" in source
