"""Read-only and estimand tests for the state-matched E3/E6 probe."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.gradient_reuse_reviewer_extension import unit_runner
from experiments.gradient_reuse_reviewer_extension.protocol import ExtensionUnit


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "reviewer_extension_overlay/models/utils/state_matched_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "gradient_reuse_state_matched_audit_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load state-matched helper: {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class _Selector:
    def __init__(self) -> None:
        self.counter = 7


class _Buffer:
    def __init__(self) -> None:
        self.examples = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.num_seen_examples = 6
        self.sample_selection_fn = _Selector()


def _optimizer_with_state(model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    targets = torch.tensor([0, 1])
    torch.nn.functional.cross_entropy(model(inputs), targets).backward()
    optimizer.step()
    return optimizer


def test_counter_keyed_permutations_are_reproducible_and_rng_independent() -> None:
    def draw(noise_seed: int) -> torch.Tensor:
        random.seed(noise_seed)
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)
        _ = [random.random() for _ in range(5)]
        _ = np.random.random(20)
        _ = torch.rand(20)
        return AUDIT.deterministic_permutation(32, run_seed=3401, boundary=7, replica=2)

    first = draw(3)
    second = draw(999)
    torch.testing.assert_close(first, second)
    assert sorted(first.tolist()) == list(range(32))
    assert not torch.equal(
        first,
        AUDIT.deterministic_permutation(32, run_seed=3401, boundary=7, replica=3),
    )


def test_live_state_guard_restores_rng_modes_gradients_and_loader_generator() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4, 2),
    ).train()
    optimizer = _optimizer_with_state(model)
    buffer = _Buffer()
    generator = torch.Generator().manual_seed(991)
    loader = DataLoader(
        TensorDataset(torch.arange(20).reshape(10, 2).float(), torch.arange(10)),
        batch_size=2,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    python_state = random.getstate()
    expected_python = random.random()
    random.setstate(python_state)
    numpy_state = np.random.get_state()
    expected_numpy = float(np.random.random())
    np.random.set_state(numpy_state)
    torch_state = torch.get_rng_state().clone()
    expected_torch = torch.rand(4)
    torch.set_rng_state(torch_state)
    loader_state = generator.get_state().clone()
    expected_loader_labels = torch.cat([labels for _, labels in loader])
    generator.set_state(loader_state)
    gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]
    model_hash = AUDIT.model_state_sha256(model)
    optimizer_hash = AUDIT.optimizer_state_sha256(optimizer)
    buffer_hash = AUDIT.buffer_state_sha256(buffer)

    with AUDIT.preserve_live_state(
        model,
        optimizer=optimizer,
        buffer=buffer,
        loaders=(loader,),
    ) as state:
        assert state["model"] == model_hash
        assert state["optimizer"] == optimizer_hash
        assert state["buffer"] == buffer_hash
        _ = random.random()
        _ = np.random.random()
        _ = torch.rand(10)
        _ = list(loader)
        model.eval()
        for parameter in model.parameters():
            parameter.grad = None

    assert model.training is True
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch)
    observed_loader_labels = torch.cat([labels for _, labels in loader])
    torch.testing.assert_close(observed_loader_labels, expected_loader_labels)
    for parameter, expected in zip(model.parameters(), gradients):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected)
    assert AUDIT.model_state_sha256(model) == model_hash
    assert AUDIT.optimizer_state_sha256(optimizer) == optimizer_hash
    assert AUDIT.buffer_state_sha256(buffer) == buffer_hash


def test_live_state_guard_rejects_persistent_loader_workers() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="persistent DataLoader workers"):
        with AUDIT.preserve_live_state(
            model,
            optimizer=optimizer,
            buffer=_Buffer(),
            loaders=(SimpleNamespace(persistent_workers=True),),
        ):
            pass


def test_state_matched_branches_obey_value_identity_and_first_order_formula() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.8, -0.2], [-0.3, 0.7]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = _Buffer()
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    directions = {
        "paired": torch.tensor([1.0, 0.0, -1.0, 0.0]),
        "permuted_0": torch.tensor([-1.0, 0.0, 1.0, 0.0]),
        "permuted_1": torch.tensor([0.0, 1.0, 0.0, -1.0]),
        "permuted_2": torch.tensor([1.0, 1.0, -1.0, -1.0]),
        "permuted_3": torch.tensor([-1.0, 1.0, 1.0, -1.0]),
    }
    radius = 0.05
    model_before = AUDIT.model_state_sha256(model)

    with AUDIT.preserve_live_state(
        model,
        optimizer=optimizer,
        buffer=buffer,
        loaders=(loader,),
    ):
        result = AUDIT.evaluate_old_task_branches(
            model, (loader,), directions, radius=radius
        )

    assert result["examples"] == 4
    assert set(result["branches"]) == {
        "no_step",
        "paired",
        "permuted_0",
        "permuted_1",
        "permuted_2",
        "permuted_3",
    }
    for endpoint in ("cross_entropy", "accuracy"):
        values = result[endpoint]
        assert values["identity_residual"] == pytest.approx(0.0, abs=1.0e-14)
        assert values["V_total"] == pytest.approx(values["V_perm"] + values["V_pair"])
    assert result["cross_entropy"]["V_total"] == pytest.approx(
        result["branches"]["no_step"]["cross_entropy"]
        - result["branches"]["paired"]["cross_entropy"]
    )
    assert result["accuracy"]["V_total"] == pytest.approx(
        result["branches"]["paired"]["accuracy"]
        - result["branches"]["no_step"]["accuracy"]
    )

    model.zero_grad(set_to_none=True)
    risk = torch.nn.functional.cross_entropy(model(inputs), targets)
    risk_gradient = torch.autograd.grad(risk, tuple(model.parameters()))[0].reshape(-1)
    normalized = {
        name: AUDIT.normalize_gradient(value) for name, value in directions.items()
    }
    mean_permuted = torch.stack(
        [normalized[f"permuted_{index}"] for index in range(4)]
    ).mean(dim=0)
    expected_first_order = (
        radius
        * torch.dot(
            risk_gradient.double(),
            (normalized["paired"] - mean_permuted).double(),
        ).item()
    )
    assert result["first_order"]["radius_scaled_pairing_term"] == pytest.approx(
        expected_first_order
    )
    assert AUDIT.model_state_sha256(model) == model_before


class _MammothStyleAliasedMLP(torch.nn.Module):
    """Mirror the repeated module registration in pinned BaseMNISTMLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(3, 3)
        self._features = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(3, 2)
        self.net = torch.nn.Sequential(self._features, self.classifier)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(inputs))


def test_functional_branches_preserve_repeatedly_registered_mnist_modules() -> None:
    """A discarded branch must not become the aliased live MLP state."""

    torch.manual_seed(91)
    model = _MammothStyleAliasedMLP().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = _Buffer()
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    parameter_count = sum(value.numel() for value in model.parameters())
    directions = {
        name: torch.linspace(-1.0 + index, 1.0 + index, parameter_count)
        for index, name in enumerate(
            ("paired", "permuted_0", "permuted_1", "permuted_2", "permuted_3")
        )
    }
    radius = 0.05
    model_before = AUDIT.model_state_sha256(model)
    parameters = AUDIT.trainable_parameters(model)
    paired = AUDIT.branch_parameters(
        parameters, AUDIT.normalize_gradient(directions["paired"]), radius
    )
    with torch.no_grad():
        hidden = torch.nn.functional.linear(
            inputs, paired["fc1.weight"], paired["fc1.bias"]
        ).relu()
        hidden = torch.nn.functional.linear(
            hidden, paired["fc2.weight"], paired["fc2.bias"]
        ).relu()
        expected_logits = torch.nn.functional.linear(
            hidden, paired["classifier.weight"], paired["classifier.bias"]
        )
        expected_cross_entropy = float(
            torch.nn.functional.cross_entropy(expected_logits, targets).item()
        )

    with AUDIT.preserve_live_state(
        model,
        optimizer=optimizer,
        buffer=buffer,
        loaders=(loader,),
    ):
        result = AUDIT.evaluate_old_task_branches(
            model, (loader,), directions, radius=radius
        )

    assert result["branches"]["paired"]["cross_entropy"] == pytest.approx(
        expected_cross_entropy
    )
    assert AUDIT.model_state_sha256(model) == model_before
    assert model.training is True


class _LogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.anchor * 0.0


def test_class_il_confusion_partitions_every_old_example_exactly_once() -> None:
    model = _LogitModel().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    old_inputs = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [4.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ]
    )
    old_targets = torch.tensor([0, 1, 0, 1])
    old_loader = DataLoader(
        TensorDataset(old_inputs, old_targets), batch_size=2, shuffle=False
    )
    newest_loader = DataLoader(
        TensorDataset(torch.tensor([[0.0, 0.0, 1.0, 0.0]]), torch.tensor([2])),
        batch_size=1,
        shuffle=False,
    )

    with AUDIT.preserve_live_state(
        model,
        optimizer=optimizer,
        buffer=_Buffer(),
        loaders=(old_loader, newest_loader),
    ):
        result = AUDIT.final_confusion_diagnostic(
            model, (old_loader,), newest_loader, setting="class-il"
        )

    assert model.training is True
    assert result["schema_version"] == "gradient_reuse.cifar_confusion.v1"
    assert result["examples"] == 4
    assert result["accuracy"] == 25.0
    assert result["correct_old_class_fraction"] == 0.25
    assert result["other_old_class_fraction"] == 0.25
    assert result["newest_task_class_fraction"] == 0.25
    assert result["other_unseen_class_fraction"] == 0.25
    assert result["partition_residual"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["mean_correct_old_logit"] == pytest.approx(2.0)
    assert result["mean_other_old_logit"] == pytest.approx(1.0)
    assert result["mean_newest_task_logit"] == pytest.approx(1.25)
    assert result["mean_incorrect_label_logit"] == pytest.approx(14.0 / 12.0)


def test_domain_il_confusion_uses_the_prespecified_common_diagnostics() -> None:
    model = _LogitModel()
    old_loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, 0.0], [1.0, 3.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    newest_loader = DataLoader(
        TensorDataset(torch.tensor([[0.0, 1.0]]), torch.tensor([1])),
        batch_size=1,
    )
    result = AUDIT.final_confusion_diagnostic(
        model, (old_loader,), newest_loader, setting="domain-il"
    )

    assert result["schema_version"] == "gradient_reuse.pmnist_confusion.v1"
    assert result["examples"] == 2
    assert result["accuracy"] == 100.0
    assert "correct_old_class_fraction" not in result
    assert result["incorrect_prediction_label_entropy"] == 0.0


def test_state_audit_summary_rejects_unknown_jsonl_records(tmp_path: Path) -> None:
    unit = ExtensionUnit(
        experiment="E3_E6_state_matched_bridge",
        priority=2,
        benchmark="permuted_mnist",
        method="experience_replay",
        mammoth_dataset="perm-mnist",
        mammoth_model="er_state_matched_audit",
        intervention="permuted",
        seed=3401,
        permutation_replica=0,
        epochs=1,
        buffer_size=500,
        state_matched_probe=False,
        final_confusion=True,
        resource_weight=2,
    )
    audit = tmp_path / "state_matched_audit.jsonl"
    rows = [
        {"schema_version": "gradient_reuse.state_matched_effective_args.v1"},
        {"schema_version": "gradient_reuse.final_confusion_record.v1"},
        {"schema_version": "gradient_reuse.unexpected_record.v1"},
    ]
    audit.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unknown|schema|record count"):
        unit_runner._state_audit_summary(unit, audit)
