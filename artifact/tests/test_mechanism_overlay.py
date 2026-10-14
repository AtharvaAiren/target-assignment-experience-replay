"""Focused tests for replay-target interventions and the PCGrad adaptation."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "mammoth_overlay" / "models"


def _load(relative: str, module_name: str):
    path = OVERLAY / relative
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load overlay module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


INTERVENTION = _load(
    "utils/pairing_intervention.py", "gradient_reuse_pairing_intervention_test"
)
PCGRAD = _load("utils/pcgrad_replay.py", "gradient_reuse_pcgrad_replay_test")


def _args(**overrides):
    values = {
        "pairing_intervention": "permuted",
        "pairing_permutation_seed": 17,
        "pairing_replica": 2,
        "pairing_audit_path": None,
        "pairing_audit_interval": 1,
        "pairing_audit_permutations": 4,
        "pairing_benchmark": "split_cifar100",
        "pairing_run_seed": 3401,
        "dataset": "seq-cifar100",
        "seed": 3401,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_paired_intervention_is_an_identity_view() -> None:
    intervention = INTERVENTION.PairingIntervention(
        _args(pairing_intervention="paired"), "er_pairing"
    )
    targets = torch.tensor([4, 1, 3, 2])
    intervention.begin_observation()
    transformed, permutation = intervention.intervene(
        targets, channel="replay_labels"
    )
    assert torch.equal(transformed, targets)
    assert torch.equal(permutation, torch.arange(len(targets)))


def test_permuted_intervention_is_counter_keyed_and_rng_independent() -> None:
    targets = torch.tensor([10, 20, 30, 40, 50, 60])

    def run_after_noise(noise_seed: int):
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)
        _ = np.random.random(100)
        _ = torch.randn(100)
        intervention = INTERVENTION.PairingIntervention(_args(), "er_pairing")
        intervention.begin_observation()
        return intervention.intervene(targets, channel="replay_labels")

    first_targets, first_permutation = run_after_noise(3)
    second_targets, second_permutation = run_after_noise(999)
    assert torch.equal(first_permutation, second_permutation)
    assert torch.equal(first_targets, second_targets)
    assert sorted(first_targets.tolist()) == sorted(targets.tolist())
    assert not torch.equal(first_permutation, torch.arange(len(targets)))


def test_stratified_permutation_never_crosses_gem_task_boundaries() -> None:
    intervention = INTERVENTION.PairingIntervention(_args(), "gem_pairing")
    strata = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    intervention.begin_observation()
    permutation = intervention.permutation(
        len(strata), channel="constraint", strata=strata
    )
    assert torch.equal(strata[permutation], strata)


def test_intervention_record_preserves_multiset_and_current_ids(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    intervention = INTERVENTION.PairingIntervention(
        _args(pairing_audit_path=str(path)), "der_pairing"
    )
    logits = torch.tensor(
        [[1.0, -1.0], [0.2, 0.8], [-0.4, 0.5], [2.0, 1.0]]
    )
    indexes = torch.tensor([19, 3, 8, 11])
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    intervention.begin_observation()
    transformed, permutation = intervention.intervene(
        logits, channel="dark_logits"
    )
    intervention.record_intervention(
        logits,
        transformed,
        permutation,
        channel="dark_logits",
        sampled_indexes=indexes,
        sampled_inputs=inputs,
    )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    record = next(
        row
        for row in records
        if row["schema_version"] == "gradient_reuse.mammoth_intervention.v1"
    )
    assert record["channel"] == "dark_logits"
    assert record["rows"] == len(logits)
    assert record["target_multiset_sha256"]
    assert record["sampled_indexes_sha256"]
    assert record["input_sha256"]
    assert record["benchmark"] == "split_cifar100"
    assert record["run_seed"] == 3401


def test_gradient_diagnostic_restores_rng_state_buffers_and_mode() -> None:
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.register_buffer("counter", torch.tensor([7.0]))

        def forward(self, inputs):
            return self.linear(inputs)

    torch.manual_seed(31)
    model = Predictor().train()
    inputs = torch.randn(5, 3)
    targets = torch.tensor([0, 1, 0, 1, 1])
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    buffers_before = {
        name: value.detach().clone() for name, value in model.named_buffers()
    }
    rng_before = torch.get_rng_state().clone()
    intervention = INTERVENTION.PairingIntervention(_args(), "er_pairing")
    intervention.begin_observation()
    record = intervention.gradient_diagnostic(
        model,
        inputs,
        targets,
        INTERVENTION.cross_entropy_objective,
        channel="replay_labels",
    )
    assert record is not None
    assert record["permutation_draws"] == 4
    assert record["finite_permutation_variance"] >= 0.0
    assert record["pairing_component_norm"] >= 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is True
    for name, value in model.named_parameters():
        assert torch.equal(value.detach(), parameters_before[name])
        assert value.grad is None
    for name, value in model.named_buffers():
        assert torch.equal(value.detach(), buffers_before[name])


def test_pcgrad_nonconflicting_pair_is_unmodified_and_averaged() -> None:
    current = torch.tensor([1.0, 2.0, -1.0])
    replay = torch.tensor([2.0, 1.0, 0.5])
    result = PCGRAD.pcgrad_replay_merge(current, replay)
    assert result.conflict is False
    torch.testing.assert_close(result.current_projected, current)
    torch.testing.assert_close(result.replay_projected, replay)
    torch.testing.assert_close(result.merged, 0.5 * (current + replay))
    assert result.pre_dot >= 0.0
    assert result.post_dot == pytest.approx(result.pre_dot)


def test_pcgrad_conflict_is_removed_symmetrically() -> None:
    current = torch.tensor([1.0, 0.0])
    replay = torch.tensor([-1.0, 1.0])
    current_before = current.clone()
    replay_before = replay.clone()
    result = PCGRAD.pcgrad_replay_merge(current, replay)
    assert result.conflict is True
    assert result.pre_dot < 0.0
    assert result.post_dot >= -1.0e-12
    assert torch.dot(result.current_projected, replay_before).item() == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert torch.dot(result.replay_projected, current_before).item() == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert torch.equal(current, current_before)
    assert torch.equal(replay, replay_before)


@pytest.mark.parametrize(
    "current,replay",
    [
        (torch.ones(2, 2), torch.ones(4)),
        (torch.ones(3), torch.ones(4)),
        (torch.tensor([1, 2]), torch.tensor([3, 4])),
        (torch.tensor([float("nan")]), torch.tensor([1.0])),
    ],
)
def test_pcgrad_rejects_invalid_gradient_pairs(current, replay) -> None:
    with pytest.raises(ValueError):
        PCGRAD.pcgrad_replay_merge(current, replay)


def test_pcgrad_overlay_is_explicitly_an_adaptation_and_never_permutates_current_labels() -> None:
    source = (OVERLAY / "pcgrad_replay_pairing.py").read_text(encoding="utf-8")
    assert 'NAME = "pcgrad_replay_pairing"' in source
    assert "PCGrad replay adaptation" in source
    assert source.count("self.pairing.intervene(") == 1
    assert 'replay_labels, channel="replay_labels"' in source
    assert "labels=labels" in source


def test_mer_current_only_matches_the_no_replay_meta_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContinualModel(torch.nn.Module):
        def __init__(self, backbone, loss, args, transform, dataset=None):
            super().__init__()
            self.net = backbone
            self.loss = loss
            self.args = args
            self.transform = transform
            self.device = next(backbone.parameters()).device
            self.opt = torch.optim.SGD(self.net.parameters(), lr=float(args.lr))

    class FlatLinear(torch.nn.Linear):
        def get_params(self):
            return torch.cat([parameter.reshape(-1) for parameter in self.parameters()])

        def set_params(self, values):
            cursor = 0
            with torch.no_grad():
                for parameter in self.parameters():
                    count = parameter.numel()
                    parameter.copy_(values[cursor : cursor + count].reshape_as(parameter))
                    cursor += count

    modules = {
        "models": ModuleType("models"),
        "models.utils": ModuleType("models.utils"),
        "models.utils.continual_model": ModuleType("models.utils.continual_model"),
        "utils": ModuleType("utils"),
        "utils.args": ModuleType("utils.args"),
    }
    modules["models.utils.continual_model"].ContinualModel = ContinualModel
    modules["utils.args"].ArgumentParser = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    wrapper = _load("mer_current_only.py", "gradient_reuse_mer_control_test")

    torch.manual_seed(29)
    backbone = FlatLinear(2, 2, bias=False)
    before = backbone.weight.detach().clone()
    inputs = torch.tensor([[1.0, -0.5]])
    labels = torch.tensor([1])
    args = SimpleNamespace(
        batch_size=1,
        batch_num=1,
        beta=0.2,
        gamma=0.3,
        lr=0.1,
    )
    reference = FlatLinear(2, 2, bias=False)
    reference.weight.data.copy_(before)
    loss = torch.nn.functional.cross_entropy(reference(inputs), labels)
    gradient = torch.autograd.grad(loss, reference.weight)[0]
    expected = before - args.lr * args.beta * args.gamma * gradient

    model = wrapper.MerCurrentOnly(
        backbone,
        torch.nn.functional.cross_entropy,
        args,
        transform=lambda value: value,
    )
    model.observe(inputs, labels, inputs.clone())
    torch.testing.assert_close(model.net.weight.detach(), expected)
    source = (OVERLAY / "mer_current_only.py").read_text(encoding="utf-8")
    assert "Buffer" not in source
    assert "pairing" not in source.lower()


def test_gem_domain_buffer_is_task_balanced_across_two_permuted_mnist_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContinualModel(torch.nn.Module):
        pass

    class PlaceholderBuffer:
        pass

    class FakeBuffer:
        def __init__(self, capacity):
            self.buffer_size = int(capacity)
            self.num_seen_examples = 0

        def __len__(self):
            return self.num_seen_examples

        def is_empty(self):
            return self.num_seen_examples == 0

        def empty(self):
            for name in ("examples", "labels", "task_labels"):
                if hasattr(self, name):
                    delattr(self, name)
            self.num_seen_examples = 0

        def add_data(self, *, examples, labels, task_labels):
            for name, value in (
                ("examples", examples),
                ("labels", labels),
                ("task_labels", task_labels),
            ):
                old = getattr(self, name, None)
                setattr(self, name, value.clone() if old is None else torch.cat((old, value)))
            self.num_seen_examples += len(examples)

        def get_all_data(self, device=None):
            return self.examples, self.labels, self.task_labels

    class Dataset:
        SETTING = "domain-il"

        def __init__(self, offset):
            self.args = SimpleNamespace(seed=3401)
            inputs = torch.arange(6, dtype=torch.float32).reshape(6, 1) + offset
            labels = torch.tensor([0, 1, 0, 1, 0, 1])
            self.train_loader = [(inputs.clone(), labels, inputs.clone())]

    calls = []
    modules = {
        "models": ModuleType("models"),
        "models.gem": ModuleType("models.gem"),
        "models.utils": ModuleType("models.utils"),
        "models.utils.continual_model": ModuleType("models.utils.continual_model"),
        "utils": ModuleType("utils"),
        "utils.args": ModuleType("utils.args"),
        "utils.buffer": ModuleType("utils.buffer"),
        "utils.conf": ModuleType("utils.conf"),
    }
    modules["models.gem"].overwrite_grad = lambda *args, **kwargs: None
    modules["models.gem"].project2cone2 = lambda *args, **kwargs: None
    modules["models.gem"].store_grad = lambda *args, **kwargs: None
    modules["models.utils.continual_model"].ContinualModel = ContinualModel
    modules["utils.args"].ArgumentParser = object
    modules["utils.args"].add_rehearsal_args = lambda parser: parser
    modules["utils.buffer"].Buffer = PlaceholderBuffer
    modules["utils.buffer"].fill_buffer = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    )
    modules["utils.conf"].warn_once = lambda *args, **kwargs: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, "models.utils.pairing_intervention", INTERVENTION
    )
    wrapper = _load("gem_pairing.py", "gradient_reuse_gem_domain_buffer_test")

    buffer = FakeBuffer(4)
    wrapper.fill_domain_task_balanced_buffer(buffer, Dataset(0.0), 0)
    assert torch.equal(buffer.task_labels, torch.zeros(4, dtype=torch.long))
    first_task_examples = buffer.examples.clone()
    wrapper.fill_domain_task_balanced_buffer(buffer, Dataset(100.0), 1)
    tasks, counts = torch.unique(buffer.task_labels, return_counts=True)
    assert dict(zip(tasks.tolist(), counts.tolist())) == {0: 2, 1: 2}
    assert torch.equal(buffer.examples[buffer.task_labels == 0], first_task_examples[:2])
    assert torch.all(buffer.examples[buffer.task_labels == 1] >= 100.0)

    class ClassDataset:
        SETTING = "class-il"

    fake_model = SimpleNamespace(
        grads_cs=[],
        grad_dims=[1],
        device="cpu",
        buffer=buffer,
        current_task=1,
    )
    wrapper.GemPairing.end_task(fake_model, ClassDataset())
    assert len(calls) == 1
    assert calls[0][1]["required_attributes"] == [
        "examples",
        "labels",
        "task_labels",
    ]


def test_domain_il_sitecustomize_patch_retains_and_validates_full_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Logger:
        def __init__(self, setting):
            self.setting = setting
            self.fullaccs = []
            self.delegated = []

        def log_fullacc(self, accs):
            self.delegated.append(accs)

    utils = ModuleType("utils")
    loggers = ModuleType("utils.loggers")
    loggers.Logger = Logger
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.loggers", loggers)
    monkeypatch.setenv("GRADIENT_REUSE_DETERMINISTIC", "1")
    monkeypatch.setenv("GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH", "v1")
    _load("../sitecustomize.py", "gradient_reuse_sitecustomize_test")

    domain = Logger("domain-il")
    domain.log_fullacc(([61.0], []))
    domain.log_fullacc(([60.0, 55.0], []))
    assert domain.fullaccs == [[61.0], [60.0, 55.0]]
    with pytest.raises(RuntimeError, match="expected 3"):
        domain.log_fullacc(([50.0], []))

    class_il = Logger("class-il")
    payload = ([70.0], [80.0])
    class_il.log_fullacc(payload)
    assert class_il.delegated == [payload]
    assert Logger._gradient_reuse_domain_il_fullacc_patch == "v1"


def test_pcgrad_overlay_observe_matches_the_two_objective_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the Mammoth-facing wrapper against a small API-compatible stub."""

    class ContinualModel(torch.nn.Module):
        def __init__(self, backbone, loss, args, transform, dataset=None):
            super().__init__()
            self.net = backbone
            self.loss = loss
            self.args = args
            self.transform = transform
            self.device = next(backbone.parameters()).device
            self.opt = torch.optim.SGD(self.net.parameters(), lr=float(args.lr))

    def store_grad(parameters, destination, dimensions):
        cursor = 0
        for parameter, count in zip(parameters(), dimensions):
            gradient = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad
            )
            destination[cursor : cursor + count].copy_(gradient.reshape(-1))
            cursor += count

    def overwrite_grad(parameters, source, dimensions):
        cursor = 0
        for parameter, count in zip(parameters(), dimensions):
            parameter.grad = source[cursor : cursor + count].reshape_as(parameter).clone()
            cursor += count

    class EmptyBuffer:
        def __init__(self, capacity):
            self.capacity = int(capacity)

        def is_empty(self):
            return True

    modules = {
        "models": ModuleType("models"),
        "models.gem": ModuleType("models.gem"),
        "models.utils": ModuleType("models.utils"),
        "models.utils.continual_model": ModuleType("models.utils.continual_model"),
        "utils": ModuleType("utils"),
        "utils.args": ModuleType("utils.args"),
        "utils.buffer": ModuleType("utils.buffer"),
    }
    modules["models.gem"].store_grad = store_grad
    modules["models.gem"].overwrite_grad = overwrite_grad
    modules["models.utils.continual_model"].ContinualModel = ContinualModel
    modules["utils.args"].ArgumentParser = object
    modules["utils.args"].add_rehearsal_args = lambda parser: parser
    modules["utils.buffer"].Buffer = EmptyBuffer
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, "models.utils.pairing_intervention", INTERVENTION
    )
    monkeypatch.setitem(sys.modules, "models.utils.pcgrad_replay", PCGRAD)
    wrapper = _load(
        "pcgrad_replay_pairing.py", "gradient_reuse_pcgrad_wrapper_test"
    )

    class FixedReplayBuffer:
        def __init__(self, replay_inputs, replay_labels):
            self.replay_inputs = replay_inputs
            self.replay_labels = replay_labels
            self.added = None

        def is_empty(self):
            return False

        def get_data(self, size, **kwargs):
            assert int(size) == len(self.replay_labels)
            return (
                torch.arange(len(self.replay_labels)),
                self.replay_inputs,
                self.replay_labels,
            )

        def add_data(self, **kwargs):
            self.added = kwargs

    torch.manual_seed(43)
    backbone = torch.nn.Linear(2, 2, bias=False)
    before = copy.deepcopy(backbone)
    args = _args(
        pairing_intervention="paired",
        buffer_size=4,
        minibatch_size=4,
        lr=0.2,
    )
    model = wrapper.PCGradReplayPairing(
        backbone,
        torch.nn.functional.cross_entropy,
        args,
        transform=lambda value: value,
    )
    current_inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    )
    current_labels = torch.tensor([0, 1, 0, 1])
    replay_inputs = torch.tensor(
        [[-0.5, 1.0], [1.5, -1.0], [0.2, 0.8], [-1.0, -1.0]]
    )
    replay_labels = torch.tensor([0, 0, 1, 1])
    fixed_buffer = FixedReplayBuffer(replay_inputs, replay_labels)
    model.buffer = fixed_buffer

    current_loss = torch.nn.functional.cross_entropy(
        before(current_inputs), current_labels
    )
    current_gradient = torch.autograd.grad(
        current_loss, tuple(before.parameters()), retain_graph=False
    )[0].reshape(-1)
    replay_loss = torch.nn.functional.cross_entropy(
        before(replay_inputs), replay_labels
    )
    replay_gradient = torch.autograd.grad(
        replay_loss, tuple(before.parameters()), retain_graph=False
    )[0].reshape(-1)
    expected_merge = PCGRAD.pcgrad_replay_merge(
        current_gradient, replay_gradient
    ).merged.reshape_as(before.weight)
    expected_weight = before.weight.detach() - args.lr * expected_merge

    reported = model.observe(
        current_inputs,
        current_labels,
        current_inputs.clone(),
    )
    torch.testing.assert_close(model.net.weight.detach(), expected_weight)
    assert reported == pytest.approx(
        0.5 * (float(current_loss.item()) + float(replay_loss.item()))
    )
    assert fixed_buffer.added is not None
    assert torch.equal(fixed_buffer.added["labels"], current_labels)
    assert torch.equal(fixed_buffer.added["examples"], current_inputs)
