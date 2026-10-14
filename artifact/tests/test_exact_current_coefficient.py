"""Focused tests for the exact current-loss-coefficient control."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
UTIL_PATH = (
    ROOT
    / "exact_reference_overlay"
    / "models"
    / "utils"
    / "exact_current_coefficient.py"
)
MODEL_PATH = (
    ROOT
    / "exact_reference_overlay"
    / "models"
    / "sgd_exact_current_coefficient.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


UTIL = _load(UTIL_PATH, "test_exact_current_coefficient_util")


def _args(path: Path, **overrides):
    values = {
        "reference_buffer_size": 500,
        "reference_minibatch_size": 32,
        "exact_coefficient_audit_path": str(path),
        "dataset": "perm-mnist",
        "seed": 4101,
        "lr": 0.1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _expected_header(**overrides):
    values = {
        "model": "sgd_exact_current_coefficient",
        "dataset": "perm-mnist",
        "seed": 4101,
        "reference_buffer_size": 500,
        "reference_minibatch_size": 32,
        "learning_rate": 0.1,
    }
    values.update(overrides)
    return values


def test_empty_partial_and_full_shadow_buffer_coefficients() -> None:
    empty = UTIL.coefficient_plan(
        update=0,
        b_current=32,
        seen_examples=0,
        reference_buffer_size=500,
        reference_minibatch_size=32,
    )
    partial = UTIL.coefficient_plan(
        update=1,
        b_current=32,
        seen_examples=9,
        reference_buffer_size=500,
        reference_minibatch_size=32,
    )
    full_with_partial_current = UTIL.coefficient_plan(
        update=20,
        b_current=17,
        seen_examples=700,
        reference_buffer_size=500,
        reference_minibatch_size=32,
    )

    assert (empty.b_replay, empty.coefficient) == (0, 1.0)
    assert partial.buffer_occupancy_before == 9
    assert partial.b_replay == 9
    assert partial.coefficient == pytest.approx(32 / 41)
    assert full_with_partial_current.buffer_occupancy_before == 500
    assert full_with_partial_current.b_replay == 32
    assert full_with_partial_current.coefficient == pytest.approx(17 / 49)


def test_audit_tracks_shadow_occupancy_and_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "coefficient.jsonl"
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    audit = UTIL.ExactCurrentCoefficientAudit(
        _args(path),
        "sgd_exact_current_coefficient",
        optimizer_contract=UTIL.assert_plain_sgd(optimizer),
        objective_contract=UTIL.assert_mean_cross_entropy_objective(
            torch.nn.functional.cross_entropy
        ),
    )
    first = audit.plan(9)
    audit.complete(first, current_mean_loss=2.0, scaled_loss=2.0)
    second = audit.plan(4)
    audit.complete(second, current_mean_loss=1.3, scaled_loss=0.4)
    audit.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["schema_version"] for row in rows] == [
        UTIL.HEADER_SCHEMA,
        UTIL.UPDATE_SCHEMA,
        UTIL.UPDATE_SCHEMA,
    ]
    assert rows[1]["b_replay"] == 0
    assert rows[1]["seen_examples_after"] == 9
    assert rows[2]["b_current"] == 4
    assert rows[2]["b_replay"] == 9
    assert rows[2]["coefficient_numerator"] == 4
    assert rows[2]["coefficient_denominator"] == 13
    assert rows[2]["current_loss_coefficient"] == pytest.approx(4 / 13)
    assert rows[2]["previous_record_sha256"] == rows[1]["record_sha256"]
    summary = UTIL.summarize_exact_coefficient_audit(
        path,
        expected_header=_expected_header(),
        expected_updates=2,
        expected_seen_examples=13,
    )
    assert summary["status"] == "PASS"
    assert summary["updates"] == 2
    assert summary["seen_examples"] == 13
    assert summary["total_reference_replay_rows"] == 9
    assert summary["final_record_sha256"] == rows[-1]["record_sha256"]
    with pytest.raises(RuntimeError, match="header does not match"):
        UTIL.summarize_exact_coefficient_audit(
            path,
            expected_header=_expected_header(learning_rate=0.2),
        )
    with pytest.raises(RuntimeError, match="final example count"):
        UTIL.summarize_exact_coefficient_audit(
            path,
            expected_header=_expected_header(),
            expected_seen_examples=12,
        )


def test_plain_sgd_contract_rejects_optimizer_state_changes() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="torch.optim.SGD"):
        UTIL.assert_plain_sgd(torch.optim.Adam([parameter], lr=0.1))
    with pytest.raises(RuntimeError, match="momentum"):
        UTIL.assert_plain_sgd(
            torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        )
    with pytest.raises(RuntimeError, match="weight_decay"):
        UTIL.assert_plain_sgd(
            torch.optim.SGD([parameter], lr=0.1, weight_decay=0.01)
        )
    with pytest.raises(RuntimeError, match="foreach"):
        UTIL.assert_plain_sgd(
            torch.optim.SGD([parameter], lr=0.1, foreach=True)
        )


def test_mean_cross_entropy_contract_rejects_noncanonical_objectives() -> None:
    assert UTIL.assert_mean_cross_entropy_objective(
        torch.nn.functional.cross_entropy
    )["reduction"] == "mean"
    assert UTIL.assert_mean_cross_entropy_objective(
        torch.nn.CrossEntropyLoss()
    )["reduction"] == "mean"
    with pytest.raises(RuntimeError, match="mean cross-entropy"):
        UTIL.assert_mean_cross_entropy_objective(
            torch.nn.CrossEntropyLoss(reduction="sum")
        )
    with pytest.raises(RuntimeError, match="canonical mean"):
        UTIL.assert_mean_cross_entropy_objective(torch.nn.functional.mse_loss)


def _load_model_with_mammoth_stubs(monkeypatch: pytest.MonkeyPatch):
    models = ModuleType("models")
    models.__path__ = []
    models_utils = ModuleType("models.utils")
    models_utils.__path__ = []
    continual = ModuleType("models.utils.continual_model")
    utils = ModuleType("utils")
    utils.__path__ = []
    args_module = ModuleType("utils.args")

    class ContinualModel:
        def __init__(self, backbone, loss, args, transform, dataset=None):
            self.net = backbone
            self.loss = loss
            self.args = args
            self.transform = transform
            self.dataset = dataset
            self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

    class ArgumentParser:
        pass

    continual.ContinualModel = ContinualModel
    args_module.ArgumentParser = ArgumentParser
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "models.utils", models_utils)
    monkeypatch.setitem(sys.modules, "models.utils.continual_model", continual)
    monkeypatch.setitem(
        sys.modules, "models.utils.exact_current_coefficient", UTIL
    )
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.args", args_module)
    return _load(MODEL_PATH, "test_sgd_exact_current_coefficient_model")


def test_model_parameter_update_uses_exact_partial_batch_coefficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_model_with_mammoth_stubs(monkeypatch)
    torch.manual_seed(19)
    network = torch.nn.Linear(3, 3)
    model = module.SgdExactCurrentCoefficient(
        network,
        torch.nn.CrossEntropyLoss(),
        _args(tmp_path / "updates.jsonl", lr=0.07),
        transform=None,
    )

    warm_inputs = torch.randn(9, 3)
    warm_labels = torch.arange(9) % 3
    model.observe(warm_inputs, warm_labels, warm_inputs)

    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    weight_before = network.weight.detach().clone()
    bias_before = network.bias.detach().clone()
    logits = network(inputs)
    mean_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    expected_gradients = torch.autograd.grad(
        (4 / 13) * mean_loss, tuple(network.parameters())
    )

    observed_loss = model.observe(inputs, labels, inputs)
    assert observed_loss == pytest.approx(float(((4 / 13) * mean_loss).item()))
    assert torch.allclose(
        network.weight,
        weight_before - 0.07 * expected_gradients[0],
        rtol=0,
        atol=2e-7,
    )
    assert torch.allclose(
        network.bias,
        bias_before - 0.07 * expected_gradients[1],
        rtol=0,
        atol=2e-7,
    )
    assert model.exact_coefficient.seen_examples == 13
    model.exact_coefficient.close()


def test_model_reasserts_optimizer_contract_on_every_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_model_with_mammoth_stubs(monkeypatch)
    network = torch.nn.Linear(3, 3)
    model = module.SgdExactCurrentCoefficient(
        network,
        torch.nn.CrossEntropyLoss(),
        _args(tmp_path / "optimizer-drift.jsonl", lr=0.07),
        transform=None,
    )
    # Simulate a task-boundary optimizer rebuild by the upstream trainer.
    model.opt = torch.optim.SGD(network.parameters(), lr=0.07, momentum=0.9)
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    with pytest.raises(RuntimeError, match="momentum"):
        model.observe(inputs, labels, inputs)
    model.exact_coefficient.close()


def test_audit_refuses_relative_or_existing_paths(tmp_path: Path) -> None:
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    contract = UTIL.assert_plain_sgd(optimizer)
    with pytest.raises(ValueError, match="absolute"):
        UTIL.ExactCurrentCoefficientAudit(
            _args(Path("relative.jsonl")),
            "sgd_exact_current_coefficient",
            optimizer_contract=contract,
            objective_contract=UTIL.assert_mean_cross_entropy_objective(
                torch.nn.functional.cross_entropy
            ),
        )
    existing = tmp_path / "existing.jsonl"
    existing.write_text("do not append\n")
    with pytest.raises(FileExistsError):
        UTIL.ExactCurrentCoefficientAudit(
            _args(existing),
            "sgd_exact_current_coefficient",
            optimizer_contract=contract,
            objective_contract=UTIL.assert_mean_cross_entropy_objective(
                torch.nn.functional.cross_entropy
            ),
        )


def _rehash_rows(rows: list[dict]) -> list[str]:
    previous = None
    rendered = []
    for row in rows:
        payload = {
            key: value
            for key, value in row.items()
            if key not in {"record_sha256", "previous_record_sha256"}
        }
        payload["previous_record_sha256"] = previous
        digest = UTIL._digest(payload)
        complete = {**payload, "record_sha256": digest}
        rendered.append(UTIL._canonical_json(complete))
        previous = digest
    return rendered


def test_validator_rejects_digest_and_even_rehashed_arithmetic_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    audit = UTIL.ExactCurrentCoefficientAudit(
        _args(path),
        "sgd_exact_current_coefficient",
        optimizer_contract=UTIL.assert_plain_sgd(optimizer),
        objective_contract=UTIL.assert_mean_cross_entropy_objective(
            torch.nn.functional.cross_entropy
        ),
    )
    plan = audit.plan(8)
    audit.complete(plan, current_mean_loss=1.0, scaled_loss=1.0)
    plan = audit.plan(3)
    audit.complete(plan, current_mean_loss=1.0, scaled_loss=3 / 11)
    audit.close()
    original = path.read_text()

    rows = [json.loads(line) for line in original.splitlines()]
    rows[-1]["record_sha256"] = "0" * 64
    path.write_text("\n".join(UTIL._canonical_json(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="record digest"):
        UTIL.summarize_exact_coefficient_audit(
            path,
            expected_header=_expected_header(),
        )

    rows = [json.loads(line) for line in original.splitlines()]
    rows[-1]["b_replay"] = 7
    path.write_text("\n".join(_rehash_rows(rows)) + "\n")
    with pytest.raises(RuntimeError, match="b_replay arithmetic"):
        UTIL.summarize_exact_coefficient_audit(
            path,
            expected_header=_expected_header(),
        )


def test_validator_rejects_header_only_audit(tmp_path: Path) -> None:
    path = tmp_path / "header-only.jsonl"
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    audit = UTIL.ExactCurrentCoefficientAudit(
        _args(path),
        "sgd_exact_current_coefficient",
        optimizer_contract=UTIL.assert_plain_sgd(optimizer),
        objective_contract=UTIL.assert_mean_cross_entropy_objective(
            torch.nn.functional.cross_entropy
        ),
    )
    audit.close()
    with pytest.raises(RuntimeError, match="no optimizer updates"):
        UTIL.summarize_exact_coefficient_audit(
            path,
            expected_header=_expected_header(),
        )
