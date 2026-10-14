from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

from experiments.gradient_reuse_mechanisms.acquire_public import (
    RECEIPT_NAME,
    acquire_public_data,
)


class _FakeDataset:
    rows = {"CIFAR100": (50_000, 10_000, 100), "MNIST": (60_000, 10_000, 10)}

    def __init__(self, root: str, train: bool, download: bool) -> None:
        assert download is True
        location = Path(root)
        location.mkdir(parents=True, exist_ok=True)
        name = location.name
        train_rows, test_rows, classes = self.rows[name]
        self._length = train_rows if train else test_rows
        self.classes = list(range(classes))
        (location / ("train.bin" if train else "test.bin")).write_bytes(
            f"{name}:{train}".encode("ascii")
        )

    def __len__(self) -> int:
        return self._length


def _install_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = types.ModuleType("torchvision.datasets")
    datasets.CIFAR100 = _FakeDataset
    datasets.MNIST = _FakeDataset
    package = types.ModuleType("torchvision")
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "torchvision", package)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets)
    monkeypatch.setattr(
        "experiments.gradient_reuse_mechanisms.acquire_public.importlib.metadata.version",
        lambda name: "test-version" if name == "torchvision" else "unexpected",
    )


def test_public_acquisition_is_deterministic_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_torchvision(monkeypatch)
    path = acquire_public_data(tmp_path)
    assert path == tmp_path / RECEIPT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["datasets"]["cifar100"]["train_rows"] == 50_000
    assert payload["datasets"]["mnist"]["test_rows"] == 10_000
    assert acquire_public_data(tmp_path) == path


def test_public_acquisition_refuses_changed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_torchvision(monkeypatch)
    path = acquire_public_data(tmp_path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        acquire_public_data(tmp_path)
