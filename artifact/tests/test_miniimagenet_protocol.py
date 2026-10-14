"""Data-integrity tests for the local-only Split-miniImageNet protocol."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image
import pytest
import torch
import torchvision.transforms as transforms

from experiments.gradient_reuse_mechanisms import miniimagenet as mini


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SOURCE = ROOT / "mammoth_overlay" / "datasets" / "seq_miniimg.py"
OVERLAY_CONFIG = (
    ROOT
    / "mammoth_overlay"
    / "datasets"
    / "configs"
    / "seq-miniimg"
    / "20tasks.yaml"
)


SMALL_PROTOCOL = mini.MiniImageNetProtocol(
    source_train_classes=2,
    source_val_classes=1,
    source_test_classes=1,
    source_examples_per_class=3,
    train_examples_per_class=2,
    test_examples_per_class=1,
)


def _write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (84, 84), color=color).save(path, format="JPEG", quality=91)


def _make_small_source(root: Path) -> dict[str, list[tuple[str, str]]]:
    (root / "splits").mkdir(parents=True)
    (root / "images").mkdir()
    split_labels = {
        "train": ["n00000003", "n00000001"],
        "val": ["n00000004"],
        "test": ["n00000002"],
    }
    rows: dict[str, list[tuple[str, str]]] = {}
    for split, labels in split_labels.items():
        current: list[tuple[str, str]] = []
        for label in labels:
            for index in (2, 0, 1):
                filename = f"{label}{index:08d}.jpg"
                current.append((filename, label))
                value = int(label[-2:]) * 17 + index
                _write_jpeg(
                    root / "images" / filename,
                    (value % 256, (value * 3) % 256, (value * 7) % 256),
                )
        rows[split] = current
        with (root / "splits" / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("filename", "label"))
            writer.writerows(current)
    return rows


def _small_manifest(root: Path) -> dict:
    return mini._build_manifest(
        root,
        root / "splits",
        root / "images",
        protocol=SMALL_PROTOCOL,
    )


def _load_overlay(monkeypatch: pytest.MonkeyPatch):
    datasets = ModuleType("datasets")
    dataset_transforms = ModuleType("datasets.transforms")
    denormalization = ModuleType("datasets.transforms.denormalization")
    dataset_utils = ModuleType("datasets.utils")
    continual = ModuleType("datasets.utils.continual_dataset")
    utilities = ModuleType("utils")
    configuration = ModuleType("utils.conf")

    class ContinualDataset:
        class_names = None

    class DeNormalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    def set_default_from_args(name):
        return lambda function: function

    continual.ContinualDataset = ContinualDataset
    continual.fix_class_names_order = lambda names, args: names
    continual.store_masked_loaders = lambda train, test, setting: (train, test)
    denormalization.DeNormalize = DeNormalize
    dataset_utils.set_default_from_args = set_default_from_args
    configuration.base_path = lambda: "./data/"
    for name, module in {
        "datasets": datasets,
        "datasets.transforms": dataset_transforms,
        "datasets.transforms.denormalization": denormalization,
        "datasets.utils": dataset_utils,
        "datasets.utils.continual_dataset": continual,
        "utils": utilities,
        "utils.conf": configuration,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    name = "gradient_reuse_seq_miniimg_test"
    specification = importlib.util.spec_from_file_location(name, OVERLAY_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load overlay source: {OVERLAY_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, module)
    specification.loader.exec_module(module)
    return module


def test_canonical_profile_is_exactly_100_classes_with_500_100_partition() -> None:
    protocol = mini.CANONICAL_PROTOCOL
    assert protocol.source_split_classes == {"train": 64, "val": 16, "test": 20}
    assert protocol.classes == 100
    assert protocol.source_examples_per_class == 600
    assert protocol.train_examples_per_class == 500
    assert protocol.test_examples_per_class == 100
    assert protocol.train_samples == 50_000
    assert protocol.test_samples == 10_000
    assert (protocol.image_width, protocol.image_height, protocol.image_mode) == (
        84,
        84,
        "RGB",
    )


def test_miniimagenet_manifest_parser_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"LOCKED","status":"CHANGED"}\n', encoding="utf-8")
    with pytest.raises(mini.MiniImageNetIntegrityError, match="strict JSON"):
        mini._strict_json(path)


def test_small_profile_manifest_is_stable_complete_and_disjoint(tmp_path: Path) -> None:
    _make_small_source(tmp_path)
    first = _small_manifest(tmp_path)
    second = _small_manifest(tmp_path)
    assert first == second
    assert first["status"] == "TEST_ONLY"
    assert first["classes"] == 4
    assert first["train_samples"] == 8
    assert first["test_samples"] == 4
    assert [row["label"] for row in first["class_records"]] == [
        "n00000001",
        "n00000002",
        "n00000003",
        "n00000004",
    ]
    train = first["partitions"]["train"]
    test = first["partitions"]["test"]
    assert {row["path"] for row in train}.isdisjoint(
        {row["path"] for row in test}
    )
    assert {row["target"] for row in train} == set(range(4))
    assert {row["target"] for row in test} == set(range(4))
    for target in range(4):
        train_names = [row["path"] for row in train if row["target"] == target]
        test_names = [row["path"] for row in test if row["target"] == target]
        assert len(train_names) == 2
        assert len(test_names) == 1
        assert train_names + test_names == sorted(train_names + test_names)
    assert str(tmp_path.resolve()) not in json.dumps(first)
    mini.validate_manifest_payload(
        first, protocol=SMALL_PROTOCOL, require_locked=False
    )


def test_manifest_hashes_detect_any_partition_edit(tmp_path: Path) -> None:
    _make_small_source(tmp_path)
    payload = _small_manifest(tmp_path)
    changed = copy.deepcopy(payload)
    changed["partitions"]["train"][0]["target"] = 1
    with pytest.raises(mini.MiniImageNetIntegrityError, match="content hash"):
        mini.validate_manifest_payload(
            changed, protocol=SMALL_PROTOCOL, require_locked=False
        )


def test_manifest_validator_rejects_overlap_even_with_recomputed_hashes(
    tmp_path: Path,
) -> None:
    _make_small_source(tmp_path)
    changed = _small_manifest(tmp_path)
    # Preserve the affected class count while duplicating a train image into
    # test, then recompute the public hashes.  Structural validation must still
    # reject the leakage rather than relying only on stale hashes.
    target = changed["partitions"]["test"][0]["target"]
    replacement = next(
        row for row in changed["partitions"]["train"] if row["target"] == target
    )
    changed["partitions"]["test"][0] = copy.deepcopy(replacement)
    changed["partition_bundle_sha256"] = mini._canonical_sha256(
        changed["partitions"]
    )
    without_content = dict(changed)
    without_content.pop("manifest_content_sha256")
    changed["manifest_content_sha256"] = mini._canonical_sha256(without_content)
    with pytest.raises(mini.MiniImageNetIntegrityError, match="duplicate paths"):
        mini.validate_manifest_payload(
            changed, protocol=SMALL_PROTOCOL, require_locked=False
        )


def test_preparation_rejects_classes_shared_across_source_splits(
    tmp_path: Path,
) -> None:
    rows = _make_small_source(tmp_path)
    shared_label = rows["train"][0][1]
    replacement_rows: list[tuple[str, str]] = []
    for index in range(3, 6):
        filename = f"{shared_label}{index:08d}.jpg"
        replacement_rows.append((filename, shared_label))
        _write_jpeg(tmp_path / "images" / filename, (index, index, index))
    with (tmp_path / "splits" / "val.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("filename", "label"))
        writer.writerows(replacement_rows)
    with pytest.raises(mini.MiniImageNetIntegrityError, match="occurs in both"):
        _small_manifest(tmp_path)


def test_preparation_rejects_incomplete_class_coverage(tmp_path: Path) -> None:
    rows = _make_small_source(tmp_path)
    with (tmp_path / "splits" / "train.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("filename", "label"))
        writer.writerows(rows["train"][:-1])
    with pytest.raises(mini.MiniImageNetIntegrityError, match="source images"):
        _small_manifest(tmp_path)


@pytest.mark.parametrize(
    "unsafe",
    ["../n0000000100000000.jpg", "/tmp/n0000000100000000.jpg", "a\\b.jpg"],
)
def test_preparation_rejects_unsafe_csv_paths(tmp_path: Path, unsafe: str) -> None:
    rows = _make_small_source(tmp_path)
    rows["train"][0] = (unsafe, rows["train"][0][1])
    with (tmp_path / "splits" / "train.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("filename", "label"))
        writer.writerows(rows["train"])
    with pytest.raises(mini.MiniImageNetIntegrityError, match="unsafe CSV filename"):
        _small_manifest(tmp_path)


def test_preparation_rejects_symlink_escape(tmp_path: Path) -> None:
    rows = _make_small_source(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.jpg"
    _write_jpeg(outside, (10, 20, 30))
    victim = tmp_path / "images" / rows["train"][0][0]
    victim.unlink()
    try:
        victim.symlink_to(outside)
    except OSError:
        pytest.skip("filesystem does not permit symlinks")
    with pytest.raises(mini.MiniImageNetIntegrityError, match="escapes the image root"):
        _small_manifest(tmp_path)


def test_preparation_rejects_noncanonical_image_bytes(tmp_path: Path) -> None:
    rows = _make_small_source(tmp_path)
    victim = tmp_path / "images" / rows["train"][0][0]
    Image.new("RGB", (80, 80), color=(1, 2, 3)).save(victim, format="JPEG")
    with pytest.raises(mini.MiniImageNetIntegrityError, match="has size"):
        _small_manifest(tmp_path)


def test_mammoth_overlay_loads_the_locked_manifest_and_checks_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_small_source(tmp_path)
    payload = _small_manifest(tmp_path)
    # The production loader accepts LOCKED only.  Rebind its immutable protocol
    # profile to the small fixture while retaining the same validation path.
    payload["status"] = "LOCKED"
    payload_without_hash = dict(payload)
    payload_without_hash.pop("manifest_content_sha256")
    payload["manifest_content_sha256"] = mini._canonical_sha256(payload_without_hash)
    (tmp_path / mini.MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overlay = _load_overlay(monkeypatch)
    monkeypatch.setattr(overlay, "PROTOCOL", mini._protocol_payload(SMALL_PROTOCOL))
    overlay._MANIFEST_CACHE.clear()

    locked = overlay._load_locked_manifest(tmp_path)
    assert len(locked["partitions"]["train"]) == 8
    assert len(locked["partitions"]["test"]) == 4
    dataset = overlay.LocalMiniImageNet(
        str(tmp_path), train=True, transform=transforms.ToTensor()
    )
    image, target, not_augmented = dataset[0]
    assert image.shape == (3, 84, 84)
    assert not_augmented.shape == (3, 84, 84)
    assert int(target) in range(4)

    selected = Path(dataset.root) / str(dataset.data[0])
    selected.write_bytes(selected.read_bytes() + b"changed")
    with pytest.raises(overlay.MiniImageNetManifestError, match="size changed"):
        dataset[0]

    # CSV provenance is rechecked even when the large manifest parse is cached.
    split_csv = tmp_path / "splits" / "train.csv"
    split_csv.write_bytes(split_csv.read_bytes() + b"\n")
    with pytest.raises(overlay.MiniImageNetManifestError, match="bytes changed"):
        overlay._load_locked_manifest(tmp_path)


def test_overlay_is_local_only_and_mammoth_protocol_matches_lock() -> None:
    source = OVERLAY_SOURCE.read_text(encoding="utf-8")
    assert 'NAME = "seq-miniimg"' in source
    assert 'SETTING = "class-il"' in source
    assert "N_CLASSES_PER_TASK = 5" in source
    assert "N_TASKS = 20" in source
    assert "N_CLASSES = 100" in source
    assert "requests" not in source
    assert "urlretrieve" not in source
    assert "ensure_required_files_from_hf" not in source
    assert "MINIIMAGENET_MANIFEST.json" in source
    assert OVERLAY_CONFIG.read_text(encoding="utf-8") == (
        "N_TASKS: 20\nN_CLASSES_PER_TASK: 5\nN_CLASSES: 100\n"
    )


def test_overlay_protocol_exactly_matches_the_preparation_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _load_overlay(monkeypatch)
    assert overlay.PROTOCOL == mini._protocol_payload(mini.CANONICAL_PROTOCOL)
    assert overlay.SequentialMiniImageNet.NAME == "seq-miniimg"
    assert overlay.SequentialMiniImageNet.N_TASKS == 20
    assert overlay.SequentialMiniImageNet.N_CLASSES_PER_TASK == 5
    assert overlay.SequentialMiniImageNet.N_CLASSES == 100
