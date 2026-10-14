"""Local-only, manifest-locked sequential miniImageNet for Mammoth.

The dataset deliberately has no download path.  It accepts only the
``MINIIMAGENET_MANIFEST.json`` produced by the mechanism-audit preparation
utility and verifies the manifest, canonical split CSVs, path containment,
coverage, and file sizes before use.  Image bytes are SHA-256 checked while
they are read, so modified local data fail before contributing an example.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from utils.conf import base_path


SCHEMA_VERSION = "gradient_reuse.miniimagenet_manifest.v1"
MANIFEST_NAME = "MINIIMAGENET_MANIFEST.json"
SOURCE_SPLITS = ("train", "val", "test")
PROTOCOL = {
    "source_train_classes": 64,
    "source_val_classes": 16,
    "source_test_classes": 20,
    "source_examples_per_class": 600,
    "train_examples_per_class": 500,
    "test_examples_per_class": 100,
    "image_width": 84,
    "image_height": 84,
    "image_mode": "RGB",
    "classes": 100,
    "source_split_classes": {"train": 64, "val": 16, "test": 20},
    "class_order": "lexicographic_wordnet_id",
    "within_class_order": "lexicographic_relative_image_path",
    "partition_rule": "first_train_count_then_test_count",
}
_HEX = re.compile(r"[0-9a-f]{64}")
_WNID = re.compile(r"n[0-9]{8}")
_MANIFEST_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


class MiniImageNetManifestError(RuntimeError):
    """Raised when the local miniImageNet evidence is incomplete or changed."""


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_payload(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise MiniImageNetManifestError(
            f"miniImageNet manifest is not strict UTF-8 JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise MiniImageNetManifestError("miniImageNet manifest must be a JSON object")
    return payload


def _safe_relative(value: Any, *, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MiniImageNetManifestError(f"{description} must be a nonempty relative path")
    if "\\" in value or "\x00" in value:
        raise MiniImageNetManifestError(f"unsafe {description}: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise MiniImageNetManifestError(f"unsafe {description}: {value!r}")
    return relative


def _resolved_below(root: Path, relative: PurePosixPath, *, description: str) -> Path:
    try:
        resolved = root.joinpath(*relative.parts).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise MiniImageNetManifestError(
            f"{description} is missing or escapes the dataset root: {relative}"
        ) from error
    return resolved


def _require_int(value: Any, expected: int, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise MiniImageNetManifestError(
            f"{description} is {value!r}, expected {expected}"
        )


def _validate_class_records(payload: Mapping[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    records = payload.get("class_records")
    classes = int(PROTOCOL["classes"])
    if not isinstance(records, list) or len(records) != classes:
        raise MiniImageNetManifestError(
            f"miniImageNet must contain {classes} class records"
        )
    labels: list[str] = []
    label_by_target: dict[int, str] = {}
    source_by_target: dict[int, str] = {}
    source_counts = {split: 0 for split in SOURCE_SPLITS}
    for expected_target, row in enumerate(records):
        if not isinstance(row, dict):
            raise MiniImageNetManifestError("miniImageNet class record is not an object")
        label = row.get("label")
        if not isinstance(label, str) or _WNID.fullmatch(label) is None:
            raise MiniImageNetManifestError("miniImageNet class label is not a WordNet ID")
        _require_int(row.get("target"), expected_target, "miniImageNet class target")
        _require_int(
            row.get("train_samples"),
            int(PROTOCOL["train_examples_per_class"]),
            "class training count",
        )
        _require_int(
            row.get("test_samples"),
            int(PROTOCOL["test_examples_per_class"]),
            "class test count",
        )
        source = row.get("source_split")
        if source not in source_counts:
            raise MiniImageNetManifestError("miniImageNet class source split is invalid")
        source_counts[source] += 1
        labels.append(label)
        label_by_target[expected_target] = label
        source_by_target[expected_target] = str(source)
    if labels != sorted(labels) or len(set(labels)) != classes:
        raise MiniImageNetManifestError("miniImageNet classes are not unique lexicographic IDs")
    if source_counts != PROTOCOL["source_split_classes"]:
        raise MiniImageNetManifestError("miniImageNet source class coverage mismatch")
    return label_by_target, source_by_target


def _validate_split_files(
    payload: Mapping[str, Any], root: Path, split_root: Path
) -> None:
    files = payload.get("split_files")
    if not isinstance(files, dict) or set(files) != set(SOURCE_SPLITS):
        raise MiniImageNetManifestError("miniImageNet split-file inventory mismatch")
    split_hashes: dict[str, str] = {}
    for split in SOURCE_SPLITS:
        row = files[split]
        if not isinstance(row, dict):
            raise MiniImageNetManifestError("miniImageNet split record is not an object")
        relative = _safe_relative(row.get("path"), description="split CSV path")
        source = _resolved_below(root, relative, description="split CSV")
        try:
            source.relative_to(split_root)
        except ValueError as error:
            raise MiniImageNetManifestError("split CSV lies outside the split directory") from error
        if not source.is_file():
            raise MiniImageNetManifestError(f"split CSV is not a file: {source}")
        expected_rows = (
            PROTOCOL["source_split_classes"][split]
            * PROTOCOL["source_examples_per_class"]
        )
        _require_int(row.get("rows"), expected_rows, f"{split}.csv row count")
        _require_int(
            row.get("classes"),
            PROTOCOL["source_split_classes"][split],
            f"{split}.csv class count",
        )
        digest = row.get("sha256")
        if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
            raise MiniImageNetManifestError("split CSV digest is invalid")
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise MiniImageNetManifestError(f"split CSV bytes changed: {source}")
        split_hashes[relative.as_posix()] = digest
    if payload.get("split_bundle_sha256") != _canonical_sha256(split_hashes):
        raise MiniImageNetManifestError("miniImageNet split bundle hash mismatch")


def _validate_partitions(
    payload: Mapping[str, Any], root: Path, image_root: Path
) -> dict[str, tuple[dict[str, Any], ...]]:
    label_by_target, source_by_target = _validate_class_records(payload)
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "test"}:
        raise MiniImageNetManifestError("miniImageNet partitions are malformed")
    if payload.get("partition_bundle_sha256") != _canonical_sha256(partitions):
        raise MiniImageNetManifestError("miniImageNet partition bundle hash mismatch")

    classes = int(PROTOCOL["classes"])
    per_class_expected = {
        "train": int(PROTOCOL["train_examples_per_class"]),
        "test": int(PROTOCOL["test_examples_per_class"]),
    }
    expected_counts = {
        partition: classes * count
        for partition, count in per_class_expected.items()
    }
    paths_seen: set[str] = set()
    resolved_seen: set[Path] = set()
    inventory: dict[str, dict[str, Any]] = {}
    validated: dict[str, tuple[dict[str, Any], ...]] = {}
    for partition in ("train", "test"):
        rows = partitions[partition]
        if not isinstance(rows, list) or len(rows) != expected_counts[partition]:
            raise MiniImageNetManifestError(
                f"miniImageNet {partition} partition coverage mismatch"
            )
        counts = {target: 0 for target in range(classes)}
        order: list[tuple[int, str]] = []
        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise MiniImageNetManifestError("miniImageNet sample row is not an object")
            relative = _safe_relative(row.get("path"), description="image path")
            source = _resolved_below(root, relative, description="miniImageNet image")
            try:
                source.relative_to(image_root)
            except ValueError as error:
                raise MiniImageNetManifestError(
                    "manifest image lies outside the image directory"
                ) from error
            if not source.is_file():
                raise MiniImageNetManifestError(f"miniImageNet image is not a file: {source}")
            relative_text = relative.as_posix()
            if relative_text in paths_seen or source in resolved_seen:
                raise MiniImageNetManifestError("miniImageNet partitions contain duplicate images")
            paths_seen.add(relative_text)
            resolved_seen.add(source)
            target = row.get("target")
            if isinstance(target, bool) or not isinstance(target, int) or target not in label_by_target:
                raise MiniImageNetManifestError("miniImageNet sample target is invalid")
            if row.get("label") != label_by_target[target]:
                raise MiniImageNetManifestError("miniImageNet sample label/target mismatch")
            if row.get("source_split") != source_by_target[target]:
                raise MiniImageNetManifestError("miniImageNet sample source split mismatch")
            digest = row.get("sha256")
            size = row.get("size")
            if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
                raise MiniImageNetManifestError("miniImageNet image digest is invalid")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise MiniImageNetManifestError("miniImageNet image size is invalid")
            if source.stat().st_size != size:
                raise MiniImageNetManifestError(f"miniImageNet image size changed: {source}")
            counts[target] += 1
            order.append((target, relative_text))
            inventory[relative_text] = {"sha256": digest, "size": size}
            clean_rows.append(dict(row))
        if order != sorted(order):
            raise MiniImageNetManifestError(
                f"miniImageNet {partition} partition order changed"
            )
        if any(count != per_class_expected[partition] for count in counts.values()):
            raise MiniImageNetManifestError(
                f"miniImageNet {partition} per-class coverage mismatch"
            )
        validated[partition] = tuple(clean_rows)
    if payload.get("image_bundle_sha256") != _canonical_sha256(inventory):
        raise MiniImageNetManifestError("miniImageNet image bundle hash mismatch")
    return validated


def _load_locked_manifest(root: Path) -> dict[str, Any]:
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise MiniImageNetManifestError(
            f"local miniImageNet root is missing (downloads are disabled): {root}"
        ) from error
    if not root.is_dir():
        raise MiniImageNetManifestError(f"miniImageNet root is not a directory: {root}")
    path = root / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise MiniImageNetManifestError(
            f"local miniImageNet manifest is missing (downloads are disabled): {path}"
        )
    raw = path.read_bytes()
    cache_key = (
        str(root),
        hashlib.sha256(raw).hexdigest() + _canonical_sha256(PROTOCOL),
    )
    if cache_key in _MANIFEST_CACHE:
        locked = _MANIFEST_CACHE[cache_key]
        # CSVs are small and provenance-bearing.  Recheck their bytes even
        # when the large 60k-row manifest structure is cached across tasks.
        _validate_split_files(
            locked["payload"], locked["root"], locked["split_root"]
        )
        return locked
    payload = _strict_payload(raw, path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MiniImageNetManifestError("miniImageNet manifest schema mismatch")
    if payload.get("status") != "LOCKED":
        raise MiniImageNetManifestError("miniImageNet manifest is not locked")
    if payload.get("images_redistributed") is not False:
        raise MiniImageNetManifestError("miniImageNet redistribution flag changed")
    if payload.get("protocol") != PROTOCOL:
        raise MiniImageNetManifestError("miniImageNet protocol mismatch")
    classes = int(PROTOCOL["classes"])
    _require_int(payload.get("classes"), classes, "miniImageNet class count")
    _require_int(
        payload.get("train_samples"),
        classes * int(PROTOCOL["train_examples_per_class"]),
        "training sample count",
    )
    _require_int(
        payload.get("test_samples"),
        classes * int(PROTOCOL["test_examples_per_class"]),
        "test sample count",
    )
    content = dict(payload)
    observed_content = content.pop("manifest_content_sha256", None)
    if observed_content != _canonical_sha256(content):
        raise MiniImageNetManifestError("miniImageNet manifest content hash mismatch")

    split_relative = _safe_relative(
        payload.get("split_directory"), description="split directory"
    )
    image_relative = _safe_relative(
        payload.get("image_directory"), description="image directory"
    )
    split_root = _resolved_below(root, split_relative, description="split directory")
    image_root = _resolved_below(root, image_relative, description="image directory")
    if not split_root.is_dir() or not image_root.is_dir():
        raise MiniImageNetManifestError("miniImageNet split/image roots must be directories")
    _validate_split_files(payload, root, split_root)
    partitions = _validate_partitions(payload, root, image_root)
    locked = {
        "payload": payload,
        "root": root,
        "split_root": split_root,
        "image_root": image_root,
        "partitions": partitions,
        "class_names": tuple(
            row["label"] for row in payload["class_records"]
        ),
    }
    _MANIFEST_CACHE[cache_key] = locked
    return locked


def _verified_image(
    root: Path, image_root: Path, row: Mapping[str, Any]
) -> Image.Image:
    relative = _safe_relative(row.get("path"), description="image path")
    source = _resolved_below(root, relative, description="miniImageNet image")
    try:
        source.relative_to(image_root)
    except ValueError as error:
        raise MiniImageNetManifestError(
            "miniImageNet image moved outside the image directory"
        ) from error
    payload = source.read_bytes()
    if len(payload) != row["size"]:
        raise MiniImageNetManifestError(f"miniImageNet image size changed: {source}")
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise MiniImageNetManifestError(f"miniImageNet image bytes changed: {source}")
    try:
        with Image.open(BytesIO(payload)) as opened:
            image = opened.convert("RGB")
            image.load()
    except Exception as error:
        raise MiniImageNetManifestError(f"invalid miniImageNet image: {source}") from error
    expected_size = (int(PROTOCOL["image_width"]), int(PROTOCOL["image_height"]))
    if image.size != expected_size:
        raise MiniImageNetManifestError(f"miniImageNet image dimensions changed: {source}")
    return image


class LocalMiniImageNet(Dataset):
    """A manifest-backed split with Mammoth-compatible return values."""

    def __init__(self, root: str, *, train: bool, transform=None) -> None:
        locked = _load_locked_manifest(Path(root))
        self.root = locked["root"]
        self.image_root = locked["image_root"]
        self.train = bool(train)
        self.transform = transform
        self.not_aug_transform = transforms.ToTensor()
        self.class_names = list(locked["class_names"])
        rows = locked["partitions"]["train" if self.train else "test"]
        self._fingerprints = {row["path"]: row for row in rows}
        self.data = np.asarray([row["path"] for row in rows])
        self.targets = np.asarray([row["target"] for row in rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        relative = str(self.data[index])
        target = self.targets[index]
        original = _verified_image(
            self.root, self.image_root, self._fingerprints[relative]
        )
        image = self.transform(original.copy()) if self.transform is not None else original
        if not self.train:
            return image, target
        not_augmented = self.not_aug_transform(original)
        if hasattr(self, "logits"):
            return image, target, not_augmented, self.logits[index]
        return image, target, not_augmented


class SequentialMiniImageNet(ContinualDataset):
    """Twenty 5-way tasks over all 100 canonical miniImageNet classes."""

    NAME = "seq-miniimg"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = 100
    SIZE = (84, 84)
    MEAN = (0.472, 0.453, 0.410)
    STD = (0.277, 0.268, 0.285)
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        root = str(Path(base_path()) / "MINIIMG")
        train_dataset = LocalMiniImageNet(
            root, train=True, transform=self.TRANSFORM
        )
        test_dataset = LocalMiniImageNet(
            root, train=False, transform=self.TEST_TRANSFORM
        )
        self._manifest_class_names = train_dataset.class_names
        return store_masked_loaders(train_dataset, test_dataset, self)

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_transform():
        return transforms.Compose(
            [transforms.ToPILImage(), SequentialMiniImageNet.TRANSFORM]
        )

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(
            SequentialMiniImageNet.MEAN, SequentialMiniImageNet.STD
        )

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialMiniImageNet.MEAN, SequentialMiniImageNet.STD)

    @set_default_from_args("n_epochs")
    def get_epochs():
        return 1

    @set_default_from_args("batch_size")
    def get_batch_size():
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        if not hasattr(self, "_manifest_class_names"):
            locked = _load_locked_manifest(Path(base_path()) / "MINIIMG")
            self._manifest_class_names = list(locked["class_names"])
        self.class_names = fix_class_names_order(
            list(self._manifest_class_names), self.args
        )
        return self.class_names
