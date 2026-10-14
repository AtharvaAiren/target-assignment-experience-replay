"""Prepare a byte-locked, local-only miniImageNet continual-learning manifest.

This module does not download or copy images.  It consumes the canonical
``train.csv``, ``val.csv``, and ``test.csv`` class splits together with a
locally acquired ``images`` directory.  The three source splits contain
64/16/20 disjoint classes and 600 images per class.  For the continual
benchmark, classes are ordered by WordNet identifier and each class is split
deterministically into 500 training and 100 test images by relative filename.

Every path, split byte, and image byte is authenticated.  A different existing
manifest is never overwritten, making data preparation fail closed and
leaving ImageNet-licensed images outside the repository.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .protocol import strict_json


SCHEMA_VERSION = "gradient_reuse.miniimagenet_manifest.v1"
MANIFEST_NAME = "MINIIMAGENET_MANIFEST.json"
SOURCE_SPLITS = ("train", "val", "test")
_WNID = re.compile(r"n[0-9]{8}")


class MiniImageNetIntegrityError(RuntimeError):
    """Raised when local miniImageNet inputs violate the locked protocol."""


@dataclass(frozen=True)
class MiniImageNetProtocol:
    """Expected structure of a miniImageNet source and derived benchmark."""

    source_train_classes: int = 64
    source_val_classes: int = 16
    source_test_classes: int = 20
    source_examples_per_class: int = 600
    train_examples_per_class: int = 500
    test_examples_per_class: int = 100
    image_width: int = 84
    image_height: int = 84
    image_mode: str = "RGB"

    @property
    def source_split_classes(self) -> dict[str, int]:
        return {
            "train": self.source_train_classes,
            "val": self.source_val_classes,
            "test": self.source_test_classes,
        }

    @property
    def classes(self) -> int:
        return sum(self.source_split_classes.values())

    @property
    def train_samples(self) -> int:
        return self.classes * self.train_examples_per_class

    @property
    def test_samples(self) -> int:
        return self.classes * self.test_examples_per_class

    def validate(self) -> None:
        values = [
            *self.source_split_classes.values(),
            self.source_examples_per_class,
            self.train_examples_per_class,
            self.test_examples_per_class,
            self.image_width,
            self.image_height,
        ]
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise MiniImageNetIntegrityError("miniImageNet counts must be positive integers")
        if (
            self.train_examples_per_class + self.test_examples_per_class
            != self.source_examples_per_class
        ):
            raise MiniImageNetIntegrityError(
                "per-class train/test counts do not cover the source class"
            )
        if not self.image_mode:
            raise MiniImageNetIntegrityError("image mode must be nonempty")


CANONICAL_PROTOCOL = MiniImageNetProtocol()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> Any:
    try:
        return strict_json(path)
    except RuntimeError as error:
        raise MiniImageNetIntegrityError(
            f"miniImageNet manifest is not strict JSON: {path}"
        ) from error


def _below(path: Path, root: Path, *, description: str) -> tuple[Path, str]:
    """Resolve ``path`` below ``root`` and return its portable POSIX path."""

    root = root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise MiniImageNetIntegrityError(
            f"{description} must exist below the miniImageNet root: {path}"
        ) from error
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise MiniImageNetIntegrityError(f"unsafe {description}: {path}")
    return resolved, relative.as_posix()


def _safe_csv_filename(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MiniImageNetIntegrityError("CSV filename must be a nonempty trimmed string")
    if "\\" in value or "\x00" in value:
        raise MiniImageNetIntegrityError(f"unsafe CSV filename: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise MiniImageNetIntegrityError(f"unsafe CSV filename: {value!r}")
    if path.suffix != ".jpg":
        raise MiniImageNetIntegrityError(
            f"canonical miniImageNet files must use lowercase .jpg: {value!r}"
        )
    return path


def _safe_csv_filename_for_manifest(value: Any, *, description: str) -> PurePosixPath:
    """Validate a portable manifest path without imposing an image suffix."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise MiniImageNetIntegrityError(f"{description} must be a nonempty relative path")
    if "\\" in value or "\x00" in value:
        raise MiniImageNetIntegrityError(f"unsafe {description}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise MiniImageNetIntegrityError(f"unsafe {description}: {value!r}")
    return path


def _read_source_csv(path: Path, split: str) -> list[tuple[PurePosixPath, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["filename", "label"]:
                raise MiniImageNetIntegrityError(
                    f"{split}.csv must have exactly the header filename,label"
                )
            rows: list[tuple[PurePosixPath, str]] = []
            for line_number, row in enumerate(reader, 2):
                if None in row or set(row) != {"filename", "label"}:
                    raise MiniImageNetIntegrityError(
                        f"malformed row at {path}:{line_number}"
                    )
                label = row["label"]
                if not isinstance(label, str) or _WNID.fullmatch(label) is None:
                    raise MiniImageNetIntegrityError(
                        f"invalid WordNet class at {path}:{line_number}: {label!r}"
                    )
                filename = _safe_csv_filename(row["filename"])
                if not filename.name.startswith(label):
                    raise MiniImageNetIntegrityError(
                        f"filename/class mismatch at {path}:{line_number}"
                    )
                rows.append((filename, label))
    except UnicodeError as error:
        raise MiniImageNetIntegrityError(f"split CSV is not UTF-8: {path}") from error
    if not rows:
        raise MiniImageNetIntegrityError(f"split CSV is empty: {path}")
    return rows


def _image_record(path: Path, relative: str, protocol: MiniImageNetProtocol) -> dict[str, Any]:
    """Read, validate, and fingerprint one local image without copying it."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MiniImageNetIntegrityError(f"cannot read miniImageNet image: {path}") from error
    if not payload:
        raise MiniImageNetIntegrityError(f"miniImageNet image is empty: {path}")
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            observed_format = image.format
            observed_size = image.size
            observed_mode = image.mode
            image.verify()
    except Exception as error:
        raise MiniImageNetIntegrityError(f"invalid miniImageNet image: {path}") from error
    if observed_format != "JPEG":
        raise MiniImageNetIntegrityError(f"miniImageNet image is not JPEG: {path}")
    if observed_size != (protocol.image_width, protocol.image_height):
        raise MiniImageNetIntegrityError(
            f"miniImageNet image has size {observed_size}, expected "
            f"{(protocol.image_width, protocol.image_height)}: {path}"
        )
    if observed_mode != protocol.image_mode:
        raise MiniImageNetIntegrityError(
            f"miniImageNet image has mode {observed_mode}, expected "
            f"{protocol.image_mode}: {path}"
        )
    return {
        "path": relative,
        "sha256": _sha256_bytes(payload),
        "size": len(payload),
    }


def _protocol_payload(protocol: MiniImageNetProtocol) -> dict[str, Any]:
    return {
        **asdict(protocol),
        "classes": protocol.classes,
        "source_split_classes": protocol.source_split_classes,
        "class_order": "lexicographic_wordnet_id",
        "within_class_order": "lexicographic_relative_image_path",
        "partition_rule": "first_train_count_then_test_count",
    }


def _build_manifest(
    dataset_root: Path,
    split_root: Path,
    image_root: Path,
    *,
    protocol: MiniImageNetProtocol,
) -> dict[str, Any]:
    """Build a manifest under ``protocol``; noncanonical profiles are test-only."""

    protocol.validate()
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise MiniImageNetIntegrityError(f"miniImageNet root is not a directory: {root}")
    splits, split_directory = _below(split_root, root, description="split directory")
    images, image_directory = _below(image_root, root, description="image directory")
    if not splits.is_dir() or not images.is_dir():
        raise MiniImageNetIntegrityError("split and image locations must be directories")

    source_rows: dict[str, list[tuple[PurePosixPath, str]]] = {}
    split_files: dict[str, dict[str, Any]] = {}
    label_source: dict[str, str] = {}
    seen_csv_names: set[str] = set()
    for split in SOURCE_SPLITS:
        csv_path = splits / f"{split}.csv"
        resolved_csv, relative_csv = _below(
            csv_path, root, description=f"{split} split CSV"
        )
        if not resolved_csv.is_file():
            raise MiniImageNetIntegrityError(f"split CSV is not a file: {csv_path}")
        rows = _read_source_csv(resolved_csv, split)
        source_rows[split] = rows
        labels = sorted({label for _, label in rows})
        expected_classes = protocol.source_split_classes[split]
        if len(labels) != expected_classes:
            raise MiniImageNetIntegrityError(
                f"{split}.csv has {len(labels)} classes, expected {expected_classes}"
            )
        for label in labels:
            previous = label_source.setdefault(label, split)
            if previous != split:
                raise MiniImageNetIntegrityError(
                    f"class {label} occurs in both {previous}.csv and {split}.csv"
                )
        names = [filename.as_posix() for filename, _ in rows]
        duplicates = seen_csv_names.intersection(names)
        if duplicates:
            raise MiniImageNetIntegrityError(
                f"image filename occurs in multiple split rows: {sorted(duplicates)[0]}"
            )
        if len(names) != len(set(names)):
            raise MiniImageNetIntegrityError(f"duplicate filename within {split}.csv")
        seen_csv_names.update(names)
        split_payload = resolved_csv.read_bytes()
        split_files[split] = {
            "path": relative_csv,
            "sha256": _sha256_bytes(split_payload),
            "rows": len(rows),
            "classes": len(labels),
        }

    if len(label_source) != protocol.classes:
        raise MiniImageNetIntegrityError(
            f"source splits cover {len(label_source)} classes, expected {protocol.classes}"
        )

    by_label: dict[str, list[PurePosixPath]] = {label: [] for label in label_source}
    for split in SOURCE_SPLITS:
        for filename, label in source_rows[split]:
            by_label[label].append(filename)
    for label, names in by_label.items():
        if len(names) != protocol.source_examples_per_class:
            raise MiniImageNetIntegrityError(
                f"class {label} has {len(names)} source images, expected "
                f"{protocol.source_examples_per_class}"
            )

    class_order = sorted(by_label)
    class_records: list[dict[str, Any]] = []
    partitions: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    image_inventory: dict[str, dict[str, Any]] = {}
    resolved_images: set[Path] = set()
    for target, label in enumerate(class_order):
        names = sorted(by_label[label], key=lambda item: item.as_posix())
        train_names = names[: protocol.train_examples_per_class]
        test_names = names[protocol.train_examples_per_class :]
        if len(test_names) != protocol.test_examples_per_class:
            raise MiniImageNetIntegrityError(f"derived test coverage changed for {label}")
        class_records.append(
            {
                "label": label,
                "target": target,
                "source_split": label_source[label],
                "train_samples": len(train_names),
                "test_samples": len(test_names),
            }
        )
        for partition, selected in (("train", train_names), ("test", test_names)):
            for csv_name in selected:
                lexical = images.joinpath(*csv_name.parts)
                try:
                    resolved = lexical.resolve(strict=True)
                    resolved.relative_to(images)
                except (FileNotFoundError, ValueError) as error:
                    raise MiniImageNetIntegrityError(
                        f"image path is missing or escapes the image root: {csv_name}"
                    ) from error
                if not resolved.is_file():
                    raise MiniImageNetIntegrityError(f"image is not a file: {csv_name}")
                if resolved in resolved_images:
                    raise MiniImageNetIntegrityError(
                        f"multiple CSV paths resolve to one image: {csv_name}"
                    )
                resolved_images.add(resolved)
                relative = resolved.relative_to(root).as_posix()
                fingerprint = _image_record(resolved, relative, protocol)
                image_inventory[relative] = {
                    "sha256": fingerprint["sha256"],
                    "size": fingerprint["size"],
                }
                partitions[partition].append(
                    {
                        **fingerprint,
                        "label": label,
                        "target": target,
                        "source_split": label_source[label],
                    }
                )

    for partition in partitions:
        partitions[partition].sort(key=lambda row: (row["target"], row["path"]))
    if len(partitions["train"]) != protocol.train_samples:
        raise MiniImageNetIntegrityError("derived training coverage is incomplete")
    if len(partitions["test"]) != protocol.test_samples:
        raise MiniImageNetIntegrityError("derived test coverage is incomplete")
    train_paths = {row["path"] for row in partitions["train"]}
    test_paths = {row["path"] for row in partitions["test"]}
    if train_paths.intersection(test_paths):
        raise MiniImageNetIntegrityError("derived train and test images overlap")
    if len(image_inventory) != protocol.classes * protocol.source_examples_per_class:
        raise MiniImageNetIntegrityError("image inventory coverage is incomplete")

    split_hashes = {
        row["path"]: row["sha256"] for row in split_files.values()
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCKED" if protocol == CANONICAL_PROTOCOL else "TEST_ONLY",
        "images_redistributed": False,
        "classes": protocol.classes,
        "train_samples": protocol.train_samples,
        "test_samples": protocol.test_samples,
        "split_directory": split_directory,
        "image_directory": image_directory,
        "protocol": _protocol_payload(protocol),
        "split_files": split_files,
        "class_records": class_records,
        "partitions": partitions,
        "split_bundle_sha256": _canonical_sha256(split_hashes),
        "image_bundle_sha256": _canonical_sha256(image_inventory),
        "partition_bundle_sha256": _canonical_sha256(partitions),
    }
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    return manifest


def build_manifest(
    dataset_root: Path,
    *,
    split_root: Path | None = None,
    image_root: Path | None = None,
) -> dict[str, Any]:
    """Build the canonical 100-class manifest without writing or downloading."""

    root = dataset_root.resolve(strict=True)
    return _build_manifest(
        root,
        root / "splits" if split_root is None else split_root,
        root / "images" if image_root is None else image_root,
        protocol=CANONICAL_PROTOCOL,
    )


def validate_manifest_payload(
    payload: Mapping[str, Any],
    *,
    protocol: MiniImageNetProtocol = CANONICAL_PROTOCOL,
    require_locked: bool = True,
) -> None:
    """Validate the immutable structure and hashes of an in-memory manifest."""

    expected_status = "LOCKED" if require_locked else (
        "LOCKED" if protocol == CANONICAL_PROTOCOL else "TEST_ONLY"
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MiniImageNetIntegrityError("miniImageNet manifest schema mismatch")
    if payload.get("status") != expected_status:
        raise MiniImageNetIntegrityError("miniImageNet manifest is not locked")
    if payload.get("protocol") != _protocol_payload(protocol):
        raise MiniImageNetIntegrityError("miniImageNet manifest protocol mismatch")
    def require_int(value: Any, expected: int, description: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise MiniImageNetIntegrityError(
                f"{description} is {value!r}, expected {expected}"
            )

    require_int(payload.get("classes"), protocol.classes, "miniImageNet classes")
    require_int(
        payload.get("train_samples"),
        protocol.train_samples,
        "miniImageNet training coverage",
    )
    require_int(
        payload.get("test_samples"),
        protocol.test_samples,
        "miniImageNet test coverage",
    )
    content = dict(payload)
    observed_content = content.pop("manifest_content_sha256", None)
    if observed_content != _canonical_sha256(content):
        raise MiniImageNetIntegrityError("miniImageNet manifest content hash mismatch")

    split_files = payload.get("split_files")
    if not isinstance(split_files, dict) or set(split_files) != set(SOURCE_SPLITS):
        raise MiniImageNetIntegrityError("miniImageNet split-file inventory mismatch")
    split_directory = _safe_csv_filename_for_manifest(
        payload.get("split_directory"), description="split directory"
    )
    image_directory = _safe_csv_filename_for_manifest(
        payload.get("image_directory"), description="image directory"
    )
    split_hashes: dict[str, str] = {}
    for split in SOURCE_SPLITS:
        row = split_files[split]
        if not isinstance(row, dict):
            raise MiniImageNetIntegrityError("miniImageNet split record is malformed")
        relative = _safe_csv_filename_for_manifest(
            row.get("path"), description="split CSV path"
        )
        if relative.parts[: len(split_directory.parts)] != split_directory.parts:
            raise MiniImageNetIntegrityError("split CSV lies outside split directory")
        digest = row.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise MiniImageNetIntegrityError("miniImageNet split digest is invalid")
        require_int(
            row.get("rows"),
            protocol.source_split_classes[split]
            * protocol.source_examples_per_class,
            f"{split}.csv rows",
        )
        require_int(
            row.get("classes"),
            protocol.source_split_classes[split],
            f"{split}.csv classes",
        )
        split_hashes[relative.as_posix()] = digest
    if len(split_hashes) != len(SOURCE_SPLITS) or payload.get(
        "split_bundle_sha256"
    ) != _canonical_sha256(split_hashes):
        raise MiniImageNetIntegrityError("miniImageNet split bundle hash mismatch")

    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "test"}:
        raise MiniImageNetIntegrityError("miniImageNet partitions are malformed")
    if payload.get("partition_bundle_sha256") != _canonical_sha256(partitions):
        raise MiniImageNetIntegrityError("miniImageNet partition bundle hash mismatch")
    if len(partitions["train"]) != protocol.train_samples or len(
        partitions["test"]
    ) != protocol.test_samples:
        raise MiniImageNetIntegrityError("miniImageNet partition coverage mismatch")
    inventory: dict[str, dict[str, Any]] = {}
    partition_paths: dict[str, set[str]] = {"train": set(), "test": set()}
    per_class: dict[str, dict[int, int]] = {
        "train": {index: 0 for index in range(protocol.classes)},
        "test": {index: 0 for index in range(protocol.classes)},
    }
    class_records = payload.get("class_records")
    if not isinstance(class_records, list) or len(class_records) != protocol.classes:
        raise MiniImageNetIntegrityError("miniImageNet class records are malformed")
    if not all(isinstance(row, dict) for row in class_records):
        raise MiniImageNetIntegrityError("miniImageNet class record is malformed")
    labels = [row.get("label") for row in class_records]
    if labels != sorted(labels) or len(set(labels)) != protocol.classes:
        raise MiniImageNetIntegrityError("miniImageNet class order is unstable")
    label_by_target: dict[int, str] = {}
    source_by_target: dict[int, str] = {}
    source_counts = {split: 0 for split in SOURCE_SPLITS}
    for expected_target, row in enumerate(class_records):
        label = row.get("label")
        if not isinstance(label, str) or _WNID.fullmatch(label) is None:
            raise MiniImageNetIntegrityError("miniImageNet class label is invalid")
        require_int(row.get("target"), expected_target, "miniImageNet target")
        require_int(
            row.get("train_samples"),
            protocol.train_examples_per_class,
            "class training samples",
        )
        require_int(
            row.get("test_samples"),
            protocol.test_examples_per_class,
            "class test samples",
        )
        source = row.get("source_split")
        if source not in source_counts:
            raise MiniImageNetIntegrityError("miniImageNet class source split is invalid")
        source_counts[source] += 1
        label_by_target[expected_target] = label
        source_by_target[expected_target] = str(source)
    if source_counts != protocol.source_split_classes:
        raise MiniImageNetIntegrityError("miniImageNet source class coverage mismatch")
    for partition, rows in partitions.items():
        if not isinstance(rows, list):
            raise MiniImageNetIntegrityError(f"miniImageNet {partition} rows are malformed")
        observed_order: list[tuple[int, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise MiniImageNetIntegrityError("miniImageNet sample row is not an object")
            relative = row.get("path")
            if not isinstance(relative, str):
                raise MiniImageNetIntegrityError("miniImageNet sample path is missing")
            safe = PurePosixPath(relative)
            if (
                safe.is_absolute()
                or not safe.parts
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in safe.parts)
            ):
                raise MiniImageNetIntegrityError(f"unsafe manifest image path: {relative}")
            if safe.parts[: len(image_directory.parts)] != image_directory.parts:
                raise MiniImageNetIntegrityError("manifest image lies outside image directory")
            target = row.get("target")
            if (
                isinstance(target, bool)
                or not isinstance(target, int)
                or target not in label_by_target
            ):
                raise MiniImageNetIntegrityError("miniImageNet sample target is invalid")
            if row.get("label") != label_by_target[target]:
                raise MiniImageNetIntegrityError("miniImageNet sample label/target mismatch")
            if row.get("source_split") != source_by_target[target]:
                raise MiniImageNetIntegrityError("miniImageNet sample source split mismatch")
            digest = row.get("sha256")
            size = row.get("size")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not isinstance(size, int)
                or size <= 0
            ):
                raise MiniImageNetIntegrityError("miniImageNet image fingerprint is invalid")
            if relative in partition_paths[partition] or relative in inventory:
                raise MiniImageNetIntegrityError("miniImageNet manifest contains duplicate paths")
            partition_paths[partition].add(relative)
            inventory[relative] = {"sha256": digest, "size": size}
            per_class[partition][target] += 1
            observed_order.append((target, relative))
        if observed_order != sorted(observed_order):
            raise MiniImageNetIntegrityError(
                f"miniImageNet {partition} rows are not in canonical order"
            )
    if partition_paths["train"].intersection(partition_paths["test"]):
        raise MiniImageNetIntegrityError("miniImageNet train/test partitions overlap")
    for target in range(protocol.classes):
        if per_class["train"][target] != protocol.train_examples_per_class:
            raise MiniImageNetIntegrityError("miniImageNet per-class training coverage mismatch")
        if per_class["test"][target] != protocol.test_examples_per_class:
            raise MiniImageNetIntegrityError("miniImageNet per-class test coverage mismatch")
    if payload.get("image_bundle_sha256") != _canonical_sha256(inventory):
        raise MiniImageNetIntegrityError("miniImageNet image bundle hash mismatch")


def write_manifest(
    dataset_root: Path,
    *,
    split_root: Path | None = None,
    image_root: Path | None = None,
) -> Path:
    """Atomically write the canonical manifest, refusing differing outputs."""

    root = dataset_root.resolve(strict=True)
    output = root / MANIFEST_NAME
    payload = build_manifest(root, split_root=split_root, image_root=image_root)
    validate_manifest_payload(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists() or output.is_symlink():
        if (
            output.is_symlink()
            or not output.is_file()
            or output.read_text(encoding="utf-8") != encoded
        ):
            raise MiniImageNetIntegrityError(
                f"refusing to overwrite a different miniImageNet manifest: {output}"
            )
        return output
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local MINIIMG root; images are read but never copied.",
    )
    parser.add_argument(
        "--split-root",
        help="Directory containing train.csv, val.csv, test.csv (default: ROOT/splits).",
    )
    parser.add_argument(
        "--image-root",
        help="Directory containing local JPEGs (default: ROOT/images).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(arguments.dataset_root)
    output = write_manifest(
        root,
        split_root=Path(arguments.split_root) if arguments.split_root else None,
        image_root=Path(arguments.image_root) if arguments.image_root else None,
    )
    payload = _strict_json(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "manifest": str(output),
                "classes": payload["classes"],
                "train_samples": payload["train_samples"],
                "test_samples": payload["test_samples"],
                "split_bundle_sha256": payload["split_bundle_sha256"],
                "image_bundle_sha256": payload["image_bundle_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
