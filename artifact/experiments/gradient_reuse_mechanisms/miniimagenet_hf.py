"""Convert gated Hugging Face ImageNet Parquet shards into locked miniImageNet.

The official ``ILSVRC/imagenet-1k`` Hugging Face repository stores each image
as bytes plus a path of the form ``WNID_SOURCEINDEX_WNID.JPEG``.  The canonical
Ravi--Larochelle miniImageNet CSV files encode a one-based rank into the sorted
source-image indices for each class.  This module reproduces that selection,
resizes the selected images to 84 x 84 with OpenCV area interpolation, and then
delegates to :mod:`miniimagenet` to authenticate every output byte.

Conversion is local-only and resumable.  Source JPEGs for the 100 requested
classes are staged by numeric ImageNet index, and a shard is marked complete
only after all of its selected records have been durably written.  Existing
files are accepted only when their bytes match, so retries fail closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from . import miniimagenet


STATE_SCHEMA = "gradient_reuse.miniimagenet_hf_conversion.v1"
STATE_NAME = "HF_CONVERSION_STATE.json"
SOURCE_PATTERN = re.compile(
    r"^(?P<left>n[0-9]{8})_(?P<index>[0-9]+)_(?P<right>n[0-9]{8})[.]JPEG$"
)
CSV_PATTERN = re.compile(r"^(?P<label>n[0-9]{8})(?P<selection>[0-9]{8})[.]jpg$")


class HuggingFaceMiniImageNetError(RuntimeError):
    """Raised when the Hugging Face conversion cannot be authenticated."""


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Create ``path`` atomically, or verify an identical existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise HuggingFaceMiniImageNetError(
                f"refusing a differing existing conversion file: {path}"
            )
        return
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": path.name, "size": stat.st_size}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": STATE_SCHEMA, "completed_shards": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HuggingFaceMiniImageNetError(
            f"invalid Hugging Face conversion state: {path}"
        ) from error
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != STATE_SCHEMA
        or not isinstance(state.get("completed_shards"), dict)
    ):
        raise HuggingFaceMiniImageNetError("Hugging Face conversion state mismatch")
    return state


def _canonical_selections(split_root: Path) -> dict[str, dict[int, str]]:
    """Return ``WNID -> one-based source rank -> canonical output filename``."""

    selections: dict[str, dict[int, str]] = {}
    observed_classes: dict[str, str] = {}
    for split in miniimagenet.SOURCE_SPLITS:
        path = split_root / f"{split}.csv"
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["filename", "label"]:
                    raise HuggingFaceMiniImageNetError(
                        f"{path} must have exactly the header filename,label"
                    )
                for row in reader:
                    filename = row.get("filename")
                    label = row.get("label")
                    if not isinstance(filename, str) or not isinstance(label, str):
                        raise HuggingFaceMiniImageNetError(f"malformed split row: {path}")
                    match = CSV_PATTERN.fullmatch(filename)
                    if match is None or match.group("label") != label:
                        raise HuggingFaceMiniImageNetError(
                            f"unexpected canonical miniImageNet filename: {filename!r}"
                        )
                    previous = observed_classes.setdefault(label, split)
                    if previous != split:
                        raise HuggingFaceMiniImageNetError(
                            f"miniImageNet class occurs in two source splits: {label}"
                        )
                    rank = int(match.group("selection")[-4:])
                    if rank <= 0:
                        raise HuggingFaceMiniImageNetError(
                            f"invalid one-based source rank in {filename!r}"
                        )
                    current = selections.setdefault(label, {})
                    if rank in current or filename in current.values():
                        raise HuggingFaceMiniImageNetError(
                            f"duplicate miniImageNet selection for {label}: {filename}"
                        )
                    current[rank] = filename
        except FileNotFoundError as error:
            raise HuggingFaceMiniImageNetError(f"missing split CSV: {path}") from error
    expected = miniimagenet.CANONICAL_PROTOCOL
    if len(selections) != expected.classes:
        raise HuggingFaceMiniImageNetError(
            f"canonical splits contain {len(selections)} classes, expected {expected.classes}"
        )
    for label, rows in selections.items():
        if len(rows) != expected.source_examples_per_class:
            raise HuggingFaceMiniImageNetError(
                f"canonical split selects {len(rows)} images for {label}, expected "
                f"{expected.source_examples_per_class}"
            )
    return selections


def _parse_source_path(value: Any) -> tuple[str, int]:
    if not isinstance(value, str):
        raise HuggingFaceMiniImageNetError("Hugging Face image path is missing")
    name = Path(value).name
    match = SOURCE_PATTERN.fullmatch(name)
    if match is None or match.group("left") != match.group("right"):
        raise HuggingFaceMiniImageNetError(
            f"unexpected Hugging Face ImageNet path: {value!r}"
        )
    return match.group("left"), int(match.group("index"))


def _parquet_files(root: Path) -> list[Path]:
    paths = sorted(root.glob("train-*.parquet"))
    if len(paths) != 294:
        raise HuggingFaceMiniImageNetError(
            f"found {len(paths)} ImageNet training shards below {root}, expected 294"
        )
    return paths


def _stage_sources(
    parquet_root: Path,
    staging_root: Path,
    labels: set[str],
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise HuggingFaceMiniImageNetError("pyarrow is required for conversion") from error

    state_path = staging_root / STATE_NAME
    state = _load_state(state_path)
    completed: dict[str, Any] = state["completed_shards"]
    source_root = staging_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    shards = _parquet_files(parquet_root)
    for sequence, shard in enumerate(shards, 1):
        identity = _source_identity(shard)
        previous = completed.get(shard.name)
        if previous is not None:
            if previous != identity:
                raise HuggingFaceMiniImageNetError(
                    f"completed Parquet shard changed: {shard}"
                )
            print(f"[stage] shard {sequence:03d}/294 already complete: {shard.name}")
            continue
        selected = 0
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["image"], batch_size=256):
            for row in batch.to_pylist():
                image = row.get("image")
                if not isinstance(image, dict):
                    raise HuggingFaceMiniImageNetError(
                        f"malformed image record in {shard}"
                    )
                label, source_index = _parse_source_path(image.get("path"))
                if label not in labels:
                    continue
                payload = image.get("bytes")
                if not isinstance(payload, bytes) or not payload:
                    raise HuggingFaceMiniImageNetError(
                        f"missing JPEG bytes in {shard}: {image.get('path')!r}"
                    )
                destination = source_root / label / f"{source_index:08d}.JPEG"
                _atomic_bytes(destination, payload)
                selected += 1
        completed[shard.name] = identity
        _atomic_json(state_path, state)
        print(
            f"[stage] shard {sequence:03d}/294 complete: "
            f"{shard.name}; selected={selected}"
        )


def _resize_jpeg(payload: bytes) -> bytes:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise HuggingFaceMiniImageNetError(
            "opencv-python-headless and numpy are required for conversion"
        ) from error
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise HuggingFaceMiniImageNetError("cannot decode ImageNet source as color JPEG")
    resized = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    succeeded, result = cv2.imencode(".jpg", resized)
    if not succeeded:
        raise HuggingFaceMiniImageNetError("cannot encode resized miniImageNet JPEG")
    return result.tobytes()


def _materialize_images(
    staging_root: Path,
    image_root: Path,
    selections: dict[str, dict[int, str]],
) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    for sequence, label in enumerate(sorted(selections), 1):
        sources = sorted((staging_root / "source" / label).glob("*.JPEG"))
        indexed: list[tuple[int, Path]] = []
        seen: set[int] = set()
        for source in sources:
            try:
                source_index = int(source.stem)
            except ValueError as error:
                raise HuggingFaceMiniImageNetError(
                    f"invalid staged source filename: {source}"
                ) from error
            if source_index in seen:
                raise HuggingFaceMiniImageNetError(
                    f"duplicate staged ImageNet index for {label}: {source_index}"
                )
            seen.add(source_index)
            indexed.append((source_index, source))
        indexed.sort()
        requested = selections[label]
        if not indexed or max(requested) > len(indexed):
            raise HuggingFaceMiniImageNetError(
                f"ImageNet source coverage is insufficient for {label}: "
                f"available={len(indexed)}, requested_rank={max(requested)}"
            )
        for rank, filename in sorted(requested.items()):
            source = indexed[rank - 1][1]
            resized = _resize_jpeg(source.read_bytes())
            _atomic_bytes(image_root / filename, resized)
        print(
            f"[resize] class {sequence:03d}/{len(selections)} complete: "
            f"{label}; source={len(indexed)} selected={len(requested)}"
        )


def convert(
    parquet_root: Path,
    split_root: Path,
    output_root: Path,
    staging_root: Path,
) -> Path:
    parquet_root = parquet_root.resolve(strict=True)
    split_root = split_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    output_splits = output_root / "splits"
    output_splits.mkdir(parents=True, exist_ok=True)
    selections = _canonical_selections(split_root)
    for split in miniimagenet.SOURCE_SPLITS:
        source = split_root / f"{split}.csv"
        destination = output_splits / source.name
        _atomic_bytes(destination, source.read_bytes())
    _stage_sources(parquet_root, staging_root, set(selections))
    _materialize_images(staging_root, output_root / "images", selections)
    return miniimagenet.write_manifest(output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--staging-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = convert(
        Path(arguments.parquet_root),
        Path(arguments.split_root),
        Path(arguments.output_root),
        Path(arguments.staging_root),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
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
