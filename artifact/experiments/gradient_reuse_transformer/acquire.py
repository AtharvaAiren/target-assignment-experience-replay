"""Acquire and materialize the exact external snapshots locked by the audit.

This is the only network-facing module in the Transformer extension.  The
scientific runner never imports it and always loads local content-addressed
assets with ``local_files_only=True``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from .assets import sha256_file, sha256_tree
from .data import iter_json_records
from .run import (
    DEFAULT_CONFIG,
    _package_version,
    read_config,
    validate_transformer_source_manifest,
)


SnapshotDownload = Callable[..., str]


def _materialize_regular_tree(source: Path, destination: Path) -> int:
    """Dereference Hub cache links and omit client-specific cache metadata."""
    destination.mkdir(parents=True, exist_ok=False)
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".cache" in relative.parts:
            continue
        if not path.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile reads through a source symlink but always creates a regular
        # destination file, which makes the final tree portable and hashable.
        shutil.copyfile(path, target)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"snapshot materialized no files from {source}")
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise RuntimeError("materialized snapshot unexpectedly contains a symlink")
    return copied


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"dataset value is not canonically JSON serializable: {type(value)}")


def _iter_parquet(root: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - raw JSON is the primary source.
        raise RuntimeError(
            "the pinned dataset snapshot contains only Parquet; install pinned pyarrow"
        ) from error
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise RuntimeError("dataset snapshot contains neither JSONL nor Parquet records")
    for path in files:
        reader = parquet.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=2048):
            for record in batch.to_pylist():
                if not isinstance(record, dict):
                    raise RuntimeError(f"Parquet row is not an object: {path}")
                yield record


def _source_records(root: Path) -> tuple[Iterable[dict[str, Any]], str]:
    json_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name.endswith((".jsonl", ".jsonl.gz", ".json.gz"))
    ]
    if json_files:
        return iter_json_records(root), "json_lines"
    return _iter_parquet(root), "parquet"


def _canonicalize_dataset(
    source: Path,
    destination: Path,
    *,
    repository_id: str,
    resolved_revision: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "news.jsonl"
    records, source_format = _source_records(source)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            normalized = _jsonable(record)
            handle.write(
                json.dumps(
                    normalized,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise RuntimeError("canonical dataset is empty")
    derivation = {
        "schema_version": "fmmt.canonical_news_derivation.v1",
        "source_repository_id": repository_id,
        "source_resolved_revision": resolved_revision,
        "source_tree_sha256": source_tree_sha256,
        "source_format": source_format,
        "canonical_format": "utf8_json_lines_sorted_object_keys_compact",
        "canonical_file": output.name,
        "canonical_file_sha256": sha256_file(output),
        "rows": count,
        "row_order": "source shard lexical order, then source row order",
    }
    (destination / "DERIVATION.json").write_text(
        json.dumps(derivation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derivation


def acquire_assets(
    *,
    config_path: str | Path,
    output_root: str | Path,
    source_manifest: str | Path,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> Path:
    """Download exact revisions, materialize them, and emit their manifest."""
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    source_provenance = validate_transformer_source_manifest(
        source_manifest, config_path=config_path, required=True
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite asset root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # These must be set before huggingface_hub is imported.  Xet reconstruction
    # has previously been a source of opaque failures on the target server.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=str(output.parent))
    )
    try:
        downloads = stage / "_downloads"
        downloads.mkdir()
        model_lock = config["assets"]["model"]
        dataset_lock = config["assets"]["dataset"]
        model_source = Path(
            snapshot_download_fn(
                repo_id=str(model_lock["repository_id"]),
                revision=str(model_lock["resolved_revision"]),
                repo_type="model",
                local_dir=str(downloads / "model"),
                allow_patterns=[
                    "*.json",
                    "*.bin",
                    "*.safetensors",
                    "*.model",
                    "*.txt",
                    "*.tiktoken",
                ],
                max_workers=4,
            )
        ).resolve()
        dataset_source = Path(
            snapshot_download_fn(
                repo_id=str(dataset_lock["repository_id"]),
                revision=str(dataset_lock["resolved_revision"]),
                repo_type="dataset",
                local_dir=str(downloads / "dataset"),
                allow_patterns=[
                    "v0/**",
                    "v0/documents/**",
                    "*.jsonl",
                    "*.jsonl.gz",
                    "*.json.gz",
                    "*.parquet",
                    "**/*.parquet",
                ],
                max_workers=4,
            )
        ).resolve()

        model_destination = stage / "model"
        _materialize_regular_tree(model_source, model_destination)
        raw_dataset = stage / "_dataset_source"
        _materialize_regular_tree(dataset_source, raw_dataset)
        source_dataset_hash = sha256_tree(raw_dataset)
        derivation = _canonicalize_dataset(
            raw_dataset,
            stage / "dataset",
            repository_id=str(dataset_lock["repository_id"]),
            resolved_revision=str(dataset_lock["resolved_revision"]),
            source_tree_sha256=source_dataset_hash,
        )
        shutil.rmtree(downloads)
        shutil.rmtree(raw_dataset)

        manifest = {
            "schema_version": "fmmt.transformer_assets.v1",
            "xet_disabled": True,
            "config_sha256": sha256_file(config_path),
            "source_provenance": source_provenance,
            "assets": {
                str(model_lock["asset_name"]): {
                    "path": "model",
                    "kind": "directory",
                    "repository_id": str(model_lock["repository_id"]),
                    "requested_revision": str(model_lock["resolved_revision"]),
                    "resolved_revision": str(model_lock["resolved_revision"]),
                    "sha256": sha256_tree(model_destination),
                },
                str(dataset_lock["asset_name"]): {
                    "path": "dataset",
                    "kind": "directory",
                    "repository_id": str(dataset_lock["repository_id"]),
                    "requested_revision": str(dataset_lock["resolved_revision"]),
                    "resolved_revision": str(dataset_lock["resolved_revision"]),
                    "sha256": sha256_tree(stage / "dataset"),
                    "canonical_file_sha256": derivation["canonical_file_sha256"],
                    "source_tree_sha256": source_dataset_hash,
                },
            },
            "dependencies": {
                name: _package_version(name)
                for name in ("numpy", "huggingface_hub", "pyarrow")
            },
        }
        (stage / "ASSET_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output / "ASSET_MANIFEST.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    manifest = acquire_assets(
        config_path=arguments.config,
        output_root=arguments.output_root,
        source_manifest=arguments.source_manifest,
    )
    print(json.dumps({"status": "PASS", "manifest": str(manifest)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
