"""Offline, content-addressed external-asset contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class AssetContractError(RuntimeError):
    """An external asset is absent, mutable, or differs from its lock."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AssetContractError(f"duplicate JSON key in asset manifest: {key!r}")
        value[key] = item
    return value


@dataclass(frozen=True)
class ValidatedAsset:
    name: str
    path: Path
    kind: str
    repository_id: str
    resolved_revision: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash relative names, sizes, and bytes; reject symlink indirection."""
    digest = hashlib.sha256()
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise AssetContractError(f"asset directory is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise AssetContractError(f"asset trees may not contain symlinks: {path}")
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AssetContractError(
                    f"non-finite JSON constant in asset manifest: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AssetContractError(f"cannot read asset manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssetContractError("asset manifest must be a JSON object")
    return value


def validate_asset_manifest(
    manifest_path: str | Path,
    requirements: Mapping[str, Mapping[str, Any]],
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, ValidatedAsset]:
    """Validate exact local snapshots without making a network request.

    ``requirements`` comes from the experiment lock.  The separate acquisition
    manifest must bind each local path to the same repository and full 40-hex
    resolved revision, then authenticate its current bytes with SHA-256.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    payload = _read_object(manifest)
    if payload.get("schema_version") != "fmmt.transformer_assets.v1":
        raise AssetContractError("unsupported asset-manifest schema")
    if expected_config_sha256 is not None:
        if not _SHA256.fullmatch(expected_config_sha256):
            raise AssetContractError("caller supplied an invalid config SHA-256")
        if payload.get("config_sha256") != expected_config_sha256:
            raise AssetContractError("asset manifest was created from a different lock")
    records = payload.get("assets")
    if not isinstance(records, dict):
        raise AssetContractError("asset manifest has no assets object")

    validated: dict[str, ValidatedAsset] = {}
    for role, requirement in requirements.items():
        name = str(requirement.get("asset_name", ""))
        record = records.get(name)
        if not name or not isinstance(record, dict):
            raise AssetContractError(f"missing required {role} asset {name!r}")
        repository_id = str(record.get("repository_id", ""))
        revision = str(record.get("resolved_revision", ""))
        requested_revision = str(record.get("requested_revision", ""))
        expected_revision = str(requirement.get("resolved_revision", ""))
        if repository_id != str(requirement.get("repository_id", "")):
            raise AssetContractError(f"repository mismatch for {name}")
        if (
            not _REVISION.fullmatch(revision)
            or revision != expected_revision
            or requested_revision != expected_revision
        ):
            raise AssetContractError(f"resolved revision mismatch for {name}")
        kind = str(record.get("kind", ""))
        if kind != str(requirement.get("kind", "")) or kind not in {"file", "directory"}:
            raise AssetContractError(f"asset kind mismatch for {name}")
        claimed_hash = str(record.get("sha256", ""))
        if not _SHA256.fullmatch(claimed_hash):
            raise AssetContractError(f"invalid SHA-256 for {name}")
        raw_path = Path(str(record.get("path", ""))).expanduser()
        path = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
        path = path.resolve()
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            raise AssetContractError(f"missing local {kind} for {name}: {path}")
        observed_hash = sha256_file(path) if kind == "file" else sha256_tree(path)
        if observed_hash != claimed_hash:
            raise AssetContractError(
                f"content hash mismatch for {name}: expected {claimed_hash}, got {observed_hash}"
            )
        validated[role] = ValidatedAsset(
            name=name,
            path=path,
            kind=kind,
            repository_id=repository_id,
            resolved_revision=revision,
            sha256=observed_hash,
        )
    return validated
