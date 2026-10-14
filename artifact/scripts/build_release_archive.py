#!/usr/bin/env python3
"""Build a fail-closed deterministic public archive from ``SHA256SUMS``.

``SHA256SUMS`` is the sole payload allowlist. Every member is read once,
checked against its declared digest, and then written from those checked bytes.
The output must live outside the repository and must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
ROW = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
RESERVED_MEMBERS = frozenset({"SHA256SUMS"})


@dataclass(frozen=True)
class CheckedMember:
    relative: str
    payload: bytes
    sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_member(text: str) -> PurePosixPath:
    if not text or "\\" in text or any(ord(character) < 32 for character in text):
        raise RuntimeError(f"unsafe manifest path: {text!r}")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or relative.as_posix() != text
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ".git" in relative.parts
        or relative.as_posix() in RESERVED_MEMBERS
    ):
        raise RuntimeError(f"unsafe manifest path: {text!r}")
    return relative


def _checked_regular_file(root: Path, relative: PurePosixPath) -> Path:
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise RuntimeError(f"manifest member is symbolic: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"manifest member is missing: {relative}") from error
    if root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"manifest member is not a regular in-tree file: {relative}")
    return resolved


def load_manifest(
    *, root: Path = ROOT, manifest: Path = MANIFEST
) -> tuple[list[CheckedMember], bytes]:
    root = Path(root).resolve(strict=True)
    manifest = Path(manifest)
    if manifest.is_symlink():
        raise RuntimeError("SHA256SUMS must not be symbolic")
    manifest = manifest.resolve(strict=True)
    if manifest != root / "SHA256SUMS" or not manifest.is_file():
        raise RuntimeError("manifest must be the regular in-tree SHA256SUMS file")
    manifest_bytes = manifest.read_bytes()
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("SHA256SUMS is not UTF-8") from error

    rows: list[CheckedMember] = []
    seen: set[PurePosixPath] = set()
    ordered_paths: list[str] = []
    for line_number, raw in enumerate(manifest_text.splitlines(), start=1):
        match = ROW.fullmatch(raw)
        if match is None:
            raise RuntimeError(f"malformed SHA256SUMS row {line_number}")
        expected, rel_text = match.groups()
        relative = _relative_member(rel_text)
        if relative in seen:
            raise RuntimeError(f"duplicate manifest path: {rel_text!r}")
        seen.add(relative)
        path = _checked_regular_file(root, relative)
        payload = path.read_bytes()
        observed = sha256_bytes(payload)
        if observed != expected:
            raise RuntimeError(
                f"digest mismatch for {relative}: {observed} != {expected}"
            )
        ordered_paths.append(relative.as_posix())
        rows.append(CheckedMember(relative.as_posix(), payload, expected))
    if not rows:
        raise RuntimeError("SHA256SUMS is empty")
    if ordered_paths != sorted(ordered_paths):
        raise RuntimeError("SHA256SUMS paths are not in canonical sorted order")
    return rows, manifest_bytes


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _output_path(output: Path, *, root: Path) -> Path:
    raw = Path(output).expanduser()
    if raw.suffix.lower() != ".zip":
        raise RuntimeError("release output must use a .zip suffix")
    if raw.is_symlink() or raw.exists():
        raise RuntimeError(f"release output already exists or is symbolic: {raw}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    parent = raw.parent.resolve(strict=True)
    resolved = parent / raw.name
    if root == resolved or root in resolved.parents:
        raise RuntimeError("release output must be outside the repository")
    return resolved


def _verify_archive(
    archive_path: Path,
    *,
    rows: list[CheckedMember],
    manifest_bytes: bytes,
) -> None:
    expected_names = [row.relative for row in rows] + ["SHA256SUMS"]
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != expected_names or len(names) != len(set(names)):
            raise RuntimeError("release archive inventory differs from the allowlist")
        if archive.testzip() is not None:
            raise RuntimeError("release archive failed its CRC check")
        for row in rows:
            payload = archive.read(row.relative)
            if sha256_bytes(payload) != row.sha256:
                raise RuntimeError(f"archived digest mismatch for {row.relative}")
        if archive.read("SHA256SUMS") != manifest_bytes:
            raise RuntimeError("archived SHA256SUMS differs from the checked manifest")


def build(
    output: Path,
    *,
    root: Path = ROOT,
    manifest: Path = MANIFEST,
) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    output = _output_path(output, root=root)
    rows, manifest_bytes = load_manifest(root=root, manifest=manifest)

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for row in rows:
                archive.writestr(_zip_info(row.relative), row.payload)
            archive.writestr(_zip_info("SHA256SUMS"), manifest_bytes)
        _verify_archive(temporary, rows=rows, manifest_bytes=manifest_bytes)

        # Hard-link publication is atomic and fails rather than overwriting if
        # another process creates the requested output after the preflight.
        os.link(temporary, output)
    except FileExistsError as error:
        raise RuntimeError(f"release output already exists: {output}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "schema_version": 1,
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "payload_file_count": len(rows),
        "zip_member_count": len(rows) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
