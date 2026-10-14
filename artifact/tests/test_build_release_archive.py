"""Fail-closed tests for the public release-archive builder."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts import build_release_archive as release


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "nested").mkdir()
    (root / "nested/b.txt").write_bytes(b"beta\n")
    manifest = root / "SHA256SUMS"
    alpha_sha256 = _digest(b"alpha\n")
    beta_sha256 = _digest(b"beta\n")
    manifest.write_text(
        f"{alpha_sha256}  a.txt\n"
        f"{beta_sha256}  nested/b.txt\n",
        encoding="utf-8",
    )
    return root, manifest


def test_build_is_deterministic_and_contains_only_checked_members(tmp_path: Path) -> None:
    root, manifest = _tree(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = release.build(first, root=root, manifest=manifest)
    two = release.build(second, root=root, manifest=manifest)

    assert first.read_bytes() == second.read_bytes()
    assert one["archive_sha256"] == two["archive_sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "nested/b.txt", "SHA256SUMS"]
        assert archive.read("a.txt") == b"alpha\n"
        assert archive.read("nested/b.txt") == b"beta\n"
        assert archive.read("SHA256SUMS") == manifest.read_bytes()


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "/absolute",
        r"C:\\escape",
        "SHA256SUMS",
        ".git/config",
        "a//b",
        "a/./b",
        "line\tbreak",
    ],
)
def test_manifest_rejects_unsafe_member_names(tmp_path: Path, relative: str) -> None:
    root, manifest = _tree(tmp_path)
    manifest.write_text(f"{'0' * 64}  {relative}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe manifest path"):
        release.load_manifest(root=root, manifest=manifest)


def test_manifest_rejects_duplicates_unsorted_rows_and_stale_bytes(tmp_path: Path) -> None:
    root, manifest = _tree(tmp_path)
    alpha_sha256 = _digest(b"alpha\n")
    beta_sha256 = _digest(b"beta\n")
    row = f"{alpha_sha256}  a.txt\n"
    manifest.write_text(row + row, encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        release.load_manifest(root=root, manifest=manifest)

    manifest.write_text(
        f"{beta_sha256}  nested/b.txt\n{row}", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="sorted"):
        release.load_manifest(root=root, manifest=manifest)

    manifest.write_text(
        f"{alpha_sha256}  a.txt\n"
        f"{beta_sha256}  nested/b.txt\n",
        encoding="utf-8",
    )
    (root / "a.txt").write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        release.load_manifest(root=root, manifest=manifest)


def test_manifest_rejects_symbolic_payload(tmp_path: Path) -> None:
    root, manifest = _tree(tmp_path)
    target = root / "a.txt"
    target.unlink()
    target.symlink_to(root / "nested/b.txt")
    beta_sha256 = _digest(b"beta\n")
    manifest.write_text(
        f"{beta_sha256}  a.txt\n"
        f"{beta_sha256}  nested/b.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="symbolic"):
        release.load_manifest(root=root, manifest=manifest)


def test_builder_refuses_output_collision_or_in_tree_output(tmp_path: Path) -> None:
    root, manifest = _tree(tmp_path)
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"do not overwrite")
    with pytest.raises(RuntimeError, match="already exists"):
        release.build(existing, root=root, manifest=manifest)
    assert existing.read_bytes() == b"do not overwrite"

    with pytest.raises(RuntimeError, match="outside the repository"):
        release.build(root / "release.zip", root=root, manifest=manifest)


def test_builder_refuses_symbolic_output(tmp_path: Path) -> None:
    root, manifest = _tree(tmp_path)
    target = tmp_path / "target.zip"
    target.write_bytes(b"do not overwrite")
    symbolic = tmp_path / "symbolic.zip"
    symbolic.symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic"):
        release.build(symbolic, root=root, manifest=manifest)
    assert target.read_bytes() == b"do not overwrite"
