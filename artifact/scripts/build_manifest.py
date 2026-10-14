#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for the public artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"
SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
}
SKIP_NAMES = {".DS_Store", "SHA256SUMS"}
PUBLIC_EXCLUSIONS = {
    "provenance/crossover/HANDOFF_SHA256SUMS",
    "provenance/crossover/SHA256SUMS",
    "provenance/crossover/crossover_source_closure.tar.gz",
}

rows: list[str] = []
for path in sorted(ROOT.rglob("*")):
    rel = path.relative_to(ROOT)
    relative = rel.as_posix()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.name in SKIP_NAMES
        or path.name.startswith("._")
        or any(part in SKIP_PARTS or part.endswith(".egg-info") for part in rel.parts)
        or relative in PUBLIC_EXCLUSIONS
        or path.suffix in {".pyc", ".pyo", ".log", ".pid"}
    ):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(f"{digest}  {relative}")
OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Wrote {len(rows)} entries to {OUTPUT.name}")
