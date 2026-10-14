#!/usr/bin/env python3
"""Fail on identity, machine-path, credential, symlink, or release-clutter leaks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
MANIFEST_ROW = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
TEXT_LIMIT = 20 * 1024 * 1024

parser = argparse.ArgumentParser()
parser.add_argument(
    "--private-token",
    action="append",
    default=[],
    help="release-local identity/host token; may be repeated",
)
parser.add_argument(
    "--private-token-file",
    type=Path,
    help="external newline-delimited private-token checklist",
)
args = parser.parse_args()

private_tokens = list(args.private_token)
if args.private_token_file is not None:
    private_tokens.extend(
        line.strip()
        for line in args.private_token_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
if any(len(token.strip()) < 4 for token in private_tokens):
    parser.error("private tokens must contain at least four characters")

# Machine-path patterns are safe to publish. Author, account, and host names
# belong in an external release checklist supplied with the options above.
forbidden = [
    "/home" + "2/",
    "/" + "Users/",
    "/scr" + "atch/",
    *sorted({token.strip() for token in private_tokens if token.strip()}),
]
credential_patterns = [
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY", re.I),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]
clutter_suffixes = (".pyc", ".pyo", ".log", ".pid", ".tar.gz")
clutter_fragments = ("failure", "handoff", "launcher")
disclosure_allowlist = {
    "docs/POST_FAILURE_DROP_LAST_AUDIT_AMENDMENT.md",
}

problems: list[str] = []


def inspect_text(text: str, rel: Path, *, location: str = "") -> None:
    low = text.lower()
    where = f"{rel}{location}"
    for token in forbidden:
        if token.lower() in low:
            problems.append(f"private identity/path token in {where}: {token!r}")
    for pattern in credential_patterns:
        if pattern.search(text):
            problems.append(f"credential pattern in {where}: {pattern.pattern}")


members: list[tuple[Path, Path]] = []
seen: set[PurePosixPath] = set()
manifest_lines = MANIFEST.read_text(encoding="utf-8").splitlines()
for line_number, line in enumerate(manifest_lines, start=1):
    match = MANIFEST_ROW.fullmatch(line)
    if match is None:
        problems.append(f"malformed SHA256SUMS row {line_number}")
        continue
    expected, relative_text = match.groups()
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_text
        or any(part in {"", ".", "..", ".git"} for part in relative.parts)
        or relative in seen
    ):
        problems.append(f"unsafe or duplicate manifest path: {relative_text!r}")
        continue
    seen.add(relative)
    path = ROOT.joinpath(*relative.parts)
    rel = Path(relative.as_posix())
    if path.is_symlink() or not path.is_file():
        problems.append(f"missing, nonregular, or symbolic manifest member: {rel}")
        continue
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        problems.append(f"manifest digest mismatch: {rel}")
        continue
    members.append((path, rel))

if [rel.as_posix() for _, rel in members] != sorted(rel.as_posix() for _, rel in members):
    problems.append("SHA256SUMS paths are not in canonical sorted order")

for path, rel in members:
    inspect_text(rel.as_posix(), rel, location=":path")
    lowered = path.name.lower()
    if (
        rel.as_posix() not in disclosure_allowlist
        and (lowered.endswith(clutter_suffixes) or any(x in lowered for x in clutter_fragments))
    ):
        problems.append(f"release clutter: {rel}")
    if path.stat().st_size > 95 * 1024 * 1024:
        problems.append(f"file exceeds 95 MiB: {rel}")
    if path.suffix == ".gz":
        try:
            with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    inspect_text(line, rel, location=f":{line_number}")
        except OSError as exc:
            problems.append(f"unreadable gzip member: {rel}: {exc}")
        continue
    if path.stat().st_size > TEXT_LIMIT:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    inspect_text(text, rel)

if problems:
    print("ANONYMITY AUDIT: FAIL", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    raise SystemExit(1)
print("ANONYMITY AUDIT: PASS")
