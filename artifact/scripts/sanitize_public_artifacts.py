#!/usr/bin/env python3
"""Create portable public JSON artifacts without changing scientific values.

Only ``results/`` and ``provenance/`` are in scope.  String values and string
keys are rewritten when they expose a machine-local absolute path, host,
network address, email address, or caller-supplied private token.  Numbers,
booleans, nulls, list structure, and object values are left unchanged.  A
deterministic manifest records the source/public SHA-256 lineage and
replacement counts. Private tokens are supplied at release time and are never
embedded in the distributed source.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "SANITIZATION_MANIFEST.json"
SCOPES = (ROOT / "results", ROOT / "provenance")
FORMATS = (".jsonl.gz", ".json.gz", ".jsonl", ".json")


_ROOT_NAMES = (
    "Users",
    "home",
    "home2",
    "scratch",
    "tmp",
    "private",
    "var",
    "mnt",
    "data",
    "opt",
    "usr",
)

_ROOT_ALTERNATION = "|".join(re.escape(value) for value in _ROOT_NAMES)
_EXACT_ABSOLUTE = re.compile(rf"^/(?:{_ROOT_ALTERNATION})(?:/|$)")
_INLINE_ABSOLUTE = re.compile(
    rf"(?<![A-Za-z0-9])/(?:{_ROOT_ALTERNATION})(?:/[^\s\"'<>|]+)?"
)
_WINDOWS_ABSOLUTE = re.compile(
    rf"(?i)\b[A-Z]:\\(?:{re.escape(_ROOT_NAMES[0])})\\[^\s\"'<>|]+"
)
_IP_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = ()
_TRAILING_PUNCTUATION = ",.;:)]}"


def configure_private_tokens(tokens: list[str]) -> None:
    """Configure release-local redactions without publishing the checklist."""

    normalized = sorted({token.strip() for token in tokens if token.strip()})
    if any(len(token) < 4 for token in normalized):
        raise ValueError("private tokens must contain at least four characters")
    global _IDENTITY_PATTERNS
    _IDENTITY_PATTERNS = tuple(
        re.compile(re.escape(token), re.I) for token in normalized
    )


def private_tokens_from_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable_leaf(raw: str) -> str:
    value = raw
    suffix = ""
    while value and value[-1] in _TRAILING_PUNCTUATION:
        suffix = value[-1] + suffix
        value = value[:-1]
    value = value.replace("\\", "/").rstrip("/")
    leaf = value.rsplit("/", 1)[-1] if value else "root"
    for pattern in _IDENTITY_PATTERNS:
        leaf = pattern.sub("anonymous", leaf)
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip("-.") or "artifact"
    return f"artifact://local/{leaf}{suffix}"


def sanitize_string(value: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    stripped = value.strip()
    if "\n" not in value and (
        _EXACT_ABSOLUTE.match(stripped) or _WINDOWS_ABSOLUTE.match(stripped)
    ):
        prefix = value[: len(value) - len(value.lstrip())]
        suffix = value[len(value.rstrip()) :]
        return prefix + _portable_leaf(stripped) + suffix, Counter({"absolute_path": 1})

    def replace_path(match: re.Match[str]) -> str:
        counts["absolute_path"] += 1
        return _portable_leaf(match.group(0))

    result = _INLINE_ABSOLUTE.sub(replace_path, value)
    result = _WINDOWS_ABSOLUTE.sub(replace_path, result)

    def replace_ip(_: re.Match[str]) -> str:
        counts["network_address"] += 1
        return "artifact-host"

    result = _IP_ADDRESS.sub(replace_ip, result)

    def replace_email(_: re.Match[str]) -> str:
        counts["email_address"] += 1
        return "artifact://identity/anonymous"

    result = _EMAIL.sub(replace_email, result)
    for pattern in _IDENTITY_PATTERNS:
        result, replacements = pattern.subn("anonymous", result)
        if replacements:
            counts["identity_or_host"] += replacements
    return result, counts


def sanitize_value(value: Any) -> tuple[Any, Counter[str]]:
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        output = []
        counts: Counter[str] = Counter()
        for item in value:
            sanitized, item_counts = sanitize_value(item)
            output.append(sanitized)
            counts.update(item_counts)
        return output, counts
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        counts = Counter()
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object key is not a string")
            public_key, key_counts = sanitize_string(key)
            if public_key in output:
                raise ValueError(f"sanitization would collide object keys: {public_key!r}")
            public_item, item_counts = sanitize_value(item)
            output[public_key] = public_item
            counts.update(key_counts)
            counts.update(item_counts)
        return output, counts
    return value, Counter()


def numeric_signature(value: Any) -> list[tuple[str, str]]:
    """Record every non-string scalar in traversal order."""

    output: list[tuple[str, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, bool):
            output.append(("bool", repr(item)))
        elif item is None:
            output.append(("null", "null"))
        elif isinstance(item, int):
            output.append(("int", str(item)))
        elif isinstance(item, float):
            output.append(("float", item.hex()))

    visit(value)
    return output


def artifact_format(path: Path) -> str | None:
    name = path.name.lower()
    return next((suffix for suffix in FORMATS if name.endswith(suffix)), None)


def decode_payload(path: Path, raw: bytes, kind: str) -> tuple[Any, int]:
    if kind.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8")
    if kind.startswith(".jsonl"):
        records: list[Any | None] = []
        count = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                records.append(None)
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            count += 1
        return records, count
    try:
        return json.loads(text), 1
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}: {error}") from error


def encode_payload(value: Any, kind: str) -> bytes:
    if kind.startswith(".jsonl"):
        lines = [
            "" if record is None else json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for record in value
        ]
        plain = ("\n".join(lines) + "\n").encode("utf-8")
    else:
        plain = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if not kind.endswith(".gz"):
        return plain
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(plain)
    return buffer.getvalue()


def artifact_paths() -> list[Path]:
    paths: list[Path] = []
    for scope in SCOPES:
        if not scope.exists():
            continue
        for path in scope.rglob("*"):
            if not path.is_file() or path == MANIFEST:
                continue
            if artifact_format(path) is not None:
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def load_previous_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST.is_file():
        return {}
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {record["path"]: record for record in payload.get("files", [])}


def build_plan() -> tuple[list[tuple[Path, bytes]], dict[str, Any]]:
    previous = load_previous_manifest()
    writes: list[tuple[Path, bytes]] = []
    records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for path in artifact_paths():
        relative = path.relative_to(ROOT).as_posix()
        kind = artifact_format(path)
        assert kind is not None
        raw = path.read_bytes()
        current_sha = sha256_bytes(raw)
        parsed, record_count = decode_payload(path, raw, kind)
        public_value, current_counts = sanitize_value(parsed)
        if numeric_signature(parsed) != numeric_signature(public_value):
            raise RuntimeError(f"numeric/scientific scalar changed: {relative}")
        # Preserve already-portable plain artifacts byte for byte.  Retained
        # gzip artifacts are always normalized to canonical JSON plus a fixed
        # gzip header, even when no sensitive string occurs in their payload.
        public = (
            encode_payload(public_value, kind)
            if sum(current_counts.values()) or kind.endswith(".gz")
            else raw
        )
        public_sha = sha256_bytes(public)

        old = previous.get(relative)
        already_public = old is not None and old.get("public_sha256") == current_sha
        if already_public:
            original_sha = str(old["original_sha256"])
            original_bytes = int(old["original_bytes"])
            replacement_counts = Counter(old.get("replacements", {}))
        else:
            original_sha = current_sha
            original_bytes = len(raw)
            replacement_counts = current_counts
        if public != raw:
            writes.append((path, public))

        totals.update(replacement_counts)
        records.append(
            {
                "path": relative,
                "format": kind.lstrip("."),
                "records": record_count,
                "original_bytes": original_bytes,
                "public_bytes": len(public),
                "original_sha256": original_sha,
                "public_sha256": public_sha,
                "changed": original_sha != public_sha,
                "replacements": dict(sorted(replacement_counts.items())),
            }
        )

    manifest = {
        "schema_version": "anonymous_artifact.sanitization.v1",
        "status": "PASS",
        "scope": ["results", "provenance"],
        "policy": {
            "strings_only": True,
            "numeric_and_scientific_scalars_preserved": True,
            "internal_recorded_hashes_preserved_as_execution_lineage": True,
            "modified_gzip_recompressed_deterministically": True,
            "omitted_gzip_artifacts": [],
        },
        "file_count": len(records),
        "changed_file_count": sum(bool(record["changed"]) for record in records),
        "replacement_totals": dict(sorted(totals.items())),
        "files": records,
    }
    return writes, manifest


def verify_no_remaining_leaks() -> None:
    findings: list[str] = []
    for path in artifact_paths():
        kind = artifact_format(path)
        assert kind is not None
        parsed, _ = decode_payload(path, path.read_bytes(), kind)

        def visit(value: Any) -> None:
            if isinstance(value, str):
                _, counts = sanitize_string(value)
                if counts:
                    findings.append(path.relative_to(ROOT).as_posix())
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(key)
                    visit(item)

        visit(parsed)
    if findings:
        raise RuntimeError("unsanitized artifact strings remain in: " + ", ".join(sorted(set(findings))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without modifying files")
    parser.add_argument(
        "--private-token",
        action="append",
        default=[],
        help="release-local identity/host token to redact; may be repeated",
    )
    parser.add_argument(
        "--private-token-file",
        type=Path,
        help="external newline-delimited private-token checklist",
    )
    args = parser.parse_args(argv)
    configure_private_tokens(
        [*args.private_token, *private_tokens_from_file(args.private_token_file)]
    )

    writes, manifest = build_plan()
    if args.check:
        if writes:
            raise SystemExit(f"sanitization required for {len(writes)} artifact files")
        verify_no_remaining_leaks()
        print(json.dumps({"status": "PASS", "files": manifest["file_count"]}, sort_keys=True))
        return 0

    for path, payload in writes:
        path.write_bytes(payload)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_no_remaining_leaks()
    print(
        json.dumps(
            {
                "status": "PASS",
                "files": manifest["file_count"],
                "changed_files": manifest["changed_file_count"],
                "replacements": manifest["replacement_totals"],
                "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
