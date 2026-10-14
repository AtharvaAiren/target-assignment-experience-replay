"""Leakage-resistant chronological next-token stream construction."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch

from .audit import NextTokenPacket


class TokenizerLike(Protocol):
    vocab_size: int

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class StreamExample:
    example_id: str
    timestamp_utc: str
    source: str
    input_ids: tuple[int, ...]
    target_id: int
    window_start: int
    target_position: int
    text_sha256: str


@dataclass(frozen=True)
class StreamBuild:
    warmup: tuple[StreamExample, ...]
    online: tuple[StreamExample, ...]
    manifest: dict[str, Any]


_PUBLISHED = re.compile(r"(?:published\s+on\s+)?(.+)", re.IGNORECASE)
_LICENSE_DASHES = str.maketrans({"\N{EN DASH}": "-", "\N{EM DASH}": "-"})
_CC_VERSIONS = ("1.0", "2.0", "2.5", "3.0", "4.0")


def _license_allowlist() -> frozenset[str]:
    """Exact normalized spellings accepted by the locked CC BY/BY-SA filter."""

    values: set[str] = set()
    for family, long_names in (
        ("by", ("attribution",)),
        (
            "by-sa",
            (
                "attribution-sharealike",
                "attribution sharealike",
                "attribution-share alike",
            ),
        ),
    ):
        for version in _CC_VERSIONS:
            urls = (
                f"https://creativecommons.org/licenses/{family}/{version}/",
                f"http://creativecommons.org/licenses/{family}/{version}/",
            )
            values.update({f"cc {family} {version}", f"cc-{family}-{version}"})
            for long_name in long_names:
                values.update(
                    {
                        f"creative commons {long_name} {version}",
                        f"creative commons {long_name} {version} international",
                    }
                )
                for url in urls:
                    values.update({url, f"creative commons - {long_name} - {url}"})
    return frozenset(values)


_ALLOWED_LICENSES = _license_allowlist()


def _strict_record_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_record_json(line: str, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_strict_record_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"invalid strict JSON at {context}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"record is not an object at {context}")
    return value


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing publication timestamp")
    text = value.strip()
    match = _PUBLISHED.fullmatch(text)
    if match:
        text = match.group(1).strip()
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is None:
        for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError(f"unparseable publication timestamp: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_allowed_license(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    label = re.sub(r"\s+", " ", value.translate(_LICENSE_DASHES).strip().lower())
    return label in _ALLOWED_LICENSES


def _stable_identity(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    parts = (
        str(record.get("source", "")).strip(),
        str(record.get("id", "")).strip(),
        str(metadata.get("url", "")).strip(),
    )
    if not any(parts):
        raise ValueError("record has no stable identity")
    return "\x1f".join(parts)


def _token_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    values = encoded.get("input_ids")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise RuntimeError("tokenizer returned more than one sequence")
        values = values[0]
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError("tokenizer did not return a flat integer input_ids list")
    vocab_size = int(getattr(tokenizer, "vocab_size"))
    if any(value < 0 or value >= vocab_size for value in values):
        raise RuntimeError("tokenizer emitted an out-of-vocabulary token ID")
    return values


def _deterministic_offset(
    *, stable_id: str, choices: int, salt: str
) -> int:
    if choices <= 0:
        raise ValueError("offset requires at least one valid window")
    # Content—including the held-out token—must not influence which position
    # is selected.  The content hash is recorded for provenance, not sampling.
    digest = hashlib.sha256(f"{salt}\0{stable_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % choices


def build_chronological_stream(
    records: Iterable[Mapping[str, Any]],
    tokenizer: TokenizerLike,
    *,
    context_length: int,
    warmup_examples: int,
    online_examples: int,
    earliest_utc: str,
    offset_salt: str,
) -> StreamBuild:
    """Create one fixed context and held-out next token per unique article.

    Selection, ordering, and window offsets do not accept a model seed.  The
    target is always token ``start + context_length`` and is never included in
    ``input_ids``.  Targets are therefore safe to permute while conditioning
    tokens remain fixed.
    """
    context_length = int(context_length)
    required = int(warmup_examples) + int(online_examples)
    if context_length < 2 or min(int(warmup_examples), int(online_examples)) < 1:
        raise ValueError("invalid stream sizes")
    earliest = _parse_timestamp(earliest_utc)
    candidates: list[tuple[datetime, str, str, str, str]] = []
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    counts = {
        "records_seen": 0,
        "rejected_timestamp": 0,
        "rejected_license": 0,
        "rejected_duplicate": 0,
        "rejected_empty": 0,
    }
    for record in records:
        counts["records_seen"] += 1
        try:
            timestamp = _parse_timestamp(record.get("created"))
        except ValueError:
            counts["rejected_timestamp"] += 1
            continue
        if timestamp < earliest:
            counts["rejected_timestamp"] += 1
            continue
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if not _is_allowed_license(metadata.get("license")):
            counts["rejected_license"] += 1
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            counts["rejected_empty"] += 1
            continue
        stable_id = _stable_identity(record)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if stable_id in seen_ids or text_hash in seen_text:
            counts["rejected_duplicate"] += 1
            continue
        seen_ids.add(stable_id)
        seen_text.add(text_hash)
        candidates.append(
            (timestamp, str(record.get("source", "")), stable_id, text_hash, text)
        )
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))

    examples: list[StreamExample] = []
    short = 0
    for timestamp, source, stable_id, text_hash, text in candidates:
        tokens = _token_ids(tokenizer, text)
        choices = len(tokens) - context_length
        if choices <= 0:
            short += 1
            continue
        start = _deterministic_offset(
            stable_id=stable_id,
            choices=choices,
            salt=str(offset_salt),
        )
        stop = start + context_length
        context = tuple(tokens[start:stop])
        target = int(tokens[stop])
        if len(context) != context_length or stop >= len(tokens):
            raise RuntimeError("next-token window construction failed")
        examples.append(
            StreamExample(
                example_id=hashlib.sha256(stable_id.encode("utf-8")).hexdigest(),
                timestamp_utc=timestamp.isoformat(),
                source=source,
                input_ids=context,
                target_id=target,
                window_start=start,
                target_position=stop,
                text_sha256=text_hash,
            )
        )
        if len(examples) == required:
            break
    if len(examples) != required:
        raise RuntimeError(
            f"only {len(examples)} eligible tokenized articles; {required} required"
        )
    identity_payload = [
        {
            "id": item.example_id,
            "timestamp": item.timestamp_utc,
            "context": list(item.input_ids),
            "target": item.target_id,
            "start": item.window_start,
            "target_position": item.target_position,
        }
        for item in examples
    ]
    stream_hash = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "fmmt.fixed_conditioning_stream.v1",
        **counts,
        "rejected_too_short_after_tokenization": short,
        "context_length": context_length,
        "warmup_examples": int(warmup_examples),
        "online_examples": int(online_examples),
        "one_window_per_article": True,
        "selection_depends_on_model_seed": False,
        "conditioning_tokens_are_never_permuted": True,
        "target_position_strictly_after_context": all(
            item.target_position == item.window_start + context_length for item in examples
        ),
        "stream_sha256": stream_hash,
        "first_timestamp_utc": examples[0].timestamp_utc,
        "last_timestamp_utc": examples[-1].timestamp_utc,
    }
    warmup = tuple(examples[: int(warmup_examples)])
    online = tuple(examples[int(warmup_examples) :])
    return StreamBuild(warmup=warmup, online=online, manifest=manifest)


def packetize(
    examples: Sequence[StreamExample], packet_size: int
) -> Iterator[NextTokenPacket]:
    packet_size = int(packet_size)
    if packet_size < 2:
        raise ValueError("packet size must be at least two")
    if len(examples) % packet_size:
        raise ValueError("stream size must be divisible by packet size")
    for packet_number, start in enumerate(range(0, len(examples), packet_size)):
        values = examples[start : start + packet_size]
        lengths = {len(item.input_ids) for item in values}
        if len(lengths) != 1:
            raise RuntimeError("packet has unequal context lengths")
        yield NextTokenPacket(
            input_ids=torch.tensor([item.input_ids for item in values], dtype=torch.long),
            targets=np.asarray([item.target_id for item in values], dtype=np.int64),
            ids=tuple(item.example_id for item in values),
            timestamps=tuple(item.timestamp_utc for item in values),
            packet_number=packet_number,
        )


def iter_json_records(root: str | Path) -> Iterator[dict[str, Any]]:
    """Read a pinned file or snapshot containing JSONL/JSONL.GZ shards."""
    path = Path(root)
    files = [path] if path.is_file() else sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.name.endswith((".jsonl", ".jsonl.gz", ".json.gz"))
    )
    if not files:
        raise RuntimeError(f"no JSON record shards found under {path}")
    for file_path in files:
        opener = gzip.open if file_path.name.endswith(".gz") else open
        with opener(file_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield _strict_record_json(
                    line,
                    context=f"{file_path}:{line_number}",
                )
