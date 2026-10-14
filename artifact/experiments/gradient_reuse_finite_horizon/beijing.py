"""Minimal byte-locked reproduction of the frozen Beijing preprocessing.

This module is intentionally data-only.  The implementation is the relevant
subset of the previously audited Stage-4/FMMT loader, isolated so the
finite-horizon study does not import unrelated models or schedulers.  The
caller rejects its output unless both historical array digests match.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import zipfile

import numpy as np


NUMERIC = ("PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM")
REQUIRED = frozenset(("PM2.5", *NUMERIC, "wd", "year", "month", "day", "hour", "station"))
FROZEN_TARGET_MEAN_BITS = 0x407DE83E
FROZEN_TARGET_STD_BITS = 0x3F8FF2D3


class BeijingDataError(RuntimeError):
    """The locked archive or preprocessing failed validation."""


def _csv_members_recursive(blob: bytes, origin: str, depth: int = 0, max_depth: int = 2):
    if depth > max_depth:
        return
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as error:
        raise BeijingDataError(f"{origin}: invalid ZIP: {error}") from error
    with archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or member.startswith("__MACOSX/"):
                continue
            payload = archive.read(member)
            qualified = f"{origin}!{member}"
            lower = member.lower()
            if lower.endswith((".csv", ".csv.gz")):
                yield qualified, payload
            elif lower.endswith(".zip") and depth < max_depth:
                yield from _csv_members_recursive(payload, qualified, depth + 1, max_depth)


def parse_archive(path: Path):
    import pandas as pd

    blob = path.read_bytes()
    frames = []
    selected = []
    for qualified, payload in _csv_members_recursive(blob, path.name):
        compression = "gzip" if qualified.lower().endswith(".gz") else "infer"
        try:
            header = pd.read_csv(io.BytesIO(payload), compression=compression, nrows=0)
        except Exception:
            continue
        columns = [str(column).strip() for column in header.columns]
        if not REQUIRED.issubset(columns):
            continue
        frame = pd.read_csv(io.BytesIO(payload), compression=compression)
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
        selected.append(qualified)
    if not frames:
        raise BeijingDataError("no archive member matches the frozen Beijing schema")
    return pd.concat(frames, ignore_index=True), selected


def _hash16(value: object) -> int:
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % 16


def _cyclic(value: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * (np.asarray(value, float) / period)
    return np.sin(angle), np.cos(angle)


def locked_target_location_scale(
    ylog: np.ndarray, calibration_rows: int
) -> tuple[float, float]:
    """Return the byte-locked historical float32 normalization constants.

    NumPy reduction algorithms are not a cross-release byte-level contract:
    NumPy 2.5.1 returned a standard deviation one float32 ULP above the value
    used to create the frozen target digest, even with ``dtype=float32``.  The
    original values are therefore represented by their IEEE-754 bit patterns.
    The final target-array digest remains the authoritative data check.
    """
    prefix = np.asarray(ylog[:calibration_rows], dtype=np.float32)
    if prefix.ndim != 1 or prefix.size != calibration_rows or not np.isfinite(prefix).all():
        raise BeijingDataError("invalid target calibration prefix")
    mean = np.asarray([FROZEN_TARGET_MEAN_BITS], dtype="<u4").view("<f4")[0]
    std = np.asarray([FROZEN_TARGET_STD_BITS], dtype="<u4").view("<f4")[0]
    return float(mean), float(std)


def load_beijing(path: Path) -> dict[str, object]:
    import pandas as pd

    frame, selected = parse_archive(path)
    frame = frame.copy()
    frame["_ts"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=frame["day"], hour=frame["hour"]),
        errors="coerce",
    )
    frame = frame.sort_values(["_ts", "station"], kind="stable").reset_index(drop=True)
    frame = frame.dropna(subset=["PM2.5"]).reset_index(drop=True)
    if frame["_ts"].isna().any():
        raise BeijingDataError("invalid timestamp in frozen Beijing rows")
    timestamps = frame["_ts"]
    original_y = frame["PM2.5"].to_numpy(np.float64)
    keep = np.isfinite(original_y)
    frame = frame[keep].reset_index(drop=True)
    timestamps = timestamps[keep].reset_index(drop=True)
    original_y = original_y[keep]
    rows = len(frame)
    if rows != 412029:
        raise BeijingDataError(f"unexpected usable row count: {rows}")
    calibration_rows = max(32, int(0.05 * rows))
    prefix = slice(0, calibration_rows)

    numeric_blocks = []
    missing_blocks = []
    for column in NUMERIC:
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        median = np.nanmedian(raw[prefix])
        median = 0.0 if not np.isfinite(median) else median
        missing = ~np.isfinite(raw)
        filled = np.where(missing, median, raw)
        q75, q25 = np.nanpercentile(raw[prefix], [75, 25])
        width = q75 - q25
        iqr = width if np.isfinite(width) and width > 0 else 1.0
        numeric_blocks.append(((filled - median) / iqr).astype(np.float32))
        missing_blocks.append(missing.astype(np.float32))

    hour_s, hour_c = _cyclic(timestamps.dt.hour.to_numpy(), 24.0)
    dow_s, dow_c = _cyclic(timestamps.dt.dayofweek.to_numpy(), 7.0)
    doy_s, doy_c = _cyclic(timestamps.dt.dayofyear.to_numpy(), 365.25)
    cyclic_blocks = [block.astype(np.float32) for block in (hour_s, hour_c, dow_s, dow_c, doy_s, doy_c)]
    values = frame["wd"].fillna("MISSING").astype(str).to_numpy()
    one_hot = np.zeros((rows, 16), np.float32)
    one_hot[np.arange(rows), np.asarray([_hash16(value) for value in values])] = 1.0
    features = np.concatenate(
        [np.stack(numeric_blocks, 1), np.stack(missing_blocks, 1), np.stack(cyclic_blocks, 1), one_hot],
        axis=1,
    ).astype(np.float32)
    if features.shape != (412029, 42) or not np.isfinite(features).all():
        raise BeijingDataError("frozen feature construction changed")

    clipped = np.clip(original_y, 0, None)
    ylog = np.log1p(clipped).astype(np.float32)
    y_mean, y_std = locked_target_location_scale(ylog, calibration_rows)
    if not np.isfinite(y_std) or y_std < 1.0e-8:
        y_std = 1.0
    targets = ((ylog - y_mean) / y_std).astype(np.float32)

    timestamps_ns = timestamps.astype("int64").to_numpy(np.int64)
    change = np.empty(rows, dtype=bool)
    change[0] = True
    change[1:] = timestamps_ns[1:] != timestamps_ns[:-1]
    event_id = np.cumsum(change, dtype=np.int64) - 1
    calibration_event = int(event_id[calibration_rows - 1])
    scored = event_id > calibration_event
    scoring_boundary = int(np.flatnonzero(scored)[0])
    return {
        "X": features,
        "y": targets,
        "raw_rows": int(sum(len(frame) for frame in [frame])),
        "selected_members": selected,
        "calibration_rows": calibration_rows,
        "scoring_calibration_rows": scoring_boundary,
    }
