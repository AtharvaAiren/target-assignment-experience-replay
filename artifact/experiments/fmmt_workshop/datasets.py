"""Chronological, calibration-only loaders for the workshop breadth streams."""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.fmmt.common import REPO, load_regression_stream


EXTRA_SPECS: dict[str, dict[str, Any]] = {
    "APPLIANCES": {
        "uci_id": 374,
        "url": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        "page": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        "license": "CC BY 4.0",
        "timestamp": "date",
        "target": "Appliances",
        "required": ["date", "Appliances", "T1", "RH_1"],
        "exclude": ["rv1", "rv2"],
    },
    "TETOUAN": {
        "uci_id": 849,
        "url": "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip",
        "page": "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city",
        "license": "CC BY 4.0",
        "timestamp": "DateTime",
        "target": "Zone 1 Power Consumption",
        "required": [
            "DateTime",
            "Temperature",
            "Humidity",
            "Wind Speed",
            "Zone 1 Power Consumption",
        ],
        "exclude_prefixes": ["Zone 2", "Zone 3"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive(name: str) -> tuple[Path, dict[str, Any]]:
    spec = EXTRA_SPECS[name]
    directory = REPO / "data_raw" / "uci"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.lower()}_workshop.zip"
    if not path.exists():
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "Mozilla/5.0 fmmt-workshop-audit"}
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    manifest = {
        "dataset": name,
        "uci_id": spec["uci_id"],
        "source_url": spec["url"],
        "dataset_page": spec["page"],
        "license": spec["license"],
        "sha256": _sha256(path),
        "zip_bytes": path.stat().st_size,
    }
    return path, manifest


def _read_matching_csv(path: Path, required: list[str]):
    import pandas as pd

    required_set = {str(value).strip() for value in required}
    matches: list[tuple[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or not member.lower().endswith((".csv", ".txt")):
                continue
            payload = archive.read(member)
            parsed = None
            for encoding in ("utf-8", "latin1"):
                for separator in (",", ";"):
                    try:
                        candidate = pd.read_csv(
                            io.BytesIO(payload), encoding=encoding, sep=separator
                        )
                    except Exception:
                        continue
                    candidate.columns = [str(column).strip() for column in candidate.columns]
                    if required_set.issubset(candidate.columns):
                        parsed = candidate
                        break
                if parsed is not None:
                    break
            if parsed is not None:
                matches.append((member, parsed))
    if len(matches) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one matching CSV, found "
            f"{[name for name, _ in matches]} for columns {sorted(required_set)}"
        )
    return matches[0]


def _cyclic(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / float(period)
    return np.sin(angle), np.cos(angle)


def _from_frame(name: str, frame: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    spec = EXTRA_SPECS[name]
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    timestamp = pd.to_datetime(data[spec["timestamp"]], errors="coerce", dayfirst=False)
    target = pd.to_numeric(data[spec["target"]], errors="coerce")
    valid = timestamp.notna() & target.notna() & np.isfinite(target.to_numpy(float))
    data = data.loc[valid].copy()
    data["_timestamp"] = timestamp.loc[valid]
    data["_target"] = target.loc[valid].astype(float)
    data = data.sort_values("_timestamp", kind="stable").reset_index(drop=True)
    if len(data) < 512:
        raise RuntimeError(f"{name}: too few usable chronological rows ({len(data)})")

    candidate_columns = []
    excluded = {spec["timestamp"], spec["target"], *spec.get("exclude", [])}
    prefixes = tuple(spec.get("exclude_prefixes", ()))
    for column in data.columns:
        if column in excluded or column.startswith(prefixes) or column.startswith("_"):
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().sum() >= max(32, int(0.8 * len(data))):
            candidate_columns.append(column)
            data[column] = numeric
    if not candidate_columns:
        raise RuntimeError(f"{name}: no usable numeric predictors")

    n = len(data)
    row_calibration = max(32, int(round(0.05 * n)))
    timestamps_ns = data["_timestamp"].astype("int64").to_numpy(np.int64)
    event_change = np.r_[True, timestamps_ns[1:] != timestamps_ns[:-1]]
    event_id = np.cumsum(event_change, dtype=np.int64) - 1
    calibration_event = int(event_id[min(row_calibration - 1, n - 1)])
    scored_mask = event_id > calibration_event
    if not scored_mask.any():
        raise RuntimeError(f"{name}: calibration consumes the complete stream")
    cal_n = int(np.flatnonzero(scored_mask)[0])

    raw_features = data[candidate_columns].to_numpy(dtype=np.float64)
    calibration = raw_features[:cal_n]
    medians = np.nanmedian(calibration, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    q25 = np.nanpercentile(calibration, 25, axis=0)
    q75 = np.nanpercentile(calibration, 75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    missing = ~np.isfinite(raw_features)
    filled = np.where(missing, medians[None, :], raw_features)
    normalized = np.clip((filled - medians[None, :]) / scale[None, :], -20.0, 20.0)
    features = [normalized]
    feature_names = [f"scaled:{column}" for column in candidate_columns]
    if missing.any():
        used_missing = missing.any(axis=0)
        features.append(missing[:, used_missing].astype(np.float64))
        feature_names.extend(
            f"missing:{column}"
            for column, used in zip(candidate_columns, used_missing)
            if used
        )

    ts = data["_timestamp"].dt
    for label, values, period in (
        ("hour", ts.hour.to_numpy(), 24.0),
        ("weekday", ts.dayofweek.to_numpy(), 7.0),
        ("yearday", ts.dayofyear.to_numpy(), 366.0),
    ):
        sine, cosine = _cyclic(values, period)
        features.extend([sine[:, None], cosine[:, None]])
        feature_names.extend([f"time:{label}:sin", f"time:{label}:cos"])
    X = np.concatenate(features, axis=1).astype(np.float32)

    raw_y = data["_target"].to_numpy(np.float64)
    transform = "log1p" if float(np.min(raw_y)) >= 0 else "identity"
    y_transformed = np.log1p(raw_y) if transform == "log1p" else raw_y
    y_mean = float(np.mean(y_transformed[:cal_n]))
    y_std = float(np.std(y_transformed[:cal_n], ddof=1))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    y = ((y_transformed - y_mean) / y_std).astype(np.float32)

    iso = data["_timestamp"].dt.isocalendar()
    week_key = (iso.year.to_numpy(np.int64) * 100 + iso.week.to_numpy(np.int64))
    ordered_weeks = {int(value): index for index, value in enumerate(np.unique(week_key))}
    cluster = np.asarray([ordered_weeks[int(value)] for value in week_key], dtype=np.int64)
    if len(np.unique(cluster[scored_mask])) < 3:
        raise RuntimeError(f"{name}: fewer than three scored weekly clusters")

    chronology = {
        "timestamp_sha256": hashlib.sha256(timestamps_ns.tobytes()).hexdigest(),
        "n_rows": int(n),
        "n_acquisition_events": int(event_id[-1] + 1),
        "timestamps_nondecreasing": bool(np.all(np.diff(timestamps_ns) >= 0)),
        "no_event_crosses_calibration": bool(
            not np.any(event_id[:cal_n] == event_id[cal_n]) if cal_n < n else True
        ),
        "time_cluster": "ISO calendar week",
    }
    if not all(chronology[key] for key in ("timestamps_nondecreasing", "no_event_crosses_calibration")):
        raise RuntimeError(f"{name}: chronology integrity failure")
    return {
        "name": name,
        "X": X,
        "y": y,
        "month_id": cluster,
        "scored_mask": np.asarray(scored_mask, dtype=bool),
        "cal_n": cal_n,
        "event_id": event_id,
        "event_time_seconds": timestamps_ns.astype(np.float64) / 1.0e9,
        "chronology_audit": chronology,
        "chronology_preserved": True,
        "all_preprocessed_rows_retained": True,
        "data_manifest": manifest,
        "feature_names": feature_names,
        "target_transform": transform,
        "target_calibration_mean": y_mean,
        "target_calibration_std": y_std,
    }


def load_workshop_stream(name: str, *, smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    canonical = str(name).upper()
    if smoke:
        stream = load_regression_stream("METRO", smoke=True, seed=seed)
        stream["name"] = f"SMOKE_{canonical}"
        return stream
    if canonical in {"METRO", "BEIJING"}:
        return load_regression_stream(canonical, smoke=False, seed=seed)
    if canonical not in EXTRA_SPECS:
        raise ValueError(f"unsupported workshop stream: {name!r}")
    path, manifest = _download_archive(canonical)
    member, frame = _read_matching_csv(path, EXTRA_SPECS[canonical]["required"])
    manifest = dict(manifest, members=[member])
    return _from_frame(canonical, frame, manifest)


def dataset_manifest_json(stream: dict[str, Any]) -> str:
    """Stable compact identity used by cross-process and cross-stage audits."""
    payload = {
        "name": stream["name"],
        "shape": list(stream["X"].shape),
        "manifest": stream.get("data_manifest"),
        "chronology": stream.get("chronology_audit"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
