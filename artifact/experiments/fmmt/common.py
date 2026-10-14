"""Shared, deliberately small utilities for the two FMMT preliminary studies.

The scientific runners use the already-audited Stage-4 regression loader.  This
module adds only target standardisation, packetisation, clustered inference,
and fail-closed artifact helpers.  It does not alter stream order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.validation import stage4_regression as stage4


REPO = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: str | Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    encoded = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bundle_sha256(paths: Iterable[str | Path]) -> tuple[str, dict[str, str]]:
    hashes = {str(Path(p).relative_to(REPO)): sha256_file(p) for p in paths}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), hashes


def _chronology_metadata(name: str, frame: Any, n: int) -> dict[str, Any]:
    """Reproduce the frozen loader's row order and retain acquisition groups.

    The Stage-4 arrays intentionally discard timestamps.  The prospective
    audits need them to prevent two records from one physical acquisition
    event (duplicate Metro rows or Beijing stations in one hour) from landing
    on opposite support/query sides.
    """
    import pandas as pd

    spec = stage4.DS[name]
    ordered = frame.copy()
    ordered.columns = [str(column).strip() for column in ordered.columns]
    if name == "METRO":
        ordered["_ts"] = pd.to_datetime(ordered[spec["ts"]], errors="coerce")
        ordered = ordered.sort_values("_ts", kind="stable").reset_index(drop=True)
    else:
        ordered["_ts"] = pd.to_datetime(
            dict(
                year=ordered["year"], month=ordered["month"],
                day=ordered["day"], hour=ordered["hour"],
            ),
            errors="coerce",
        )
        ordered = ordered.sort_values(["_ts", "station"], kind="stable").reset_index(drop=True)
        ordered = ordered.dropna(subset=[spec["target"]]).reset_index(drop=True)
    target = pd.to_numeric(ordered[spec["target"]], errors="coerce").to_numpy(float)
    ordered = ordered[np.isfinite(target)].reset_index(drop=True)
    timestamps_ns = ordered["_ts"].astype("int64").to_numpy(np.int64)[: int(n)]
    if len(timestamps_ns) != int(n) or np.any(timestamps_ns == np.iinfo(np.int64).min):
        raise RuntimeError(f"{name}: timestamp metadata does not align with frozen arrays")
    change = np.empty(len(timestamps_ns), dtype=bool)
    change[0] = True
    change[1:] = timestamps_ns[1:] != timestamps_ns[:-1]
    event_id = np.cumsum(change, dtype=np.int64) - 1
    counts = np.bincount(event_id)
    duplicate_rows = int(np.sum(counts[event_id] > 1))
    digest = hashlib.sha256(timestamps_ns.tobytes()).hexdigest()
    positive_deltas = np.diff(timestamps_ns[change]) / 1.0e9
    positive_deltas = positive_deltas[positive_deltas > 0]
    return {
        "event_id": event_id,
        "event_time_seconds": timestamps_ns.astype(np.float64) / 1.0e9,
        "audit": {
            "timestamp_sha256": digest,
            "n_rows": int(len(timestamps_ns)),
            "n_acquisition_events": int(len(counts)),
            "duplicate_event_row_fraction": duplicate_rows / max(len(timestamps_ns), 1),
            "maximum_rows_per_event": int(counts.max()),
            "timestamps_nondecreasing": bool(np.all(np.diff(timestamps_ns) >= 0)),
            "median_positive_event_interval_seconds": (
                float(np.median(positive_deltas)) if len(positive_deltas) else None
            ),
            "support_query_group_key": "exact acquisition timestamp",
        },
    }


def grouped_event_folds(event_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, acquisition-event-disjoint two-fold packet split."""
    event_ids = np.asarray(list(event_ids), dtype=np.int64)
    if event_ids.ndim != 1 or event_ids.size < 2:
        raise ValueError("event_ids must be a one-dimensional packet with >=2 rows")
    unique = np.unique(event_ids)
    if len(unique) < 2:
        raise ValueError("packet has fewer than two acquisition events")
    first = np.flatnonzero((event_ids & 1) == 0)
    second = np.flatnonzero((event_ids & 1) == 1)
    if not len(first) or not len(second):
        # Consecutive global event IDs normally make parity sufficient.  This
        # deterministic fallback assigns whole groups by their ordered rank.
        lookup = {int(group): rank & 1 for rank, group in enumerate(unique)}
        assignment = np.asarray([lookup[int(group)] for group in event_ids])
        first, second = np.flatnonzero(assignment == 0), np.flatnonzero(assignment == 1)
    if set(event_ids[first]).intersection(set(event_ids[second])):
        raise RuntimeError("an acquisition event crossed support/query folds")
    return first.astype(np.int64), second.astype(np.int64)


def load_regression_stream(name: str, smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    """Load a chronological stream and add calibration-only target scaling.

    ``y`` is standardised log1p(target), with mean/std estimated exclusively on
    the leading calibration prefix.  ``month_id`` and the original arrays are
    retained for cluster-aware reporting.  No row is sorted here.
    """
    canonical = str(name).upper()
    if smoke:
        raw = stage4._smoke_arrays(seed)
        manifest = {"source": "deterministic_smoke", "seed": int(seed)}
        event_id = np.arange(int(raw["N"]), dtype=np.int64)
        event_time_seconds = event_id.astype(np.float64)
        event_audit = {
            "timestamp_sha256": hashlib.sha256(event_id.tobytes()).hexdigest(),
            "n_rows": int(raw["N"]),
            "n_acquisition_events": int(raw["N"]),
            "duplicate_event_row_fraction": 0.0,
            "maximum_rows_per_event": 1,
            "timestamps_nondecreasing": True,
            "median_positive_event_interval_seconds": 1.0,
            "support_query_group_key": "synthetic unique arrival",
        }
    else:
        if canonical not in {"METRO", "BEIJING"}:
            raise ValueError(f"unsupported regression stream: {name!r}")
        frame, manifest = stage4._download(canonical)
        raw = stage4._build_arrays(
            canonical, frame, drop_final_short_packet=False
        )
        chronology = _chronology_metadata(canonical, frame, int(raw["N"]))
        event_id = chronology["event_id"]
        event_time_seconds = chronology["event_time_seconds"]
        event_audit = chronology["audit"]
    scale_cal_n = max(2, int(raw["cal_n"]))
    ylog = np.asarray(raw["ylog"], dtype=np.float32)
    y_mean = float(np.mean(ylog[:scale_cal_n]))
    y_std = float(np.std(ylog[:scale_cal_n], ddof=1))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    # The Stage-4 loader freezes calibration by row count.  Move only the
    # *scoring* boundary to the end of that row's acquisition group so no
    # timestamp can be partly used for calibration/training and partly scored.
    calibration_event = int(event_id[min(scale_cal_n - 1, len(event_id) - 1)])
    scored_mask = np.asarray(event_id > calibration_event, dtype=bool)
    cal_n = int(np.flatnonzero(scored_mask)[0]) if scored_mask.any() else len(event_id)
    event_audit = dict(event_audit)
    event_audit.update(
        {
            "row_count_calibration_boundary": scale_cal_n,
            "event_aligned_scoring_boundary": cal_n,
            "calibration_event_id": calibration_event,
            "no_acquisition_event_crosses_scoring_boundary": bool(
                not np.any(event_id[:cal_n] == event_id[cal_n]) if cal_n < len(event_id) else True
            ),
        }
    )
    out = dict(raw)
    out.update(
        name=("SMOKE" if smoke else canonical),
        X=np.asarray(raw["X"], dtype=np.float32),
        y=((ylog - y_mean) / y_std).astype(np.float32),
        ylog=ylog,
        y_calibration_mean=y_mean,
        y_calibration_std=y_std,
        y_scale_calibration_rows=scale_cal_n,
        scoring_calibration_rows=cal_n,
        scored_mask=scored_mask,
        data_manifest=manifest,
        event_id=event_id,
        event_time_seconds=event_time_seconds,
        chronology_audit=event_audit,
        chronology_preserved=True,
        all_preprocessed_rows_retained=(
            int(raw.get("N", len(raw["X"]))) == int(raw.get("n_rows", len(raw["X"])))
        ),
    )
    if not np.isfinite(out["X"]).all() or not np.isfinite(out["y"]).all():
        raise RuntimeError(f"{canonical}: non-finite value after frozen preprocessing")
    if not bool(out["all_preprocessed_rows_retained"]):
        raise RuntimeError(f"{canonical}: preprocessing dropped a final short packet")
    if not bool(event_audit["no_acquisition_event_crosses_scoring_boundary"]):
        raise RuntimeError(f"{canonical}: an acquisition event crosses the scoring boundary")
    return out


def packetize_stream(data: dict[str, Any], packet_size: int = 32) -> list[dict[str, Any]]:
    """Create chronological, acquisition-event-atomic packets.

    ``packet_size`` is a target, not permission to split simultaneous records.
    A packet contains at least two timestamp groups whenever the stream itself
    has at least two groups; this is required by the cross-fitted diagnostics.
    The final partial packet is retained rather than silently discarding data.
    """
    n = len(data["X"])
    target = int(packet_size)
    if target < 2 or n < 2:
        raise ValueError("packet_size and stream length must both be at least two")
    all_event_ids = np.asarray(
        data.get("event_id", np.arange(n, dtype=np.int64)),
        dtype=np.int64,
    )
    all_event_times = np.asarray(
        data.get("event_time_seconds", np.arange(n, dtype=np.float64)),
        dtype=np.float64,
    )
    if len(all_event_ids) != n or len(all_event_times) != n:
        raise ValueError("event metadata length does not match the stream")
    if np.any(np.diff(all_event_times) < 0):
        raise RuntimeError("event times are not chronological")
    # Each acquisition event must occupy one contiguous run.  A repeated ID in
    # disjoint runs would make an atomic packetization ambiguous and fails shut.
    boundaries = np.r_[0, 1 + np.flatnonzero(all_event_ids[1:] != all_event_ids[:-1]), n]
    group_ranges = [(int(left), int(right)) for left, right in zip(boundaries[:-1], boundaries[1:])]
    if len({int(all_event_ids[left]) for left, _ in group_ranges}) != len(group_ranges):
        raise RuntimeError("an acquisition event appears in disjoint chronology runs")
    if len(group_ranges) < 2:
        raise RuntimeError("stream has fewer than two acquisition events")

    packet_ranges: list[tuple[int, int]] = []
    current_left = group_ranges[0][0]
    current_right = group_ranges[0][1]
    current_groups = 1
    for left, right in group_ranges[1:]:
        prospective_size = right - current_left
        if current_groups >= 2 and prospective_size > target:
            packet_ranges.append((current_left, current_right))
            current_left, current_right, current_groups = left, right, 1
        else:
            current_right = right
            current_groups += 1
    if current_groups < 2 and packet_ranges:
        previous_left, _ = packet_ranges.pop()
        packet_ranges.append((previous_left, current_right))
    else:
        packet_ranges.append((current_left, current_right))

    packets: list[dict[str, Any]] = []
    for packet_id, (start, stop) in enumerate(packet_ranges):
        index = np.arange(start, stop, dtype=np.int64)
        event_count = int(len(np.unique(all_event_ids[start:stop])))
        if event_count < 2:
            raise RuntimeError("packet has fewer than two acquisition events")
        packets.append(
            {
                "packet_id": packet_id,
                "index": index,
                "x": data["X"][start:stop],
                "y": data["y"][start:stop],
                "month": data["month_id"][start:stop],
                "event_id": all_event_ids[start:stop],
                "event_time_seconds": all_event_times[start:stop],
                "scored": data["scored_mask"][start:stop],
                "n_acquisition_events": event_count,
            }
        )
    covered = np.concatenate([packet["index"] for packet in packets])
    if not np.array_equal(covered, np.arange(n, dtype=np.int64)):
        raise RuntimeError("packetization did not preserve every stream row exactly once")
    return packets


def month_cluster_bootstrap(
    effects: Iterable[float],
    clusters: Iterable[Any],
    seed: int = 271828,
    n_boot: int = 4000,
    block_length: int = 3,
) -> tuple[float, float, float]:
    """Circular moving-block bootstrap over ordered calendar-month clusters.

    Invalid inputs are an integrity failure.  In particular, silently dropping
    a non-finite effect would let different rows be evaluated on different
    month supports and could change a scientific gate.  Each sampled block
    contains consecutive months (wrapping at the end), and all observations in
    a selected month cluster are retained.
    """
    effects = np.asarray(list(effects), dtype=np.float64)
    if effects.ndim != 1:
        raise ValueError("month_cluster_bootstrap expects one-dimensional effects")
    if effects.size == 0:
        raise ValueError("effects must be nonempty")
    if not np.isfinite(effects).all():
        raise ValueError("effects contain non-finite values")
    clusters, unique = _validated_clusters(clusters, effects.shape[0])
    n_boot = _validated_bootstrap_draws(n_boot)
    effective_block_length = _validated_block_length(block_length, len(unique))
    rng = np.random.default_rng(int(seed))
    boot = np.empty(n_boot, dtype=np.float64)
    grouped = [effects[clusters == c] for c in unique]
    for b in range(n_boot):
        chosen = _sample_circular_month_blocks(
            rng, len(grouped), effective_block_length
        )
        boot[b] = np.concatenate([grouped[i] for i in chosen]).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(effects.mean()), float(lo), float(hi)


def _validated_bootstrap_draws(n_boot: int) -> int:
    n_boot = int(n_boot)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    return n_boot


def _validated_block_length(block_length: int, n_clusters: int) -> int:
    if isinstance(block_length, (float, np.floating)) and not float(block_length).is_integer():
        raise ValueError("block_length must be a positive integer")
    block_length = int(block_length)
    if block_length <= 0:
        raise ValueError("block_length must be a positive integer")
    if int(n_clusters) <= 0:
        raise ValueError("at least one month cluster is required")
    return min(block_length, int(n_clusters))


def _sample_circular_month_blocks(
    rng: np.random.Generator,
    n_clusters: int,
    block_length: int,
) -> np.ndarray:
    """Draw ordered circular blocks and return exactly ``n_clusters`` IDs."""
    blocks_needed = int(math.ceil(int(n_clusters) / int(block_length)))
    starts = rng.integers(0, int(n_clusters), size=blocks_needed)
    sampled = np.concatenate(
        [
            (int(start) + np.arange(int(block_length), dtype=np.int64))
            % int(n_clusters)
            for start in starts
        ]
    )
    return sampled[: int(n_clusters)]


def _validated_clusters(
    clusters: Iterable[Any], expected_columns: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(clusters))
    if labels.ndim != 1:
        raise ValueError("clusters must be one-dimensional")
    if labels.shape[0] != int(expected_columns):
        raise ValueError("cluster labels do not match the month dimension")
    if labels.size == 0:
        raise ValueError("clusters must be nonempty")
    if np.issubdtype(labels.dtype, np.number):
        if not np.isfinite(labels.astype(np.float64, copy=False)).all():
            raise ValueError("clusters contain non-finite values")
    else:
        for label in labels.tolist():
            if label is None or (isinstance(label, float) and not math.isfinite(label)):
                raise ValueError("clusters contain invalid values")
    try:
        unique, first = np.unique(labels, return_index=True)
        unique = unique[np.argsort(first)]
    except (TypeError, ValueError) as exc:
        raise ValueError("cluster labels must be mutually comparable") from exc
    if unique.size == 0:  # Defensive: labels.size already rules this out.
        raise ValueError("clusters must contain at least one month")
    return labels, unique


def _two_way_cluster_bootstrap(
    effects: np.ndarray,
    clusters: Iterable[Any],
    *,
    seed: int,
    n_boot: int,
    block_length: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Resample initialization rows and circular month blocks independently.

    The seed-by-month matrix is retained throughout every draw.  Cells are
    never flattened and treated as independent replicates.  This propagates
    initialization uncertainty while preserving ordered temporal blocks and
    all columns belonging to each sampled calendar-month cluster.
    """
    effects = np.asarray(effects, dtype=np.float64)
    if effects.ndim != 2:
        raise ValueError("two-way bootstrap expects a seed-by-month matrix")
    if effects.shape[0] == 0 or effects.shape[1] == 0:
        raise ValueError("seed-by-month effects must be nonempty on both axes")
    if not np.isfinite(effects).all():
        raise ValueError("effects contain non-finite values")
    labels, unique = _validated_clusters(clusters, effects.shape[1])
    n_boot = _validated_bootstrap_draws(n_boot)
    effective_block_length = _validated_block_length(block_length, len(unique))
    grouped_columns = [np.flatnonzero(labels == label) for label in unique]
    rng = np.random.default_rng(int(seed))
    boot = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        sampled_seeds = rng.integers(0, effects.shape[0], size=effects.shape[0])
        sampled_clusters = _sample_circular_month_blocks(
            rng, len(grouped_columns), effective_block_length
        )
        sampled_columns = np.concatenate(
            [grouped_columns[index] for index in sampled_clusters]
        )
        boot[draw] = effects[np.ix_(sampled_seeds, sampled_columns)].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    month_effects = np.asarray(
        [effects[:, columns].mean() for columns in grouped_columns], dtype=np.float64
    )
    seed_effects = effects.mean(axis=1)
    return float(effects.mean()), float(lo), float(hi), month_effects, seed_effects


def _numeric_array(values: Iterable[float], name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64)
    else:
        array = np.asarray(list(values), dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be one- or two-dimensional")
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def paired_summary(
    baseline: Iterable[float],
    candidate: Iterable[float],
    clusters: Iterable[Any] | None = None,
    *,
    lower_is_better: bool = True,
    seed: int = 271828,
    n_boot: int = 4000,
    block_length: int = 3,
) -> dict[str, Any]:
    """Summarize a paired lower-is-better contrast.

    One-dimensional inputs retain the original month-cluster API.  Two-
    dimensional inputs must be shaped ``[initialization_seed, calendar_month]``.
    Both paths use an ordered circular moving-block bootstrap over months; the
    two-dimensional path independently resamples initialization rows.  The
    reported ``wins_of_n`` remains a data-side quantity (wins among seed-
    averaged calendar months); seed robustness is reported separately as
    ``seed_wins_of_n``.
    """
    baseline = _numeric_array(baseline, "baseline")
    candidate = _numeric_array(candidate, "candidate")
    if baseline.shape != candidate.shape:
        raise ValueError("paired arrays have different shapes")
    improvement = baseline - candidate if lower_is_better else candidate - baseline
    if clusters is None:
        clusters = np.arange(improvement.shape[-1])
    if improvement.ndim == 1:
        labels, unique = _validated_clusters(clusters, improvement.shape[0])
        effective_block_length = _validated_block_length(block_length, len(unique))
        mean, lo, hi = month_cluster_bootstrap(
            improvement,
            labels,
            seed,
            n_boot,
            block_length=block_length,
        )
        wins = int(np.sum(improvement > 0))
        n = int(len(improvement))
        seed_fields: dict[str, Any] = {}
    else:
        labels, unique = _validated_clusters(clusters, improvement.shape[1])
        effective_block_length = _validated_block_length(block_length, len(unique))
        mean, lo, hi, month_effects, seed_effects = _two_way_cluster_bootstrap(
            improvement,
            labels,
            seed=int(seed),
            n_boot=int(n_boot),
            block_length=block_length,
        )
        wins = int(np.sum(month_effects > 0))
        n = int(len(month_effects))
        seed_fields = {
            "seed_wins_of_n": int(np.sum(seed_effects > 0)),
            "n_seeds": int(len(seed_effects)),
            "seed_averaged_month_effects": month_effects.tolist(),
            "month_averaged_seed_effects": seed_effects.tolist(),
        }
    denom = max(abs(float(np.mean(baseline))), 1e-12)
    result = {
        "improvement": mean,
        "ci95": [lo, hi],
        "relative_improvement": mean / denom,
        "wins_of_n": wins,
        "n": n,
        "paired_effects": improvement.tolist(),
        "bootstrap": {
            "method": "paired_circular_moving_block",
            "ordered_calendar_month_clusters": True,
            "requested_block_length": int(block_length),
            "effective_block_length": int(effective_block_length),
            "n_month_clusters": int(len(unique)),
            "initialization_rows_resampled": improvement.ndim == 2,
        },
    }
    result.update(seed_fields)
    return result


def metric_by_month(pred: np.ndarray, target: np.ndarray, month: np.ndarray, scored: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    month = np.asarray(month)
    scored = np.asarray(scored, bool)
    per_month: dict[str, Any] = {}
    squared, absolute, clusters = [], [], []
    for key in np.unique(month[scored]):
        keep = scored & (month == key) & np.isfinite(pred)
        if not keep.any():
            continue
        error = pred[keep] - target[keep]
        mse = float(np.mean(error * error))
        mae = float(np.mean(np.abs(error)))
        per_month[str(int(key))] = {"rmse": math.sqrt(mse), "mae": mae, "n": int(keep.sum())}
        squared.append(mse)
        absolute.append(mae)
        clusters.append(int(key))
    return {
        "macro_month_nrmse": float(np.mean(np.sqrt(squared))) if squared else float("nan"),
        "macro_month_mae": float(np.mean(absolute)) if absolute else float("nan"),
        "per_month": per_month,
        "month_ids": clusters,
    }


def provenance(device: str, lock_path: str | Path | None = None) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        git_commit = "unavailable"
    return {
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None,
        "lock_sha256": sha256_file(lock_path) if lock_path and Path(lock_path).is_file() else None,
        "server_wrapper_sha256": os.environ.get("FMMT_SERVER_WRAPPER_SHA256"),
    }
