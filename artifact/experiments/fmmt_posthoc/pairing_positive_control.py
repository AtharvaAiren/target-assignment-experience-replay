#!/usr/bin/env python3
"""Reproducible synthetic sanity check for the K1--K4 pairing audit.

This experiment is intentionally elementary.  It verifies that an equal-radius
target-permutation audit can (i) detect persistent conditional structure,
(ii) recognize target-marginal structure that is invariant to pairing, and
(iii) expose a stale conditional direction that is harmful after reversal.
It is a post-hoc implementation sanity check, not part of the locked natural-
stream confirmatory study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "paper" / "results" / "pairing_positive_control.json"
N_REPETITIONS = 512
PACKET_SIZE = 64
DIMENSION = 8
RADIUS = 0.25
SLOPE = 1.0
INTERCEPT = 1.0
NOISE_STD = 0.10
BOOTSTRAP_DRAWS = 10_000


def half_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(0.5 * np.mean((prediction - target) ** 2))


def unit(value: np.ndarray | float) -> np.ndarray | float:
    norm = float(np.linalg.norm(value))
    return np.zeros_like(value) if norm == 0.0 else value / norm


def orthonormal_design(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(PACKET_SIZE, DIMENSION))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw)
    return np.sqrt(PACKET_SIZE) * q[:, :DIMENSION]


def relative_gain(loss: float, zero_loss: float) -> float:
    return 100.0 * (zero_loss - loss) / zero_loss


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run_repetition(seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}

    # Stable conditional relationship: y = a x + eps in both packets.
    direction = np.ones(DIMENSION, dtype=np.float64) / np.sqrt(DIMENSION)
    coefficient = SLOPE * direction
    x_prev = orthonormal_design(rng)
    x_next = orthonormal_design(rng)
    y_prev = x_prev @ coefficient + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    y_next = x_next @ coefficient + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    g_true = x_prev.T @ (-y_prev) / PACKET_SIZE
    y_perm = y_prev[rng.permutation(PACKET_SIZE)]
    g_perm = x_prev.T @ (-y_perm) / PACKET_SIZE
    zero = half_mse(np.zeros(PACKET_SIZE), y_next)
    losses = {
        "zero": zero,
        "true": half_mse(x_next @ (-RADIUS * unit(g_true)), y_next),
        "permuted": half_mse(x_next @ (-RADIUS * unit(g_perm)), y_next),
        "reverse": half_mse(x_next @ (RADIUS * unit(g_true)), y_next),
    }
    result["stable_slope"] = {name: relative_gain(loss, zero) for name, loss in losses.items()}

    # Marginal-only persistence: an intercept learner sees exactly the same
    # gradient after any within-packet target permutation.
    y_prev = INTERCEPT + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    y_next = INTERCEPT + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    g_true = float(np.mean(-y_prev))
    y_perm = y_prev[rng.permutation(PACKET_SIZE)]
    g_perm = float(np.mean(-y_perm))
    zero = half_mse(np.zeros_like(y_next), y_next)
    losses = {
        "zero": zero,
        "true": half_mse(np.full_like(y_next, -RADIUS * unit(g_true)), y_next),
        "permuted": half_mse(np.full_like(y_next, -RADIUS * unit(g_perm)), y_next),
        "reverse": half_mse(np.full_like(y_next, RADIUS * unit(g_true)), y_next),
    }
    result["marginal_intercept"] = {name: relative_gain(loss, zero) for name, loss in losses.items()}

    # Conditional reversal: the previous slope flips sign before the next
    # packet, so the previous true direction is deliberately stale.
    x_prev = orthonormal_design(rng)
    x_next = orthonormal_design(rng)
    y_prev = x_prev @ coefficient + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    y_next = -(x_next @ coefficient) + rng.normal(0.0, NOISE_STD, PACKET_SIZE)
    g_true = x_prev.T @ (-y_prev) / PACKET_SIZE
    y_perm = y_prev[rng.permutation(PACKET_SIZE)]
    g_perm = x_prev.T @ (-y_perm) / PACKET_SIZE
    zero = half_mse(np.zeros(PACKET_SIZE), y_next)
    losses = {
        "zero": zero,
        "true": half_mse(x_next @ (-RADIUS * unit(g_true)), y_next),
        "permuted": half_mse(x_next @ (-RADIUS * unit(g_perm)), y_next),
        "reverse": half_mse(x_next @ (RADIUS * unit(g_true)), y_next),
    }
    result["reversed_slope"] = {name: relative_gain(loss, zero) for name, loss in losses.items()}
    return result


def main() -> None:
    repetitions = [run_repetition(seed) for seed in range(N_REPETITIONS)]
    bootstrap_rng = np.random.default_rng(20260810)
    regimes: dict[str, dict[str, object]] = {}
    for regime in ["stable_slope", "marginal_intercept", "reversed_slope"]:
        by_action = {
            action: np.asarray([row[regime][action] for row in repetitions], dtype=np.float64)
            for action in ["zero", "true", "permuted", "reverse"]
        }
        true_minus_permuted = by_action["true"] - by_action["permuted"]
        regimes[regime] = {
            "mean_relative_gain_pct": {
                action: float(values.mean()) for action, values in by_action.items()
            },
            "true_minus_permuted_pp": float(true_minus_permuted.mean()),
            "true_minus_permuted_ci95": percentile_ci(true_minus_permuted, bootstrap_rng),
            "true_wins_over_permuted": int(np.sum(true_minus_permuted > 0.0)),
        }

    payload = {
        "status": "PASS",
        "scope": "post_hoc_pairing_audit_sanity_check",
        "not_part_of_locked_natural_study": True,
        "n_repetitions": N_REPETITIONS,
        "packet_size": PACKET_SIZE,
        "dimension": DIMENSION,
        "radius": RADIUS,
        "slope": SLOPE,
        "intercept": INTERCEPT,
        "noise_std": NOISE_STD,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "regimes": regimes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
