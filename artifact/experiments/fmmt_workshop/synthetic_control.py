"""Synthetic audit that directly controls the observed useful-gradient cosine."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ACTIONS = ("zero", "previous", "reverse")


def _next_direction(
    previous: np.ndarray, rho: float, rng: np.random.Generator
) -> np.ndarray:
    innovation = rng.standard_normal(previous.shape)
    innovation = innovation - float(innovation @ previous) * previous
    norm = float(np.linalg.norm(innovation))
    if norm <= 1e-12:
        raise RuntimeError("degenerate orthogonal innovation")
    innovation /= norm
    direction = float(rho) * previous + np.sqrt(max(0.0, 1.0 - rho * rho)) * innovation
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction


def run_synthetic_unit(
    rho: float,
    seed: int,
    *,
    dimension: int,
    steps: int,
    burn_in: int,
    selector_eta: float,
) -> dict[str, Any]:
    rng = np.random.default_rng([int(seed), int(round((rho + 1.0) * 10000)), 0xC05])
    previous = rng.standard_normal(int(dimension))
    previous /= np.linalg.norm(previous)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    losses = {name: [] for name in (*ACTIONS, "selector")}
    cosines: list[float] = []
    mass_trace: list[np.ndarray] = []

    for _ in range(int(steps)):
        current = _next_direction(previous, float(rho), rng)
        cosines.append(float(previous @ current))
        optimum = -current
        displacements = {
            "zero": np.zeros_like(previous),
            "previous": -previous,
            "reverse": previous,
        }
        action_losses = np.asarray(
            [float(np.square(displacements[name] - optimum).sum()) for name in ACTIONS],
            dtype=np.float64,
        )
        mixture = sum(weights[index] * displacements[name] for index, name in enumerate(ACTIONS))
        selector_loss = float(np.square(mixture - optimum).sum())
        for index, name in enumerate(ACTIONS):
            losses[name].append(float(action_losses[index]))
        losses["selector"].append(selector_loss)
        weights *= np.exp(-float(selector_eta) * action_losses)
        weights /= max(float(weights.sum()), 1e-300)
        mass_trace.append(weights.copy())
        previous = current

    start = min(max(int(burn_in), 0), int(steps) - 1)
    tail = max(1, int(steps) // 5)
    mass = np.mean(np.asarray(mass_trace)[-tail:], axis=0)
    return {
        "rho": float(rho),
        "seed": int(seed),
        "realized_mean_cosine": float(np.mean(cosines[start:])),
        "losses": {
            name: float(np.mean(values[start:])) for name, values in losses.items()
        },
        "selector_mass_last_20pct": {
            name: float(mass[index]) for index, name in enumerate(ACTIONS)
        },
        "audit": {
            "observed_gradient_is_directly_constructed": True,
            "unit_norm_gradients": True,
            "selector_updates_after_current_gradient": True,
            "independent_seed_rng": True,
        },
    }


def _paired_seed_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_seed: int,
    draws: int,
) -> dict[str, Any]:
    effects = np.asarray(baseline, float) - np.asarray(candidate, float)
    order = np.argsort(effects, kind="stable")
    baseline = np.asarray(baseline, float)[order]
    candidate = np.asarray(candidate, float)[order]
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        chosen = rng.integers(0, len(effects), size=len(effects))
        samples[index] = float(np.mean(baseline[chosen] - candidate[chosen]))
    improvement = float(np.mean(baseline - candidate))
    denominator = max(float(np.mean(baseline)), 1e-12)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "baseline_mean": float(np.mean(baseline)),
        "candidate_mean": float(np.mean(candidate)),
        "improvement": improvement,
        "relative_improvement": improvement / denominator,
        "ci95": [float(lo), float(hi)],
        "ci95_relative": [float(lo / denominator), float(hi / denominator)],
        "wins_of_n": int(np.sum(candidate < baseline)),
        "n": int(len(baseline)),
        "bootstrap": "paired_iid_generator_seed",
    }


def summarize_synthetic(
    units_by_rho: Mapping[str, list[dict[str, Any]]],
    lock: Mapping[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    specification = lock["corrected_synthetic"]
    runtime = lock["runtime"]
    regimes: dict[str, Any] = {}
    for regime_index, (name, units) in enumerate(sorted(units_by_rho.items())):
        zero = np.asarray([unit["losses"]["zero"] for unit in units], dtype=float)
        regimes[name] = {
            "rho": float(units[0]["rho"]),
            "realized_mean_cosine": float(
                np.mean([unit["realized_mean_cosine"] for unit in units])
            ),
            "losses": {
                action: float(np.mean([unit["losses"][action] for unit in units]))
                for action in (*ACTIONS, "selector")
            },
            "contrasts_vs_zero": {
                action: _paired_seed_summary(
                    zero,
                    np.asarray([unit["losses"][action] for unit in units], dtype=float),
                    bootstrap_seed=int(runtime["cluster_bootstrap_seed"])
                    + int(seed_offset)
                    + 10 * regime_index
                    + action_index,
                    draws=int(runtime["cluster_bootstrap_draws"]),
                )
                for action_index, action in enumerate(("previous", "reverse", "selector"))
            },
            "selector_mass_last_20pct": {
                action: float(
                    np.mean([unit["selector_mass_last_20pct"][action] for unit in units])
                )
                for action in ACTIONS
            },
        }

    tolerance = float(specification["correlation_tolerance"])
    positive = regimes["rho_+0.9"]
    zero_regime = regimes["rho_+0.0"]
    negative = regimes["rho_-0.9"]
    required_gain = float(specification["minimum_persistent_relative_gain"])
    noninferiority = float(specification["selector_noninferiority"])
    mass_minimum = float(specification["correct_action_mass_minimum"])
    gates = {
        "realized_positive_correlation": abs(positive["realized_mean_cosine"] - 0.9) <= tolerance,
        "realized_zero_correlation": abs(zero_regime["realized_mean_cosine"]) <= tolerance,
        "realized_negative_correlation": abs(negative["realized_mean_cosine"] + 0.9) <= tolerance,
        "previous_helps_persistent": (
            positive["contrasts_vs_zero"]["previous"]["relative_improvement"] >= required_gain
        ),
        "previous_hurts_antipersistent": (
            negative["contrasts_vs_zero"]["previous"]["relative_improvement"] < 0
        ),
        "reverse_helps_antipersistent": (
            negative["contrasts_vs_zero"]["reverse"]["relative_improvement"] >= required_gain
        ),
        "selector_zero_noninferior": (
            zero_regime["contrasts_vs_zero"]["selector"]["relative_improvement"]
            >= -noninferiority
        ),
        "selector_selects_previous_positive": (
            positive["selector_mass_last_20pct"]["previous"] >= mass_minimum
        ),
        "selector_selects_zero_independent": (
            zero_regime["selector_mass_last_20pct"]["zero"] >= mass_minimum
        ),
        "selector_selects_reverse_negative": (
            negative["selector_mass_last_20pct"]["reverse"] >= mass_minimum
        ),
        "integrity": all(
            all(unit["audit"].values()) for units in units_by_rho.values() for unit in units
        ),
    }
    return {
        "schema_version": "fmmt.workshop.synthetic.summary.v1",
        "regimes": regimes,
        "gates": gates,
        "all_pass": bool(all(gates.values())),
        "verdict": "SYNTHETIC_CONTROL_PASS" if all(gates.values()) else "SYNTHETIC_CONTROL_FAIL",
    }


def run_synthetic_suite(lock: Mapping[str, Any], seeds: list[int], *, seed_offset: int) -> dict[str, Any]:
    specification = lock["corrected_synthetic"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for rho in specification["observed_gradient_correlations"]:
        name = f"rho_{float(rho):+0.1f}"
        groups[name] = [
            run_synthetic_unit(
                float(rho),
                int(seed),
                dimension=int(specification["dimension"]),
                steps=int(specification["steps"]),
                burn_in=int(specification["burn_in"]),
                selector_eta=float(specification["selector_eta"]),
            )
            for seed in seeds
        ]
    return {
        "units": groups,
        "summary": summarize_synthetic(groups, lock, seed_offset=seed_offset),
    }
