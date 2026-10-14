from __future__ import annotations

import numpy as np
import pytest

from experiments.fmmt.common import month_cluster_bootstrap, paired_summary


def test_paired_summary_preserves_one_dimensional_behavior() -> None:
    baseline = np.asarray([4.0, 4.0, 4.0, 4.0])
    candidate = np.asarray([3.0, 5.0, 2.0, 4.0])

    result = paired_summary(
        baseline,
        candidate,
        clusters=np.asarray([10, 11, 12, 13]),
        seed=9,
        n_boot=100,
        block_length=2,
    )

    assert result["improvement"] == pytest.approx(0.5)
    assert result["relative_improvement"] == pytest.approx(0.125)
    assert result["wins_of_n"] == 2
    assert result["n"] == 4
    assert result["paired_effects"] == [1.0, -1.0, 2.0, 0.0]
    assert "seed_wins_of_n" not in result
    assert result["bootstrap"] == {
        "method": "paired_circular_moving_block",
        "ordered_calendar_month_clusters": True,
        "requested_block_length": 2,
        "effective_block_length": 2,
        "n_month_clusters": 4,
        "initialization_rows_resampled": False,
    }


def _manual_circular_blocks(
    rng: np.random.Generator, n_clusters: int, block_length: int
) -> np.ndarray:
    starts = rng.integers(
        0, n_clusters, size=int(np.ceil(n_clusters / block_length))
    )
    sampled = np.concatenate(
        [
            (int(start) + np.arange(block_length, dtype=np.int64)) % n_clusters
            for start in starts
        ]
    )
    return sampled[:n_clusters]


def _manual_two_way_draws(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    block_length: int,
) -> np.ndarray:
    unique, first = np.unique(clusters, return_index=True)
    unique = unique[np.argsort(first)]
    groups = [np.flatnonzero(clusters == value) for value in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for draw in range(n_boot):
        seed_rows = rng.integers(0, effects.shape[0], size=effects.shape[0])
        month_blocks = _manual_circular_blocks(rng, len(groups), block_length)
        columns = np.concatenate([groups[index] for index in month_blocks])
        draws[draw] = effects[np.ix_(seed_rows, columns)].mean()
    return draws


def test_paired_summary_two_way_bootstrap_keeps_seed_month_structure() -> None:
    effects = np.asarray(
        [
            [1.0, 3.0, -2.0, 4.0],
            [3.0, 1.0, -4.0, 2.0],
        ]
    )
    baseline = np.full_like(effects, 10.0)
    candidate = baseline - effects
    clusters = np.asarray([10, 10, 11, 12])
    seed = 17
    n_boot = 101
    block_length = 2

    result = paired_summary(
        baseline,
        candidate,
        clusters=clusters,
        seed=seed,
        n_boot=n_boot,
        block_length=block_length,
    )
    expected_draws = _manual_two_way_draws(
        effects,
        clusters,
        seed=seed,
        n_boot=n_boot,
        block_length=block_length,
    )

    assert result["improvement"] == pytest.approx(float(effects.mean()))
    assert result["ci95"] == pytest.approx(
        np.quantile(expected_draws, [0.025, 0.975]).tolist()
    )
    # Seed-averaged cluster effects are [2, -3, 3].
    assert result["seed_averaged_month_effects"] == pytest.approx([2.0, -3.0, 3.0])
    assert result["wins_of_n"] == 2
    assert result["n"] == 3
    # Per-seed grand means are [1.5, 0.5].
    assert result["month_averaged_seed_effects"] == pytest.approx([1.5, 0.5])
    assert result["seed_wins_of_n"] == 2
    assert result["n_seeds"] == 2
    assert np.asarray(result["paired_effects"]).shape == (2, 4)
    assert result["bootstrap"]["method"] == "paired_circular_moving_block"
    assert result["bootstrap"]["effective_block_length"] == 2
    assert result["bootstrap"]["initialization_rows_resampled"] is True


def test_one_dimensional_circular_block_ci_matches_manual_draws() -> None:
    effects = np.asarray([1.0, -2.0, 4.0, -1.0, 3.0])
    clusters = np.arange(len(effects))
    seed = 71
    n_boot = 79
    block_length = 3
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            effects[_manual_circular_blocks(rng, len(effects), block_length)].mean()
            for _ in range(n_boot)
        ]
    )

    result = paired_summary(
        np.full_like(effects, 10.0),
        np.full_like(effects, 10.0) - effects,
        clusters=clusters,
        seed=seed,
        n_boot=n_boot,
        block_length=block_length,
    )

    assert result["ci95"] == pytest.approx(
        np.quantile(draws, [0.025, 0.975]).tolist()
    )
    assert result["bootstrap"]["requested_block_length"] == 3
    assert result["bootstrap"]["effective_block_length"] == 3


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([], [], "nonempty"),
        ([1.0, np.nan], [1.0, 1.0], "non-finite"),
        ([1.0, 1.0], [1.0, np.inf], "non-finite"),
        (np.empty((2, 0)), np.empty((2, 0)), "nonempty"),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), "one- or two-dimensional"),
    ],
)
def test_paired_summary_rejects_empty_or_nonfinite_inputs(
    baseline: object, candidate: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_summary(baseline, candidate, n_boot=10)  # type: ignore[arg-type]


def test_paired_summary_rejects_shape_and_cluster_mismatches() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        paired_summary([1.0, 2.0], [1.0], n_boot=10)
    with pytest.raises(ValueError, match="month dimension"):
        paired_summary(np.ones((2, 3)), np.zeros((2, 3)), clusters=[0, 1], n_boot=10)
    with pytest.raises(ValueError, match="positive"):
        paired_summary([1.0], [0.0], clusters=[0], n_boot=0)
    with pytest.raises(ValueError, match="positive integer"):
        paired_summary([1.0], [0.0], clusters=[0], n_boot=10, block_length=0)
    with pytest.raises(ValueError, match="positive integer"):
        paired_summary([1.0], [0.0], clusters=[0], n_boot=10, block_length=1.5)


def test_month_cluster_bootstrap_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        month_cluster_bootstrap([], [], n_boot=10)
    with pytest.raises(ValueError, match="non-finite"):
        month_cluster_bootstrap([1.0, np.nan], [0, 1], n_boot=10)
    with pytest.raises(ValueError, match="month dimension"):
        month_cluster_bootstrap([1.0, 2.0], [0], n_boot=10)
