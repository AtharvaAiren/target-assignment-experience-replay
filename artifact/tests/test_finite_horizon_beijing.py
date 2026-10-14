from __future__ import annotations

import numpy as np

from experiments.gradient_reuse_finite_horizon.beijing import (
    FROZEN_TARGET_MEAN_BITS,
    FROZEN_TARGET_STD_BITS,
    locked_target_location_scale,
)
from experiments.gradient_reuse_finite_horizon.study import (
    assignment_law,
    assignment_moments,
    risk,
    support_size,
)


def test_target_location_scale_locks_historical_float32_bits() -> None:
    values = np.linspace(0.0, 7.0, 20601, dtype=np.float32)
    mean, std = locked_target_location_scale(values, len(values))
    assert int(np.float32(mean).view(np.uint32)) == FROZEN_TARGET_MEAN_BITS
    assert int(np.float32(std).view(np.uint32)) == FROZEN_TARGET_STD_BITS
    assert float(mean).hex() == "0x1.fbd07c0000000p+1"
    assert float(std).hex() == "0x1.1fe5a60000000p+0"


def test_support_size_and_law_cardinality() -> None:
    assert [support_size(6, value) for value in (0, 2500, 5000, 7500, 10000)] == [0, 2, 4, 5, 6]
    assert [len(assignment_law(6, value)) for value in (0, 2500, 5000, 7500, 10000)] == [1, 30, 360, 720, 720]


def test_exhaustive_mean_matches_q_lambda() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(6, 5))
    y = rng.normal(size=6)
    for strength in (0, 2500, 5000, 7500, 10000):
        plans = assignment_law(6, strength)
        mean_b, covariance, _ = assignment_moments(x, y, plans)
        m = support_size(6, strength)
        lam = 0.0 if m == 0 else (m - 1) / 5
        expected_y = (1 - lam) * y + lam * y.mean()
        assert np.allclose(mean_b, x.T @ expected_y / 6, atol=2e-14, rtol=0)
        assert np.linalg.eigvalsh(covariance)[0] >= -1e-12


def test_finite_horizon_risk_identity_matches_complete_one_step_law() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=6)
    plans = assignment_law(6, 5000)
    mean_b, omega, b_values = assignment_moments(x, y, plans)
    h = x.T @ x / 6
    eta = 0.07
    mu = eta * mean_b
    covariance = eta * eta * omega
    eval_x = rng.normal(size=(11, 3))
    eval_y = rng.normal(size=11)
    h_star = eval_x.T @ eval_x / 11
    b_star = eval_x.T @ eval_y / 11
    constant = 0.5 * float(np.mean(eval_y * eval_y))
    exact = float(np.mean(risk(eta * b_values, h_star, b_star, constant)))
    predicted = float(risk(mu, h_star, b_star, constant)) + 0.5 * float(np.trace(h_star @ covariance))
    assert np.isclose(exact, predicted, atol=2e-14, rtol=0)
