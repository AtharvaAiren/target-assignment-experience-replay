"""Two-channel, prospective controllers for reusing a stored lag-one gradient.

The implementation is deliberately self-contained.  It uses a linear squared-loss
model so that the target-permutation decomposition is exact and can be tested
without relying on the manuscript's larger experimental stack.

At packet ``t`` every policy follows the same chronology:

1. predict with the state carried into the packet;
2. reveal the packet targets and score the prequential prediction;
3. compute the ordinary gradient and its preserved/pairing decomposition;
4. compare the current decomposition with the stored decomposition from packet
   ``t-1``;
5. take the ordinary optimizer step; and
6. optionally reuse the stored packet-``t-1`` gradient at fixed radius.  This
   action can affect packet ``t+1`` but can never affect the already-recorded
   prediction on packet ``t``.

The pairing-aware policy treats the two decomposition channels separately.  It
keeps a temporally aligned preserved component (which contains target-mean
information) while requiring both temporal alignment and finite-permutation
signal-to-noise for the pairing component.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import (
    ValidatedEvidence,
    evidence_payload,
    observed_runtime,
    source_identity as bound_source_identity,
)


CONTROLLER_POLICIES = ("never", "always", "cosine", "pairing_aware")
_STREAM_KINDS = ("mean_drift", "recurrent_reversal", "weak_noisy_pairing")
_LOCK_STATUS = "LOCKED_BEFORE_RUN"


class ControllerConfigError(ValueError):
    """Raised when a controller campaign configuration is not prespecified."""


@dataclass(frozen=True)
class Stream:
    """A deterministic sequence of supervised packets."""

    name: str
    features: np.ndarray
    targets: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [packets, rows, dimension]")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape [packets, rows]")
        if self.features.shape[:2] != self.targets.shape:
            raise ValueError("feature and target packet shapes do not match")
        if self.features.shape[0] < 2 or self.features.shape[1] < 2:
            raise ValueError("a stream needs at least two packets and two rows")
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError("stream contains non-finite values")


def _require_exact_keys(
    mapping: Mapping[str, Any], required: set[str], context: str
) -> None:
    missing = required.difference(mapping)
    extra = set(mapping).difference(required)
    if missing or extra:
        raise ControllerConfigError(
            f"{context} keys differ from the lock; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ControllerConfigError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControllerConfigError(f"{context} must be a nonnegative integer")
    return value


def _finite_float(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerConfigError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ControllerConfigError(f"{context} must be {qualifier}")
    return result


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable smoke/full campaign boundary.

    Unknown keys are rejected so a misspelled option cannot silently change the
    scientific run.  Smoke seeds must be disjoint from full-run seeds, and smoke
    resources must be strictly smaller.  A full run is evidence-eligible only
    after this function accepts a ``LOCKED_BEFORE_RUN`` configuration.
    """

    if not isinstance(config, Mapping):
        raise ControllerConfigError("configuration must be a JSON object")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "status",
            "campaign_seed",
            "policies",
            "model",
            "permutations",
            "controller",
            "streams",
            "ablations",
            "full",
            "smoke",
        },
        "configuration",
    )
    if config["schema_version"] != "gradient_reuse.controller.v3":
        raise ControllerConfigError("unsupported controller schema_version")
    if config["status"] != _LOCK_STATUS:
        raise ControllerConfigError(
            f"status must be {_LOCK_STATUS!r}; refusing an unlocked campaign"
        )
    _nonnegative_int(config["campaign_seed"], "campaign_seed")
    if list(config["policies"]) != list(CONTROLLER_POLICIES):
        raise ControllerConfigError(
            f"policies must be prespecified as {list(CONTROLLER_POLICIES)}"
        )

    model = config["model"]
    if not isinstance(model, Mapping):
        raise ControllerConfigError("model must be an object")
    _require_exact_keys(
        model,
        {
            "conditional_dimension",
            "packet_size",
            "base_learning_rate",
            "reuse_radius",
            "warmup_packets",
        },
        "model",
    )
    dimension = _positive_int(model["conditional_dimension"], "conditional_dimension")
    packet_size = _positive_int(model["packet_size"], "packet_size")
    if packet_size <= dimension + 1:
        raise ControllerConfigError(
            "packet_size must exceed conditional_dimension + 1"
        )
    _finite_float(model["base_learning_rate"], "base_learning_rate", positive=True)
    _finite_float(model["reuse_radius"], "reuse_radius", positive=True)
    _nonnegative_int(model["warmup_packets"], "warmup_packets")

    permutations = config["permutations"]
    if not isinstance(permutations, Mapping):
        raise ControllerConfigError("permutations must be an object")
    _require_exact_keys(
        permutations,
        {"replicas", "draws_per_replica", "seed_salt"},
        "permutations",
    )
    replicas = _positive_int(permutations["replicas"], "permutations.replicas")
    if replicas < 2:
        raise ControllerConfigError("permutations.replicas must be at least two")
    _positive_int(
        permutations["draws_per_replica"], "permutations.draws_per_replica"
    )
    _nonnegative_int(permutations["seed_salt"], "permutations.seed_salt")

    controller = config["controller"]
    if not isinstance(controller, Mapping):
        raise ControllerConfigError("controller must be an object")
    _require_exact_keys(
        controller,
        {"epsilon", "pairing_reliability_threshold"},
        "controller",
    )
    _finite_float(controller["epsilon"], "controller.epsilon", positive=True)
    reliability_threshold = _finite_float(
        controller["pairing_reliability_threshold"],
        "controller.pairing_reliability_threshold",
    )
    if not 0.0 < reliability_threshold < 1.0:
        raise ControllerConfigError(
            "controller.pairing_reliability_threshold must lie strictly between 0 and 1"
        )

    streams = config["streams"]
    if not isinstance(streams, Mapping) or set(streams) != set(_STREAM_KINDS):
        raise ControllerConfigError(
            f"streams must contain exactly {list(_STREAM_KINDS)}"
        )
    mean = streams["mean_drift"]
    reversal = streams["recurrent_reversal"]
    weak = streams["weak_noisy_pairing"]
    _require_exact_keys(
        mean,
        {"kind", "mean_start", "mean_drift_per_packet", "noise_std"},
        "streams.mean_drift",
    )
    if mean["kind"] != "mean_drift":
        raise ControllerConfigError("mean_drift stream has the wrong kind")
    _finite_float(mean["mean_start"], "mean_drift.mean_start")
    _finite_float(mean["mean_drift_per_packet"], "mean_drift.mean_drift_per_packet")
    noise = _finite_float(mean["noise_std"], "mean_drift.noise_std")
    if noise < 0.0:
        raise ControllerConfigError("mean_drift.noise_std must be nonnegative")

    _require_exact_keys(
        reversal,
        {
            "kind",
            "mean_start",
            "mean_drift_per_packet",
            "slope",
            "reversal_period_packets",
            "noise_std",
        },
        "streams.recurrent_reversal",
    )
    if reversal["kind"] != "recurrent_reversal":
        raise ControllerConfigError("recurrent_reversal stream has the wrong kind")
    _finite_float(reversal["mean_start"], "recurrent_reversal.mean_start")
    _finite_float(
        reversal["mean_drift_per_packet"],
        "recurrent_reversal.mean_drift_per_packet",
    )
    _finite_float(reversal["slope"], "recurrent_reversal.slope", positive=True)
    period = _positive_int(
        reversal["reversal_period_packets"],
        "recurrent_reversal.reversal_period_packets",
    )
    if period < 2:
        raise ControllerConfigError(
            "recurrent reversal regimes must persist for at least two packets"
        )
    noise = _finite_float(reversal["noise_std"], "recurrent_reversal.noise_std")
    if noise < 0.0:
        raise ControllerConfigError("recurrent_reversal.noise_std must be nonnegative")

    _require_exact_keys(
        weak,
        {"kind", "mean_start", "mean_drift_per_packet", "slope", "noise_std"},
        "streams.weak_noisy_pairing",
    )
    if weak["kind"] != "weak_noisy_pairing":
        raise ControllerConfigError("weak_noisy_pairing stream has the wrong kind")
    _finite_float(weak["mean_start"], "weak_noisy_pairing.mean_start")
    _finite_float(
        weak["mean_drift_per_packet"], "weak_noisy_pairing.mean_drift_per_packet"
    )
    _finite_float(weak["slope"], "weak_noisy_pairing.slope", positive=True)
    weak_noise = _finite_float(
        weak["noise_std"], "weak_noisy_pairing.noise_std", positive=True
    )
    if weak_noise <= float(weak["slope"]):
        raise ControllerConfigError(
            "weak noisy stream must have noise larger than pairing slope"
        )

    ablations = config["ablations"]
    if not isinstance(ablations, Mapping):
        raise ControllerConfigError("ablations must be an object")
    _require_exact_keys(
        ablations, {"pairing_aware_no_reliability"}, "ablations"
    )
    reliability_ablation = ablations["pairing_aware_no_reliability"]
    if not isinstance(reliability_ablation, Mapping):
        raise ControllerConfigError("reliability ablation must be an object")
    _require_exact_keys(
        reliability_ablation,
        {"enabled_streams", "pairing_reliability_threshold"},
        "ablations.pairing_aware_no_reliability",
    )
    if reliability_ablation["enabled_streams"] != ["weak_noisy_pairing"]:
        raise ControllerConfigError(
            "reliability ablation is locked to weak_noisy_pairing"
        )
    if float(reliability_ablation["pairing_reliability_threshold"]) != 0.0:
        raise ControllerConfigError("no-reliability ablation threshold must be zero")

    run_keys = {
        "seeds",
        "packets",
        "scored_after_packets",
        "bootstrap_draws",
        "inference_unit",
        "retain_packet_records",
    }
    validated_runs: dict[str, dict[str, Any]] = {}
    for name in ("full", "smoke"):
        spec = config[name]
        if not isinstance(spec, Mapping):
            raise ControllerConfigError(f"{name} must be an object")
        _require_exact_keys(spec, run_keys, name)
        seeds = spec["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ControllerConfigError(f"{name}.seeds must be distinct nonnegative integers")
        packets = _positive_int(spec["packets"], f"{name}.packets")
        scored_after = _nonnegative_int(
            spec["scored_after_packets"], f"{name}.scored_after_packets"
        )
        if scored_after >= packets:
            raise ControllerConfigError(
                f"{name}.scored_after_packets must be smaller than packets"
            )
        _positive_int(spec["bootstrap_draws"], f"{name}.bootstrap_draws")
        if spec["inference_unit"] != "independent_stream_seed":
            raise ControllerConfigError(
                f"{name}.inference_unit must be independent_stream_seed"
            )
        if not isinstance(spec["retain_packet_records"], bool):
            raise ControllerConfigError(f"{name}.retain_packet_records must be boolean")
        validated_runs[name] = dict(spec)

    if set(validated_runs["full"]["seeds"]).intersection(validated_runs["smoke"]["seeds"]):
        raise ControllerConfigError("smoke and full seeds must be disjoint")
    if validated_runs["smoke"]["packets"] >= validated_runs["full"]["packets"]:
        raise ControllerConfigError("smoke packets must be fewer than full packets")
    if (
        validated_runs["smoke"]["bootstrap_draws"]
        >= validated_runs["full"]["bootstrap_draws"]
    ):
        raise ControllerConfigError("smoke bootstrap draws must be fewer than full draws")

    # Return a detached plain dictionary so later mutation of the caller's object
    # cannot change the validated campaign during execution.
    return deepcopy(dict(config))


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(payload)


def _seed(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(part) for part in parts]))


def _centered_orthonormal_design(
    rng: np.random.Generator, rows: int, dimension: int
) -> np.ndarray:
    raw = rng.normal(size=(rows, dimension))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw, mode="reduced")
    # Cov(X) = I under the population (1/n) convention used by the gradients.
    return math.sqrt(rows) * q[:, :dimension]


def generate_stream(
    name: str,
    stream_config: Mapping[str, Any],
    *,
    seed: int,
    packets: int,
    packet_size: int,
    conditional_dimension: int,
) -> Stream:
    """Generate a deterministic mean-drift or recurrent-reversal stream."""

    if name not in _STREAM_KINDS or stream_config.get("kind") != name:
        raise ValueError(f"unknown or mismatched stream {name!r}")
    if packets < 2 or packet_size < 2 or conditional_dimension < 1:
        raise ValueError("invalid stream dimensions")
    kind_id = _STREAM_KINDS.index(name) + 1
    rng = _seed(seed, kind_id, 0xC0FFEE)

    if name == "mean_drift":
        # The target depends only on a drifting intercept.  Centered nuisance
        # covariates are generated independently of the targets, so their
        # population pairing is zero even though a finite packet generally has
        # a nonzero sample covariance.  Full-gradient cosine reuse carries that
        # sampling noise along with the useful intercept; the two-channel rule
        # can retain the preserved intercept while suppressing an unreliable
        # pairing component.
        features = np.empty(
            (packets, packet_size, conditional_dimension + 1), dtype=np.float64
        )
        targets = np.empty((packets, packet_size), dtype=np.float64)
        means: list[float] = []
        for packet in range(packets):
            nuisance = _centered_orthonormal_design(
                rng, packet_size, conditional_dimension
            )
            features[packet] = np.column_stack(
                (np.ones(packet_size), nuisance)
            )
            mean = float(stream_config["mean_start"]) + packet * float(
                stream_config["mean_drift_per_packet"]
            )
            targets[packet] = mean + rng.normal(
                scale=float(stream_config["noise_std"]), size=packet_size
            )
            means.append(mean)
        return Stream(
            name=name,
            features=features,
            targets=targets,
            metadata={
                "target_means": means,
                "population_pairing_is_zero": True,
                "realized_pairing_source": "finite_packet_sampling_noise",
                "nuisance_features_are_target_independent": True,
            },
        )

    features = np.empty(
        (packets, packet_size, conditional_dimension + 1), dtype=np.float64
    )
    targets = np.empty((packets, packet_size), dtype=np.float64)
    beta = np.ones(conditional_dimension, dtype=np.float64)
    beta /= np.linalg.norm(beta)
    period = (
        int(stream_config["reversal_period_packets"])
        if name == "recurrent_reversal"
        else packets + 1
    )
    signs: list[int] = []
    means = []
    for packet in range(packets):
        conditional = _centered_orthonormal_design(
            rng, packet_size, conditional_dimension
        )
        design = np.column_stack((np.ones(packet_size), conditional))
        sign = 1 if name == "weak_noisy_pairing" else (
            1 if (packet // period) % 2 == 0 else -1
        )
        mean = float(stream_config["mean_start"]) + packet * float(
            stream_config["mean_drift_per_packet"]
        )
        signal = sign * float(stream_config["slope"]) * (conditional @ beta)
        targets[packet] = mean + signal + rng.normal(
            scale=float(stream_config["noise_std"]), size=packet_size
        )
        features[packet] = design
        signs.append(sign)
        means.append(mean)
    reversal_boundaries = [
        packet
        for packet in range(1, packets - 1)
        if signs[packet] != signs[packet - 1]
    ]
    if name == "recurrent_reversal" and (
        not reversal_boundaries
        or any(signs[packet + 1] != signs[packet] for packet in reversal_boundaries)
    ):
        raise RuntimeError(
            "reversal regimes must persist through the action-evaluation packet"
        )
    return Stream(
        name=name,
        features=features,
        targets=targets,
        metadata={
            "target_means": means,
            "relationship_signs": signs,
            "reversal_boundaries": reversal_boundaries,
            "reversal_action_horizon_persistent": True,
            "reversal_period_packets": period,
            "weak_noisy_pairing": name == "weak_noisy_pairing",
            "conditional_direction": beta.tolist(),
        },
    )


def gradient_components(
    theta: np.ndarray, features: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    """Return the exact ordinary/preserved/pairing squared-loss decomposition.

    For ``L = (1 / 2n) sum_i (f_i - y_i)^2`` and Jacobian ``J=X``, the
    finite-permutation expectation is

    ``m = E_pi[g(X, y_pi)]`` and ``c = g - m = -Cov(J, y)``.

    The returned ``permutation_variance`` is the exact expected squared distance
    ``E_pi ||g_pi - m||^2 = S_J S_y / (n^2 (n-1))``.
    """

    theta = np.asarray(theta, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("one packet must have features [rows, dimension], targets [rows]")
    if features.shape[0] != targets.shape[0] or features.shape[1] != theta.shape[0]:
        raise ValueError("gradient component shapes do not match")
    rows = features.shape[0]
    if rows < 2:
        raise ValueError("permutation variance needs at least two rows")
    if not (
        np.isfinite(theta).all()
        and np.isfinite(features).all()
        and np.isfinite(targets).all()
    ):
        raise ValueError("gradient input contains non-finite values")

    predictions = features @ theta
    ordinary = features.T @ (predictions - targets) / rows
    preserved = features.T @ (predictions - targets.mean()) / rows
    pairing = ordinary - preserved
    centered_features = features - features.mean(axis=0, keepdims=True)
    centered_targets = targets - targets.mean()
    covariance = centered_features.T @ centered_targets / rows
    s_j = float(np.sum(centered_features * centered_features))
    s_y = float(centered_targets @ centered_targets)
    permutation_variance = s_j * s_y / (rows * rows * (rows - 1))
    pairing_strength = float(pairing @ pairing)
    residual = float(np.linalg.norm(ordinary - preserved - pairing))
    covariance_residual = float(np.linalg.norm(pairing + covariance))
    if not all(
        math.isfinite(value)
        for value in (
            s_j,
            s_y,
            permutation_variance,
            pairing_strength,
            residual,
            covariance_residual,
        )
    ):
        raise FloatingPointError("non-finite gradient decomposition")
    return {
        "ordinary": ordinary,
        "preserved": preserved,
        "pairing": pairing,
        "covariance_j_y": covariance,
        "pairing_strength": pairing_strength,
        "s_j": s_j,
        "s_y": s_y,
        "permutation_variance": permutation_variance,
        "decomposition_residual": residual,
        "covariance_residual": covariance_residual,
    }


def _cosine(left: np.ndarray | None, right: np.ndarray | None, epsilon: float) -> float:
    if left is None or right is None:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= epsilon:
        return 0.0
    value = float(left @ right / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _unit(vector: np.ndarray, epsilon: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        return np.zeros_like(vector)
    return vector / norm


def _pairing_gate_weights(
    *,
    preserved_cosine: float,
    pairing_cosine: float,
    pairing_strength: float,
    permutation_variance: float,
    reliability_threshold: float,
    epsilon: float,
) -> tuple[float, float, float, bool]:
    """Return the two channel weights with a non-cancelling reliability gate."""

    reliability = pairing_strength / (
        pairing_strength + permutation_variance + epsilon
    )
    channel_open = bool(
        pairing_cosine > 0.0 and reliability >= reliability_threshold
    )
    return (
        max(0.0, preserved_cosine),
        max(0.0, pairing_cosine) if channel_open else 0.0,
        reliability,
        channel_open,
    )


def _half_mse(theta: np.ndarray, features: np.ndarray, targets: np.ndarray) -> float:
    residual = features @ theta - targets
    return float(0.5 * np.mean(residual * residual))


def _permutation_gradients(
    theta: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    replicas: int,
    draws_per_replica: int,
    seed: int,
    packet_index: int,
    seed_salt: int,
) -> np.ndarray:
    rows = targets.shape[0]
    predictions = features @ theta
    output = np.empty(
        (replicas, draws_per_replica, theta.shape[0]), dtype=np.float64
    )
    for replica in range(replicas):
        rng = _seed(seed, seed_salt, packet_index, replica, 0xA11D17)
        for draw in range(draws_per_replica):
            permutation = rng.permutation(rows)
            output[replica, draw] = (
                features.T @ (predictions - targets[permutation]) / rows
            )
    return output


def _prediction_digest(prediction: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(prediction, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def run_policy(
    stream: Stream,
    policy: str,
    *,
    seed: int,
    model_config: Mapping[str, Any],
    permutation_config: Mapping[str, Any],
    controller_config: Mapping[str, Any],
    scored_after_packets: int,
    retain_packet_records: bool,
) -> dict[str, Any]:
    """Run one policy with prequential scores and action-aligned V shadows."""

    if policy not in CONTROLLER_POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    packets, rows, dimension = stream.features.shape
    if not 0 <= scored_after_packets < packets:
        raise ValueError("invalid scored_after_packets")
    theta = np.zeros(dimension, dtype=np.float64)
    learning_rate = float(model_config["base_learning_rate"])
    radius = float(model_config["reuse_radius"])
    warmup = int(model_config["warmup_packets"])
    epsilon = float(controller_config["epsilon"])
    reliability_threshold = float(
        controller_config["pairing_reliability_threshold"]
    )
    permutation_replicas = int(permutation_config["replicas"])
    draws_per_replica = int(permutation_config["draws_per_replica"])
    permutation_salt = int(permutation_config["seed_salt"])

    previous_ordinary: np.ndarray | None = None
    previous_preserved: np.ndarray | None = None
    previous_pairing: np.ndarray | None = None
    previous_permutations: np.ndarray | None = None
    previous_pairing_strength: float | None = None
    previous_permutation_variance: float | None = None
    pending_v_shadow: dict[str, Any] | None = None
    scored_mse: list[float] = []
    records: list[dict[str, Any]] = []
    v_identity_residuals: list[float] = []
    scored_v_total: list[float] = []
    scored_v_perm: list[float] = []
    scored_v_pair: list[float] = []
    scored_v_perm_replicas: list[np.ndarray] = []
    action_count = 0
    scored_action_count = 0
    full_cosines: list[float] = []
    preserved_cosines: list[float] = []
    pairing_cosines: list[float] = []
    preserved_weights: list[float] = []
    pairing_weights: list[float] = []
    reliabilities: list[float] = []
    pairing_channel_open_count = 0
    covariance_norms: list[float] = []
    permutation_variances: list[float] = []

    for packet_index in range(packets):
        features = stream.features[packet_index]

        # Chronology boundary: finalize the live prediction before revealing y_t.
        prediction = features @ theta
        prediction_digest = _prediction_digest(prediction)
        targets = stream.targets[packet_index]
        residual = prediction - targets
        mse = float(np.mean(residual * residual))
        half_mse = 0.5 * mse
        scored = packet_index >= scored_after_packets
        if scored:
            scored_mse.append(mse)

        # These shadows were created after packet t-1's ordinary update, at the
        # exact point where the live reuse action was applied.  Scoring them now
        # measures that action's one-packet-ahead utility rather than applying a
        # stored gradient a second time to an already adapted state.
        v_total: float | None = None
        v_perm: float | None = None
        v_pair: float | None = None
        v_perm_mc_se: float | None = None
        v_pair_mc_se: float | None = None
        v_perm_replica_values: list[float] | None = None
        v_identity_residual: float | None = None
        v_source_action_packet: int | None = None
        v_source_gradient_packet: int | None = None
        if pending_v_shadow is not None:
            baseline_risk = _half_mse(
                pending_v_shadow["baseline_theta"], features, targets
            )
            true_risk = _half_mse(
                pending_v_shadow["true_theta"], features, targets
            )
            permutation_thetas = pending_v_shadow["permutation_thetas"]
            permutation_risks = np.empty(
                (permutation_replicas, draws_per_replica), dtype=np.float64
            )
            for replica in range(permutation_replicas):
                for draw in range(draws_per_replica):
                    permutation_risks[replica, draw] = _half_mse(
                        permutation_thetas[replica, draw], features, targets
                    )
            replica_values = baseline_risk - permutation_risks.mean(axis=1)
            v_total = baseline_risk - true_risk
            v_perm = float(replica_values.mean())
            v_pair = v_total - v_perm
            v_perm_replica_values = replica_values.astype(float).tolist()
            v_perm_mc_se = float(
                np.std(replica_values, ddof=1) / math.sqrt(permutation_replicas)
            )
            v_pair_mc_se = v_perm_mc_se
            v_identity_residual = abs(v_total - (v_perm + v_pair))
            v_source_action_packet = int(pending_v_shadow["action_packet"])
            v_source_gradient_packet = int(
                pending_v_shadow["gradient_source_packet"]
            )
            v_identity_residuals.append(v_identity_residual)
            if scored:
                scored_v_total.append(v_total)
                scored_v_perm.append(v_perm)
                scored_v_pair.append(v_pair)
                scored_v_perm_replicas.append(replica_values)

        components = gradient_components(theta, features, targets)
        ordinary = components["ordinary"]
        preserved = components["preserved"]
        pairing = components["pairing"]
        covariance_norms.append(float(np.linalg.norm(components["covariance_j_y"])))
        permutation_variances.append(float(components["permutation_variance"]))
        permutations = _permutation_gradients(
            theta,
            features,
            targets,
            replicas=permutation_replicas,
            draws_per_replica=draws_per_replica,
            seed=seed,
            packet_index=packet_index,
            seed_salt=permutation_salt,
        )

        full_cosine = _cosine(previous_ordinary, ordinary, epsilon)
        preserved_cosine = _cosine(previous_preserved, preserved, epsilon)
        pairing_cosine = _cosine(previous_pairing, pairing, epsilon)
        full_cosines.append(full_cosine)
        preserved_cosines.append(preserved_cosine)
        pairing_cosines.append(pairing_cosine)

        preserved_weight = 0.0
        pairing_weight = 0.0
        reliability = 0.0
        pairing_channel_open = False
        direction = np.zeros_like(theta)
        eligible = packet_index >= warmup and previous_ordinary is not None
        if eligible and policy == "always":
            direction = previous_ordinary.copy()
        elif eligible and policy == "cosine" and full_cosine > 0.0:
            direction = previous_ordinary.copy()
        elif eligible and policy == "pairing_aware":
            if (
                previous_preserved is None
                or previous_pairing is None
                or previous_pairing_strength is None
                or previous_permutation_variance is None
            ):
                raise RuntimeError("eligible controller has no stored decomposition")
            (
                preserved_weight,
                pairing_weight,
                reliability,
                pairing_channel_open,
            ) = _pairing_gate_weights(
                preserved_cosine=preserved_cosine,
                pairing_cosine=pairing_cosine,
                pairing_strength=previous_pairing_strength,
                permutation_variance=previous_permutation_variance,
                reliability_threshold=reliability_threshold,
                epsilon=epsilon,
            )
            direction = (
                preserved_weight * previous_preserved
                + pairing_weight * previous_pairing
            )
        if pairing_channel_open:
            pairing_channel_open_count += 1

        # First take the common ordinary update.  Build all one-step audit
        # shadows from this exact post-update base, then apply the live policy.
        base_theta = theta - learning_rate * ordinary
        next_v_shadow: dict[str, Any] | None = None
        if eligible and previous_permutations is not None:
            permutation_thetas = np.empty(
                (permutation_replicas, draws_per_replica, dimension),
                dtype=np.float64,
            )
            for replica in range(permutation_replicas):
                for draw in range(draws_per_replica):
                    permutation_thetas[replica, draw] = (
                        base_theta
                        - radius
                        * _unit(previous_permutations[replica, draw], epsilon)
                    )
            next_v_shadow = {
                "baseline_theta": base_theta.copy(),
                "true_theta": base_theta
                - radius * _unit(previous_ordinary, epsilon),
                "permutation_thetas": permutation_thetas,
                "action_packet": packet_index,
                "gradient_source_packet": packet_index - 1,
            }

        theta = base_theta
        acted = float(np.linalg.norm(direction)) > epsilon
        if acted:
            theta = theta - radius * _unit(direction, epsilon)
            action_count += 1
            if scored:
                scored_action_count += 1
        preserved_weights.append(preserved_weight)
        pairing_weights.append(pairing_weight)
        reliabilities.append(reliability)

        record = {
            "packet": packet_index,
            "scored": scored,
            "prediction_sha256": prediction_digest,
            "mse": mse,
            "half_mse": half_mse,
            "gradient_norm": float(np.linalg.norm(ordinary)),
            "preserved_norm": float(np.linalg.norm(preserved)),
            "pairing_norm": float(np.linalg.norm(pairing)),
            "covariance_j_y_norm": float(
                np.linalg.norm(components["covariance_j_y"])
            ),
            "pairing_strength": float(components["pairing_strength"]),
            "s_j": float(components["s_j"]),
            "s_y": float(components["s_y"]),
            "permutation_variance": float(components["permutation_variance"]),
            "decomposition_residual": float(components["decomposition_residual"]),
            "covariance_residual": float(components["covariance_residual"]),
            "full_cosine": full_cosine,
            "preserved_cosine": preserved_cosine,
            "pairing_cosine": pairing_cosine,
            "stored_reliability_q_over_q_plus_v": reliability,
            "pairing_reliability_threshold": reliability_threshold,
            "pairing_channel_open": pairing_channel_open,
            "preserved_weight": preserved_weight,
            "pairing_weight": pairing_weight,
            "reuse_gradient_lag_packets": 1,
            "reuse_source_packet": packet_index - 1 if eligible else None,
            "reuse_direction_norm": float(np.linalg.norm(direction)),
            "reuse_displacement_norm": radius if acted else 0.0,
            "acted": acted,
            "v_total": v_total,
            "v_perm": v_perm,
            "v_pair": v_pair,
            "v_perm_mc_replica_values": v_perm_replica_values,
            "v_perm_mc_se": v_perm_mc_se,
            "v_pair_mc_se": v_pair_mc_se,
            "v_identity_residual": v_identity_residual,
            "v_source_action_packet": v_source_action_packet,
            "v_source_gradient_packet": v_source_gradient_packet,
        }
        if retain_packet_records:
            record["prediction"] = prediction.tolist()
            record["theta_after_update"] = theta.tolist()
        records.append(record)

        previous_ordinary = ordinary.copy()
        previous_preserved = preserved.copy()
        previous_pairing = pairing.copy()
        previous_permutations = permutations.copy()
        previous_pairing_strength = float(components["pairing_strength"])
        previous_permutation_variance = float(components["permutation_variance"])
        pending_v_shadow = next_v_shadow

    scored_array = np.asarray(scored_mse, dtype=np.float64)
    if scored_array.size != packets - scored_after_packets or not np.isfinite(scored_array).all():
        raise RuntimeError("prequential scoring produced an invalid horizon")
    if not scored_v_total:
        raise RuntimeError("no action-aligned V shadows reached the scored horizon")
    scored_opportunities = packets - max(scored_after_packets, warmup, 1)
    scored_opportunities = max(0, scored_opportunities)
    max_v_residual = max(v_identity_residuals, default=0.0)
    if max_v_residual > 1e-12:
        raise RuntimeError(f"V decomposition identity failed: residual={max_v_residual}")
    replica_matrix = np.stack(scored_v_perm_replicas, axis=0)
    aggregate_replica_values = replica_matrix.mean(axis=0)
    aggregate_v_perm_mc_se = float(
        np.std(aggregate_replica_values, ddof=1)
        / math.sqrt(permutation_replicas)
    )

    return {
        "stream": stream.name,
        "stream_construction": {
            "population_pairing_is_zero": bool(
                stream.metadata.get("population_pairing_is_zero", False)
            ),
            "nuisance_features_are_target_independent": bool(
                stream.metadata.get(
                    "nuisance_features_are_target_independent", False
                )
            ),
            "realized_pairing_source": stream.metadata.get(
                "realized_pairing_source"
            ),
        },
        "reversal_action_horizon_persistent": bool(
            stream.metadata.get("reversal_action_horizon_persistent", True)
        ),
        "policy": policy,
        "seed": seed,
        "packets": packets,
        "packet_size": rows,
        "dimension": dimension,
        "scored_packets": int(scored_array.size),
        "prequential_mse": float(scored_array.mean()),
        "prequential_rmse": float(math.sqrt(scored_array.mean())),
        "cumulative_squared_error": float(rows * scored_array.sum()),
        "per_packet_mse": scored_array.tolist(),
        "actions": action_count,
        "action_rate": action_count / max(1, packets - max(warmup, 1)),
        "scored_action_rate": scored_action_count / max(1, scored_opportunities),
        "pairing_channel_open_rate": pairing_channel_open_count
        / max(1, packets - max(warmup, 1)),
        "mean_full_cosine": float(np.mean(full_cosines[1:])),
        "mean_preserved_cosine": float(np.mean(preserved_cosines[1:])),
        "mean_pairing_cosine": float(np.mean(pairing_cosines[1:])),
        "fraction_positive_full_cosine": float(
            np.mean(np.asarray(full_cosines[1:]) > 0.0)
        ),
        "fraction_negative_pairing_cosine": float(
            np.mean(np.asarray(pairing_cosines[1:]) < 0.0)
        ),
        "mean_stored_reliability": float(np.mean(reliabilities)),
        "mean_preserved_weight": float(np.mean(preserved_weights)),
        "mean_pairing_weight": float(np.mean(pairing_weights)),
        "mean_covariance_j_y_norm": float(np.mean(covariance_norms)),
        "mean_permutation_variance": float(np.mean(permutation_variances)),
        "v_decomposition": {
            "definition": "post_current_update_equal_radius_action_evaluated_on_next_packet",
            "permutation_action": "mean_risk_over_normalized_full_gradient_target_permutations",
            "permutation_replicas": permutation_replicas,
            "draws_per_replica": draws_per_replica,
            "audited_scored_packets": len(scored_v_total),
            "mean_v_total": float(np.mean(scored_v_total)),
            "mean_v_perm": float(np.mean(scored_v_perm)),
            "mean_v_pair": float(np.mean(scored_v_pair)),
            "sum_v_total": float(np.sum(scored_v_total)),
            "sum_v_perm": float(np.sum(scored_v_perm)),
            "sum_v_pair": float(np.sum(scored_v_pair)),
            "v_perm_nested_mc_se": aggregate_v_perm_mc_se,
            "v_pair_nested_mc_se": aggregate_v_perm_mc_se,
            "identity_residual": abs(
                float(np.sum(scored_v_total))
                - float(np.sum(scored_v_perm))
                - float(np.sum(scored_v_pair))
            ),
        },
        "max_v_identity_residual": max_v_residual,
        "prediction_before_target_update": True,
        "reuse_gradient_lag_packets": 1,
        "gate_observes_current_target_only_after_scoring": True,
        "v_shadow_matches_post_current_update_action_timing": True,
        # Compact audit records are always retained.  Smoke optionally adds the
        # prediction vector and parameter state for prefix-invariance tests.
        "packet_records": records,
        "detailed_packet_payload_retained": bool(retain_packet_records),
    }


def paired_seed_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Deterministic paired bootstrap over independent stream seeds.

    Positive effects mean that the candidate has lower prequential MSE than the
    baseline.  Packet observations within a trajectory are not treated as
    independent replicates: each seed is first reduced to one mean and the
    paired seed effects are then resampled.
    """

    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 2 or candidate.shape != baseline.shape:
        raise ValueError("bootstrap inputs must have equal [seeds, packets] shapes")
    if baseline.size == 0 or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("bootstrap inputs must be nonempty and finite")
    if draws <= 0 or baseline.shape[0] < 2:
        raise ValueError("bootstrap requires positive draws and at least two seeds")
    if baseline.mean() <= 0.0:
        raise ValueError("relative improvement requires positive baseline MSE")

    rng = _seed(seed, 0xB00757)
    seeds = baseline.shape[0]
    baseline_by_seed = baseline.mean(axis=1)
    candidate_by_seed = candidate.mean(axis=1)
    effects = np.empty(draws, dtype=np.float64)
    relative = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed_indices = rng.integers(0, seeds, size=seeds)
        baseline_mean = float(baseline_by_seed[seed_indices].mean())
        candidate_mean = float(candidate_by_seed[seed_indices].mean())
        effects[draw] = baseline_mean - candidate_mean
        relative[draw] = effects[draw] / baseline_mean

    point_baseline = float(baseline.mean())
    point_candidate = float(candidate.mean())
    point_effect = point_baseline - point_candidate
    return {
        "baseline_mse": point_baseline,
        "candidate_mse": point_candidate,
        "mse_improvement": point_effect,
        "mse_improvement_ci95": np.quantile(effects, [0.025, 0.975]).tolist(),
        "relative_mse_improvement": point_effect / point_baseline,
        "relative_mse_improvement_ci95": np.quantile(
            relative, [0.025, 0.975]
        ).tolist(),
        "draws": draws,
        "bootstrap_unit": "independent_stream_seed",
        "independent_seed_count": seeds,
        "paired_seed_effects": (
            baseline_by_seed - candidate_by_seed
        ).astype(float).tolist(),
        "seed": seed,
    }


def _json_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_json_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_campaign(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    source_binding: ValidatedEvidence | None = None,
    runtime_binding: ValidatedEvidence | None = None,
) -> dict[str, Any]:
    """Run the prespecified two-stream controller campaign."""

    locked = validate_config(config)
    mode = "smoke" if smoke else "full"
    spec = locked[mode]
    model = locked["model"]
    all_units: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    identity_residual = 0.0

    for stream_index, (stream_name, stream_config) in enumerate(
        locked["streams"].items()
    ):
        by_policy: dict[str, list[list[float]]] = {
            policy: [] for policy in CONTROLLER_POLICIES
        }
        ablation_errors: list[list[float]] = []
        ablation_units: list[dict[str, Any]] = []
        stream_units: list[dict[str, Any]] = []
        for seed in spec["seeds"]:
            stream = generate_stream(
                stream_name,
                stream_config,
                seed=seed,
                packets=spec["packets"],
                packet_size=model["packet_size"],
                conditional_dimension=model["conditional_dimension"],
            )
            for policy in CONTROLLER_POLICIES:
                unit = run_policy(
                    stream,
                    policy,
                    seed=seed,
                    model_config=model,
                    permutation_config=locked["permutations"],
                    controller_config=locked["controller"],
                    scored_after_packets=spec["scored_after_packets"],
                    retain_packet_records=spec["retain_packet_records"],
                )
                identity_residual = max(
                    identity_residual, float(unit["max_v_identity_residual"])
                )
                by_policy[policy].append(unit["per_packet_mse"])
                stream_units.append(unit)
                all_units.append(unit)
            reliability_ablation = locked["ablations"][
                "pairing_aware_no_reliability"
            ]
            if stream_name in reliability_ablation["enabled_streams"]:
                ablation_controller = deepcopy(locked["controller"])
                ablation_controller["pairing_reliability_threshold"] = float(
                    reliability_ablation["pairing_reliability_threshold"]
                )
                ablation_unit = run_policy(
                    stream,
                    "pairing_aware",
                    seed=seed,
                    model_config=model,
                    permutation_config=locked["permutations"],
                    controller_config=ablation_controller,
                    scored_after_packets=spec["scored_after_packets"],
                    retain_packet_records=spec["retain_packet_records"],
                )
                ablation_unit["policy"] = "pairing_aware_no_reliability"
                ablation_unit["ablation_of"] = "pairing_aware"
                ablation_unit["scientific_role"] = (
                    "prespecified_finite_permutation_reliability_ablation"
                )
                ablation_errors.append(ablation_unit["per_packet_mse"])
                ablation_units.append(ablation_unit)
                all_units.append(ablation_unit)

        baseline = np.asarray(by_policy["never"], dtype=np.float64)
        stream_comparisons: dict[str, Any] = {}
        for policy_index, policy in enumerate(CONTROLLER_POLICIES[1:], start=1):
            candidate = np.asarray(by_policy[policy], dtype=np.float64)
            policy_comparison = paired_seed_bootstrap(
                baseline,
                candidate,
                seed=(
                    int(locked["campaign_seed"])
                    + 1009 * stream_index
                    + 97 * policy_index
                    + (1 if smoke else 0)
                ),
                draws=spec["bootstrap_draws"],
            )
            # This is a comparator contrast, not online regret: never-reuse is
            # not assumed to be the best action sequence in hindsight.
            policy_comparison[
                "cumulative_prequential_squared_error_difference_vs_never"
            ] = float(
                model["packet_size"] * np.sum(candidate - baseline)
            )
            policy_comparison[
                "mean_cumulative_prequential_squared_error_difference_per_seed"
            ] = float(
                model["packet_size"]
                * np.sum(candidate - baseline)
                / candidate.shape[0]
            )
            stream_comparisons[policy] = policy_comparison

        for wedge_index, (baseline_policy, label) in enumerate(
            (("cosine", "pairing_aware_vs_cosine"),
             ("always", "pairing_aware_vs_always")),
            start=1,
        ):
            stream_comparisons[label] = paired_seed_bootstrap(
                np.asarray(by_policy[baseline_policy], dtype=np.float64),
                np.asarray(by_policy["pairing_aware"], dtype=np.float64),
                seed=(
                    int(locked["campaign_seed"])
                    + 1009 * stream_index
                    + 7919 * wedge_index
                    + (1 if smoke else 0)
                ),
                draws=spec["bootstrap_draws"],
            )
            stream_comparisons[label]["baseline"] = baseline_policy
            stream_comparisons[label]["candidate"] = "pairing_aware"
            stream_comparisons[label][
                "positive_improvement_favors"
            ] = "pairing_aware"

        if ablation_units:
            reliability_comparison = paired_seed_bootstrap(
                np.asarray(ablation_errors, dtype=np.float64),
                np.asarray(by_policy["pairing_aware"], dtype=np.float64),
                seed=(
                    int(locked["campaign_seed"])
                    + 1009 * stream_index
                    + 104729
                    + (1 if smoke else 0)
                ),
                draws=spec["bootstrap_draws"],
            )
            reliability_comparison.update(
                {
                    "baseline": "pairing_aware_no_reliability",
                    "candidate": "pairing_aware",
                    "positive_improvement_favors": "pairing_aware",
                    "ablation_scope": "weak_noisy_pairing_only",
                    "mean_candidate_pairing_channel_open_rate": float(
                        np.mean(
                            [
                                unit["pairing_channel_open_rate"]
                                for unit in stream_units
                                if unit["policy"] == "pairing_aware"
                            ]
                        )
                    ),
                    "mean_ablation_pairing_channel_open_rate": float(
                        np.mean(
                            [
                                unit["pairing_channel_open_rate"]
                                for unit in ablation_units
                            ]
                        )
                    ),
                }
            )
            stream_comparisons["reliability_ablation"] = reliability_comparison

        pairing_units = [
            unit for unit in stream_units if unit["policy"] == "pairing_aware"
        ]
        stream_comparisons["controller_diagnostics"] = {
            "mean_full_cosine": float(
                np.mean([unit["mean_full_cosine"] for unit in pairing_units])
            ),
            "mean_pairing_cosine": float(
                np.mean([unit["mean_pairing_cosine"] for unit in pairing_units])
            ),
            "fraction_positive_full_cosine": float(
                np.mean(
                    [unit["fraction_positive_full_cosine"] for unit in pairing_units]
                )
            ),
            "fraction_negative_pairing_cosine": float(
                np.mean(
                    [unit["fraction_negative_pairing_cosine"] for unit in pairing_units]
                )
            ),
            "mean_pairing_weight": float(
                np.mean([unit["mean_pairing_weight"] for unit in pairing_units])
            ),
            "mean_stored_reliability": float(
                np.mean(
                    [unit["mean_stored_reliability"] for unit in pairing_units]
                )
            ),
            "mean_pairing_channel_open_rate": float(
                np.mean(
                    [unit["pairing_channel_open_rate"] for unit in pairing_units]
                )
            ),
            "mean_covariance_j_y_norm": float(
                np.mean(
                    [unit["mean_covariance_j_y_norm"] for unit in pairing_units]
                )
            ),
            "mean_permutation_variance": float(
                np.mean(
                    [unit["mean_permutation_variance"] for unit in pairing_units]
                )
            ),
        }
        stream_comparisons["v_decomposition_by_policy"] = {
            policy: {
                "mean_v_total": float(
                    np.mean(
                        [
                            unit["v_decomposition"]["mean_v_total"]
                            for unit in stream_units
                            if unit["policy"] == policy
                        ]
                    )
                ),
                "mean_v_perm": float(
                    np.mean(
                        [
                            unit["v_decomposition"]["mean_v_perm"]
                            for unit in stream_units
                            if unit["policy"] == policy
                        ]
                    )
                ),
                "mean_v_pair": float(
                    np.mean(
                        [
                            unit["v_decomposition"]["mean_v_pair"]
                            for unit in stream_units
                            if unit["policy"] == policy
                        ]
                    )
                ),
                "max_identity_residual": float(
                    max(
                        unit["v_decomposition"]["identity_residual"]
                        for unit in stream_units
                        if unit["policy"] == policy
                    )
                ),
            }
            for policy in CONTROLLER_POLICIES
        }
        comparisons[stream_name] = stream_comparisons

    integrity = {
        "locked_configuration": locked["status"] == _LOCK_STATUS,
        "all_predictions_prequential": all(
            unit["prediction_before_target_update"] for unit in all_units
        ),
        "all_v_shadows_action_aligned": all(
            unit["v_shadow_matches_post_current_update_action_timing"]
            for unit in all_units
        ),
        "persistent_reversal_regimes": all(
            unit["stream"] != "recurrent_reversal"
            or unit["reversal_action_horizon_persistent"]
            for unit in all_units
        ),
        "seed_level_inference": spec["inference_unit"] == "independent_stream_seed",
        "v_identity_exact": identity_residual <= 1e-12,
        "max_v_identity_residual": identity_residual,
        "all_outputs_finite": True,
        "mode_is_explicit": mode in ("smoke", "full"),
    }
    source_evidence = evidence_payload(source_binding, kind="source")
    runtime_evidence = evidence_payload(runtime_binding, kind="runtime")
    bindings_consistent = False
    if source_evidence is not None and runtime_evidence is not None:
        runtime_manifest = runtime_evidence.get("manifest", {})
        bindings_consistent = bool(
            runtime_manifest.get("source_identity")
            == bound_source_identity(source_evidence)
            and runtime_manifest.get("configuration_payload_sha256")
            == _canonical_sha256(locked)
            and runtime_manifest.get("observed")
            == observed_runtime(experiment="controller", config=locked)
        )
    evidence_eligible = bool(
        not smoke
        and source_evidence is not None
        and runtime_evidence is not None
        and bindings_consistent
    )
    integrity.update(
        {
            "validated_source_binding": source_evidence is not None,
            "validated_runtime_binding": runtime_evidence is not None,
            "provenance_bindings_consistent": bindings_consistent,
        }
    )
    payload = {
        "schema_version": "gradient_reuse.controller.results.v3",
        "status": "SMOKE_PASS" if smoke else "PASS",
        "mode": mode,
        "scientific_evidence_eligible": evidence_eligible,
        "warning": (
            "SMOKE OUTPUT: engineering validation only; not scientific evidence."
            if smoke
            else (
                None
                if evidence_eligible
                else "INELIGIBLE FULL OUTPUT: validated source and runtime bindings were not supplied."
            )
        ),
        "source_provenance": source_evidence,
        "runtime_provenance": runtime_evidence,
        "configuration_sha256": _canonical_sha256(locked),
        "policies": list(CONTROLLER_POLICIES),
        "prespecified_ablations": deepcopy(locked["ablations"]),
        "streams": list(_STREAM_KINDS),
        "run_spec": deepcopy(spec),
        "comparisons": comparisons,
        "integrity": integrity,
        "units": all_units,
    }
    integrity["all_outputs_finite"] = _json_finite(payload)
    required_integrity_checks = (
        "locked_configuration",
        "all_predictions_prequential",
        "all_v_shadows_action_aligned",
        "persistent_reversal_regimes",
        "seed_level_inference",
        "v_identity_exact",
        "all_outputs_finite",
        "mode_is_explicit",
    )
    if not all(bool(integrity[key]) for key in required_integrity_checks):
        raise RuntimeError(f"controller campaign integrity failure: {integrity}")
    if smoke and payload["scientific_evidence_eligible"]:
        raise RuntimeError("smoke output was incorrectly marked evidence-eligible")
    return payload
