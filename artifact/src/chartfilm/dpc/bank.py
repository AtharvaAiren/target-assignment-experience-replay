"""Bounded distribution-keyed value bank + label-free lifecycle (brief §6.3--6.4).

This holds only the DETACHED slot state and the routing/lifecycle logic:

    kappa_k   running distribution key (recognizes a context)
    n_k       effective mass / count
    a_k       causal expert log-weights (raw / plastic / stable)  [updated by the gate]
    uuid_k    stable id across serialization
    age_k     last-touch clock
    allocated whether the slot exists

The LEARNED per-slot rows (compact code e_k, FiLM value v_act_k) live in the
stable processor (torch), indexed by the same slot ids.  The key recognizes a
context; the value stores context-specific computation; neither is "the memory"
alone.

Causal ordering (brief §5): route_predict + a cached label-free structural
*proposal* happen BEFORE predictions are saved; apply_structure + route_update +
update_keys happen AFTER.  Existing keys are only ever updated with r_update
(post-structure), never contaminated by the obsolete r_predict rows.

Primary allocation/merge/eviction may use only detached keys, feature-derived
responsibility/usage, age, and capacity --- never supervised values, losses,
targets, or class/task ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .entmax import entmax15
from .keys import DistributionKey


@dataclass
class BankConfig:
    k_max: int = 8
    lifecycle: str = "fixed_k_soft"        # fixed_k_soft | penalized_dynamic
    tau_b: float = 1.0                     # aggregate routing temperature
    tau_x: float = 1.0                     # per-example routing temperature
    key_lr: float = 0.05                   # responsibility-weighted key EMA rate
    lambda_slot: float = 0.5               # new-slot cost (penalized_dynamic)
    eviction: str = "lru"                  # lru | lfu (penalized_dynamic, at capacity)
    eps: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        if self.lifecycle not in ("fixed_k_soft", "penalized_dynamic"):
            raise ValueError(f"unknown lifecycle {self.lifecycle!r}")
        if self.eviction not in ("lru", "lfu"):
            raise ValueError(f"unknown eviction {self.eviction!r}")


@dataclass
class StructuralDecision:
    """A cached, label-free lifecycle action (computed pre-prediction, applied
    post-prediction).  ``kind`` in {none, allocate}."""

    kind: str = "none"
    new_key: Optional[np.ndarray] = None   # aggregate key for the new slot
    evict_slot: Optional[int] = None       # slot to evict to make room (or None)
    recon_cost: float = 0.0


class Bank:
    def __init__(self, cfg: BankConfig, key: DistributionKey):
        self.cfg = cfg
        self.key = key
        K = cfg.k_max
        self.kappa = np.zeros((K, key.dim), dtype=np.float64)
        self.n = np.zeros(K, dtype=np.float64)
        self.a = np.zeros((K, 3), dtype=np.float64)      # expert log-weights: raw, plastic, stable
        self.uuid = np.full(K, -1, dtype=np.int64)
        self.age = np.zeros(K, dtype=np.int64)
        self.allocated = np.zeros(K, dtype=bool)
        self._next_uuid = 0
        self._clock = 0
        self._rng = np.random.default_rng(cfg.seed)

    # ---------------- initialization ---------------- #
    @property
    def n_allocated(self) -> int:
        return int(self.allocated.sum())

    def _new_slot_id(self) -> Optional[int]:
        free = np.where(~self.allocated)[0]
        return int(free[0]) if free.size else None

    def _activate(self, slot: int, kappa: np.ndarray) -> None:
        self.kappa[slot] = kappa
        self.n[slot] = 0.0
        self.a[slot] = 0.0
        self.uuid[slot] = self._next_uuid
        self.age[slot] = self._clock
        self.allocated[slot] = True
        self._next_uuid += 1

    def initialize_fixed_k(self, cal_points: np.ndarray, k: Optional[int] = None) -> None:
        """Seed exactly k keys by deterministic k-means++ on saved (already-predicted)
        calibration-prefix per-example fingerprints; no later birth in fixed_k_soft."""
        k = k or self.cfg.k_max
        pts = np.asarray(cal_points, np.float64)
        assign, centers = _kmeanspp(pts, k, self._rng)
        for j in range(k):
            members = pts[assign == j]
            if members.shape[0] == 0:
                members = centers[j : j + 1]
            kappa = self.key.aggregate_points(members)
            self._activate(j, kappa)

    # ---------------- routing ---------------- #
    def _route(self, q_agg: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two-level entmax responsibilities over the CURRENT allocated slots.

        Returns (rho [K_max], r [B, K_max]); unallocated slots are exactly 0.
        """
        cfg = self.cfg
        alloc = np.where(self.allocated)[0]
        rho = np.zeros(self.cfg.k_max, dtype=np.float64)
        r = np.zeros((psi.shape[0], self.cfg.k_max), dtype=np.float64)
        if alloc.size == 0:
            return rho, r

        ka = self.kappa[alloc]
        agg_d = self.key.agg_distances(q_agg, ka)                 # [A]
        rho_a = entmax15(-agg_d / cfg.tau_b)                      # [A]
        rho[alloc] = rho_a

        point_d = self.key.point_distances(psi, ka)              # [B, A]
        scores = np.log(rho_a[None, :] + cfg.eps) - point_d / cfg.tau_x
        r_a = entmax15(scores, axis=1)                           # [B, A], exact zeros
        r[:, alloc] = r_a
        return rho, r

    def route_predict(self, q_agg, psi):
        return self._route(q_agg, psi)

    def route_update(self, q_agg, psi):
        return self._route(q_agg, psi)

    # ---------------- structural lifecycle (label-free) ---------------- #
    def propose_structure(self, q_agg: np.ndarray) -> StructuralDecision:
        """Cache a label-free structural action from the query + PRE-batch bank."""
        if self.cfg.lifecycle == "fixed_k_soft":
            return StructuralDecision(kind="none")

        # penalized_dynamic: allocate only if no convex combination of existing
        # keys reconstructs q_agg within the new-slot cost, and capacity permits.
        if self.n_allocated == 0:
            return StructuralDecision(kind="allocate", new_key=np.asarray(q_agg, np.float64))

        cost, _ = _simplex_reconstruction_cost(q_agg, self.kappa[self.allocated])
        if cost <= self.cfg.lambda_slot:
            return StructuralDecision(kind="none", recon_cost=cost)

        evict = None
        if self.n_allocated >= self.cfg.k_max:
            evict = self._eviction_target()
            if evict is None:
                return StructuralDecision(kind="none", recon_cost=cost)
        return StructuralDecision(kind="allocate", new_key=np.asarray(q_agg, np.float64),
                                  evict_slot=evict, recon_cost=cost)

    def apply_structure(self, decision: StructuralDecision) -> Optional[int]:
        """Apply a cached decision AFTER predictions are saved. Returns new slot id."""
        if decision.kind != "allocate":
            return None
        if decision.evict_slot is not None:
            self._evict(decision.evict_slot)
        slot = self._new_slot_id()
        if slot is None:
            return None
        self._activate(slot, np.asarray(decision.new_key, np.float64))
        return slot

    def _eviction_target(self) -> Optional[int]:
        alloc = np.where(self.allocated)[0]
        if alloc.size == 0:
            return None
        if self.cfg.eviction == "lru":
            return int(alloc[np.argmin(self.age[alloc])])
        return int(alloc[np.argmin(self.n[alloc])])          # lfu

    def _evict(self, slot: int) -> None:
        """Atomically reset a slot's detached state (learned rows reset by the
        processor via the same slot id)."""
        self.kappa[slot] = 0.0
        self.n[slot] = 0.0
        self.a[slot] = 0.0
        self.uuid[slot] = -1
        self.age[slot] = 0
        self.allocated[slot] = False

    # ---------------- causal key/count update ---------------- #
    def update_keys(self, q_agg: np.ndarray, r_update: np.ndarray) -> None:
        """Responsibility-weighted running key + count update (post-prediction).

        A slot with negligible responsibility gets a negligible update --- no
        class/task exception.  Uses the aggregate key toward which the batch's
        mass (weighted by r_update) points.
        """
        mass = r_update.sum(0)                                # [K_max] responsibility mass
        for k in np.where(self.allocated)[0]:
            mk = mass[k]
            if mk <= 0:
                continue
            rate = self.cfg.key_lr * mk / (mk + 1.0)         # bounded step
            self.kappa[k] = (1.0 - rate) * self.kappa[k] + rate * np.asarray(q_agg, np.float64)
            self.n[k] += mk
            self.age[k] = self._clock

    def advance_clock(self) -> None:
        self._clock += 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _kmeanspp(points: np.ndarray, k: int, rng: np.random.Generator,
              n_lloyd: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic-given-rng k-means++ init + a few Lloyd iterations."""
    n = points.shape[0]
    k = min(k, n)
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = points[first]
    d2 = np.sum((points - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[j] = points[idx]
        d2 = np.minimum(d2, np.sum((points - centers[j]) ** 2, axis=1))
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(n_lloyd):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = dists.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = points[m].mean(0)
    return assign, centers


def _simplex_reconstruction_cost(q: np.ndarray, keys: np.ndarray,
                                 n_iter: int = 50) -> tuple[float, np.ndarray]:
    """min_{w in simplex} 1/2 ||q - sum_k w_k kappa_k||^2  via Frank-Wolfe."""
    q = np.asarray(q, np.float64)
    K = np.asarray(keys, np.float64)                        # [k, dim]
    k = K.shape[0]
    w = np.full(k, 1.0 / k)
    for t in range(n_iter):
        resid = K.T @ w - q                                 # [dim]
        grad = K @ resid                                    # [k]
        j = int(np.argmin(grad))
        gamma = 2.0 / (t + 2.0)
        w = (1.0 - gamma) * w
        w[j] += gamma
    resid = K.T @ w - q
    return 0.5 * float(resid @ resid), w
