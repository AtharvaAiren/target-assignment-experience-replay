"""Strictly-causal soft contextual arbitration over {raw, plastic, stable} (§6.8).

Every gate here is DETACHED (numpy) and is NEVER an optimizer parameter: the only
mutation of the soft-MW log-weights is the explicit post-reveal multiplicative-
weights rule.  Weights used for batch t existed at the end of t-1 (causal).

Forward (pre-reveal):
    A_ib = sum_k r_predict_ik a_kb ;  w_ib = softmax_b(A_ib / tau_g)
    h_i  = w_i0 z_i + w_iP h^P_i + w_iS h^S_i ;  y_hat = g(h_i)   [in the learner]

Update (post-reveal), with unreduced nonneg per-expert losses l_ib:
    lbar_ib = l_ib / (sum_c l_ic + eps)  in [0,1]
    a_kb <- a_kb - eta * (1/B) * sum_i r_update_ik * lbar_ib      (occupancy-scaled)
then subtract the row mean of a_k (softmax-invariant, for numerical stability).
Zero-mass / unallocated slots are bitwise unchanged.

Experts are indexed 0=raw, 1=plastic, 2=stable.
"""

from __future__ import annotations

import numpy as np

from .entmax import softmax

RAW, PLASTIC, STABLE = 0, 1, 2
_NEG = -1e30


class Gate:
    """Common interface. ``mix_weights`` is causal (no labels); ``update`` is a
    no-op unless the gate has post-reveal state."""

    name = "gate"

    def mix_weights(self, r_predict: np.ndarray) -> np.ndarray:  # -> [B,3]
        raise NotImplementedError

    def update(self, r_update: np.ndarray, losses: np.ndarray) -> None:
        return None

    def reset_slot(self, slot: int) -> None:
        return None


class SoftMWGate(Gate):
    name = "soft_mw"

    def __init__(self, k_max: int, tau_g: float = 1.0, eta: float = 0.5,
                 use_raw: bool = True):
        self.k_max = k_max
        self.tau_g = float(tau_g)
        self.eta = float(eta)
        self.use_raw = bool(use_raw)
        self.a = np.zeros((k_max, 3), dtype=np.float64)     # detached log-weights
        if not use_raw:
            self.name = "soft_mw_no_raw"

    def _mask(self):
        m = np.ones(3, dtype=bool)
        if not self.use_raw:
            m[RAW] = False
        return m

    def mix_weights(self, r_predict: np.ndarray) -> np.ndarray:
        A = np.asarray(r_predict, np.float64) @ self.a       # [B,3]
        mask = np.broadcast_to(self._mask(), A.shape)
        return softmax(A / self.tau_g, axis=1, mask=mask)

    def reset_slot(self, slot: int) -> None:
        """Neutral (equal) expert log-weights for a (re)allocated slot."""
        self.a[slot] = 0.0

    def update(self, r_update: np.ndarray, losses: np.ndarray) -> None:
        r_update = np.asarray(r_update, np.float64)
        losses = np.asarray(losses, np.float64)
        B = losses.shape[0]
        lbar = losses / (losses.sum(1, keepdims=True) + 1e-8)   # [B,3] in [0,1]
        grad = (r_update.T @ lbar) / B                          # [K,3]
        mass = r_update.sum(0)                                  # [K]
        active = mass > 0
        if active.any():
            self.a[active] -= self.eta * grad[active]
            # subtract row mean over the *active experts* (softmax-invariant).
            m = self._mask()
            self.a[np.ix_(active, m)] -= self.a[np.ix_(active, m)].mean(1, keepdims=True)


class UniformGate(Gate):
    name = "uniform"

    def __init__(self, use_raw: bool = True):
        self.use_raw = use_raw

    def mix_weights(self, r_predict):
        B = np.asarray(r_predict).shape[0]
        w = np.zeros((B, 3))
        idx = [RAW, PLASTIC, STABLE] if self.use_raw else [PLASTIC, STABLE]
        w[:, idx] = 1.0 / len(idx)
        return w


class FixedSimplexGate(Gate):
    name = "fixed_simplex"

    def __init__(self, w: np.ndarray):
        w = np.asarray(w, np.float64)
        self.w = w / w.sum()

    def mix_weights(self, r_predict):
        B = np.asarray(r_predict).shape[0]
        return np.broadcast_to(self.w, (B, 3)).copy()


class AlwaysGate(Gate):
    def __init__(self, expert: int):
        self.expert = expert
        self.name = {RAW: "raw", PLASTIC: "plastic", STABLE: "stable"}[expert]

    def mix_weights(self, r_predict):
        B = np.asarray(r_predict).shape[0]
        w = np.zeros((B, 3))
        w[:, self.expert] = 1.0
        return w


class RandomGate(Gate):
    name = "random"

    def __init__(self, use_raw: bool = True, seed: int = 0):
        self.use_raw = use_raw
        self._rng = np.random.default_rng(seed)

    def mix_weights(self, r_predict):
        B = np.asarray(r_predict).shape[0]
        idx = [RAW, PLASTIC, STABLE] if self.use_raw else [PLASTIC, STABLE]
        w = np.zeros((B, 3))
        draw = self._rng.dirichlet(np.ones(len(idx)), size=B)
        w[:, idx] = draw
        return w


class OracleHindsightGate(Gate):
    """ILLEGAL diagnostic: picks, per example, the expert with the lowest CURRENT
    loss (uses the target before prediction).  Never a deployable comparator ---
    only an upper-bound diagnostic; the learner must pass current losses in."""

    name = "oracle_hindsight"

    def mix_weights(self, r_predict):
        raise RuntimeError("oracle_hindsight needs current losses; call pick_from_losses")

    @staticmethod
    def pick_from_losses(losses: np.ndarray) -> np.ndarray:
        losses = np.asarray(losses, np.float64)
        w = np.zeros_like(losses)
        w[np.arange(losses.shape[0]), losses.argmin(1)] = 1.0
        return w


def make_gate(name: str, k_max: int, **kw) -> Gate:
    name = name.lower()
    if name == "soft_mw":
        return SoftMWGate(k_max, use_raw=True, **{k: v for k, v in kw.items() if k in ("tau_g", "eta")})
    if name == "soft_mw_no_raw":
        return SoftMWGate(k_max, use_raw=False, **{k: v for k, v in kw.items() if k in ("tau_g", "eta")})
    if name == "uniform":
        return UniformGate(use_raw=kw.get("use_raw", True))
    if name == "fixed_simplex":
        return FixedSimplexGate(kw.get("w", np.array([1 / 3, 1 / 3, 1 / 3])))
    if name in ("raw", "plastic", "stable"):
        return AlwaysGate({"raw": RAW, "plastic": PLASTIC, "stable": STABLE}[name])
    if name == "random":
        return RandomGate(use_raw=kw.get("use_raw", True), seed=kw.get("seed", 0))
    if name == "oracle_hindsight":
        return OracleHindsightGate()
    raise ValueError(f"unknown gate {name!r}")
