"""Addressing transform + label-free distribution keys (brief §6.1--6.2).

All of this is *detached* (numpy): keys recognize and address a context, they
never receive online supervised gradients and never repair a feature.  The
addressing coordinate system is frozen before the first persistent slot is
created, and never re-fit on the stream.

Descriptor families
-------------------
``mean``          : q = mean_i z_tilde                          (dim = d_r)
``diag_gaussian`` : q = [mean, causally-shrunk log-var]         (dim = 2*d_r)
``rff_kernel``    : q = mean_i sqrt(2/m) cos(Omega z_tilde + b) (dim = m)  [PRIMARY]
``deepsets``      : learned + frozen control                    (stub)

Each family owns its own discrepancy so distances are never compared across
families (brief §6.3): RFF -> 1/2||.||^2 (already m^-1/2 normalized, NOT divided
by m again); diag_gaussian -> normalized symmetric (Jeffreys) diagonal KL;
mean -> dim-normalized squared Euclidean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _sq_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances ||a_i - b_j||^2 -> [n_a, n_b], numerically clipped."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.clip(d2, 0.0, None)


# --------------------------------------------------------------------------- #
# Frozen addressing transform:  z_tilde = P D0^{-1} (z - mu0)
# --------------------------------------------------------------------------- #
@dataclass
class AddressTransform:
    """Frozen standardize-then-(optional)-project map into the address space.

    ``mode='identity'`` keeps z unchanged (primary when no reference stats exist).
    ``mode='random_projection'`` applies a fixed seeded Gaussian projection to
    ``d_r`` dims.  Per-example LayerNorm is intentionally NOT offered here (it
    erases the mean/scale shifts under study); it belongs in an ablation only.
    """

    in_dim: int
    mode: str = "identity"                 # identity | random_projection
    d_r: Optional[int] = None              # target dim for random_projection
    mu0: Optional[np.ndarray] = None       # frozen global mean (or None = 0)
    inv_d0: Optional[np.ndarray] = None    # frozen global 1/std (or None = 1)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("identity", "random_projection"):
            raise ValueError(f"unknown address mode {self.mode!r}")
        self._mu0 = np.zeros(self.in_dim, np.float64) if self.mu0 is None else np.asarray(self.mu0, np.float64)
        self._inv = np.ones(self.in_dim, np.float64) if self.inv_d0 is None else np.asarray(self.inv_d0, np.float64)
        if self.mode == "random_projection":
            dr = self.d_r or self.in_dim
            rng = np.random.default_rng(self.seed)
            self._P = rng.standard_normal((dr, self.in_dim)) / np.sqrt(self.in_dim)
            self.out_dim = dr
        else:
            self._P = None
            self.out_dim = self.in_dim

    def __call__(self, z: np.ndarray) -> np.ndarray:
        zt = (np.asarray(z, np.float64) - self._mu0) * self._inv
        if self._P is not None:
            zt = zt @ self._P.T
        return zt


# --------------------------------------------------------------------------- #
# Distribution keys
# --------------------------------------------------------------------------- #
class DistributionKey:
    """Base: encode a batch to per-example fingerprints + an aggregate key."""

    dim: int

    def encode(self, z_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def distance(self, qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        """Discrepancy D(qa, qb); broadcasts over a leading slot axis on qb."""
        raise NotImplementedError


class MeanKey(DistributionKey):
    def __init__(self, in_dim: int):
        self.dim = in_dim
        self.point_dim = in_dim

    def encode(self, z_tilde):
        zt = np.asarray(z_tilde, np.float64)
        return zt, zt.mean(0)

    def distance(self, qa, qb):
        diff = np.asarray(qb, np.float64) - np.asarray(qa, np.float64)
        return 0.5 * np.sum(diff * diff, axis=-1) / self.dim

    def agg_distances(self, q, keys):
        diff = np.asarray(keys, np.float64) - np.asarray(q, np.float64)[None, :]
        return 0.5 * np.sum(diff * diff, axis=1) / self.dim

    def point_distances(self, psi, keys):
        return _sq_dist_matrix(psi, keys) * (0.5 / self.dim)

    def aggregate_points(self, points):
        """Slot key from per-example point fingerprints (= their mean here)."""
        return np.asarray(points, np.float64).mean(0)


class DiagGaussianKey(DistributionKey):
    """Aggregate = [mean, causally-shrunk log-var]; distance = normalized sym-KL."""

    def __init__(self, in_dim: int, var_prior: float = 1.0, shrink: float = 4.0):
        self.in_dim = in_dim
        self.dim = 2 * in_dim
        self.point_dim = in_dim          # per-example fingerprint is the point z_tilde
        self.var_prior = float(var_prior)
        self.shrink = float(shrink)      # pseudo-count toward the prior (small-B safe)

    def encode(self, z_tilde):
        zt = np.asarray(z_tilde, np.float64)
        n = zt.shape[0]
        mean = zt.mean(0)
        var_batch = zt.var(0) if n > 1 else np.zeros(self.in_dim)
        var = (n * var_batch + self.shrink * self.var_prior) / (n + self.shrink)
        agg = np.concatenate([mean, np.log(var + 1e-8)])
        # per-example fingerprint: just the standardized coordinate (used rarely)
        return zt, agg

    def _split(self, q):
        q = np.asarray(q, np.float64)
        m = q[..., : self.in_dim]
        v = np.exp(q[..., self.in_dim :]) + 1e-8
        return m, v

    def distance(self, qa, qb):
        ma, va = self._split(qa)
        mb, vb = self._split(qb)
        d2 = (ma - mb) ** 2
        # symmetric (Jeffreys) diagonal-Gaussian KL; log terms cancel.
        sym = 0.5 * ((va + d2) / vb + (vb + d2) / va - 2.0)
        return np.sum(sym, axis=-1) / self.in_dim

    def agg_distances(self, q, keys):
        return np.array([self.distance(q, k) for k in np.asarray(keys, np.float64)])

    def point_distances(self, psi, keys):
        # per-example z_tilde compared to the mean part of each slot key.
        means = np.asarray(keys, np.float64)[:, : self.in_dim]
        return _sq_dist_matrix(psi, means) * (0.5 / self.in_dim)

    def aggregate_points(self, points):
        """Slot key = [mean, causally-shrunk log-var] from point fingerprints."""
        zt = np.asarray(points, np.float64)
        n = zt.shape[0]
        mean = zt.mean(0)
        var_batch = zt.var(0) if n > 1 else np.zeros(self.in_dim)
        var = (n * var_batch + self.shrink * self.var_prior) / (n + self.shrink)
        return np.concatenate([mean, np.log(var + 1e-8)])


class RFFKey(DistributionKey):
    """Random-Fourier-feature kernel mean embedding (PRIMARY, brief §6.2)."""

    def __init__(self, in_dim: int, m: int = 256, sigma: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Omega = rng.normal(0.0, 1.0 / sigma, size=(m, in_dim))  # rows ~ N(0, sigma^-2 I)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=m)
        self.m = m
        self.dim = m
        self.point_dim = m
        self.sigma = float(sigma)
        self._scale = np.sqrt(2.0 / m)                              # m^-1/2 normalization

    def embed(self, z_tilde: np.ndarray) -> np.ndarray:
        zt = np.asarray(z_tilde, np.float64)
        return self._scale * np.cos(zt @ self.Omega.T + self.b)

    def encode(self, z_tilde):
        psi = self.embed(z_tilde)
        return psi, psi.mean(0)

    def distance(self, qa, qb):
        # 1/2 || . ||^2 ; already m^-1/2 normalized, do NOT divide by m again.
        diff = np.asarray(qb, np.float64) - np.asarray(qa, np.float64)
        return 0.5 * np.sum(diff * diff, axis=-1)

    def agg_distances(self, q, keys):
        diff = np.asarray(keys, np.float64) - np.asarray(q, np.float64)[None, :]
        return 0.5 * np.sum(diff * diff, axis=1)

    def point_distances(self, psi, keys):
        return _sq_dist_matrix(psi, keys) * 0.5

    def aggregate_points(self, points):
        """Slot key = mean of the per-example RFF fingerprints (= the kernel mean)."""
        return np.asarray(points, np.float64).mean(0)

    @staticmethod
    def median_bandwidth(z_tilde: np.ndarray, max_pairs: int = 2000, seed: int = 0) -> float:
        """Median-heuristic RBF bandwidth from a causal calibration prefix."""
        zt = np.asarray(z_tilde, np.float64)
        n = zt.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(min(max_pairs, n * n), 2))
        d = np.linalg.norm(zt[idx[:, 0]] - zt[idx[:, 1]], axis=1)
        med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
        return med if med > 0 else 1.0


class DeepSetsKey(DistributionKey):
    """Learned permutation-invariant descriptor, frozen after label-free pretraining.

    Not implemented yet: it is a control that must be trained through a documented
    label-free objective on *reference/dev* data and then frozen (never on hidden
    contexts/classes, never drifting on the measured stream).  Stubbed to stay
    honest until that pretraining harness exists.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("deepsets key: build with the label-free pretrainer")


# --------------------------------------------------------------------------- #
# Plastic conditioner (separate from addressing; brief §6.2)
# --------------------------------------------------------------------------- #
class PlasticConditioner:
    """Conditioning input for the PLASTIC processor --- kept explicitly separate
    from the address key.  In the primary RFF candidate it uses the same *form*
    as an RFF key but with its own independent parameters/state, so a
    mean-vs-gaussian-vs-rff *address* comparison never also changes the actuator
    input.
    """

    def __init__(self, in_dim: int, kind: str = "rff", m: int = 128,
                 sigma: float = 1.0, seed: int = 1234):
        self.kind = kind
        if kind == "rff":
            self._k = RFFKey(in_dim, m=m, sigma=sigma, seed=seed)
        elif kind == "mean":
            self._k = MeanKey(in_dim)
        else:
            raise ValueError(f"unknown plastic conditioner kind {kind!r}")
        self.dim = self._k.dim

    def encode(self, z_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._k.encode(z_tilde)


def make_key(kind: str, in_dim: int, **kw) -> DistributionKey:
    kind = kind.lower()
    if kind == "mean":
        return MeanKey(in_dim)
    if kind in ("diag_gaussian", "gaussian"):
        return DiagGaussianKey(in_dim, **{k: v for k, v in kw.items() if k in ("var_prior", "shrink")})
    if kind in ("rff", "rff_kernel"):
        return RFFKey(in_dim, **{k: v for k, v in kw.items() if k in ("m", "sigma", "seed")})
    if kind == "deepsets":
        return DeepSetsKey()
    raise ValueError(f"unknown key kind {kind!r}")
