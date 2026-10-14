"""entmax-1.5 (alpha=1.5 Tsallis) via exact bisection --- numpy, detached routing.

entmax gives continuous responsibilities on an *exact sparse support*: low-scoring
slots receive probability exactly 0, unlike softmax.  Routing coefficients are
detached buffers (they are multiplied against graph-connected learned values
downstream), so a numpy implementation is appropriate.

Solves  p = argmax_{p in simplex} <p, s> + H_{1.5}(p), whose solution is
    p_i = [ (alpha-1) s_i - tau ]_+^{1/(alpha-1)}     (alpha = 1.5 => exponent 2)
for the unique tau making the entries sum to 1.  We find tau by bisection on the
monotone-decreasing constraint g(tau) = sum_i [X_i - tau]_+^2 - 1, X = 0.5 s.
"""

from __future__ import annotations

import numpy as np

_ALPHA = 1.5
_EXP = 1.0 / (_ALPHA - 1.0)        # = 2
_NEG = -1e30                        # stand-in for -inf on masked slots


def entmax15(
    scores: np.ndarray,
    axis: int = -1,
    mask: np.ndarray | None = None,
    n_iter: int = 50,
) -> np.ndarray:
    """entmax-1.5 probabilities along ``axis`` with exact zeros.

    ``mask`` (bool, broadcastable to ``scores``): False entries are excluded
    (assigned probability exactly 0) *before* normalization.
    """
    s = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        s = np.where(mask, s, _NEG)

    X = (_ALPHA - 1.0) * s             # = 0.5 * s
    max_val = np.max(X, axis=axis, keepdims=True)

    tau_lo = max_val - 1.0             # g(tau_lo) >= 0  (max term alone gives p=1)
    tau_hi = max_val.copy()            # g(tau_hi) = -1  (all terms clipped to 0)

    for _ in range(n_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        p = np.clip(X - tau, 0.0, None) ** _EXP
        g = p.sum(axis=axis, keepdims=True) - 1.0
        pos = g > 0                    # g decreasing in tau -> raise the floor
        tau_lo = np.where(pos, tau, tau_lo)
        tau_hi = np.where(pos, tau_hi, tau)

    tau = 0.5 * (tau_lo + tau_hi)
    p = np.clip(X - tau, 0.0, None) ** _EXP
    denom = p.sum(axis=axis, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)      # degenerate guard (all masked)
    return p / denom


def softmax(scores: np.ndarray, axis: int = -1, mask: np.ndarray | None = None) -> np.ndarray:
    """Dense softmax control (temperature folded into scores upstream)."""
    s = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        s = np.where(mask, s, _NEG)
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    if mask is not None:
        e = np.where(mask, e, 0.0)
    return e / e.sum(axis=axis, keepdims=True)
