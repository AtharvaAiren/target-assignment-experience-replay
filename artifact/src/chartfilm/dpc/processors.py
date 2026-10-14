"""Plastic + stable-memory processors + shared predictor (brief §6.5--6.6, §8 actuators).

Two learned processors (count fixed at 2, independent of slot count) plus a
raw/identity fallback, all read by ONE shared predictor:

  * Plastic P: a global residual actuator conditioned on the *current* causal
    conditioner --- fast, no memory lookup, overwrites its own past.
  * Stable S:  a global residual actuator whose "message" is retrieved from soft
    context-slot values (compact code e_k + value v_k), so an old key's return
    recalls its modulation.

Every actuator implements a unified interface ``forward(z, message) -> h`` and
exposes ``.msg_dim`` (the width of the per-example message it consumes):

  * ``film``       : bottleneck FiLM, message = (dgamma, beta), channel-affine.
  * ``direct``     : legacy V5 per-feature affine, message = (dgamma, beta).
  * ``concat_mlp`` : CROSS-coordinate MLP on [z, message] --- can express
    permutations / rotations that channel-affine FiLM cannot.

All are identity-initialized (h = z at step 0).  Inactive slot rows and their Adam
state are frozen BITWISE via ``index_copy_`` (never advanced-indexing ``copy_`` ---
the V3 bug), momentum included.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class FiLMBottleneck(nn.Module):
    """h = z + W_up * phi((1 + dgamma) ⊙ (W_down LN(z)) + beta); identity at init.
    message = concat(dgamma, beta), each of width r."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.r = r
        self.msg_dim = 2 * r
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)              # identity init: h == z

    def forward(self, z, message):
        dgamma, beta = message[:, : self.r], message[:, self.r :]
        u = self.down(self.ln(z))
        m = (1.0 + dgamma) * u + beta
        return z + self.up(self.act(m))


class DirectFeatureFiLM(nn.Module):
    """Legacy V5 per-feature affine: h = (1 + 3 tanh(dg)) ⊙ z + 3 tanh(db).
    Parameter-free; identity at message=0.  message = concat(dg, db), width d each."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.msg_dim = 2 * d

    def forward(self, z, message):
        dg, db = message[:, : self.d], message[:, self.d :]
        return (1.0 + 3.0 * torch.tanh(dg)) * z + 3.0 * torch.tanh(db)


class ConcatMLPActuator(nn.Module):
    """CROSS-coordinate residual MLP: h = z + W2 * phi(W1 [z, message]).

    Unlike channel-affine FiLM this mixes coordinates, so a per-context message can
    steer a per-context linear-ish transform (e.g. undo a coordinate permutation).
    Identity at init (zero output layer)."""

    def __init__(self, d: int, msg_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.d = d
        self.msg_dim = msg_dim
        self.fc1 = nn.Linear(d + msg_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)               # identity init: h == z

    def forward(self, z, message):
        return z + self.fc2(self.act(self.fc1(torch.cat([z, message], dim=1))))


def _make_actuator(actuator: str, d: int, r: int, msg_dim: int = 16):
    if actuator == "film":
        return FiLMBottleneck(d, r)
    if actuator == "direct":
        return DirectFeatureFiLM(d)
    if actuator == "concat_mlp":
        return ConcatMLPActuator(d, msg_dim=msg_dim)
    raise ValueError(f"unknown actuator {actuator!r}")


class PlasticProcessor(nn.Module):
    """Global fast actuator conditioned on the current causal conditioner (no memory)."""

    def __init__(self, d: int, r: int, cond_dim: int, hidden: int = 64,
                 actuator: str = "film", msg_dim: int = 16):
        super().__init__()
        self.film = _make_actuator(actuator, d, r, msg_dim)
        self.gen = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, self.film.msg_dim)
        )
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)           # message starts at 0

    def forward(self, z, cond):                     # cond [B, cond_dim] (detached input)
        return self.film(z, self.gen(cond))


class StableProcessor(nn.Module):
    """Global slow actuator whose message is retrieved from soft context slots.

    Per-slot learned rows live here as [k_max, *] parameters, indexed by slot id
    (shared with the bank).  Retrieval:
        c_i     = sum_k r_ik [e_k, log(1+n_k)]      (graph-connected via e_k)
        v_bar   = sum_k r_ik v_k                     (graph-connected via v_k)
        message = H_S(c_i) + v_bar
    """

    def __init__(self, d: int, r: int, k_max: int, code_dim: int = 8, hidden: int = 64,
                 actuator: str = "film", msg_dim: int = 16, slot_init_std: float = 0.0):
        super().__init__()
        self.r = r
        self.k_max = k_max
        self.code_dim = code_dim
        self.film = _make_actuator(actuator, d, r, msg_dim)
        self.mdim = self.film.msg_dim
        # neutral SEEDED slot init: small per-slot random messages break symmetry so
        # joint routing can specialize; identity (h=z) still holds because the actuator's
        # output projection is zero-initialized regardless of the message.
        self.e_rows = nn.Parameter(torch.randn(k_max, code_dim) * slot_init_std)
        self.v_rows = nn.Parameter(torch.randn(k_max, self.mdim) * slot_init_std)
        self.gen = nn.Sequential(
            nn.Linear(code_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, self.mdim)
        )
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)

    def retrieve(self, r_resp: torch.Tensor, n_log: torch.Tensor):
        """Per-example message [B, mdim] from detached responsibilities + counts."""
        code_full = torch.cat([self.e_rows, n_log[:, None]], dim=1)  # [K, code_dim+1]
        c_i = r_resp @ code_full                                     # [B, code_dim+1]
        v_bar = r_resp @ self.v_rows                                 # [B, mdim]
        return self.gen(c_i) + v_bar

    def forward(self, z, r_resp, n_log):
        return self.film(z, self.retrieve(r_resp, n_log))

    def apply_cv(self, z, c, v):
        """Stable residual from EXPLICIT retrieval inputs c [.,code_dim+1], v [.,mdim].
        Used by anchoring: c,v are frozen snapshots while shared params drift."""
        return self.film(z, self.gen(c) + v)

    @torch.no_grad()
    def slot_code_value(self, slot: int, n_log: torch.Tensor):
        """Detached (c_k, v_k) snapshot for a pure route to ``slot`` (anchoring)."""
        c = torch.cat([self.e_rows[slot].detach(), n_log[slot].reshape(1)])
        v = self.v_rows[slot].detach().clone()
        return c, v

    # -- bitwise inactive-row isolation (params + Adam state) -- #
    @property
    def _row_params(self):
        return {"e": self.e_rows, "v": self.v_rows}

    def snapshot_rows(self, opt) -> dict:
        snap = {}
        for name, p in self._row_params.items():
            snap[name + "_p"] = p.data.clone()
            st = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    snap[f"{name}_{key}"] = st[key].clone()
        return snap

    @torch.no_grad()
    def freeze_inactive(self, opt, active_mask: torch.Tensor, snap: dict) -> None:
        inactive = torch.where(~active_mask)[0]
        if inactive.numel() == 0:
            return
        for name, p in self._row_params.items():
            p.data.index_copy_(0, inactive, snap[name + "_p"].index_select(0, inactive))
            st = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                skey = f"{name}_{key}"
                if key in st and skey in snap:
                    st[key].index_copy_(0, inactive, snap[skey].index_select(0, inactive))

    @torch.no_grad()
    def reset_slot(self, slot: int, opt=None) -> None:
        self.e_rows[slot] = 0.0
        self.v_rows[slot] = 0.0
        if opt is not None:
            for p in self._row_params.values():
                st = opt.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        st[key][slot] = 0.0

    @torch.no_grad()
    def mask_inactive_grads(self, active_mask: torch.Tensor) -> None:
        inactive = torch.where(~active_mask)[0]
        if inactive.numel() == 0:
            return
        for p in self._row_params.values():
            if p.grad is not None:
                p.grad.index_fill_(0, inactive, 0.0)


@dataclass
class ProcessorDims:
    d: int
    r: int
    k_max: int
    plastic_cond_dim: int
    code_dim: int = 8
