"""Functional last-active anchors for the stable processor (brief §6.7).

For each slot we keep, from when it was last active, a small set of canonical
frozen features z_kj (chosen target-independently) plus the detached retrieval
snapshot (c_k^A, v_k^A) and the detached stable-residual target

    s_kj^anchor = stopgrad[ h^S(z_kj; c_k^A, v_k^A) - z_kj ].

While a slot is INACTIVE, we penalize drift of its stable response under the
(now-drifting) shared H_S/film params, holding (c_k^A, v_k^A) fixed:

    L_anchor = mean_{k in O_t, j} || h^S(z_kj; c_k^A, v_k^A) - z_kj - s_kj^anchor ||^2.

Active slots receive no anchor penalty (they may learn).  Snapshots are refreshed
only for active slots, only at the logged post-update event, and are bitwise
frozen throughout a slot's inactive interval.  Selection is target-independent,
so permuting labels never changes which features are anchored (§9.17).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .processors import StableProcessor


class AnchorStore:
    def __init__(self, k_max: int, d: int, m_anchor: int = 4, code_dim: int = 8,
                 r: int = 4, seed: int = 0):
        self.k_max = k_max
        self.d = d
        self.m = m_anchor
        self.code_dim = code_dim
        self.r = r
        self.has = np.zeros(k_max, dtype=bool)
        self.z = [None] * k_max            # [M, d] canonical features (tensor)
        self.c = [None] * k_max            # [code_dim+1] frozen retrieval code
        self.v = [None] * k_max            # [2r] frozen retrieval value
        self.target = [None] * k_max       # [M, d] frozen residual target
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def refresh(self, slot: int, z_members: torch.Tensor, stable: StableProcessor,
                n_log: torch.Tensor) -> None:
        """Refresh slot's last-active snapshot from features routed to it (§6.10.11).

        ``z_members`` are current-batch features with responsibility on this slot;
        selection among them is target-independent (uniform)."""
        n = z_members.shape[0]
        if n == 0:
            return
        take = min(self.m, n)
        idx = self._rng.choice(n, size=take, replace=False)
        zk = z_members[idx].detach().clone()                    # [take, d]
        c, v = stable.slot_code_value(slot, n_log)              # detached
        c_b = c.unsqueeze(0).expand(take, -1)
        v_b = v.unsqueeze(0).expand(take, -1)
        resp = (stable.apply_cv(zk, c_b, v_b) - zk).detach()    # target residual
        self.z[slot] = zk
        self.c[slot] = c
        self.v[slot] = v
        self.target[slot] = resp
        self.has[slot] = True

    def anchor_loss(self, inactive_slots, stable: StableProcessor) -> torch.Tensor:
        """Sum-normalized functional drift over inactive anchored slots (0 if none)."""
        terms = []
        total = 0
        for k in inactive_slots:
            if not self.has[k]:
                continue
            zk = self.z[k]
            m = zk.shape[0]
            c_b = self.c[k].unsqueeze(0).expand(m, -1)
            v_b = self.v[k].unsqueeze(0).expand(m, -1)
            cur = stable.apply_cv(zk, c_b, v_b) - zk
            terms.append(((cur - self.target[k]) ** 2).sum())
            total += m
        if total == 0:
            return torch.zeros((), dtype=torch.float32)
        return torch.stack(terms).sum() / float(total)

    @torch.no_grad()
    def clear(self, slot: int) -> None:
        self.has[slot] = False
        self.z[slot] = self.c[slot] = self.v[slot] = self.target[slot] = None

    def bytes(self) -> int:
        b = 0
        for k in range(self.k_max):
            if self.has[k]:
                b += self.z[k].numel() * 4 + self.target[k].numel() * 4
                b += self.c[k].numel() * 4 + self.v[k].numel() * 4
        return b
