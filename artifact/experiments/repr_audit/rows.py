"""Audit rows on the DINOv2 penultimate cache. All held-out accuracies use the identical
held-out all-regime set. Causal rows: predict before label, update after, insert to replay last.

X9 [N,257,768] fp16 (penultimate), POOLED [N,768] fp16 (frozen feature), LAB [N]. A batch is a
single regime (blocks align to batch). Oracle rows (O1,O2) use the true regime id; no other row does.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from .backbone import make_tail, LoRALinear, EMBED
from chartfilm.dpc.role_dual import RoleMLP

B, DRAW, CAP = 50, 32, 2000


class D:
    """Data + backbone handle shared by all rows (set once by run.py)."""
    X9 = POOLED = LAB = None
    m = None
    device = "cuda"

    @classmethod
    def x9(cls, idx):
        # in-RAM fp16 gather -> contiguous -> fp32 GPU. .contiguous() avoids a slow non-contig .to path.
        return cls.X9[torch.as_tensor(idx, dtype=torch.long)].contiguous().to(cls.device, torch.float32, non_blocking=True)

    @classmethod
    def pooled(cls, idx):
        return cls.POOLED[torch.as_tensor(idx, dtype=torch.long)].contiguous().to(cls.device, torch.float32, non_blocking=True)

    @classmethod
    def y(cls, idx):
        return torch.as_tensor(cls.LAB[idx], dtype=torch.long, device=cls.device)


class Reservoir:
    def __init__(self, seed):
        self.idx = np.full(CAP, -1, np.int64); self.ctx = np.zeros(CAP, np.int64)
        self.size = 0; self.seen = 0; self.rng = np.random.default_rng(seed)

    def add(self, idxs, ctxs):
        for i, c in zip(idxs, ctxs):
            self.seen += 1
            if self.size < CAP:
                s = self.size; self.size += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j >= CAP:
                    continue
                s = j
            self.idx[s] = i; self.ctx[s] = c

    def draw(self):
        if self.size == 0:
            return None
        sel = self.rng.integers(0, self.size, size=min(DRAW, self.size))
        return self.idx[sel], self.ctx[sel]

    def bytes(self):
        return int(self.size) * (257 * 768 * 2)     # penult footprint referenced


# --------------------------------------------------------------------------- #
# Multi-bank LoRA tail (K=1 unified; K=10 oracle bank; shared head)            #
# --------------------------------------------------------------------------- #
class MultiLoRALinear(nn.Module):
    def __init__(self, base, K, r, alpha):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(K, r, base.in_features))
        self.B = nn.Parameter(torch.zeros(K, base.out_features, r))
        for k in range(K):
            nn.init.kaiming_uniform_(self.A[k], a=5 ** 0.5)
        self.scale = alpha / r; self.active = 0

    def forward(self, x):
        a, b = self.A[self.active], self.B[self.active]
        return self.base(x) + (x @ a.t() @ b.t()) * self.scale


class MultiLoRATail(nn.Module):
    def __init__(self, m, n_classes, K=1, r=8, alpha=16, device="cuda"):
        super().__init__()
        self.b10 = copy.deepcopy(m.blocks[10]); self.b11 = copy.deepcopy(m.blocks[11])
        self.norm = copy.deepcopy(m.norm)
        for p in self.parameters():
            p.requires_grad_(False)
        self.mods = []
        for blk in (self.b10, self.b11):
            blk.attn.qkv = self._w(blk.attn.qkv, K, r, alpha)
            blk.mlp.fc1 = self._w(blk.mlp.fc1, K, r, alpha)
            blk.mlp.fc2 = self._w(blk.mlp.fc2, K, r, alpha)
        self.head = nn.Linear(EMBED, n_classes)
        self.K = K
        self.to(device)

    def _w(self, lin, K, r, alpha):
        w = MultiLoRALinear(lin, K, r, alpha); self.mods.append(w); return w

    def set_bank(self, k):
        for w in self.mods:
            w.active = int(k)

    def forward(self, x9):
        with torch.autocast("cuda", torch.bfloat16, enabled=(x9.device.type == "cuda")):
            x = self.b11(self.b10(x9)); x = self.norm(x)
            feat = x[:, 0]
            logits = self.head(feat)
        return logits.float(), feat.float()

    def lora_params(self):
        return [w.A for w in self.mods] + [w.B for w in self.mods]

    def n_lora_params(self):
        return sum(p.numel() for w in self.mods for p in (w.A, w.B))


def _ce(logits, y):
    return nn.functional.cross_entropy(logits, y)


# --------------------------------------------------------------------------- #
# F0 — frozen-feature reference (RoleMLP adapter + head on POOLED)             #
# --------------------------------------------------------------------------- #
class FrozenRef:
    name = "F0"

    def __init__(self, n_classes, seed, hidden=64, lr=1e-3, device="cuda"):
        torch.manual_seed(seed)
        self.adapter = RoleMLP(EMBED, hidden).to(device)
        self.head = nn.Linear(EMBED, n_classes).to(device)
        self.opt = torch.optim.AdamW(list(self.adapter.parameters()) + list(self.head.parameters()), lr=lr)
        self.res = Reservoir(seed + 1); self.dev = device

    def _logits(self, idx):
        return self.head(self.adapter(D.pooled(idx)))

    @torch.no_grad()
    def predict(self, idx, ctx):
        return self._logits(idx)

    def observe(self, idx, ctx):
        loss = _ce(self._logits(idx), D.y(idx))
        rb = self.res.draw()
        if rb is not None:
            loss = loss + _ce(self._logits(rb[0]), D.y(rb[0]))
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        self.res.add(idx, ctx)

    @torch.no_grad()
    def heldout_logits(self, idx):
        return self._logits(idx)

    def ledger(self):
        p = sum(x.numel() for x in list(self.adapter.parameters()) + list(self.head.parameters()))
        return dict(trainable=p, lora=0, replay_bytes=self.res.bytes())


# --------------------------------------------------------------------------- #
# LoRA rows: P0/Uflop (K1 r8), Ubyte (K1 r80), O1 (K10 oracle r8)              #
# --------------------------------------------------------------------------- #
class LoRARow:
    def __init__(self, name, n_classes, seed, K=1, r=8, lr=1e-3, oracle=False, device="cuda"):
        torch.manual_seed(seed)
        self.name = name; self.oracle = oracle; self.K = K; self.dev = device
        self.tail = MultiLoRATail(D.m, n_classes, K=K, r=r, device=device)
        self.opt = torch.optim.AdamW(self.tail.lora_params() + list(self.tail.head.parameters()), lr=lr)
        self.res = Reservoir(seed + 1)

    def _bank(self, ctx_batch):
        return int(ctx_batch[0]) if self.oracle else 0

    @torch.no_grad()
    def predict(self, idx, ctx):
        self.tail.set_bank(self._bank(ctx))
        return self.tail(D.x9(idx))[0]

    def observe(self, idx, ctx):
        self.tail.set_bank(self._bank(ctx))
        loss = _ce(self.tail(D.x9(idx))[0], D.y(idx))
        rb = self.res.draw()
        if rb is not None:
            if self.oracle:
                ri, rc = rb
                for c in np.unique(rc):
                    self.tail.set_bank(int(c)); mask = rc == c
                    loss = loss + _ce(self.tail(D.x9(ri[mask]))[0], D.y(ri[mask])) * (mask.mean())
                self.tail.set_bank(self._bank(ctx))
            else:
                loss = loss + _ce(self.tail(D.x9(rb[0]))[0], D.y(rb[0]))
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        self.res.add(idx, ctx)

    @torch.no_grad()
    def heldout_logits(self, idx, ctx=None):
        if self.oracle:
            out = torch.zeros(len(idx), self.tail.head.out_features, device=self.dev)
            for c in np.unique(ctx):
                self.tail.set_bank(int(c)); mask = ctx == c
                out[mask] = self.tail(D.x9(idx[mask]))[0]
            return out
        self.tail.set_bank(0)
        return self.tail(D.x9(idx))[0]

    def ledger(self):
        p = self.tail.n_lora_params() + sum(x.numel() for x in self.tail.head.parameters())
        return dict(trainable=p, lora=self.tail.n_lora_params(), replay_bytes=self.res.bytes())


# --------------------------------------------------------------------------- #
# P1 — shared bottleneck encoder adapter (byte-matched to P0's LoRA), no LoRA  #
# --------------------------------------------------------------------------- #
class BottleneckTail(nn.Module):
    def __init__(self, m, n_classes, d_b, device="cuda"):
        super().__init__()
        self.b10 = copy.deepcopy(m.blocks[10]); self.b11 = copy.deepcopy(m.blocks[11]); self.norm = copy.deepcopy(m.norm)
        for p in self.parameters():
            p.requires_grad_(False)
        self.down = nn.Linear(EMBED, d_b); self.up = nn.Linear(d_b, EMBED); self.act = nn.GELU()
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)      # identity init
        self.head = nn.Linear(EMBED, n_classes); self.to(device)

    def set_bank(self, k):
        pass

    def forward(self, x9):
        with torch.autocast("cuda", torch.bfloat16, enabled=(x9.device.type == "cuda")):
            x9 = x9 + self.up(self.act(self.down(x9)))                    # bottleneck adapter on tokens
            x = self.b11(self.b10(x9)); x = self.norm(x); feat = x[:, 0]
            logits = self.head(feat)
        return logits.float(), feat.float()

    def adapter_params(self):
        return list(self.down.parameters()) + list(self.up.parameters())


class BottleneckRow(LoRARow):
    def __init__(self, name, n_classes, seed, d_b=112, lr=1e-3, device="cuda"):
        torch.manual_seed(seed)
        self.name = name; self.oracle = False; self.K = 1; self.dev = device
        self.tail = BottleneckTail(D.m, n_classes, d_b, device=device)
        self.opt = torch.optim.AdamW(self.tail.adapter_params() + list(self.tail.head.parameters()), lr=lr)
        self.res = Reservoir(seed + 1)

    def ledger(self):
        ad = sum(x.numel() for x in self.tail.adapter_params())
        p = ad + sum(x.numel() for x in self.tail.head.parameters())
        return dict(trainable=p, lora=ad, replay_bytes=self.res.bytes())


# --------------------------------------------------------------------------- #
# O2 — oracle best-of-two nonexchangeable roles (FAST/SLOW)                    #
# --------------------------------------------------------------------------- #
class DualRoleRow:
    name = "O2"

    def __init__(self, n_classes, seed, r=8, lr_fast=3e-3, lr_slow=3e-4, device="cuda"):
        torch.manual_seed(seed)
        self.fast = MultiLoRATail(D.m, n_classes, K=1, r=r, device=device)   # current-heavy, high lr
        torch.manual_seed(seed + 100)
        self.slow = MultiLoRATail(D.m, n_classes, K=1, r=r, device=device)   # replay-consolidated, low lr
        self.optF = torch.optim.AdamW(self.fast.lora_params() + list(self.fast.head.parameters()), lr=lr_fast)
        self.optS = torch.optim.AdamW(self.slow.lora_params() + list(self.slow.head.parameters()), lr=lr_slow)
        self.res = Reservoir(seed + 1); self.dev = device

    @torch.no_grad()
    def predict(self, idx, ctx):
        return self.fast(D.x9(idx))[0]                      # deployed online pred = FAST

    def observe(self, idx, ctx):
        y = D.y(idx); rb = self.res.draw()
        lf = _ce(self.fast(D.x9(idx))[0], y)                # FAST current-heavy
        if rb is not None:
            lf = lf + 0.25 * _ce(self.fast(D.x9(rb[0]))[0], D.y(rb[0]))
        self.optF.zero_grad(set_to_none=True); lf.backward(); self.optF.step()
        ls = 0.25 * _ce(self.slow(D.x9(idx))[0], y)         # SLOW replay-consolidated
        if rb is not None:
            ls = ls + _ce(self.slow(D.x9(rb[0]))[0], D.y(rb[0]))
        self.optS.zero_grad(set_to_none=True); ls.backward(); self.optS.step()
        self.res.add(idx, ctx)

    @torch.no_grad()
    def heldout_logits_oracle(self, idx):
        """Oracle best-of-two: per example pick the role with lower loss vs its own label."""
        y = D.y(idx)
        lf = self.fast(D.x9(idx))[0]; ls = self.slow(D.x9(idx))[0]
        lossF = nn.functional.cross_entropy(lf, y, reduction="none")
        lossS = nn.functional.cross_entropy(ls, y, reduction="none")
        return torch.where((lossF <= lossS).unsqueeze(1), lf, ls)

    def ledger(self):
        p = self.fast.n_lora_params() + self.slow.n_lora_params() + \
            sum(x.numel() for x in list(self.fast.head.parameters()) + list(self.slow.head.parameters()))
        return dict(trainable=p, lora=self.fast.n_lora_params() + self.slow.n_lora_params(), replay_bytes=self.res.bytes())


# --------------------------------------------------------------------------- #
# OL — Online-LoRA (faithful-spirit reimplementation: online loss-weighted     #
#      parameter anchoring on the LoRA params; no task ids)                    #
# --------------------------------------------------------------------------- #
class OnlineLoRA(LoRARow):
    name = "OL"

    def __init__(self, n_classes, seed, r=8, lr=1e-3, device="cuda", lam=1.0):
        super().__init__("OL", n_classes, seed, K=1, r=r, lr=lr, oracle=False, device=device)
        self.lam = lam
        self.anchor = [p.detach().clone() for p in self.tail.lora_params()]
        self.imp = [torch.zeros_like(p) for p in self.tail.lora_params()]
        self.loss_ema = None

    def observe(self, idx, ctx):
        self.tail.set_bank(0)
        base = _ce(self.tail(D.x9(idx))[0], D.y(idx))
        rb = self.res.draw()
        loss = base
        if rb is not None:
            loss = loss + _ce(self.tail(D.x9(rb[0]))[0], D.y(rb[0]))
        # online importance-weighted anchor (Online-LoRA style regularizer)
        reg = sum((im * (p - a) ** 2).sum() for im, p, a in zip(self.imp, self.tail.lora_params(), self.anchor))
        self.opt.zero_grad(set_to_none=True)
        (loss + self.lam * reg).backward()
        # surprise-weighted importance update from the current loss landscape (grads BEFORE step)
        s = float(base.detach())
        self.loss_ema = s if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * s
        surprise = max(0.0, s - self.loss_ema)
        with torch.no_grad():
            for k, p in enumerate(self.tail.lora_params()):
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                self.imp[k] = 0.99 * self.imp[k] + 0.01 * surprise * g.abs()
        self.opt.step()
        with torch.no_grad():
            for k, p in enumerate(self.tail.lora_params()):
                self.anchor[k] = p.detach().clone()
        self.res.add(idx, ctx)
