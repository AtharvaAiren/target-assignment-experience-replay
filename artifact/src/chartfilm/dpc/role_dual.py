"""Role-anchored dual learner (context-route-free).

Three FIXED, permanent, non-exchangeable roles arbitrated by a causal functional
router trained from previously-revealed generic losses + router replay:
  RAW     : identity frozen features (no trainable actuator)
  PLASTIC : one global cross-coordinate residual MLP, FAST timescale
  STABLE  : one global cross-coordinate residual MLP, SLOW/consolidating (EMA)

No context slots / chart keys / clustering / lifecycle / slot responsibilities /
context ids. P and S are updated by fixed gate-independent rules, independent of router
probabilities (the router never starves either). Same code for cls and reg. Prediction
is strictly pre-label; the current target never affects its own prediction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .keys import AddressTransform, RFFKey
from .replay import SharedReservoir
from ..methods.base import SharedPredictor, to_tensor
from .entmax import softmax

RAW, P, S = 0, 1, 2


class RoleMLP(nn.Module):
    """Global cross-coordinate residual adapter; identity-initialized."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return z + self.fc2(self.act(self.fc1(z)))


@dataclass
class RoleConfig:
    d: int
    n_classes: int
    task: str = "both"
    role_router: str = "functional"    # functional | reactive_r2 | uniform | fixed | oracle | always_P | always_S
    router_replay: bool = True
    shuffle_targets: bool = False
    identical_lr: bool = False          # D9: S lr = P lr
    widened_single: bool = False        # D4: one 2x-hidden adapter
    fixed_pi: Optional[tuple] = None     # D3
    hidden: int = 64
    rff_m: int = 256
    eta: float = 3.0
    eps: float = 0.05
    lr_p: float = 2e-2
    lr_s: float = 5e-3
    lr_pred: float = 1e-2
    router_lr: float = 1e-2
    ema_beta: float = 0.95
    lambda_preserve: float = 1e-3
    replay_capacity: int = 2000
    replay_size: int = 32
    replay_lambda: float = 1.0
    replay_k: int = 16
    calib_frac: float = 0.06
    device: str = "cpu"
    seed: int = 0


class RoleAnchoredDual:
    protocol = "minibatch_query"

    def __init__(self, cfg: RoleConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self._rng = np.random.default_rng(cfg.seed + 3)
        self.addr = AddressTransform(in_dim=cfg.d, mode="identity")
        self.predictor = SharedPredictor(cfg.d, cfg.n_classes).to(cfg.device)
        self.plastic = RoleMLP(cfg.d, cfg.hidden * (2 if cfg.widened_single else 1)).to(cfg.device)
        self.stable = None if cfg.widened_single else RoleMLP(cfg.d, cfg.hidden).to(cfg.device)
        lr_s = cfg.lr_p if cfg.identical_lr else cfg.lr_s
        groups = [{"params": self.predictor.parameters(), "lr": cfg.lr_pred},
                  {"params": self.plastic.parameters(), "lr": cfg.lr_p}]
        if self.stable is not None:
            groups.append({"params": self.stable.parameters(), "lr": lr_s})
        self.opt = torch.optim.Adam(groups)
        self.reservoir = SharedReservoir(cfg.replay_capacity, cfg.d, seed=cfg.seed + 1)
        self.key = None; self.router = None; self.router_opt = None
        self.p_bar = self.s_bar = self.router_bar = None
        self.s_scale = 1.0
        self._val_perm = self._rng.permutation(3)
        self._calibrating = True; self._calib_buf = []; self._seen = 0; self._n_calib = None
        self.wants_context = (cfg.role_router == "oracle")
        self._ctx = None
        self.r2_w = np.ones(3) / 3      # reactive fixed-share state
        self.stats = {"router_fwd": 0, "router_bwd": 0, "replay_role_fwd": 0}

    def set_context(self, ctx):
        self._ctx = np.asarray(ctx)

    def _finalize(self):
        cfg = self.cfg
        pref = np.concatenate(self._calib_buf) if self._calib_buf else np.zeros((1, cfg.d))
        zt = self.addr(pref)
        self.key = RFFKey(self.addr.out_dim, m=cfg.rff_m,
                          sigma=RFFKey.median_bandwidth(zt, seed=cfg.seed), seed=cfg.seed + 2)
        self.router = nn.Sequential(nn.LayerNorm(self.key.dim), nn.Linear(self.key.dim, 64),
                                    nn.GELU(), nn.Linear(64, 3)).to(cfg.device)
        self.router_opt = torch.optim.Adam(self.router.parameters(), lr=cfg.router_lr)
        self.router_bar = copy.deepcopy(self.router)
        for p in self.router_bar.parameters():
            p.requires_grad_(False)
        self.p_bar = copy.deepcopy(self.plastic); self.s_bar = None if self.stable is None else copy.deepcopy(self.stable)
        for m in (self.p_bar, self.s_bar):
            if m is not None:
                for p in m.parameters(): p.requires_grad_(False)
        self._calibrating = False; self._calib_buf = []

    # ---- role features / predictions ---- #
    def _roles(self, z_t, plastic=None, stable=None):
        pl = plastic or self.plastic; st = stable if stable is not None else self.stable
        hP = pl(z_t)
        hS = z_t if st is None else st(z_t)          # widened_single: S == raw placeholder (unused)
        return z_t, hP, hS

    def _pi(self, psi):
        cfg = self.cfg
        if cfg.role_router == "uniform":
            return np.full((psi.shape[0], 3), 1 / 3)
        if cfg.role_router == "fixed":
            return np.broadcast_to(np.asarray(cfg.fixed_pi), (psi.shape[0], 3)).copy()
        if cfg.role_router == "always_P":
            a = np.zeros((psi.shape[0], 3)); a[:, P] = 1.0; return a
        if cfg.role_router == "always_S":
            a = np.zeros((psi.shape[0], 3)); a[:, S] = 1.0; return a
        if cfg.role_router == "reactive_r2":
            return np.broadcast_to((1 - cfg.eps) * self.r2_w + cfg.eps / 3, (psi.shape[0], 3)).copy()
        if cfg.role_router == "oracle":
            return None                              # filled with per-example losses in predict-after-reveal? no -> handled specially
        with torch.no_grad():                        # functional
            self.stats["router_fwd"] += 1
            a = torch.softmax(self.router(to_tensor(psi, cfg.device)), dim=1).cpu().numpy()
        return (1 - cfg.eps) * a + cfg.eps / 3

    def predict(self, z):
        cfg = self.cfg
        self._seen += z.shape[0]
        if self._n_calib is None:
            self._n_calib = max(256, z.shape[0] * 2)
        if self._calibrating:
            self._calib_buf.append(np.asarray(z, np.float32))
            with torch.no_grad():
                lo, rg = self.predictor(to_tensor(z, cfg.device))
            self._stash = dict(boot=True, z=z)
            return lo.cpu().numpy(), rg.cpu().numpy()
        zt = self.addr(z); psi = self.key.embed(zt)
        z_t = to_tensor(z, cfg.device)
        with torch.no_grad():
            _, hP, hS = self._roles(z_t)
            logits = torch.stack([self.predictor(z_t)[0], self.predictor(hP)[0], self.predictor(hS)[0]])  # [3,B,C]
            regs = torch.stack([self.predictor(z_t)[1], self.predictor(hP)[1], self.predictor(hS)[1]])     # [3,B]
        if cfg.role_router == "oracle":
            pi = self._oracle_pi(z)                   # uses hidden context proxy -> role by which role fits; diag
        else:
            pi = self._pi(psi)
        pit = to_tensor(pi, cfg.device)
        mixed_logits = (pit.T[:, :, None] * logits).sum(0)   # [B,C]
        mixed_reg = (pit.T * regs).sum(0)                    # [B]
        self._stash = dict(boot=False, z=z, psi=psi, pi=pi)
        return mixed_logits.cpu().numpy(), mixed_reg.cpu().numpy()

    def _oracle_pi(self, z):
        # diagnostic: route each example to the role with lowest CURRENT loss is illegal (needs y);
        # instead route by hidden context is meaningless for fixed roles. Use uniform+ctx-free -> treat
        # as a nondeployable UPPER bound computed post-hoc in the runner. Here fall back to uniform.
        return np.full((z.shape[0], 3), 1 / 3)

    def _role_losses(self, z_np, yc, yr, plastic=None, stable=None):
        """Per-example loss of each role -> [B,3]."""
        cfg = self.cfg
        z_t = to_tensor(z_np, cfg.device)
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device); yr_t = to_tensor(yr, cfg.device)
        _, hP, hS = self._roles(z_t, plastic, stable)
        out = np.zeros((z_np.shape[0], 3))
        for b, h in ((RAW, z_t), (P, hP), (S, hS)):
            lo, rg = self.predictor(h)
            li = torch.zeros(z_t.shape[0], device=cfg.device)
            if cfg.task in ("classification", "both"):
                li = li + nn.functional.cross_entropy(lo, yc_t, reduction="none")
            if cfg.task in ("regression", "both"):
                li = li + (rg - yr_t) ** 2
            out[:, b] = li.detach().cpu().numpy()
        return out

    def _posterior(self, pi, losses):
        u = (losses - losses.min(1, keepdims=True)) / (self.s_scale + 1e-6)
        rho = pi * np.exp(-self.cfg.eta * u)
        rho = rho / rho.sum(1, keepdims=True)
        if self.cfg.shuffle_targets:
            rho = rho[:, self._val_perm]
        return rho

    def _sup(self, h, yc, yr):
        cfg = self.cfg
        lo, rg = self.predictor(h)
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device); yr_t = to_tensor(yr, cfg.device)
        loss = h.new_zeros(())
        if cfg.task in ("classification", "both"):
            loss = loss + nn.functional.cross_entropy(lo, yc_t)
        if cfg.task in ("regression", "both"):
            loss = loss + nn.functional.mse_loss(rg, yr_t)
        return loss

    def observe(self, z, y_cls, y_reg):
        cfg = self.cfg; st = self._stash
        if st["boot"]:
            loss = self._sup(to_tensor(z, cfg.device), y_cls, y_reg)
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
            self.reservoir.add_batch(z, y_cls, y_reg)
            if self._seen >= self._n_calib:
                self._finalize()
            return

        replay = self.reservoir.draw(cfg.replay_size)

        # ---- reactive R2 fixed-share update (state) ----
        if cfg.role_router == "reactive_r2":
            L = self._role_losses(z, y_cls, y_reg).mean(0)
            w = self.r2_w * np.exp(-cfg.eta * (L - L.min()) / (self.s_scale + 1e-6))
            self.r2_w = (1 - cfg.eps) * w / w.sum() + cfg.eps / 3

        # ---- gate-independent P/S/predictor training (router-independent) ----
        z_t = to_tensor(z, cfg.device)
        _, hP, hS = self._roles(z_t)
        L = (self._sup(z_t, y_cls, y_reg) + self._sup(hP, y_cls, y_reg) + self._sup(hS, y_cls, y_reg)) / 3.0
        if replay is not None:
            rz = to_tensor(replay[0], cfg.device); _, rP, rS = self._roles(rz)
            L = L + cfg.replay_lambda * (self._sup(rz, replay[1], replay[2])
                                         + self._sup(rP, replay[1], replay[2]) + self._sup(rS, replay[1], replay[2])) / 3.0
        if self.stable is not None and cfg.lambda_preserve > 0:      # S consolidation toward EMA
            L = L + cfg.lambda_preserve * sum(((p - q) ** 2).sum()
                                              for p, q in zip(self.stable.parameters(), self.s_bar.parameters()))
        self.opt.zero_grad(set_to_none=True); L.backward(); self.opt.step()

        # ---- EMA lagged copies ----
        b = cfg.ema_beta
        with torch.no_grad():
            for tp, sp in zip(self.p_bar.parameters(), self.plastic.parameters()):
                tp.mul_(b).add_(sp, alpha=1 - b)
            if self.stable is not None:
                for tp, sp in zip(self.s_bar.parameters(), self.stable.parameters()):
                    tp.mul_(b).add_(sp, alpha=1 - b)

        # ---- functional router training: KL(rho || pi), rho from LAGGED copies; router replay ----
        if cfg.role_router == "functional":
            losses = self._role_losses(z, y_cls, y_reg, self.p_bar, self.s_bar)   # lagged roles
            rho = self._posterior(st["pi"], losses)
            self.s_scale = 0.9 * self.s_scale + 0.1 * float((losses.max(1) - losses.min(1)).mean() + 1e-6)
            psis = [st["psi"]]; rhos = [rho]
            if replay is not None and cfg.router_replay:
                psi_r = self.key.embed(self.addr(replay[0]))
                with torch.no_grad():
                    pir = torch.softmax(self.router_bar(to_tensor(psi_r, cfg.device)), dim=1).cpu().numpy()
                lr_ = self._role_losses(replay[0], replay[1], replay[2], self.p_bar, self.s_bar)
                self.stats["replay_role_fwd"] += 1
                psis.append(psi_r); rhos.append(self._posterior((1 - cfg.eps) * pir + cfg.eps / 3, lr_))
            logp = torch.log_softmax(self.router(to_tensor(np.concatenate(psis), cfg.device)), dim=1)
            Lr = -(to_tensor(np.concatenate(rhos), cfg.device) * logp).sum(1).mean()
            self.router_opt.zero_grad(set_to_none=True); Lr.backward(); self.router_opt.step()
            self.stats["router_fwd"] += 1; self.stats["router_bwd"] += 1
            with torch.no_grad():                                    # router EMA teacher
                for tp, sp in zip(self.router_bar.parameters(), self.router.parameters()):
                    tp.mul_(b).add_(sp, alpha=1 - b)

        self.reservoir.add_batch(z, y_cls, y_reg)
