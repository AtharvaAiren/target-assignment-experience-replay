"""The live DPC-FiLM online learner --- exact step order of brief §6.10.

Assembles: frozen addressing + distribution key, plastic conditioner, bounded
key->value bank (routing + lifecycle), plastic + stable FiLM processors, ONE
shared predictor, soft-MW arbitration, functional anchors, and a shared uniform
reservoir.

Causality contract (enforced structurally by the harness + this ordering):
  predict(z)  -> steps 2-6 : route on the PRE-batch bank, run experts with
                 PRE-update params, emit + save the deployed mixture prediction,
                 then apply the cached label-free lifecycle action, compute
                 r_update, and update keys/counts. No labels touched.
  observe(z,y)-> steps 7-13: reveal, per-expert losses -> causal MW update,
                 one gate-independent optimization step (equal raw/plastic/stable
                 current+replay + stable anchor/code), inactive-row bitwise freeze,
                 anchor refresh for active slots, reservoir admission.

A causal calibration prefix (already-predicted, raw-only bootstrap) fixes the RFF
bandwidth and seeds the bank; the address transform + bandwidth freeze before the
first persistent slot and never change afterward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .keys import AddressTransform, RFFKey, make_key, PlasticConditioner
from .bank import Bank, BankConfig
from .processors import PlasticProcessor, StableProcessor
from .arbitration import make_gate
from .anchoring import AnchorStore
from .replay import SharedReservoir
from ..methods.base import SharedPredictor, to_tensor

RAW, PLASTIC, STABLE = 0, 1, 2


@dataclass
class DPCConfig:
    d: int
    n_classes: int
    task: str = "both"                 # classification | regression | both
    # architecture
    r: int = 8                         # FiLM bottleneck dim
    k_max: int = 8
    code_dim: int = 8
    memory: str = "fixed_k_soft"       # fixed_k_soft | penalized_dynamic | single
    actuator: str = "film"             # film (bottleneck) | direct (per-feature affine) | concat_mlp (cross-coordinate)
    msg_dim: int = 16                  # message width for concat_mlp actuator
    slot_init_std: float = 0.0         # neutral seeded slot init (breaks symmetry for joint routing)
    address_descriptor: str = "rff"    # mean | diag_gaussian | rff
    plastic_conditioner: str = "rff"
    arbitration: str = "soft_mw"       # see arbitration.make_gate
    use_raw: bool = True
    anchor: str = "functional_last_active"   # or "off"
    # key / routing
    rff_m: int = 256
    rff_sigma: Optional[float] = None  # None => median-heuristic on the calib prefix
    plastic_m: int = 128
    tau_b: float = 0.5
    tau_x: float = 0.5
    tau_g: float = 1.0
    eta_gate: float = 0.5
    lambda_slot: float = 0.5
    # optimization
    lr: float = 1e-2
    lr_stable_ratio: float = 0.25      # eta_S / eta_P
    replay_capacity: int = 2000
    replay_size: int = 32
    replay_lambda: float = 1.0
    anchor_weight: float = 1.0
    anchor_features_per_slot: int = 4
    lambda_e: float = 0.0
    w_cls: float = 1.0
    w_reg: float = 1.0
    # Phase-2 joint functional router
    router_mode: str = "density"       # density | functional (causal loss-taught router)
    eta_router: float = 3.0
    eps_router: float = 0.05
    router_lr: float = 1e-2
    router_replay_k: int = 16
    # routing granularity + controls / ablations
    route_granularity: str = "per_example"   # per_example | batch_aggregate (context-coherent minibatch)
    shuffle_messages: bool = False     # permute per-example routing (freq-preserving) -> tests whether CORRECT routing matters
    shuffle_value_ids: bool = False    # B3: fixed seeded permutation of slot->value identity
    memory_off: bool = False           # B9: route as usual but zero the stable memory modulation
    oracle_routing: bool = False       # DIAGNOSTIC: route by the hidden context id (one-hot). Upper bound, not deployable.
    # protocol / misc
    calib_frac: float = 0.08
    online_protocol: str = "minibatch_query"
    device: str = "cpu"
    seed: int = 0


class DPCFiLM:
    def __init__(self, cfg: DPCConfig):
        self.cfg = cfg
        self.protocol = cfg.online_protocol
        torch.manual_seed(cfg.seed)
        self._rng = np.random.default_rng(cfg.seed + 7)

        self.addr = AddressTransform(in_dim=cfg.d, mode="identity")
        self.predictor = SharedPredictor(cfg.d, cfg.n_classes).to(cfg.device)
        self.plastic = PlasticProcessor(cfg.d, cfg.r, cond_dim=2 * self._plastic_dim(),
                                        actuator=cfg.actuator, msg_dim=cfg.msg_dim).to(cfg.device)
        self.stable = StableProcessor(cfg.d, cfg.r, cfg.k_max, code_dim=cfg.code_dim,
                                      actuator=cfg.actuator, msg_dim=cfg.msg_dim,
                                      slot_init_std=cfg.slot_init_std).to(cfg.device)
        self.gate = make_gate(cfg.arbitration, cfg.k_max, tau_g=cfg.tau_g, eta=cfg.eta_gate,
                              use_raw=cfg.use_raw)
        self.anchors = AnchorStore(cfg.k_max, cfg.d, m_anchor=cfg.anchor_features_per_slot,
                                   code_dim=cfg.code_dim, r=cfg.r, seed=cfg.seed + 3)
        self.reservoir = SharedReservoir(cfg.replay_capacity, cfg.d, seed=cfg.seed + 1)

        # separate LR for the slow stable shared params vs the fast plastic.
        self.opt = torch.optim.Adam([
            {"params": self.predictor.parameters(), "lr": cfg.lr},
            {"params": self.plastic.parameters(), "lr": cfg.lr},
            {"params": self.stable.parameters(), "lr": cfg.lr * cfg.lr_stable_ratio},
        ])

        # calibration bootstrap state
        self._calibrating = True
        self._calib_target = None                    # set on first observe (need stream length? no)
        self._calib_buf: list[np.ndarray] = []
        self._seen = 0
        self._n_calib = None                         # resolved lazily from calib_frac & a min
        self.key = None
        self.pcond = PlasticConditioner(cfg.d, kind=cfg.plastic_conditioner,
                                        m=cfg.plastic_m, seed=cfg.seed + 5)
        self.bank: Optional[Bank] = None
        self._stash: dict = {}
        self.wants_context = cfg.oracle_routing         # harness feeds ctx iff oracle
        self._ctx = None
        # B3: fixed seeded permutation of slot->value identity.
        self._val_perm = np.random.default_rng(cfg.seed + 9).permutation(cfg.k_max)
        # Phase-2 functional router g_theta (causal loss-taught); built post-calibration.
        self.g_router = None
        self.g_router_opt = None
        self._s_router = 1.0

    def set_context(self, ctx):
        """Harness hook: hidden context ids for the current chunk (oracle diagnostic only)."""
        self._ctx = np.asarray(ctx)

    def _oracle_r(self, n):
        r = np.zeros((n, self.cfg.k_max), dtype=np.float64)
        r[np.arange(n), self._ctx % self.cfg.k_max] = 1.0
        return r

    def _final_r(self, rho, r_pe, n):
        """Resolve the responsibilities used for retrieval, per config.

        Order: oracle (diagnostic) > granularity > example-shuffle > value-id-shuffle.
        ``batch_aggregate`` broadcasts the single aggregate responsibility to every
        example in the arrival, so one retrieved value is shared across the batch
        (context-coherent minibatch assumption)."""
        cfg = self.cfg
        if cfg.oracle_routing:
            r = self._oracle_r(n)
        elif cfg.route_granularity == "batch_aggregate":
            r = np.broadcast_to(rho, (n, cfg.k_max)).copy()
        else:
            r = r_pe
        if cfg.shuffle_messages:                          # scramble example<->context
            r = r[self._rng.permutation(n)]
        if cfg.shuffle_value_ids:                         # scramble slot->value identity (B3)
            r = r[:, self._val_perm]
        return r

    def _plastic_dim(self):
        return self.cfg.plastic_m if self.cfg.plastic_conditioner == "rff" else self.cfg.d

    # ------------------------------------------------------------------ #
    def _finalize_calibration(self) -> None:
        cfg = self.cfg
        pref = np.concatenate(self._calib_buf, axis=0) if self._calib_buf else np.zeros((1, cfg.d))
        zt = self.addr(pref)
        if cfg.address_descriptor in ("rff", "rff_kernel"):
            sigma = cfg.rff_sigma or RFFKey.median_bandwidth(zt, seed=cfg.seed)
            self.key = RFFKey(self.addr.out_dim, m=cfg.rff_m, sigma=sigma, seed=cfg.seed + 2)
        else:
            self.key = make_key(cfg.address_descriptor, self.addr.out_dim)
        psi = self.key.encode(zt)[0]

        k = 1 if cfg.memory == "single" else cfg.k_max
        life = "fixed_k_soft" if cfg.memory in ("fixed_k_soft", "single") else cfg.memory
        self.bank = Bank(BankConfig(k_max=cfg.k_max, lifecycle=life, tau_b=cfg.tau_b,
                                    tau_x=cfg.tau_x, lambda_slot=cfg.lambda_slot,
                                    seed=cfg.seed + 4), self.key)
        self.bank.initialize_fixed_k(psi, k=k)
        if cfg.router_mode == "functional":
            self.g_router = nn.Sequential(
                nn.LayerNorm(self.key.dim), nn.Linear(self.key.dim, 64), nn.GELU(),
                nn.Linear(64, cfg.k_max)).to(cfg.device)
            self.g_router_opt = torch.optim.Adam(self.g_router.parameters(), lr=cfg.router_lr)
        self._calibrating = False
        self._calib_buf = []

    # ------------------------------------------------------------------ #
    def _build_cond(self, zt: np.ndarray) -> torch.Tensor:
        psi_p, q_p = self.pcond.encode(zt)          # [B,dp], [dp]
        B = psi_p.shape[0]
        q_b = np.broadcast_to(q_p, (B, q_p.shape[0]))
        cond = np.concatenate([q_b, psi_p], axis=1)  # [B, 2dp]
        return to_tensor(cond, self.cfg.device)

    @torch.no_grad()
    def _experts_eval(self, z_np, r_resp, cond_t, n_log):
        """Expert features under no_grad (for the emitted prediction)."""
        z = to_tensor(z_np, self.cfg.device)
        hP = self.plastic(z, cond_t)
        hS = z if self.cfg.memory_off else self.stable(z, to_tensor(r_resp, self.cfg.device), n_log)
        return z, hP, hS

    def _mix_and_predict(self, z, hP, hS, w):
        wt = to_tensor(w, self.cfg.device)
        h = wt[:, RAW:RAW + 1] * z + wt[:, PLASTIC:PLASTIC + 1] * hP + wt[:, STABLE:STABLE + 1] * hS
        return self.predictor(h)

    # ------------------------------------------------------------------ #
    def predict(self, z: np.ndarray):
        cfg = self.cfg
        self._seen += z.shape[0]
        if self._n_calib is None:
            # resolve a small calibration prefix length once (min 256 rows or 2 batches).
            self._n_calib = max(256, z.shape[0] * 2)

        if self._calibrating:
            self._calib_buf.append(np.asarray(z, np.float32))
            with torch.no_grad():
                logits, reg = self.predictor(to_tensor(z, cfg.device))  # raw bootstrap
            self._stash = {"bootstrap": True, "z": z}
            return logits.cpu().numpy(), reg.cpu().numpy()

        zt = self.addr(z)
        psi, q = self.key.encode(zt)
        cond_t = self._build_cond(zt)
        proposal = self.bank.propose_structure(q)
        rho, r_pe = self.bank.route_predict(q, psi)
        if cfg.router_mode == "functional" and self.g_router is not None:
            a = self._functional_prior(q)                # causal loss-taught prior (pre-reveal)
            r_predict = np.broadcast_to(a, (z.shape[0], cfg.k_max)).copy()
        else:
            r_predict = self._final_r(rho, r_pe, z.shape[0])
        n_log = to_tensor(np.log1p(self.bank.n), cfg.device)

        with torch.no_grad():
            z_t, hP, hS = self._experts_eval(z, r_predict, cond_t, n_log)
            w = self.gate.mix_weights(r_predict)
            logits, reg = self._mix_and_predict(z_t, hP, hS, w)

        # ----- step 6: apply cached label-free lifecycle + r_update + key update -----
        new_slot = self.bank.apply_structure(proposal)
        if new_slot is not None:
            self.stable.reset_slot(new_slot, self.opt)
            self.gate.reset_slot(new_slot)
            self.anchors.clear(new_slot)
        rho_u, r_pe_u = self.bank.route_update(q, psi)
        r_update = self._final_r(rho_u, r_pe_u, z.shape[0])
        self.bank.update_keys(q, r_update)
        self.bank.advance_clock()

        self._stash = {
            "bootstrap": False, "z": z, "zt": zt, "q": q, "psi": psi,
            "cond_t": cond_t, "r_predict": r_predict, "r_update": r_update,
            "hP": hP.detach(), "hS": hS.detach(), "n_log": n_log,
        }
        return logits.cpu().numpy(), reg.cpu().numpy()

    # ------------------------------------------------------------------ #
    def _per_expert_losses(self, z_np, yc, yr, hP, hS):
        """Unreduced nonneg per-example losses for raw/plastic/stable -> [B,3]."""
        cfg = self.cfg
        z = to_tensor(z_np, cfg.device)
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device)
        yr_t = to_tensor(yr, cfg.device)
        out = np.zeros((z.shape[0], 3), dtype=np.float64)
        for idx, h in ((RAW, z), (PLASTIC, hP), (STABLE, hS)):
            logits, reg = self.predictor(h)
            li = z.new_zeros(z.shape[0])
            if cfg.task in ("classification", "both"):
                li = li + cfg.w_cls * torch.nn.functional.cross_entropy(logits, yc_t, reduction="none")
            if cfg.task in ("regression", "both"):
                li = li + cfg.w_reg * (reg - yr_t) ** 2
            out[:, idx] = li.detach().cpu().numpy()
        return out

    def _sup_loss(self, h, yc, yr):
        cfg = self.cfg
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device)
        yr_t = to_tensor(yr, cfg.device)
        logits, reg = self.predictor(h)
        loss = h.new_zeros(())
        if cfg.task in ("classification", "both"):
            loss = loss + cfg.w_cls * torch.nn.functional.cross_entropy(logits, yc_t)
        if cfg.task in ("regression", "both"):
            loss = loss + cfg.w_reg * torch.nn.functional.mse_loss(reg, yr_t)
        return loss

    def _expert_train_features(self, z_np, r_resp):
        """Graph-connected (raw, plastic, stable) features for the training loss."""
        zt = self.addr(z_np)
        cond_t = self._build_cond(zt)
        n_log = to_tensor(np.log1p(self.bank.n), self.cfg.device)
        z = to_tensor(z_np, self.cfg.device)
        hP = self.plastic(z, cond_t)
        hS = z if self.cfg.memory_off else self.stable(z, to_tensor(r_resp, self.cfg.device), n_log)
        return z, hP, hS

    def _functional_prior(self, q):
        with torch.no_grad():
            a = torch.softmax(self.g_router(to_tensor(q[None, :], self.cfg.device)), dim=1)
        a = a.cpu().numpy()[0]
        return (1 - self.cfg.eps_router) * a + self.cfg.eps_router / self.cfg.k_max

    @torch.no_grad()
    def _per_slot_losses(self, z_np, yc, yr, n_log):
        """Loss of EACH slot's stable prediction on (z,y) -> [K] (functional target)."""
        cfg = self.cfg
        z = to_tensor(z_np, cfg.device)
        B = z.shape[0]
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device)
        yr_t = to_tensor(yr, cfg.device)
        ell = np.zeros(cfg.k_max)
        for k in range(cfg.k_max):
            r = np.zeros((B, cfg.k_max), np.float64); r[:, k] = 1.0
            hS = self.stable(z, to_tensor(r, cfg.device), n_log)
            logits, reg = self.predictor(hS)
            li = 0.0
            if cfg.task in ("classification", "both"):
                li += cfg.w_cls * torch.nn.functional.cross_entropy(logits, yc_t).item()
            if cfg.task in ("regression", "both"):
                li += cfg.w_reg * torch.nn.functional.mse_loss(reg, yr_t).item()
            ell[k] = li
        return ell

    def _loss_optimal_rho(self, ell):
        u = np.clip((ell - ell.min()) / (self._s_router + 1e-6), 0.0, 5.0)
        e = np.exp(-self.cfg.eta_router * (u - u.min()))
        return e / e.sum()

    def observe(self, z: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray) -> None:
        cfg = self.cfg
        st = self._stash

        if st.get("bootstrap", False):
            # calibration phase: train only raw predictor path; admit; maybe finalize.
            h = to_tensor(z, cfg.device)
            loss = self._sup_loss(h, y_cls, y_reg)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.reservoir.add_batch(z, y_cls, y_reg)
            if self._seen >= self._n_calib:
                self._finalize_calibration()
            return

        # ----- step 7: per-expert losses -> causal MW gate update (uses r_update) -----
        losses = self._per_expert_losses(z, y_cls, y_reg, st["hP"], st["hS"])
        self.gate.update(st["r_update"], losses)

        # ----- step 8: one shared replay draw -----
        replay = self.reservoir.draw(cfg.replay_size)
        functional = cfg.router_mode == "functional" and self.g_router is not None

        if functional:
            # loss-optimal responsibilities (post-reveal) drive value specialization;
            # router replay posteriors are RECOMPUTED from the shared (z,y) reservoir.
            n_log_u = to_tensor(np.log1p(self.bank.n), cfg.device)
            ell_cur = self._per_slot_losses(z, y_cls, y_reg, n_log_u)
            rho_cur = self._loss_optimal_rho(ell_cur)
            r_update = np.broadcast_to(rho_cur, (z.shape[0], cfg.k_max)).copy()
            if replay is not None:
                q_r = self.key.encode(self.addr(replay[0]))[1]
                rho_rep = self._loss_optimal_rho(
                    self._per_slot_losses(replay[0], replay[1], replay[2], n_log_u))
                r_rep = np.broadcast_to(rho_rep, (replay[0].shape[0], cfg.k_max)).copy()
            else:
                q_r = rho_rep = r_rep = None
        else:
            r_update = st["r_update"]
            if replay is not None:
                psi_r, q_r = self.key.encode(self.addr(replay[0]))
                _, r_rep = self.bank.route_update(q_r, psi_r)
            else:
                r_rep = None
        mass = r_update.sum(0) + (r_rep.sum(0) if r_rep is not None else 0.0)
        active_mask = torch.as_tensor(mass > 0, dtype=torch.bool)

        # ----- step 9: one gate-independent optimization step -----
        z_c, hP_c, hS_c = self._expert_train_features(z, r_update)
        L = (self._sup_loss(z_c, y_cls, y_reg)
             + self._sup_loss(hP_c, y_cls, y_reg)
             + self._sup_loss(hS_c, y_cls, y_reg)) / 3.0
        if replay is not None:
            rz, ryc, ryr = replay
            z_r, hP_r, hS_r = self._expert_train_features(rz, r_rep)
            Lr = (self._sup_loss(z_r, ryc, ryr)
                  + self._sup_loss(hP_r, ryc, ryr)
                  + self._sup_loss(hS_r, ryc, ryr)) / 3.0
            L = L + cfg.replay_lambda * Lr
        # stable anchor (inactive slots) + code penalty (active rows only)
        if cfg.anchor != "off":
            inactive = [k for k in np.where(self.bank.allocated)[0]
                        if not bool(active_mask[k]) and self.anchors.has[k]]
            L = L + cfg.anchor_weight * self.anchors.anchor_loss(inactive, self.stable)
        if cfg.lambda_e > 0:
            act = torch.where(active_mask)[0]
            if act.numel():
                L = L + cfg.lambda_e * (self.stable.e_rows[act].pow(2).mean()
                                        + self.stable.v_rows[act].pow(2).mean())

        self.opt.zero_grad(set_to_none=True)
        L.backward()
        self.stable.mask_inactive_grads(active_mask)
        snap = self.stable.snapshot_rows(self.opt)
        self.opt.step()
        self.stable.freeze_inactive(self.opt, active_mask, snap)

        # ----- functional router training: KL(rho || softmax(g(q))) on current + replay -----
        if functional:
            qs = [st["q"]]; rs = [rho_cur]
            if replay is not None:
                qs.append(q_r); rs.append(rho_rep)
            logp = torch.log_softmax(self.g_router(to_tensor(np.stack(qs), cfg.device)), dim=1)
            rloss = -(to_tensor(np.stack(rs), cfg.device) * logp).sum(1).mean()
            self.g_router_opt.zero_grad(set_to_none=True)
            rloss.backward()
            self.g_router_opt.step()
            self._s_router = 0.9 * self._s_router + 0.1 * float(ell_cur.max() - ell_cur.min() + 1e-6)

        # ----- step 11: refresh anchors for active slots (post-update) -----
        if cfg.anchor != "off":
            n_log = to_tensor(np.log1p(self.bank.n), cfg.device)
            for k in np.where(active_mask.numpy())[0]:
                members = z[r_update[:, k] > 0]
                if members.shape[0]:
                    self.anchors.refresh(int(k), to_tensor(members, cfg.device), self.stable, n_log)

        # ----- step 12: admit to the shared reservoir -----
        self.reservoir.add_batch(z, y_cls, y_reg)
