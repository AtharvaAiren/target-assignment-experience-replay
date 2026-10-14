"""Causal diagnostic rows D0-D5 on the ImageNet-C DINOv2 penultimate cache.

All rows: one shared LoRA tail (K=1, r=8), AdamW lr 1e-3, bf16 autocast, no grad clip, identical
init (seed). Reservoir admission is deterministic and keyed by ORIGINAL ARRIVAL ID (order-independent
coins); replay draw is keyed by update-step index. Prediction strictly before label. B (storage) =
2000 penult vectors for D0/D1/D2/D5; D3/D4 split 1600 reservoir + 8x50 queue.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from experiments.repr_audit.rows import D, MultiLoRATail, _ce

B, DRAW, LR = 50, 32, 1e-3
CAP = 2000
W = 8                       # delayed-queue packets (D3/D4)
CAP_D34 = CAP - W * B       # 1600, so 1600 + 400 = 2000 = B
D2_PERIOD = 128             # boundary-blind Adam reset period (locked; not derived from regime structure)


class DetReservoir:
    """Reservoir with admission/slot coins keyed by ORIGINAL ARRIVAL ID (order-independent).
    Storing example with arrival id `a`: fill slot a if a<cap else admit w.p. cap/(a+1) into a
    coin-chosen slot. Replay draw keyed by update-step (identical draw coins across rows)."""

    def __init__(self, cap, seed):
        self.cap = int(cap); self.slot = np.full(self.cap, -1, np.int64); self.seed = int(seed)

    def add(self, penult_ids, arrival_ids):
        for p, a in zip(np.asarray(penult_ids), np.asarray(arrival_ids)):
            a = int(a)
            if a < self.cap:
                self.slot[a] = p
            else:
                r = np.random.default_rng([self.seed, a])
                if r.random() < self.cap / (a + 1):
                    self.slot[int(r.integers(0, self.cap))] = p

    def draw(self, k, step):
        filled = self.slot[self.slot >= 0]
        if len(filled) == 0:
            return None
        r = np.random.default_rng([self.seed, 1_000_000_000 + int(step)])
        return filled[r.integers(0, len(filled), size=min(k, len(filled)))]


class FullHistory:
    """D5: store ALL seen (memory-unmatched); uniform replay keyed by update-step."""

    def __init__(self, seed):
        self.buf = []; self.seed = int(seed)

    def add(self, penult_ids, arrival_ids=None):
        self.buf.extend(int(p) for p in np.asarray(penult_ids))

    def draw(self, k, step):
        if not self.buf:
            return None
        r = np.random.default_rng([self.seed, 1_000_000_000 + int(step)])
        b = np.asarray(self.buf)
        return b[r.integers(0, len(b), size=min(k, len(b)))]


def _new(NC, seed):
    torch.manual_seed(seed)
    tail = MultiLoRATail(D.m, NC, K=1, r=8, device=D.device)
    opt = torch.optim.AdamW(tail.lora_params() + list(tail.head.parameters()), lr=LR)
    return tail, opt


def _reset_adam(opt):
    for st in opt.state.values():
        if "exp_avg" in st:
            st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()
        if "step" in st:
            st["step"].zero_() if torch.is_tensor(st["step"]) else st.update(step=0)


def _reg_heldout(stream):
    # Large secondary streams may use a deterministic, disjoint diagnostic
    # subset for repeated old-regime-loss probes while retaining the full
    # held-out set for the primary final metric.  Existing streams do not set
    # these optional fields and are therefore bit-for-bit unchanged.
    ho = stream.get("diagnostic_heldout_idx", stream["heldout_idx"])
    hoc = stream.get("diagnostic_heldout_ctx", stream["heldout_ctx"])
    return {int(c): ho[hoc == c] for c in np.unique(hoc)}


@torch.no_grad()
def _acc(tail, idx, bs=500):
    cor = 0
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]; cor += int((tail(D.x9(bi))[0].argmax(1) == D.y(bi)).sum().item())
    return cor / len(idx)


@torch.no_grad()
def _reg_loss(tail, hidx):
    lo = tail(D.x9(hidx))[0]
    return float(nn.functional.cross_entropy(lo, D.y(hidx)).item())


def _oldrise(per_cycle):
    cycles = sorted(per_cycle); regimes = list(per_cycle[cycles[0]])
    rise = [per_cycle[cycles[-1]][c] - min(per_cycle[cy][c] for cy in cycles) for c in regimes]
    return float(np.mean(rise))


# --------------------------------------------------------------------------- #
def run_flat(stream, NC, seed, mode="D0"):
    """D0 (P0), D1 (cumulative-risk weights), D2 (periodic Adam reset), D2_oracle, D5 (full history)."""
    tail, opt = _new(NC, seed)
    res = FullHistory(seed) if mode == "D5" else DetReservoir(CAP, seed)
    idxs, ctxs = stream["stream_idx"], stream["context"]
    nstep = len(idxs) // B
    online = np.zeros(nstep * B, bool)
    reg_ho = _reg_heldout(stream)
    cyc_last_step = {(b // B) - 1 for (a, b, c, v) in stream["segments"]}
    step_cycle = {}
    for (a, bb, c, v) in stream["segments"]:
        for st_ in range(a // B, bb // B):
            step_cycle[st_] = v
    per_cycle = {}
    Nt = 0
    for s in range(nstep):
        cur = idxs[s * B:(s + 1) * B]
        aid = np.arange(s * B, s * B + len(cur))
        with torch.no_grad():
            online[s * B:(s + 1) * B] = (tail(D.x9(cur))[0].argmax(1).cpu().numpy() == D.LAB[cur])
        rep = res.draw(DRAW, s)
        Nt += len(cur)
        lc = _ce(tail(D.x9(cur))[0], D.y(cur))
        if rep is not None:
            lr_ = _ce(tail(D.x9(rep))[0], D.y(rep))
            if mode == "D1":
                loss = (len(cur) / Nt) * lc + ((Nt - len(cur)) / Nt) * lr_   # unbiased empirical risk
            else:
                loss = lc + lr_                                              # D0/D2/D5 two-term
        else:
            loss = lc
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if mode == "D2" and (s + 1) % D2_PERIOD == 0:
            _reset_adam(opt)
        if mode == "D2_oracle" and s in cyc_last_step:
            _reset_adam(opt)
        res.add(cur, aid)
        if s in cyc_last_step:
            v = step_cycle[s]
            per_cycle.setdefault(v, {})
            for c, h in reg_ho.items():
                per_cycle[v][c] = _reg_loss(tail, h)
    return dict(row=mode, acc=_acc(tail, stream["heldout_idx"]), prequential_auc=float(online.mean()),
                oldregime_loss_rise=_oldrise(per_cycle), G=nstep, E_cur=nstep * B, storage=res.cap if hasattr(res, "cap") else len(set(res.buf)))


# --------------------------------------------------------------------------- #
def run_delayed(stream, NC, seed, release="oldest"):
    """D3 (FIFO oldest) / D4 (uniform-random release). W=8 queue, reservoir cap 1600 (B matched).
    Predict at arrival (delayed model); release one pending packet per arrival once full; drain after."""
    tail, opt = _new(NC, seed)
    res = DetReservoir(CAP_D34, seed)
    idxs, ctxs = stream["stream_idx"], stream["context"]
    nstep = len(idxs) // B
    online = np.zeros(nstep * B, bool)
    reg_ho = _reg_heldout(stream)
    cyc_last_step = {(b // B) - 1 for (a, b, c, v) in stream["segments"]}
    step_cycle = {}
    for (a, bb, c, v) in stream["segments"]:
        for st_ in range(a // B, bb // B):
            step_cycle[st_] = v
    per_cycle = {}
    rng = np.random.default_rng([seed, 424242])
    queue = []          # list of (cur_idx, arrival_ids, arrival_step)
    ustep = 0; latencies = []; acc_before_drain = None

    def do_update(packet, arrival_step):
        nonlocal ustep
        cur, aid, astep = packet
        rep = res.draw(DRAW, ustep)
        loss = _ce(tail(D.x9(cur))[0], D.y(cur))
        if rep is not None:
            loss = loss + _ce(tail(D.x9(rep))[0], D.y(rep))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        res.add(cur, aid)
        latencies.append(arrival_step - astep)
        ustep += 1
        if arrival_step in cyc_last_step:
            v = step_cycle[arrival_step]; per_cycle.setdefault(v, {})
            for c, h in reg_ho.items():
                per_cycle[v][c] = _reg_loss(tail, h)

    for s in range(nstep):
        cur = idxs[s * B:(s + 1) * B]; aid = np.arange(s * B, s * B + len(cur))
        with torch.no_grad():
            online[s * B:(s + 1) * B] = (tail(D.x9(cur))[0].argmax(1).cpu().numpy() == D.LAB[cur])
        queue.append((cur, aid, s))
        if len(queue) > W:
            j = 0 if release == "oldest" else int(rng.integers(0, len(queue)))
            do_update(queue.pop(j), s)
    acc_before_drain = _acc(tail, stream["heldout_idx"])
    # drain in release order
    while queue:
        j = 0 if release == "oldest" else int(rng.integers(0, len(queue)))
        do_update(queue.pop(j), nstep - 1)
    return dict(row=("D3" if release == "oldest" else "D4"),
                acc=_acc(tail, stream["heldout_idx"]), acc_before_drain=acc_before_drain,
                prequential_auc=float(online.mean()), oldregime_loss_rise=_oldrise(per_cycle),
                latency_avg=float(np.mean(latencies)), latency_max=float(np.max(latencies)),
                G=ustep, E_cur=ustep * B, storage=CAP_D34 + W * B)


# --------------------------------------------------------------------------- #
def gradient_path_diag(stream, NC, seed, n_ckpt=6, ref_sample=5000):
    """Along the D0 trajectory, at checkpoints compute the high-precision full-past reference gradient
    g* (fp32, over a large uniform sample of history) and the alignment of current / replay / D0-mixed /
    D1-mixed update gradients with g*. Isolates the cumulative-risk gradient-path error."""
    tail, opt = _new(NC, seed)
    res = DetReservoir(CAP, seed)
    idxs = stream["stream_idx"]; nstep = len(idxs) // B
    ckpts = set(np.linspace(nstep // 4, nstep - 1, n_ckpt, dtype=int))
    params = tail.lora_params() + list(tail.head.parameters())
    rng = np.random.default_rng([seed, 55])
    rows = []
    Nt = 0
    for s in range(nstep):
        cur = idxs[s * B:(s + 1) * B]; aid = np.arange(s * B, s * B + len(cur))
        rep = res.draw(DRAW, s); Nt += len(cur)
        if s in ckpts and rep is not None:
            past = idxs[:s * B]
            samp = past[rng.integers(0, len(past), size=min(ref_sample, len(past)))]
            def g_of(idx, w=None):
                gs = torch.autograd.grad(_ce(tail(D.x9(idx))[0], D.y(idx)), params, retain_graph=False)
                return torch.cat([g.flatten() for g in gs])
            gstar = []
            for t0 in range(0, len(samp), 1024):
                bi = samp[t0:t0 + 1024]
                gs = torch.autograd.grad(_ce(tail(D.x9(bi))[0], D.y(bi)), params)
                gstar.append(torch.cat([g.flatten() for g in gs]) * (len(bi) / len(samp)))
            gstar = torch.stack(gstar).sum(0)
            gc = g_of(cur); gr = g_of(rep)
            gd0 = gc + gr
            gd1 = (len(cur) / Nt) * gc + ((Nt - len(cur)) / Nt) * gr
            cos = lambda a, b: float((a @ b) / (a.norm() * b.norm() + 1e-12))
            rows.append(dict(step=int(s),
                             cos_current_ref=cos(gc, gstar), cos_replay_ref=cos(gr, gstar),
                             cos_D0mix_ref=cos(gd0, gstar), cos_D1mix_ref=cos(gd1, gstar),
                             cos_current_replay=cos(gc, gr),
                             err_D0mix=1 - cos(gd0, gstar), err_D1mix=1 - cos(gd1, gstar)))
        # advance D0 update
        lc = _ce(tail(D.x9(cur))[0], D.y(cur))
        loss = lc + (_ce(tail(D.x9(rep))[0], D.y(rep)) if rep is not None else 0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); res.add(cur, aid)
    def m(k): return float(np.mean([r[k] for r in rows]))
    return dict(checkpoints=rows,
                mean=dict(cos_current_ref=m("cos_current_ref"), cos_replay_ref=m("cos_replay_ref"),
                          cos_D0mix_ref=m("cos_D0mix_ref"), cos_D1mix_ref=m("cos_D1mix_ref"),
                          err_D0mix=m("err_D0mix"), err_D1mix=m("err_D1mix"),
                          gradient_path_error_reduced_by_D1=bool(m("err_D1mix") < m("err_D0mix"))))
