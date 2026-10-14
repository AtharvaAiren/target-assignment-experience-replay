"""Raw online + Experience Replay baseline (no modulation, no memory).

This is the "Raw online ER" control from the V5 table --- a single shared linear
predictor trained online with a uniform replay reservoir, reading the features
directly.  It has no distribution key, no context memory, and no arbitration, so
it forgets across recurring contexts.  It is the floor the dual-process method
must clear.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import (
    MethodConfig,
    ReservoirBuffer,
    SharedPredictor,
    supervised_loss,
    to_tensor,
)


class RawOnlineER:
    protocol = "minibatch_query"

    def __init__(self, cfg: MethodConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.predictor = SharedPredictor(cfg.d, cfg.n_classes).to(cfg.device)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg.lr)
        self.buf = ReservoirBuffer(cfg.replay_capacity, cfg.d, seed=cfg.seed + 1)

    # -- prediction uses only pre-label parameters -- #
    @torch.no_grad()
    def predict(self, z: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h = to_tensor(z, self.cfg.device)
        logits, reg = self.predictor(h)
        return logits.cpu().numpy(), reg.cpu().numpy()

    def observe(self, z: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray) -> None:
        cfg = self.cfg
        # Replay sampled from the buffer as it stood BEFORE this batch entered it.
        replay = self.buf.sample(cfg.replay_size)
        if replay is not None:
            rz, ryc, ryr = replay
            zz = np.concatenate([z, rz], axis=0)
            yc = np.concatenate([y_cls, ryc], axis=0)
            yr = np.concatenate([y_reg, ryr], axis=0)
        else:
            zz, yc, yr = z, y_cls, y_reg

        h = to_tensor(zz, cfg.device)
        yc_t = torch.as_tensor(yc, dtype=torch.long, device=cfg.device)
        yr_t = to_tensor(yr, cfg.device)

        logits, reg = self.predictor(h)
        loss = supervised_loss(logits, reg, yc_t, yr_t, cfg)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.buf.add_batch(z, y_cls, y_reg)
