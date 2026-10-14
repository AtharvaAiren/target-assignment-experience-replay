"""Shared building blocks for online methods (torch).

Kept deliberately small and explicit so the fairness axes (replay ratio, bytes,
optimizer steps, compute) are countable per method.  Nothing here reads a label
before the harness has snapshotted the corresponding prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# The synthetic kill-test uses tiny (12-D) tensors; on this 256-core box torch
# otherwise fans each op across all cores and thrashes. One thread is optimal here.
# (Real GPU runs are unaffected: this caps only CPU intra-op parallelism.)
torch.set_num_threads(1)


@dataclass(frozen=True)
class MethodConfig:
    d: int
    n_classes: int
    task: str = "both"                 # classification | regression | both
    lr: float = 1e-2
    replay_capacity: int = 2000        # reservoir size (0 = replay-free)
    replay_size: int = 32              # examples drawn per update
    w_cls: float = 1.0
    w_reg: float = 1.0
    device: str = "cpu"
    seed: int = 0


def to_tensor(x: np.ndarray, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


class ReservoirBuffer:
    """Uniform reservoir sampling (Vitter's algorithm R), label-agnostic order.

    Stores (z, y_cls, y_reg).  ``add_batch`` must be called only *after* the
    corresponding predictions have been recorded by the harness.
    """

    def __init__(self, capacity: int, d: int, seed: int = 0):
        self.capacity = int(capacity)
        self.d = int(d)
        self.z = np.zeros((self.capacity, d), dtype=np.float32)
        self.yc = np.zeros(self.capacity, dtype=np.int64)
        self.yr = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, z: np.ndarray, yc: np.ndarray, yr: np.ndarray) -> None:
        if self.capacity == 0:
            return
        for i in range(z.shape[0]):
            self.seen += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j >= self.capacity:
                    continue
                slot = j
            self.z[slot] = z[i]
            self.yc[slot] = yc[i]
            self.yr[slot] = yr[i]

    def sample(self, k: int) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.size == 0 or k <= 0:
            return None
        idx = self.rng.integers(0, self.size, size=min(k, self.size))
        return self.z[idx], self.yc[idx], self.yr[idx]

    def bytes(self) -> int:
        """Physical buffer bytes actually held (for the memory-accounting axis)."""
        return int(self.size) * (self.z.itemsize * self.d + self.yc.itemsize + self.yr.itemsize)


class SharedPredictor(nn.Module):
    """One shared predictor g used by BOTH the classification and regression heads
    (brief: one shared predictor across processors; only output shape/loss change)."""

    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.cls = nn.Linear(d, n_classes)
        self.reg = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls(h), self.reg(h).squeeze(-1)


def supervised_loss(
    logits: torch.Tensor,
    reg_pred: torch.Tensor,
    yc: torch.Tensor,
    yr: torch.Tensor,
    cfg: MethodConfig,
) -> torch.Tensor:
    """Generic supervised loss = w_cls * CE + w_reg * MSE (only active heads)."""
    loss = logits.new_zeros(())
    if cfg.task in ("classification", "both"):
        loss = loss + cfg.w_cls * nn.functional.cross_entropy(logits, yc)
    if cfg.task in ("regression", "both"):
        loss = loss + cfg.w_reg * nn.functional.mse_loss(reg_pred, yr)
    return loss
