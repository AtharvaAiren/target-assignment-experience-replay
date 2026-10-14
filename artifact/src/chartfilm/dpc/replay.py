"""One physical uniform replay reservoir, shared by both processors (brief §6.9).

A single ``draw`` is reused for the raw/plastic/stable replay losses --- storage
is NOT duplicated and the buffer is NOT sampled separately per processor.
Admission is uniform over time and independent of class, target value, task, or
domain (no balancing, quotas, prototypes, or per-class allocation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from chartfilm.methods.base import ReservoirBuffer


class SharedReservoir(ReservoirBuffer):
    """Uniform reservoir (Vitter R) with an explicit single shared draw."""

    def draw(self, k: int) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """One replay minibatch reused by every expert's replay loss."""
        return self.sample(k)
