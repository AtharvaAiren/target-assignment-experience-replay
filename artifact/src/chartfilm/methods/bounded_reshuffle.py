"""Production scheduler for the frozen bounded causal packet reshuffle.

The scheduler is intentionally model- and label-agnostic.  A caller first makes
its prediction for arrival ``t``, reveals the packet's labels, and only then calls
``arrive(packet, t)``.  Once more than ``width`` packets are pending, one already
revealed packet is released for an optimizer update.  Release is uniform over the
pending queue unless the oldest packet has reached ``max_age``; in that case the
oldest packet is forced out.  End-of-stream draining is oldest-first.

This module contains no experiment instrumentation, replay implementation,
gradient diagnostics, task identifiers, or model code.  Those are deliberately
owned by callers.  The empirically locked instantiation is documented in
``METHOD_SPEC.md`` and ``MODEL_SCOPE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np


PacketT = TypeVar("PacketT")


@dataclass(frozen=True)
class BoundedReshuffleConfig:
    """Frozen scheduler hyperparameters.

    ``scheduler_tag`` is combined with the caller's model/data seed via NumPy's
    ``SeedSequence`` list convention.  This exactly preserves the candidate's
    historical RNG mapping ``default_rng([seed, 7001])``.
    """

    width: int = 8
    max_age: int = 32
    scheduler_tag: int = 7001

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or int(self.width) != self.width or self.width < 1:
            raise ValueError("width must be a positive integer")
        if isinstance(self.max_age, bool) or int(self.max_age) != self.max_age:
            raise ValueError("max_age must be an integer")
        if self.max_age < self.width:
            raise ValueError("max_age must be at least width for the locked queue semantics")
        if isinstance(self.scheduler_tag, bool) or int(self.scheduler_tag) != self.scheduler_tag:
            raise ValueError("scheduler_tag must be an integer")


@dataclass(frozen=True)
class ReleaseEvent(Generic[PacketT]):
    """One packet released for an optimizer update."""

    packet: PacketT
    enqueue_step: int
    release_step: int
    age: int
    queue_index: int
    forced_by_age: bool
    during_drain: bool


class BoundedCausalReshuffle(Generic[PacketT]):
    """Bounded, content-independent scheduler for already-labelled packets.

    The object retains only the pending queue and RNG state.  It does not retain
    a release history; callers that need diagnostic traces must record returned
    :class:`ReleaseEvent` objects themselves so production memory stays bounded.
    """

    implementation_id = "bounded_causal_reshuffle_v1"

    def __init__(
        self,
        seed: int,
        config: BoundedReshuffleConfig | None = None,
    ) -> None:
        self.config = config or BoundedReshuffleConfig()
        self.seed = int(seed)
        self._rng = np.random.default_rng([self.seed, int(self.config.scheduler_tag)])
        self._pending: list[tuple[PacketT, int]] = []
        self._last_arrival_step = -1

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def empty(self) -> bool:
        return not self._pending

    def arrive(self, packet: PacketT, arrival_step: int) -> ReleaseEvent[PacketT] | None:
        """Enqueue one post-reveal packet and possibly release one update.

        Arrival indices are required to be consecutive and zero-based.  This
        fail-closed constraint prevents ambiguous residence ages or accidental
        reuse of future scheduler state.
        """

        arrival_step = int(arrival_step)
        expected = self._last_arrival_step + 1
        if arrival_step != expected:
            raise ValueError(f"arrival_step must be consecutive: expected {expected}, got {arrival_step}")
        self._last_arrival_step = arrival_step
        self._pending.append((packet, arrival_step))
        if len(self._pending) <= self.config.width:
            return None

        oldest_age = arrival_step - self._pending[0][1]
        forced = oldest_age >= self.config.max_age
        # Preserve the frozen RNG call pattern exactly: a forced release consumes
        # no random variate; otherwise sample uniformly over W+1 pending packets.
        index = 0 if forced else int(self._rng.integers(0, len(self._pending)))
        return self._release(index, arrival_step, forced_by_age=forced, during_drain=False)

    def drain_one(self, release_step: int) -> ReleaseEvent[PacketT] | None:
        """Release the oldest pending packet during the terminal drain."""

        if not self._pending:
            return None
        release_step = int(release_step)
        if release_step < self._pending[0][1]:
            raise ValueError("drain release_step precedes the oldest packet")
        return self._release(0, release_step, forced_by_age=False, during_drain=True)

    def _release(
        self,
        index: int,
        release_step: int,
        *,
        forced_by_age: bool,
        during_drain: bool,
    ) -> ReleaseEvent[PacketT]:
        packet, enqueue_step = self._pending.pop(index)
        return ReleaseEvent(
            packet=packet,
            enqueue_step=enqueue_step,
            release_step=release_step,
            age=release_step - enqueue_step,
            queue_index=index,
            forced_by_age=bool(forced_by_age),
            during_drain=bool(during_drain),
        )


__all__ = [
    "BoundedCausalReshuffle",
    "BoundedReshuffleConfig",
    "ReleaseEvent",
]
