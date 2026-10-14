"""Runtime-only adapters for pinned Mammoth's hyphenating CLI.

The adapters preserve the scientific implementation byte-for-byte. They
bridge only parser representations during model construction and preserve an
explicit replay minibatch that pinned Mammoth otherwise overwrites whenever a
stream batch is supplied.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Callable, Type


ADAPTER_NAMES = (
    "er_pairing",
    "gem_pairing",
    "agem_pairing",
    "mer_current_only",
    "mer_pairing",
    "pcgrad_replay_pairing",
    "er_state_matched_audit",
)


def cli_compatible_class(base: Type[Any]) -> Type[Any]:
    canonical_name = str(base.NAME)
    parsed_name = canonical_name.replace("_", "-")

    class CliCompatible(base):  # type: ignore[misc, valid-type]
        NAME = canonical_name

        def __init__(self, backbone, loss, args, transform, dataset=None):
            if str(getattr(args, "model", "")) not in {
                canonical_name,
                parsed_name,
            }:
                raise RuntimeError(
                    f"adapter {canonical_name} received another model identifier"
                )
            original_model = args.model
            original_device = args.device
            args.model = canonical_name
            args.device = str(original_device)
            try:
                super().__init__(backbone, loss, args, transform, dataset=dataset)
            finally:
                args.model = original_model
                args.device = original_device

    CliCompatible.__name__ = f"{base.__name__}CliCompatible"
    CliCompatible.__qualname__ = CliCompatible.__name__
    CliCompatible.__module__ = __name__
    return CliCompatible


def register_all(
    register_model: Callable[[str], Callable[[Type[Any]], Type[Any]]],
    *,
    state_model: Type[Any],
) -> dict[str, Type[Any]]:
    """Register the locked pairing/control methods plus the external audit model."""

    from models.agem_pairing import AGemPairing
    from models.er_pairing import ErPairing
    from models.gem_pairing import GemPairing
    from models.mer_current_only import MerCurrentOnly
    from models.mer_pairing import MerPairing
    from models.pcgrad_replay_pairing import PCGradReplayPairing

    bases = {
        "er_pairing": ErPairing,
        "gem_pairing": GemPairing,
        "agem_pairing": AGemPairing,
        "mer_current_only": MerCurrentOnly,
        "mer_pairing": MerPairing,
        "pcgrad_replay_pairing": PCGradReplayPairing,
        "er_state_matched_audit": state_model,
    }
    if tuple(bases) != ADAPTER_NAMES:
        raise RuntimeError("reviewer CLI adapter inventory changed")
    adapters: dict[str, Type[Any]] = {}
    for name, base in bases.items():
        if str(base.NAME) != name:
            raise RuntimeError(f"reviewer adapter base NAME changed: {name}")
        adapter = cli_compatible_class(base)
        register_model(name)(adapter)
        adapters[name] = adapter
    return adapters


def preserve_explicit_minibatch(
    original: Callable[[Namespace, Any], None]
) -> Callable[[Namespace, Any], None]:
    """Retain an explicitly parsed minibatch after Mammoth extends arguments."""

    def corrected(args: Namespace, dataset: Any) -> None:
        explicit = getattr(args, "minibatch_size", None)
        original(args, dataset)
        if explicit is not None:
            args.minibatch_size = explicit

    corrected.__name__ = "extend_args_preserving_explicit_minibatch"
    return corrected
