"""DPC-FiLM core: distribution keys, key->value bank, dual processors, arbitration.

Kept separate from ``chartfilm.methods`` (external baselines / controls).  Nothing
here performs optimal transport, Bures/CORAL alignment, or whitening; the address
space is *frozen* and only recognizes/retrieves, never repairs features.
"""

from .learner import DPCFiLM, DPCConfig

__all__ = ["DPCFiLM", "DPCConfig"]
