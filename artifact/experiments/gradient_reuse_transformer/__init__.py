"""Fixed-conditioning next-token target-permutation audit.

The package is deliberately independent of the workshop experiment runner.
It never downloads an asset: full runs accept only locally acquired snapshots
whose repository revisions and content hashes are authenticated by a manifest.
"""

from .assets import AssetContractError, validate_asset_manifest
from .audit import NextTokenPacket, run_next_token_audit
from .data import StreamExample, build_chronological_stream, packetize
from .model import PythiaNextTokenModel, load_local_pythia

__all__ = [
    "AssetContractError",
    "NextTokenPacket",
    "PythiaNextTokenModel",
    "StreamExample",
    "build_chronological_stream",
    "load_local_pythia",
    "packetize",
    "run_next_token_audit",
    "validate_asset_manifest",
]
