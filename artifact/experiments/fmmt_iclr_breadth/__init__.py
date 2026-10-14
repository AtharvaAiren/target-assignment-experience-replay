"""ICLR breadth extension for the target-pairing audit.

This namespace is intentionally separate from the frozen workshop study.  It
contains a deeper cross-entropy classification arm and a sequential regression
arm, but does not modify or reinterpret any frozen artifact.
"""

from .core import AuditPacket, run_fixed_trajectory_audit

__all__ = ["AuditPacket", "run_fixed_trajectory_audit"]
