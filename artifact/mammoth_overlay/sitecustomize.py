"""Deterministic numerical preflight for the pinned Mammoth subprocess.

Python imports ``sitecustomize`` automatically from ``PYTHONPATH``.  The
launcher opts into this behavior with ``GRADIENT_REUSE_DETERMINISTIC=1``;
other uses of the checkout are unaffected.
"""

from __future__ import annotations

import os


if os.environ.get("GRADIENT_REUSE_DETERMINISTIC") == "1":
    import math
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda.matmul, "allow_tf32"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    if os.environ.get("GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH") == "v1":
        from utils.loggers import Logger

        _original_log_fullacc = Logger.log_fullacc

        def _gradient_reuse_log_fullacc(self, accs):
            if getattr(self, "setting", None) != "domain-il":
                return _original_log_fullacc(self, accs)
            if not isinstance(accs, (tuple, list)) or not accs:
                raise RuntimeError("domain-IL logger received no class-accuracy row")
            row = [float(value) for value in accs[0]]
            expected = len(getattr(self, "fullaccs", [])) + 1
            if len(row) != expected:
                raise RuntimeError(
                    f"domain-IL full-accuracy row has {len(row)} tasks; "
                    f"expected {expected}"
                )
            if any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in row):
                raise RuntimeError("domain-IL full-accuracy row is not a percentage")
            self.fullaccs.append(row)

        Logger.log_fullacc = _gradient_reuse_log_fullacc
        Logger._gradient_reuse_domain_il_fullacc_patch = "v1"
