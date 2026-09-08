"""Parity-corrected RMBench DP deployment.

This policy keeps the official observation encoder unchanged (the simulator's
numeric RGB array is passed through exactly as before) and applies only the two
verified parity corrections:

* OIDN, matching the renderer used by official RMBench data collection.
* A continuous observation queue without duplicating chunk-boundary frames.

The renderer is configured by the launcher before SAPIEN is imported.  This
module validates that contract so a corrected result cannot silently run with
OptiX again.
"""

from __future__ import annotations

import os

from DPFixedQueue import eval, get_model as _get_fixed_queue_model, reset_model


def get_model(usr_args):
    denoiser = os.environ.setdefault("RMBENCH_RT_DENOISER", "oidn").strip().lower()
    if denoiser != "oidn":
        raise RuntimeError(
            "DPCorrected requires RMBENCH_RT_DENOISER=oidn to match official "
            f"RMBench rendering; got {denoiser!r}"
        )
    return _get_fixed_queue_model(usr_args)


__all__ = ["eval", "get_model", "reset_model"]
