"""OIDN-guarded RMBench VPWEM deployment for corrected protocol comparisons."""

from __future__ import annotations

import os

from VPWEM.deploy_policy import eval, get_model as _get_vpwem_model, reset_model


def get_model(usr_args):
    denoiser = os.environ.setdefault("RMBENCH_RT_DENOISER", "oidn").strip().lower()
    if denoiser != "oidn":
        raise RuntimeError(
            "VPWEMCorrected requires RMBENCH_RT_DENOISER=oidn; "
            f"got {denoiser!r}"
        )
    return _get_vpwem_model(usr_args)


__all__ = ["eval", "get_model", "reset_model"]
