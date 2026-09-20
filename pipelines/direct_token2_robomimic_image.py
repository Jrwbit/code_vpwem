"""Train the Robomimic direct-history ablation with exactly two history tokens.

This is a narrow launcher around ``direct_history_robomimic_image``. It keeps
that pipeline's dataset and optimization code, while forcing the deterministic
history pooling output and the diffusion history width to two tokens.
"""

import pathlib
import runpy

import cleandiffuser.nn_condition as condition_module
import cleandiffuser.nn_diffusion as diffusion_module


_BaseCondition = condition_module.DirectSpatialHistoryCondition
_BaseDiffusion = diffusion_module.DirectHistoryChiTransformerPTP


class DirectToken2Condition(_BaseCondition):
    def __init__(self, *args, **kwargs):
        kwargs["history_output_tokens"] = 2
        super().__init__(*args, **kwargs)


class DirectToken2Diffusion(_BaseDiffusion):
    def __init__(self, *args, **kwargs):
        positional = list(args)
        if len(positional) >= 5:
            positional[4] = 2
        else:
            kwargs["history_tokens"] = 2
        super().__init__(*positional, **kwargs)


condition_module.DirectSpatialHistoryCondition = DirectToken2Condition
diffusion_module.DirectHistoryChiTransformerPTP = DirectToken2Diffusion

if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).with_name(
            "direct_history_robomimic_image.py")),
        run_name="__main__",
    )
