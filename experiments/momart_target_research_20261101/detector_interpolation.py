"""Weight-space trust region around an unchanged detector initializer."""
import math
import torch


def interpolate_detector(initial, adapted, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Interpolation alpha must be finite and within [0,1]')
    if initial.keys() != adapted.keys():
        raise ValueError('Detector state keys differ')
    result = {}
    for name, value in initial.items():
        other = adapted[name]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise ValueError('Detector state type/shape differs: ' + name)
        fixed = name.startswith('backbone.') or not value.is_floating_point()
        if fixed and not torch.equal(value, other):
            raise ValueError('Frozen detector state changed: ' + name)
        if fixed or alpha == 0:
            result[name] = value.clone()
        elif alpha == 1:
            result[name] = other.clone()
        else:
            result[name] = torch.lerp(value, other, alpha)
    return result
