"""Explicitly extend a completed cosine schedule without resetting Adam or EMA."""
import math


def extend_cosine_horizon(scheduler, total_steps):
    total_steps = int(total_steps)
    previous = int(scheduler.T_max)
    step = int(scheduler.last_epoch)
    if total_steps < previous or total_steps <= step:
        raise ValueError('The extended horizon must exceed the saved step and cannot shrink')
    if total_steps == previous:
        return False  # Ordinary restart of an already extended run: retain exact saved LR.
    scheduler.T_max = total_steps
    lrs = [scheduler.eta_min + (base - scheduler.eta_min)
           * (1 + math.cos(math.pi * step / total_steps)) / 2
           for base in scheduler.base_lrs]
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group['lr'] = lr
    scheduler._last_lr = lrs
    return True
