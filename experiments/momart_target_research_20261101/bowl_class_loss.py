"""Train-only bowl-versus-other-class risk from the pointer's own logits."""
import torch


def bowl_class_loss(logits, labels, roles, classes, weights, valid):
    # Only existing acquire/carry labels with a matched bowl proposal supervise
    # this auxiliary. Unavailable targets retain the original null CE alone.
    eligible = ((roles == 0) | (roles == 4)) & (labels >= 0) & (labels < valid.shape[-1])
    bowls = (classes == 0) & valid
    if (eligible & ~bowls.any(-1)).any():
        raise ValueError('Available acquisition/carry target has no observed bowl')
    # Finite null placeholder prevents logsumexp(-inf) for excluded rows.
    allowed = torch.cat([bowls, (~eligible)[:, None]], -1)
    loss = -logits.log_softmax(-1).masked_fill(~allowed, float('-inf')).logsumexp(-1)
    effective = torch.where(eligible, weights, torch.zeros_like(weights))
    return (loss * effective).sum() / effective.sum().clamp_min(1e-8)
