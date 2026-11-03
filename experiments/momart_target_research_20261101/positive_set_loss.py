"""Supervise any observed box matching the unchanged target instance."""
import torch


def positive_set_loss(logits, labels, positive, weights, valid, mode='marginal'):
    if mode not in ('marginal', 'uniform'):
        raise ValueError('Unknown positive-set objective: ' + mode)
    known = labels >= 0
    available = positive.any(-1)
    if (positive & ~valid).any() or (known & (available != (labels < valid.shape[-1]))).any():
        raise ValueError('Positive sets disagree with valid boxes or target availability')
    # Unknown and unavailable rows have a finite null placeholder. Unknown loss
    # weights are zero; unavailable targets still supervise null, as before.
    allowed = torch.cat([positive, (~available)[:, None]], -1)
    logp = logits.log_softmax(-1)
    if mode == 'marginal':
        loss = -logp.masked_fill(~allowed, float('-inf')).logsumexp(-1)
    else:
        loss = -torch.where(allowed, logp, torch.zeros_like(logp)).sum(-1) / allowed.sum(-1)
    effective = torch.where(known, weights, torch.zeros_like(weights))
    return (loss * effective).sum() / effective.sum().clamp_min(1e-8)
