"""Candidate permutation-equivariant selectors with optional causal context recurrence."""
import math
import torch
from torch import nn


class Pointer(nn.Module):
    def __init__(self,candidate_dim,context_dim,width=128,dropout=.1,kind='mlp',soft_role=0.):
        super().__init__();self.kind=kind;self.soft_role=soft_role;self.width=width
        if kind=='linear':
            self.score=nn.Linear(candidate_dim+context_dim,1)
        elif kind=='query':
            self.query=nn.Sequential(nn.Linear(context_dim,width),nn.Mish(),nn.Linear(width,width))
            self.key=nn.Sequential(nn.Linear(candidate_dim,width),nn.Mish(),nn.Linear(width,width))
            self.score=nn.Linear(candidate_dim,1)
        else:
            self.score=nn.Sequential(nn.Linear(candidate_dim+context_dim,width),nn.Mish(),nn.Dropout(dropout),
                                      nn.Linear(width,max(16,width//4)),nn.Mish(),nn.Linear(max(16,width//4),1))
        self.null=nn.Sequential(nn.Linear(context_dim,max(16,width//2)),nn.Mish(),nn.Linear(max(16,width//2),1))
        self.role=nn.Sequential(nn.Linear(context_dim,width),nn.Mish(),nn.Linear(width,5)) if soft_role else None

    def forward(self,candidates,context,valid,classes=None):
        if self.kind=='query':
            score=(self.key(candidates)*self.query(context)[:,None]).sum(-1)/math.sqrt(self.width)
            score=score+self.score(candidates).squeeze(-1)
        else:
            inputs=torch.cat([candidates,context[:,None].expand(-1,candidates.shape[1],-1)],-1)
            score=self.score(inputs).squeeze(-1)
        role=self.role(context) if self.role is not None else None
        if role is not None:
            if classes is None:raise ValueError('Observed detector classes required for soft role priors')
            p=role.softmax(-1)
            prior=torch.stack([p[:,0]+p[:,4],p.new_full((len(p),),.01),p[:,1],p[:,2],p.new_full((len(p),),.01),p[:,3]],-1)
            score=score+self.soft_role*prior.gather(1,classes).clamp_min(.01).log()
        return torch.cat([score.masked_fill(~valid,float('-inf')),self.null(context)],-1),role


class RecurrentPointer(nn.Module):
    def __init__(self,candidate_dim,context_dim,width=64,reset_each_frame=False,skip_current=False):
        super().__init__();self.reset_each_frame=reset_each_frame;self.skip_current=skip_current;self.project=nn.Sequential(nn.Linear(context_dim+1,width),nn.Mish())
        self.gru=nn.GRU(width,width,batch_first=True)
        self.pointer=Pointer(candidate_dim,width+context_dim if skip_current else width,width=width)

    def forward(self,candidates,context,valid,gaps):
        # Inputs [episodes,time,...]; unidirectional recurrence never sees a future observation.
        projected=self.project(torch.cat([context,gaps[:,:,None]],-1))
        if self.reset_each_frame:
            hidden,_=self.gru(projected.reshape(-1,1,self.gru.input_size));hidden=hidden.reshape(*projected.shape)
        else:hidden,_=self.gru(projected)
        batch,length=hidden.shape[:2]
        if self.skip_current:hidden=torch.cat([hidden,context],-1)
        logits,_=self.pointer(candidates.reshape(batch*length,*candidates.shape[2:]),
            hidden.reshape(batch*length,-1),valid.reshape(batch*length,-1))
        return logits.reshape(batch,length,-1)

    def step(self,candidates,context,valid,gap,hidden=None):
        projected=self.project(torch.cat([context,gap[:,None]],-1))[:,None]
        state=None if self.reset_each_frame else hidden
        output,next_state=self.gru(projected,state)
        current=torch.cat([output[:,0],context],-1) if self.skip_current else output[:,0]
        logits,_=self.pointer(candidates,current,valid)
        return logits,None if self.reset_each_frame else next_state


def weighted_loss(logits,labels,weights,valid,smoothing=0.):
    known=labels>=0;safe=torch.where(known,labels,labels.new_full(labels.shape,8))
    logp=logits.log_softmax(-1);nll=-logp.gather(-1,safe[...,None]).squeeze(-1)
    if smoothing:
        allowed=torch.cat([valid,torch.ones_like(valid[...,:1])],-1)
        uniform=-torch.where(allowed,logp,torch.zeros_like(logp)).sum(-1)/allowed.sum(-1)
        nll=(1-smoothing)*nll+smoothing*uniform
    effective=torch.where(known,weights,torch.zeros_like(weights))
    return (nll*effective).sum()/effective.sum().clamp_min(1e-8)


def hard_negative_ranking_loss(logits,labels,positive,weights,valid,margin=1.):
    """Rank the canonical positive above incorrect observed boxes, never null.

    Other boxes matching the same target are not negative examples. Unlabelled,
    unavailable and no-rival rows make no contribution; no inference labels needed.
    """
    eligible=(labels>=0)&(labels<valid.shape[-1])
    safe=labels.clamp(0,valid.shape[-1]-1)
    rivals=valid&~positive
    eligible=eligible&rivals.any(-1)&positive.gather(-1,safe[:,None]).squeeze(-1)
    # Finite placeholders avoid inf*0 for rows without eligible rivals.
    negative=logits[:,:valid.shape[-1]].masked_fill(~rivals,float('-inf')).max(-1).values
    negative=torch.where(eligible,negative,torch.zeros_like(negative))
    target=logits.gather(-1,safe[:,None]).squeeze(-1)
    target=torch.where(eligible,target,torch.zeros_like(target))
    loss=torch.nn.functional.softplus(margin+negative-target)
    effective=torch.where(eligible,weights,torch.zeros_like(weights))
    return (loss*effective).sum()/effective.sum().clamp_min(1e-8)
