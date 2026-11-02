"""Select a subject from public observations and a self-predicted short plan.

Future demonstration commands are training targets only. The forward interface
has no argument for commands, contact, target labels, or future observations.
"""
import math
import torch
from torch import nn


class ActionPlanPointer(nn.Module):
    def __init__(self,candidate_dim,context_dim,width=64,horizon=8):
        super().__init__();self.horizon=horizon;self.width=width
        self.observation=nn.Sequential(nn.Linear(context_dim,width),nn.Mish(),nn.Linear(width,width))
        self.plan=nn.Sequential(nn.Linear(width,horizon*10),nn.Tanh())
        self.query=nn.Sequential(nn.Linear(width+horizon*10,width),nn.Mish(),nn.Linear(width,width))
        self.key=nn.Sequential(nn.Linear(candidate_dim,width),nn.Mish(),nn.Linear(width,width))
        self.score=nn.Linear(candidate_dim,1);self.null=nn.Linear(width,1)

    def forward(self,candidates,context,valid,classes=None):
        hidden=self.observation(context);plan=self.plan(hidden)
        query=self.query(torch.cat([hidden,plan],-1))
        score=(self.key(candidates)*query[:,None]).sum(-1)/math.sqrt(self.width)+self.score(candidates).squeeze(-1)
        return torch.cat([score.masked_fill(~valid,float('-inf')),self.null(hidden)],-1),plan.reshape(-1,self.horizon,10)


def plan_loss(prediction,target,valid):
    valid=valid.to(prediction.dtype)
    return (((prediction-target)**2)*valid[:,:,None]).sum()/(valid.sum()*10).clamp_min(1)
