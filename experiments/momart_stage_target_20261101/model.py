"""Causal role head + instance pointer. Class gating depends on predictions only."""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
from selector import TargetSelector
from phase_logic import ROLE_CLASS


class StagePointer(nn.Module):
    def __init__(self,candidate_dim,context_dim):
        super().__init__()
        self.pointer=TargetSelector(candidate_dim,context_dim)
        self.role=nn.Sequential(nn.Linear(context_dim,128),nn.Mish(),nn.Dropout(.1),nn.Linear(128,len(ROLE_CLASS)))

    def forward(self,candidates,context,valid):
        return self.pointer(candidates,context,valid),self.role(context)


def gated_logits(pointer,role_logits,classes,valid):
    role=role_logits.argmax(-1)
    allowed=torch.tensor(ROLE_CLASS,device=role.device)[role]
    mask=valid & (classes==allowed[:,None])
    return torch.cat([pointer[:,:-1].masked_fill(~mask,float('-inf')),pointer[:,-1:]],-1)
