"""An instance pointer with a null option; labels are not model inputs."""
import torch
from torch import nn


class TargetSelector(nn.Module):
    def __init__(self,candidate_dim,context_dim):
        super().__init__()
        self.candidate=nn.Sequential(nn.Linear(candidate_dim+context_dim,128),nn.Mish(),
                                     nn.Dropout(.1),nn.Linear(128,32),nn.Mish(),nn.Linear(32,1))
        self.null=nn.Sequential(nn.Linear(context_dim,64),nn.Mish(),nn.Linear(64,1))

    def forward(self,candidates,context,valid):
        context_rows=context[:,None].expand(-1,candidates.shape[1],-1)
        logits=self.candidate(torch.cat([candidates,context_rows],-1)).squeeze(-1)
        logits=logits.masked_fill(~valid,float('-inf'))
        return torch.cat([logits,self.null(context)],-1)


class TargetLock:
    """Causal IoU identity preference; missing frames never emit a stale box."""
    def __init__(self,margin=.15,max_age_steps=16):
        self.margin=margin;self.max_age_steps=max_age_steps;self.reset()

    def reset(self):
        self.last_step=-1;self.last_seen=-1;self.box=None;self.class_id=None

    def choose(self,objects,probabilities,step):
        import numpy as np
        if step<=self.last_step:raise ValueError('Strictly increasing observation steps required')
        self.last_step=step
        scores=np.asarray(probabilities);raw=int(scores.argmax())
        raw=raw if raw<len(objects) else None
        previous=None;overlap=0.
        if self.box is not None and step-self.last_seen<=self.max_age_steps:
            a=np.asarray(self.box)
            for i,obj in enumerate(objects):
                if obj['class_id']!=self.class_id:continue
                b=np.asarray(obj['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                value=inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-8)
                if value>=.3 and value>overlap:overlap,previous=value,i
        chosen=raw
        # Null remains a meaningful decision; preference never invents detections.
        if raw is not None and previous is not None and scores[raw]<scores[previous]+self.margin:
            chosen=previous
        if chosen is not None:
            self.box=list(objects[chosen]['box']);self.class_id=objects[chosen]['class_id'];self.last_seen=step
        return chosen
