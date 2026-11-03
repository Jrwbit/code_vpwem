"""Shared main crop, trainable shared ResNet18, two aligned historical tokens."""
import copy
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.ops import roi_align
from cleandiffuser.nn_condition.base_nn_condition import BaseNNCondition
from momart_no_history_baseline.model import build_agent as baseline_agent
from momart_target_no_history.legacy_tokens import CurrentObjectTransformer


def crop_sequence(images,offsets,height=75,width=100):
    b,t,c,h,w=images.shape
    yy=offsets[:,0,None,None]+torch.arange(height,device=images.device)[None,:,None]
    xx=offsets[:,1,None,None]+torch.arange(width,device=images.device)[None,None,:]
    linear=(yy*w+xx).reshape(b,1,1,-1).expand(b,t,c,-1)
    return images.flatten(-2).gather(-1,linear).reshape(b,t,c,height,width)


def transform_boxes(boxes,offsets,height=75,width=100,source_height=90,source_width=120):
    scale=boxes.new_tensor([source_width,source_height,source_width,source_height])
    shift=offsets[:,[1,0,1,0]].to(boxes.dtype)[:,None]
    raw=boxes*scale-shift
    limit=boxes.new_tensor([width,height,width,height])
    clipped=torch.minimum(raw.clamp_min(0),limit)
    original=(raw[...,2:]-raw[...,:2]).clamp_min(0).prod(-1)
    visible=(clipped[...,2:]-clipped[...,:2]).clamp_min(0).prod(-1)
    fraction=visible/original.clamp_min(1e-8)
    return clipped/limit,visible>0,fraction


class HistoryCondition(BaseNNCondition):
    def __init__(self,scene,args):
        super().__init__();self.scene=scene;self.has_history=args.history_variant!='box';self.seed=int(args.seed)
        self.register_buffer('crop_step',torch.zeros((),dtype=torch.long))
        self.crop_height,self.crop_width=map(int,args.crop_shape)
        self.scene.encoder.key_transform_map['rgb']=nn.Identity()
        self.scene.encoder.key_transform_map['depth']=nn.Identity()
        self.scene.encoder.key_shape_map['rgb']=(3,self.crop_height,self.crop_width)
        self.scene.encoder.key_shape_map['depth']=(1,self.crop_height,self.crop_width)
        self.roi_chunk=int(args.get('roi_chunk_size',128));self.checkpoint_roi=bool(args.get('checkpoint_roi',True))
        if self.has_history:
            self.appearance=nn.Linear(512,256)
            self.register_buffer('space_frequencies',torch.tensor([1.,2.,4.,8.])*math.pi)
            self.register_buffer('time_frequencies',2.**torch.arange(-4,4)*math.pi)
            self.position=nn.Sequential(nn.Linear(64,256),nn.Mish(),nn.Linear(256,256))
            self.time=nn.Sequential(nn.Linear(16,256),nn.Mish(),nn.Linear(256,256))
            self.visibility=nn.Linear(1,256)
            self.key=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.Mish())
            self.query=nn.Parameter(torch.randn(256)*.02)
            self.image_out=nn.Linear(256,256);self.position_out=nn.Linear(256,256)

    def set_training_step(self,step):self.crop_step.fill_(step)

    def offsets(self,b,device):
        if self.training:
            generator=torch.Generator().manual_seed(self.seed+100003*int(self.crop_step.item()))
            # Preserve legacy CropRandomizer's offset distribution, now shared
            # by main t0/t1/depth and all history frames of the same sample.
            raw=torch.rand(b,2,generator=generator)
            return (raw*torch.tensor([90-self.crop_height,120-self.crop_width])).long().to(device)
        return torch.tensor([round((90-self.crop_height)/2),round((120-self.crop_width)/2)],device=device).expand(b,-1)

    def encode_history(self,history,offsets,task):
        images=history['_history_rgb'];b,t=images.shape[:2]
        boxes,inside,fraction=transform_boxes(history['_history_boxes'],offsets,self.crop_height,self.crop_width)
        valid=history['_history_valid'].bool()&inside
        cropped=crop_sequence(images,offsets,self.crop_height,self.crop_width).flatten(0,1)
        ids=valid.flatten().nonzero().flatten()
        appearance=images.new_zeros(b*t,256)
        if ids.numel():
            coords=boxes.reshape(-1,4)[ids]*boxes.new_tensor([self.crop_width,self.crop_height,self.crop_width,self.crop_height])
            rois=torch.cat([ids[:,None].to(coords.dtype),coords],-1)
            patches=roi_align(cropped,rois,output_size=(self.crop_height,self.crop_width),spatial_scale=1.,sampling_ratio=2,aligned=True)
            cnn=self.scene.encoder.key_model_map['rgb'];chunks=[]
            for chunk in patches.split(self.roi_chunk):
                encoded=checkpoint(cnn,chunk,use_reentrant=False) if self.training and self.checkpoint_roi else cnn(chunk)
                chunks.append(self.appearance(encoded))
            appearance=appearance.index_copy(0,ids,torch.cat(chunks).to(appearance.dtype))
        appearance=appearance.reshape(b,t,256)
        x1,y1,x2,y2=boxes.unbind(-1)
        corners=torch.stack([x1,y1,x2,y1,x2,y2,x1,y2],-1)
        phase=corners[...,None]*self.space_frequencies
        position=self.position(torch.cat([phase.sin(),phase.cos()],-1).flatten(-2))
        phase=history['_history_ages'][...,None]*.05*self.time_frequencies
        temporal=self.time(torch.cat([phase.sin(),phase.cos()],-1))
        keys=self.key(appearance+position+temporal+self.visibility(fraction[...,None]))
        logits=(keys*(self.query+task)[:,None]).sum(-1)/16
        weights=logits.masked_fill(~valid,-10000).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        present=valid.any(-1).to(appearance.dtype)[:,None]
        image=(self.image_out((weights[...,None]*(appearance+temporal)).sum(1))+task)*present
        location=(self.position_out((weights[...,None]*(position+temporal)).sum(1))+task)*present
        return image,location,present

    def forward(self,obs,mask=None):
        obs=dict(obs);history={key:obs.pop(key) for key in list(obs) if key.startswith('_history_')}
        b=obs['rgb'].shape[0];offsets=self.offsets(b,obs['rgb'].device)
        obs['rgb']=crop_sequence(obs['rgb'],offsets,self.crop_height,self.crop_width)
        obs['depth']=crop_sequence(obs['depth'],offsets,self.crop_height,self.crop_width)
        scene=self.scene(obs,mask)
        if not self.has_history:return scene
        task=self.scene.task_embedding(obs['_task_id'].long().reshape(-1))
        image,position,present=self.encode_history(history,offsets,task)
        packed=torch.cat([image,position,present],-1).to(scene.dtype)
        return torch.cat([scene,packed[:,None].expand(-1,scene.shape[1],-1)],-1)


class HistoryTransformer(CurrentObjectTransformer):
    def __init__(self,args):
        super().__init__(args)
        # Every historical frame is <= t0, so both history tokens are causal for
        # ALL action rows. The newest scene token remains PTP-masked at row0.
        with torch.no_grad():self.memory_mask[:,:3]=0


def build_agent(args,normalizer):
    agent=baseline_agent(args,normalizer);old=agent.model['diffusion']
    agent.model['condition']=HistoryCondition(agent.model['condition'],args).to(args.device)
    if args.history_variant!='box':
        net=HistoryTransformer(args).to(args.device)
        previous=old.state_dict();state=net.state_dict()
        for key,value in previous.items():
            if key in state and state[key].shape==value.shape:state[key]=value
        net.load_state_dict(state)
        with torch.no_grad():
            net.cond_pos_emb[:,0]=old.cond_pos_emb[:,0]
            net.cond_pos_emb[:,3:]=old.cond_pos_emb[:,1:]
        agent.model['diffusion']=net
    agent.model_ema=copy.deepcopy(agent.model).requires_grad_(False)
    agent.model.train();agent.model_ema.eval()
    agent.optimizer=torch.optim.AdamW(agent.model.parameters(),lr=args.lr,betas=(args.adam_beta1,args.adam_beta2))
    return agent
