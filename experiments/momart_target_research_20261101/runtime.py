"""Causal one-box inference from public features; no labels or scene object GT API."""
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import numpy as np
import torch
from models import Pointer,RecurrentPointer


@dataclass
class FrameFeatures:
    candidates: np.ndarray  # [8,139]: ROI128, xyxy4, detector score, class one-hot6
    context: np.ndarray  # current/past main image features, current/delta proprio, task
    valid: np.ndarray
    task: str
    extra_candidates: np.ndarray = None
    extra_context: np.ndarray = None
    geometry_candidates: np.ndarray = None
    geometry_context: np.ndarray = None
    mask_candidates: np.ndarray = None


def view(frame,name):
    def required(key):
        value=getattr(frame,key)
        if value is None:raise ValueError('Required public feature is missing: '+key)
        return value
    c=frame.candidates;x=frame.context
    if name in ['rich','rich_geometry','mask_full']:
        c=np.c_[c,required('extra_candidates')];x=np.r_[x,required('extra_context')]
    if name in ['geometry','rich_geometry','mask_full']:
        c=np.c_[c,required('geometry_candidates')];x=np.r_[x,required('geometry_context')]
    if name in ['spatial','mask_small','mask_spatial']:
        c=frame.candidates[:,128:];x=np.r_[frame.context[256:],required('geometry_context')]
        if name in ['spatial','mask_spatial']:c=np.c_[c,required('geometry_candidates')]
    if name.startswith('mask_'):c=np.c_[c,required('mask_candidates')]
    pool=(c*frame.valid[:,None]).sum(0)/max(int(frame.valid.sum()),1)
    return c.astype(np.float32),np.r_[x,pool].astype(np.float32)


class SingleBoxSelector:
    def __init__(self,selection,root):
        self.root=Path(root);selection=json.loads(Path(selection).read_text())
        self.component=selection['result']['component'];self.calibration=selection['result']['calibration']
        self.tasks=json.loads((self.root/'momart_target_selector_20261101/features.json').read_text())['tasks']
        self.models={};self.reset()

    def reset(self):
        self.episode=None;self.last_step=None;self.states={};self.cached_result=None;self.fingerprint=None

    def load(self,path):
        key=str(path)
        if key not in self.models:self.models[key]=torch.load(path,map_location='cpu',weights_only=False)
        return self.models[key]

    def parent(self,component,frame,step):
        name=component['name'];path=Path(component['screen'])/(name+'.pt')
        if name=='previous_pointer':
            path=self.root/'momart_target_selector_20261101/pilot_seed100/selector.pt';ckpt=self.load(path)
            if '_net' not in ckpt:
                sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
                from selector import TargetSelector
                net=TargetSelector(ckpt['candidate_dim'],ckpt['context_dim']);net.load_state_dict(ckpt['model']);ckpt['_net']=net.eval()
            c=(frame.candidates-ckpt['candidate_mean'])/ckpt['candidate_std'];x=(frame.context-ckpt['context_mean'])/ckpt['context_std']
            return ckpt['_net'](torch.tensor(c[None]),torch.tensor(x[None]),torch.tensor(frame.valid[None])).softmax(-1)[0].numpy()
        ckpt=self.load(path);spec=ckpt['spec'];c,x=view(frame,spec['view'])
        if 'model' in ckpt:
            slots=np.r_[np.flatnonzero(frame.valid),8];rows=[]
            for slot in slots:rows.append(np.r_[c[slot] if slot<8 else np.zeros(c.shape[-1],np.float32),x,slot==8])
            model=ckpt['model'];scores=model.predict_proba(np.asarray(rows,np.float32))[:,list(model.classes_).index(1)]
            probability=np.zeros(9,np.float32);probability[slots]=scores
            if probability.sum()==0:probability[8]=1
            return probability/probability.sum()
        n=ckpt['normalizer'];c=torch.tensor(((c-n['cm'])/n['cs'])[None]);x=torch.tensor(((x-n['xm'])/n['xs'])[None]);valid=torch.tensor(frame.valid[None])
        if '_net' not in ckpt:
            if spec['kind']=='gru':net=RecurrentPointer(c.shape[-1],x.shape[-1],spec['width'])
            else:net=Pointer(c.shape[-1],x.shape[-1],spec['width'],spec.get('dropout',.1),spec['kind'],spec.get('soft_role',0.))
            net.load_state_dict(ckpt['state']);ckpt['_net']=net.eval()
        net=ckpt['_net']
        if spec['kind']=='gru':
            key=str(path);hidden=self.states.get(key);gap=math.log1p(0 if self.last_step is None else (step-self.last_step)*.05)
            encoded=net.project(torch.cat([x,torch.tensor([[gap]],dtype=x.dtype)],-1))[:,None]
            output,hidden=net.gru(encoded,hidden);self.states[key]=hidden
            logits=net.pointer(c,output[:,0],valid)[0]
        else:logits=net(c,x,valid,torch.tensor(frame.candidates[None,:,133:].argmax(-1)))[0]
        return logits.softmax(-1)[0].numpy()

    def probability(self,component,frame,step):
        key=json.dumps(component,sort_keys=True)
        if key in self.frame_cache:return self.frame_cache[key]
        kind=component['kind']
        if kind=='parent':p=self.parent(component,frame,step)
        elif kind=='blend':p=sum(w*self.probability(c,frame,step) for w,c in zip(component['weights'],component['members']))
        elif kind=='quality':
            model=self.load(component['model'])['model'];features=np.c_[frame.candidates,frame.extra_candidates,frame.geometry_candidates]
            q=np.zeros(8,np.float32)
            if frame.valid.any():q[frame.valid]=model.predict_proba(features[frame.valid])[:,list(model.classes_).index(True)]
            p=self.probability(component['parent'],frame,step).copy();before=p[:8].copy();p[:8]*=q**component['strength'];p[8]+=(before-p[:8]).sum()
        elif kind=='role_filter':
            from scipy.linalg import expm
            ckpt=self.load(component['model']);_,x=view(frame,ckpt['view']);model=ckpt['model'];emission=np.full(5,.001,np.float32)
            emission[model.classes_]=model.predict_proba(x[None])[0];emission/=emission.sum()
            if component['temporal']:
                state=self.states.get(key,ckpt['priors'][self.tasks.index(frame.task)].copy())
                if self.last_step is not None:state=state@expm(ckpt['rates']*min(5.,(step-self.last_step)*.05))
                state*=np.clip(emission,.01,1);state/=state.sum();self.states[key]=state
            else:state=emission
            prior=np.array([state[0]+state[4],.01,state[1],state[2],.01,state[3]])
            p=self.probability(component['parent'],frame,step).copy();classes=frame.candidates[:,133:].argmax(-1)
            p[:8]*=np.where(frame.valid,np.clip(prior[classes],.01,1)**component['strength'],0);p/=p.sum()
        else:raise ValueError('Unsupported component: '+kind)
        self.frame_cache[key]=p;return p

    @torch.inference_mode()
    def select(self,frame,episode_id,sensor_step):
        if frame.task not in self.tasks:raise ValueError('Task outside validated vocabulary: '+frame.task)
        if frame.candidates.shape!=(8,139) or frame.valid.shape!=(8,) or frame.context.shape!=(280,):raise ValueError('Expected 8 candidate slots and raw 280D context')
        if not np.allclose(frame.context[-len(self.tasks):],np.eye(len(self.tasks))[self.tasks.index(frame.task)]):raise ValueError('Task name and public context one-hot disagree')
        fingerprint=hashlib.sha256()
        for value in vars(frame).values():
            if isinstance(value,np.ndarray):fingerprint.update(value.tobytes())
            else:fingerprint.update(str(value).encode())
        fingerprint=fingerprint.hexdigest()
        if episode_id!=self.episode:self.reset();self.episode=episode_id
        if self.last_step is not None:
            if sensor_step<self.last_step:raise ValueError('Sensor steps must increase within an episode')
            if sensor_step==self.last_step:
                if fingerprint!=self.fingerprint:raise ValueError('Same sensor step has different public inputs')
                return copy.deepcopy(self.cached_result)
        self.frame_cache={};p=self.probability(self.component,frame,sensor_step)
        if not np.isfinite(p).all() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid probability distribution')
        slot=int(p.argmax());order=np.sort(p);confidence=float(p[slot]);margin=float(order[-1]-order[-2])
        accepted=slot<8 and frame.valid[slot] and confidence>=self.calibration['threshold'] and margin>=self.calibration.get('margin',0.)
        result=dict(valid=bool(accepted),slot=slot if accepted else 8,box=frame.candidates[slot,128:132].copy() if accepted else None,
            class_id=int(frame.candidates[slot,133:].argmax()) if accepted else None,confidence=confidence,margin=margin,probability=p.copy())
        self.last_step=sensor_step;self.fingerprint=fingerprint;self.cached_result=result;return copy.deepcopy(result)
