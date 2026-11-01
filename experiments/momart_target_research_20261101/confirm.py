"""Freeze dev-selected components before loading confirmation labels; never refit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import load_group,attach_geometry,feature_view,metrics
from models import Pointer,RecurrentPointer
from benchmark import TargetSelector,candidate_rows,sequence_batch,write_json
from refine import quality_features,multiply_quality,role_breakdown


def predict_parent(component,data,root):
    name=component['name'];screen=Path(component['screen'])
    if name=='previous_pointer':
        ckpt=torch.load(root/'momart_target_selector_20261101/pilot_seed100/selector.pt',map_location='cpu',weights_only=False)
        net=TargetSelector(ckpt['candidate_dim'],ckpt['context_dim']).eval();net.load_state_dict(ckpt['model'])
        with torch.inference_mode():
            return net(torch.tensor((data['candidates']-ckpt['candidate_mean'])/ckpt['candidate_std']),
                torch.tensor((data['context']-ckpt['context_mean'])/ckpt['context_std']),torch.tensor(data['valid'])).softmax(-1).numpy()
    ckpt=torch.load(screen/(name+'.pt'),map_location='cpu',weights_only=False);spec=ckpt['spec'];v=spec['view']
    raw,context=feature_view(data,rich=v in ('rich','rich_geometry'),geometry=v in ('geometry','rich_geometry','spatial'),spatial=v=='spatial')
    if 'model' in ckpt:
        x,_,_,positions=candidate_rows(data,dict(raw=raw,context=context))
        model=ckpt['model'];scores=model.predict_proba(x)[:,list(model.classes_).index(1)]
        prob=np.zeros((len(data['rows']),9),np.float32)
        for (i,slot),score in zip(positions,scores):prob[i,slot]=score
        empty=prob.sum(-1)==0;prob[empty,8]=1
        return prob/prob.sum(-1,keepdims=True)
    n=ckpt['normalizer'];c=torch.tensor((raw-n['cm'])/n['cs']);x=torch.tensor((context-n['xm'])/n['xs'])
    if spec['kind']=='gru':net=RecurrentPointer(c.shape[-1],x.shape[-1],spec['width'])
    else:net=Pointer(c.shape[-1],x.shape[-1],spec['width'],spec.get('dropout',.1),spec['kind'],spec.get('soft_role',0.))
    net.load_state_dict(ckpt['state']);net.eval()
    with torch.inference_mode():
        if spec['kind']=='gru':
            seq=sequence_batch(data,dict(candidates=c,contexts=x));p=net(seq['c'],seq['x'],seq['mask'],seq['gaps']).softmax(-1).numpy()
            result=np.zeros((len(c),9),np.float32)
            for row,indices in enumerate(seq['groups']):result[indices]=p[row,:len(indices)]
            return result
        return net(c,x,torch.tensor(data['valid']),torch.tensor(data['candidates'][:,:,133:].argmax(-1)))[0].softmax(-1).numpy()


def predict_component(component,data,root):
    kind=component['kind']
    if kind=='parent':return predict_parent(component,data,root)
    if kind=='blend':return sum(w*predict_component(c,data,root) for w,c in zip(component['weights'],component['members']))
    if kind=='quality':
        model=torch.load(component['model'],map_location='cpu',weights_only=False)['model']
        q=np.zeros(data['valid'].shape,np.float32);mask=data['valid']
        q[mask]=model.predict_proba(quality_features(data)[mask])[:,list(model.classes_).index(True)]
        return multiply_quality(predict_component(component['parent'],data,root),q,component['strength'])
    raise ValueError(kind)


def apply_calibration(prob,cal):
    order=np.sort(prob,axis=-1)
    return np.where((prob.max(-1)>=cal['threshold'])&((order[:,-1]-order[:,-2])>=cal.get('margin',0)),prob.argmax(-1),8)


def main(a):
    torch.set_num_threads(2);research=a.root/'momart_target_research_20261101'
    a.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads((a.selection/'selected.json').read_text())
    baseline_screen=research/'screen_seed100';baseline=json.loads((baseline_screen/'registry.json').read_text())['previous_pointer']
    frozen=dict(primary=dict(name=selected['name'],component=selected['result']['component'],calibration=selected['result']['calibration']),
        baseline=dict(name='previous_pointer',component=dict(kind='parent',name='previous_pointer',screen=str(baseline_screen)),calibration=baseline['calibration']),
        selection_sha256=hashlib.sha256((a.selection/'selected.json').read_bytes()).hexdigest(),
        selection_policy='old dev only; frozen before confirmation labels/features are loaded',acceptance_revision=2)
    write_json(a.output/'frozen_evaluation_manifest.json',frozen)
    data=attach_geometry(load_group(research/'confirmation_corpus',research/'confirmation_features.npz',research/'confirmation_phases.json','confirm',research/'extra_confirmation.npz'),research/'geometry_confirmation.npz')
    index=np.flatnonzero(data['y']>=0);results={}
    for key in ['baseline','primary']:
        item=frozen[key];prob=predict_component(item['component'],data,a.root)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        pred=apply_calibration(prob,item['calibration']);raw=prob.argmax(-1)
        results[key]=dict(name=item['name'],raw=metrics(raw,index,data),accepted=metrics(pred,index,data),roles=role_breakdown(pred,index,data),
            episodes={ep:metrics(pred,np.array([i for i in index if data['rows'][i]['episode_id']==ep]),data) for ep in sorted({r['episode_id'] for r in data['rows']})})
        np.savez_compressed(a.output/(key+'_predictions.npz'),probability=prob,prediction=pred,ids=np.array([r['id'] for r in data['rows']]))
    p=results['primary']['accepted'];b=results['baseline']['raw']
    success=(p['precision'] or 0)>=.9 and (p['recall'] or 0)>=(b['recall'] or 0) and all(r['correct']>=1 for r in results['primary']['roles'].values() if r['visible']>=3)
    write_json(a.output/'results.json',dict(results=results,confirmation_joint_goal_met=success,total_frames=len(data['rows']),labelled_frames=len(index),
        caveat='Geometry/motion proxy labels, not human intention or closed-loop policy success. CPU batch4 proposal extraction on confirmation; all methods share proposals.'))
    print(json.dumps(dict(results=results,confirmation_joint_goal_met=success),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','selection','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
