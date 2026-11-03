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
from sequence_metrics import sequence_metrics
from calibration import apply_calibration


def predict_parent(component,data,root):
    name=component['name'];screen=Path(component['screen'])
    if name=='previous_pointer':
        ckpt=torch.load(root/'momart_target_selector_20261101/pilot_seed100/selector.pt',map_location='cpu',weights_only=False)
        net=TargetSelector(ckpt['candidate_dim'],ckpt['context_dim']).eval();net.load_state_dict(ckpt['model'])
        with torch.inference_mode():
            return net(torch.tensor((data['candidates']-ckpt['candidate_mean'])/ckpt['candidate_std']),
                torch.tensor((data['context']-ckpt['context_mean'])/ckpt['context_std']),torch.tensor(data['valid'])).softmax(-1).numpy()
    ckpt=torch.load(screen/(name+'.pt'),map_location='cpu',weights_only=False);spec=ckpt['spec'];v=spec['view']
    if spec.get('policy_features_path'):
        from policy_features import attach_policy,policy_view
        if 'policy_context' not in data:data=attach_policy(data,Path(spec['policy_features_path']))
        raw,context=policy_view(data,v)
    elif spec.get('candidate_history'):
        from candidate_history import add_candidate_history,tracking_view
        if 'candidate_history' not in data:data=add_candidate_history(data)
        raw,context=tracking_view(data,v)
    elif spec.get('history_signals'):
        from robot_history import attach_history,history_view
        if 'history_gripper' not in data:attach_history(data,Path(spec['history_path']))
        raw,context=history_view(data,v,spec['history_signals'])
    elif v.startswith('mask_'):
        from mask_benchmark import mask_view
        raw,context=mask_view(data,v)
    else:
        raw,context=feature_view(data,rich=v in ('rich','rich_geometry'),geometry=v in ('geometry','rich_geometry','spatial'),spatial=v=='spatial')
    if 'model' in ckpt:
        x,_,_,positions=candidate_rows(data,dict(raw=raw,context=context))
        model=ckpt['model'];scores=model.predict_proba(x)[:,list(model.classes_).index(1)]
        prob=np.zeros((len(data['rows']),9),np.float32)
        for (i,slot),score in zip(positions,scores):prob[i,slot]=score
        empty=prob.sum(-1)==0;prob[empty,8]=1
        return prob/prob.sum(-1,keepdims=True)
    if spec['kind']=='grid_gru':
        from recurrent_selection import predict_grid_checkpoint
        return predict_grid_checkpoint(ckpt,data)
    n=ckpt['normalizer'];c=torch.tensor((raw-n['cm'])/n['cs']);x=torch.tensor((context-n['xm'])/n['xs'])
    if spec['kind']=='plan_query':
        from plan_model import ActionPlanPointer
        net=ActionPlanPointer(c.shape[-1],x.shape[-1],spec['width'],spec['horizon'])
    elif spec['kind']=='gru':net=RecurrentPointer(c.shape[-1],x.shape[-1],spec['width'])
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
    if kind=='hard_query':
        from hard_query import predict_query
        return predict_query(component,data,root)
    if kind=='blend':return sum(w*predict_component(c,data,root) for w,c in zip(component['weights'],component['members']))
    if kind=='availability':
        from availability_features import replace_availability
        probability=predict_component(component['parent'],data,root)
        if component['model'] is None:q=probability[:,:8].sum(-1)
        else:
            from availability import feature_matrix
            ckpt=torch.load(component['model'],map_location='cpu',weights_only=False);model=ckpt['model']
            q=model.predict_proba(feature_matrix(data,ckpt['view']))[:,list(model.classes_).index(True)]
        return replace_availability(probability,q,component['mix'],component['offset'])
    if kind=='role_filter':
        from temporal import role_features,soft_role_filter
        ckpt=torch.load(component['model'],map_location='cpu',weights_only=False);net=ckpt['model']
        emission=np.full((len(data['rows']),5),.001,np.float32)
        emission[:,net.classes_]=net.predict_proba(role_features(data,ckpt['view']));emission/=emission.sum(-1,keepdims=True)
        return soft_role_filter(predict_component(component['parent'],data,root),emission,data,ckpt['rates'],ckpt['priors'],component['strength'],temporal=component['temporal'])
    if kind=='quality':
        model=torch.load(component['model'],map_location='cpu',weights_only=False)['model']
        q=np.zeros(data['valid'].shape,np.float32);mask=data['valid']
        q[mask]=model.predict_proba(quality_features(data)[mask])[:,list(model.classes_).index(True)]
        return multiply_quality(predict_component(component['parent'],data,root),q,component['strength'])
    raise ValueError(kind)


def evaluation_gate(primary,baseline,revision):
    p=primary['accepted'];b=baseline['raw']
    covered=all(v['correct']>=1 for v in primary['roles'].values() if v['visible']>=3)
    floors={role:max(0.,(v['recall'] or 0)-.05) for role,v in baseline['roles'].items() if v['visible']>=10} if revision>=3 else {}
    retained=all((primary['roles'][role]['recall'] or 0)>=floor for role,floor in floors.items())
    priority={}
    if revision>=4:
        from expanded import priority_gate
        priority=priority_gate(p,primary['sequence'],dict(metrics=baseline['accepted'],sequence=baseline['sequence']))
    success=(p['precision'] or 0)>=.9 and (p['recall'] or 0)>=(b['recall'] or 0) and covered and retained
    if revision>=4:success=success and priority['priority_errors_improved'] and priority['switching_retained']
    return dict(joint_goal_met=bool(success),priority_gate=priority,role_recall_retained=retained,role_recall_floors=floors)


def incumbent_evaluation_gate(primary,control,revision):
    # A frozen deployed configuration includes rejection calibration. The old
    # pointer gate intentionally compares raw recall; the incumbent gate must not.
    return evaluation_gate(primary,dict(control,raw=control['accepted']),revision)


def main(a):
    torch.set_num_threads(2);research=a.root/'momart_target_research_20261101'
    a.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads((a.selection/'selected.json').read_text())
    baseline_screen=research/'screen_seed100';baseline=json.loads((baseline_screen/'registry.json').read_text())['previous_pointer']
    frozen=dict(primary=dict(name=selected['name'],component=selected['result']['component'],calibration=selected['result']['calibration']),
        baseline=dict(name='previous_pointer',component=dict(kind='parent',name='previous_pointer',screen=str(baseline_screen)),calibration=baseline['calibration']),
        evaluation_purpose=a.purpose,corpus_prefix=a.prefix,public_views=a.public_views,incumbent_gate_required=selected.get('incumbent_gate_required',False),
        selection_sha256=hashlib.sha256((a.selection/'selected.json').read_bytes()).hexdigest(),
        selection_policy='Registered development cohort only; frozen before confirmation labels/features are loaded',acceptance_revision=selected['result']['calibration'].get('acceptance_revision',2))
    if a.control_selection is not None:
        control=json.loads((a.control_selection/'selected.json').read_text())
        frozen['control']=dict(name=control['name'],component=control['result']['component'],calibration=control['result']['calibration'])
    if frozen['incumbent_gate_required'] and 'control' not in frozen:
        raise ValueError('Registered incumbent-relative confirmation requires the frozen control')
    frozen['current_retention_required']=selected.get('current_retention_required',False)
    if frozen['current_retention_required']:
        if 'retention_reference' not in selected:raise ValueError('Current retention must be frozen before confirmation preprocessing')
        for path,expected in selected['frozen_weight_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen model weights changed: '+path)
        if hashlib.sha256((a.control_selection/'selected.json').read_bytes()).hexdigest()!=selected['control_selection_sha256']:
            raise ValueError('Original control selection differs from frozen comparison')
        frozen['retention']=selected['retention_reference'];frozen['retention_selection_sha256']=selected['retention_selection_sha256']
    write_json(a.output/'frozen_evaluation_manifest.json',frozen)
    extra=research/('extra_'+a.prefix+'.npz') if a.public_views=='rich' else None
    data=attach_geometry(load_group(research/(a.prefix+'_corpus'),research/(a.prefix+'_features.npz'),research/(a.prefix+'_phases.json'),'confirm',extra),research/('geometry_'+a.prefix+'.npz'))
    if (research/('mask_'+a.prefix+'.npz')).exists():
        from mask_benchmark import attach_masks
        data=attach_masks(data,[research/('mask_'+a.prefix+'.npz')])
    index=np.flatnonzero(data['y']>=0);results={}
    corpus_meta=json.loads((research/(a.prefix+'_corpus')/'manifest.json').read_text())
    partitions_by_episode={ep['episode_id']:ep.get('source_partition','valid') for ep in corpus_meta['episodes']}
    for key in ['baseline','primary']+(['control'] if 'control' in frozen else [])+(['retention'] if 'retention' in frozen else []):
        item=frozen[key];prob=predict_component(item['component'],data,a.root)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        pred=apply_calibration(prob,item['calibration'],data['candidates'][:,:,133:].argmax(-1));raw=prob.argmax(-1)
        results[key]=dict(name=item['name'],raw=metrics(raw,index,data),accepted=metrics(pred,index,data),roles=role_breakdown(pred,index,data),sequence=sequence_metrics(pred,data,index),raw_sequence=sequence_metrics(raw,data,index),
            episodes={ep:metrics(pred,np.array([i for i in index if data['rows'][i]['episode_id']==ep]),data) for ep in sorted({r['episode_id'] for r in data['rows']})})
        results[key]['source_partitions']={}
        for partition in sorted(set(partitions_by_episode.values())):
            sub=np.array([i for i in index if partitions_by_episode[data['rows'][i]['episode_id']]==partition],int)
            results[key]['source_partitions'][partition]=dict(raw=metrics(raw,sub,data),accepted=metrics(pred,sub,data),roles=role_breakdown(pred,sub,data),sequence=sequence_metrics(pred,data,sub))
        np.savez_compressed(a.output/(key+'_predictions.npz'),probability=prob,prediction=pred,ids=np.array([r['id'] for r in data['rows']]))
    revision=frozen['primary']['calibration'].get('acceptance_revision',2)
    gate=evaluation_gate(results['primary'],results['baseline'],revision)
    source_gates={partition:evaluation_gate(results['primary']['source_partitions'][partition],results['baseline']['source_partitions'][partition],revision) for partition in results['primary']['source_partitions']}
    success=gate['joint_goal_met']
    if revision>=4 and 'valid' in source_gates:success=success and source_gates['valid']['joint_goal_met']
    incumbent_gates={}
    if frozen['incumbent_gate_required']:
        incumbent_gates['overall']=incumbent_evaluation_gate(results['primary'],results['control'],revision)
        for partition in results['primary']['source_partitions']:
            incumbent_gates[partition]=incumbent_evaluation_gate(results['primary']['source_partitions'][partition],results['control']['source_partitions'][partition],revision)
        success=bool(success and all(value['joint_goal_met'] for value in incumbent_gates.values()))
    retention_gates={}
    if frozen['current_retention_required']:
        from checkpoint_selection import current_retention_gate
        retention_gates['overall']=current_retention_gate(results['primary'],results['retention'])
        for partition in results['primary']['source_partitions']:
            retention_gates[partition]=current_retention_gate(results['primary']['source_partitions'][partition],results['retention']['source_partitions'][partition])
        success=bool(success and all(value['joint_goal_met'] for value in retention_gates.values()))
    payload=dict(results=results,source_partition_gates=source_gates,overall_gate=gate,priority_gate=gate['priority_gate'],acceptance_revision=revision,role_recall_retained=gate['role_recall_retained'],role_recall_floors=gate['role_recall_floors'],confirmation_joint_goal_met=success if a.purpose=='confirmation' else None,regression_joint_goal_met=success if a.purpose=='regression' else None,evaluation_purpose=a.purpose,total_frames=len(data['rows']),labelled_frames=len(index),
        caveat='Geometry/motion proxy labels, not human intention or closed-loop policy success. All methods share the same frozen proposal arrays; extraction device/batch are in corpus metadata. Source-valid subset also required for V4; source-nonvalid unseen episodes do not establish full visual/policy holdout.')
    payload['incumbent_gates']=incumbent_gates;payload['current_retention_gates']=retention_gates
    write_json(a.output/'results.json',payload)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','selection','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--control-selection',type=Path);p.add_argument('--prefix',default='confirmation');p.add_argument('--purpose',choices=['confirmation','regression'],default='confirmation')
    p.add_argument('--public-views',choices=['rich','geometry'],default='rich')
    main(p.parse_args())
