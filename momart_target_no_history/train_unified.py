"""Jointly train ONE five-task target selector for all ten data sources."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
RESEARCH=Path(__file__).resolve().parents[1]/'experiments/momart_target_research_20261101'
sys.path.insert(0,str(RESEARCH))
from benchmark_growth import train_model
from growth_data import load_growth_data
from data import load_group,attach_geometry,feature_view,metrics
from availability_features import replace_availability
from calibration import apply_calibration
from export_runtime_bundle import export_bundle

FAMILIES=['table_cleanup_to_dishwasher','table_setup_from_dishwasher','table_setup_from_dresser','unload_dishwasher_to_dresser','table_cleanup_to_sink']


def family(task):
    for name in FAMILIES:
        if task in (name,name+'_expert',name+'_suboptimal'):return name
    raise ValueError('Unsupported task source: '+task)


def remap(data):
    d=copy.copy(data);tasks=[family(t) for t in data['tasks']]
    mapped=np.zeros((len(data['context']),5),np.float32)
    for i,name in enumerate(tasks):mapped[:,FAMILIES.index(name)]+=data['context'][:,-len(tasks)+i]
    d['context']=np.c_[data['context'][:,:-len(tasks)],mapped].astype(np.float32);d['tasks']=FAMILIES
    return d


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False);r=a.root/'momart_target_research_20261101'
    old=remap(load_growth_data(a.root))
    new=remap(attach_geometry(load_group(r/'sink_extension_20261103_corpus',r/'sink_extension_20261103_features.npz',r/'sink_extension_20261103_phases.json','old'),r/'geometry_sink_extension_20261103.npz'))
    keys=['candidates','context','valid','y','roles','positive','visible','weights','geometry_candidates','geometry_context']
    data={k:np.concatenate([old[k],new[k]]) for k in keys};data['tasks']=FAMILIES;data['rows']=old['rows']+new['rows']
    old_n=len(old['rows']);known=data['y']>=0
    splits={name:np.array([i for i,row in enumerate(data['rows']) if known[i] and row['tag']=='old' and row['split']==name],int) for name in ['train','dev','test']}
    # Only newly sampled sink test episodes are a new held-out result. Historical
    # test/regression rows must never accidentally enter training or this report.
    sink_dev=splits['dev'][splits['dev']>=old_n];sink_test=splits['test'][splits['test']>=old_n]
    if not len(sink_dev) or not len(sink_test):raise ValueError('No labelled sink dev/test observations')
    for name in ['train','dev','test']:
        episodes={data['rows'][i]['episode_id'] for i in splits[name]}
        for other in ['train','dev','test']:
            if name!=other and episodes & {data['rows'][i]['episode_id'] for i in splits[other]}:raise ValueError('Episode split leakage')
    audit=dict(train_samples=len(splits['train']),dev_samples=len(splits['dev']),sink_dev_samples=len(sink_dev),sink_test_samples=len(sink_test),
        known_sink_train=int(sum(i>=old_n for i in splits['train'])),tasks=FAMILIES,seed=100,
        routing='one shared five-task model, identical weights and inference path for all ten sources',
        label_standard='active_subject_v1 weak motion/proximity/holding; not human intent truth',
        test_used_for_selection=False,confirm10_opened=False)
    (a.output/'protocol.json').write_text(json.dumps(audit,indent=2))
    view=feature_view(data,geometry=True)
    # train_model selects an epoch only by the explicit development indices.
    # Upsample the new task through weights without excluding old replay rows.
    fit=dict(data,weights=data['weights'].copy());fit['weights'][old_n:]*=4
    best_macro = dict(score=-1.)
    def keep_macro(epoch,net,prob,normalizer):
        values=[]
        for task in FAMILIES:
            indices=np.array([i for i in splits['dev'] if family(data['rows'][i]['task'])==task],int)
            if not len(indices):raise ValueError('Every task needs a development episode')
            values.append(metrics(prob.argmax(-1),indices,data)['f1'])
        score=float(np.mean(values))
        if score>best_macro['score']:
            best_macro.update(score=score,epoch=epoch,state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},prob=prob.copy(),normalizer=normalizer)
    probability,checkpoint,history=train_model(fit,view,splits['train'],splits['dev'],100,a.device,epoch_callback=keep_macro)
    checkpoint.update(state=best_macro['state'],best_epoch=best_macro['epoch'],normalizer=best_macro['normalizer'])
    probability=best_macro['prob'];history['selection']='Maximum equal-weight five-task dev macro F1; no test selection'
    history['selected_macro_f1']=best_macro['score']
    checkpoint['spec']=dict(view='geometry',kind='mlp',width=64,training_population='original108 + sink16 train episodes',tasks=FAMILIES)
    torch.save(checkpoint,a.output/'unified_five_task_seed100.pt')
    (a.output/'training.json').write_text(json.dumps(history,indent=2))
    points=[]
    for offset in [0.,4.]:
        p=replace_availability(probability,probability[:,:8].sum(-1),0.,offset)
        for threshold in [0.,.25,.4,.5,.6,.7,.8,.9,.95]:
            cal=dict(threshold=threshold,margin=0.)
            pred=apply_calibration(p,cal);m=metrics(pred,splits['dev'],data)
            per_task={task:metrics(pred,np.array([i for i in splits['dev'] if family(data['rows'][i]['task'])==task],int),data) for task in FAMILIES}
            points.append(dict(offset=offset,calibration=cal,metrics=m,per_task=per_task,macro_f1=float(np.mean([v['f1'] for v in per_task.values()]))))
    feasible=[v for v in points if (v['metrics']['precision'] or 0)>=.9 and v['metrics']['selected']>=10]
    winner=max(feasible or points,key=lambda v:(v['macro_f1'],v['metrics']['recall'] or 0))
    component=dict(kind='availability',parent=dict(kind='parent',name='unified_five_task_seed100',screen=str(a.output)),model=None,mix=0.,offset=winner['offset'])
    selection=dict(name='unified_five_task_seed100',result=dict(component=component,calibration=winner['calibration']))
    (a.output/'selected.json').write_text(json.dumps(selection,indent=2))
    voc=a.output/'momart_target_selector_20261101';voc.mkdir();(voc/'features.json').write_text(json.dumps(dict(tasks=FAMILIES)))
    export_bundle(a.output/'selected.json',a.output,a.output/'bundle')
    # Report holdout only after the checkpoint, calibration and relocatable
    # bundle have been frozen. No subsequent retuning is performed here.
    p=replace_availability(probability,probability[:,:8].sum(-1),0.,winner['offset']);pred=apply_calibration(p,winner['calibration'])
    report=dict(protocol=audit,dev_selection=winner,sink_test=metrics(pred,sink_test,data),
        original_dev_replay=metrics(pred,splits['dev'][splits['dev']<old_n],data),
        deployment='ONE jointly trained five-task model for ALL ten sources; incumbent is offline comparison only')
    (a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');main(p.parse_args())
