"""Separate subject identity from target-box availability; V4 development selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from expanded import load_expanded_data,calibrate_v4,baseline_for
from data import partitions,feature_view,metrics
from mask_benchmark import mask_view
from benchmark import write_json
from confirm import predict_component,apply_calibration
from availability_features import public_availability_features,replace_availability


def feature_matrix(data,view):
    if view.startswith('mask_'):_,context=mask_view(data,view)
    else:_,context=feature_view(data,rich=view=='rich_geometry',geometry=True,spatial=view=='spatial')
    return public_availability_features(data['candidates'],data['valid'],data['geometry_candidates'],context)


def main(a):
    from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
    torch.set_num_threads(2);data=load_expanded_data(a.root);splits=partitions(data);train=splits['train'];a.output.mkdir(parents=True,exist_ok=False)
    r=a.root/'momart_target_research_20261101';registry_expanded=json.loads((r/'expanded_v4_seed100_300/registry.json').read_text())
    spatial=[v for name,v in registry_expanded.items() if name.startswith('spatial_') or name.startswith('mask_spatial_')]
    def merit_value(v):
        c=v['calibration'];return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else c['metrics']['f1'],-v['raw']['dev']['wrong_acquire'])
    components={'geometry_v4':json.loads((r/'frozen_for_confirm3/selected.json').read_text())['result']['component'],
        'spatial_v4':max(spatial,key=merit_value)['component'],'previous_pointer':dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100'))}
    write_json(a.output/'frozen_search_protocol.json',dict(method='separate target identity and availability',parents=components,
        train='expanded 28 episodes; no cohort3 labels fitted',dev='expanded8 episodes only',hypothesis_source='examined cohort3: conditional identity can be correct while null probability dominates; cohort3 now regression, never fresh confirmation again',
        views=['spatial','geometry','mask_spatial'],kinds=['extra leaf5','boost leaves7'],mixes=[.5,1.],offsets=[-.5,0.,.5],bias_controls=[0.,.5,1.,2.,3.,4.],acceptance_revision=4,confirmation4_not_loaded=True))
    probabilities={name:predict_component(component,data,a.root) for name,component in components.items()}
    baseline=baseline_for(probabilities['previous_pointer'],splits['dev'],data);registry={}
    def record(name,prob,component):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        cal=calibrate_v4(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,raw={k:metrics(prob.argmax(-1),v,data) for k,v in splits.items() if len(v)},reject={k:metrics(pred,v,data) for k,v in splits.items() if len(v)})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
    for parent,prob in probabilities.items():
        for offset in [0.,.5,1.,2.,3.,4.]:
            component=dict(kind='availability',parent=components[parent],model=None,mix=0.,offset=offset)
            record(parent+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),component)
    target=data['positive'].any(-1)
    for view in ['spatial','geometry','mask_spatial']:
        x=feature_matrix(data,view)
        for kind in ['extra','boost']:
            if kind=='extra':model=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=5,max_features=.7,n_jobs=2,random_state=100)
            else:model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=1.,early_stopping=False,random_state=100)
            model.fit(x[train],target[train],sample_weight=data['weights'][train]);q=model.predict_proba(x)[:,list(model.classes_).index(True)]
            path=a.output/(view+'_'+kind+'_availability.pt');torch.save(dict(model=model,view=view,seed=100,label='same-frame GT subject has a matching current detector proposal; not intent-negative'),path)
            for parent,prob in probabilities.items():
                for mix in [.5,1.]:
                    for offset in [-.5,0.,.5]:
                        name=f'{parent}_{view}_{kind}_mix{mix}_offset{offset}'
                        component=dict(kind='availability',parent=components[parent],model=str(path),mix=mix,offset=offset)
                        record(name,replace_availability(prob,q,mix,offset),component)
            print(view,kind,len(registry),flush=True)
    winner=max(registry,key=lambda name:merit_value(registry[name]));write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),acceptance_revision=4,selection_partition='expanded/dev only; no cohort3 fit or selection',confirmation4_evaluated=False))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
