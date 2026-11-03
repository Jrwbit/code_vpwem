"""Offline contact/motion competition audit; never relabels or feeds the selector.

Collision links can contain a drawer panel and two visual handles. Keep the
whole identity set: a link contact cannot resolve a specific handle instance.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from growth_data import load_growth_data,growth_partitions
from data import load_group
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detector_comparison_20261030'))
from annotate import descriptors


def collision_groups(parts):
    groups={}
    for part in parts:
        body=int(part['inst'].pybullet_uuid)
        links=[-1] if part['indices'] is None else [int(part['inst'].link_ids[i]) for i in part['indices']]
        for link in set(links):
            group=groups.setdefault((body,link),dict(body=body,link=link,identities=[],classes=[]))
            if part['identity'] not in group['identities']:group['identities'].append(part['identity'])
            if int(part['cls']) not in group['classes']:group['classes'].append(int(part['cls']))
    for group in groups.values():
        group['identities'].sort();group['classes'].sort();group['resolved_instance']=group['identities'][0] if len(group['identities'])==1 else None
    return [groups[k] for k in sorted(groups)]


def contact_facts(groups,target,contact_m=.002,far_m=.05,motion_m=.005,motion_rad=.02):
    targets=[g for g in groups if target in g['identities']]
    if len(targets)!=1:raise ValueError('Target must have one collision group for this bowl audit')
    distance=targets[0]['min_hand_distance_m'];others=[g for g in groups if target not in g['identities'] and g['min_hand_distance_m']<=contact_m]
    moving=[g for g in others if g['translation_m']>=motion_m or g['rotation_rad']>=motion_rad]
    return dict(target_distance_m=distance,target_far=distance>far_m,target_touching=distance<=contact_m,
        competing_contact=distance>far_m and bool(others),competing_moving_contact=distance>far_m and bool(moving),
        contact_groups=[dict(body=g['body'],link=g['link'],identities=g['identities'],classes=g['classes'],resolved_instance=g['resolved_instance'],
            min_hand_distance_m=g['min_hand_distance_m'],translation_m=g['translation_m'],rotation_rad=g['rotation_rad']) for g in others])


def audit_plan(data,splits):
    result=[]
    for split,index in splits.items():
        for i in index:
            if data['y'][i]<0 or data['roles'][i]!=0:continue
            row=data['rows'][i];target=row['label']['target']
            assert target['class_id']==0
            result.append(dict(id=row['id'],source=str(Path(row['source']).resolve()),demo=row['demo'],episode_id=row['episode_id'],step=int(row['step']),
                partition=split,target=target['instance_id'],label_reason=row['label']['reason'],motion_onset=target.get('onset'),
                future_motion_steps=target['onset']-row['step'] if target.get('onset') is not None else None))
    assert len({(r['partition'],r['id']) for r in result})==len(result)
    return result


def snapshot(p,robot,hand,groups):
    result=[]
    for group in groups:
        body,link=group['body'],group['link'];distances=[]
        for finger in hand:
            distances.append(min([v[8] for v in p.getClosestPoints(robot,body,.15,linkIndexA=finger,linkIndexB=link)],default=.15))
        pose=p.getLinkState(body,link,computeForwardKinematics=True)[4:6] if link>=0 else p.getBasePositionAndOrientation(body)
        result.append(dict(group,min_hand_distance_m=float(min(distances)),position=list(pose[0]),orientation=list(pose[1])))
    return result


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    data=load_growth_data(a.root);splits=growth_partitions(data);plan=audit_plan(data,{k:splits[k] for k in ['train','dev']})
    research=a.root/'momart_target_research_20261101'
    if a.include_confirm8:
        assert (research/'confirm8_recurrent_confirmation/results.json').exists(),'Confirmation8 must already be examined'
        extra=load_group(research/'confirm8_corpus',research/'confirm8_features.npz',research/'confirm8_phases.json','examined')
        plan+=audit_plan(extra,dict(examined_confirm8=np.flatnonzero(extra['y']>=0)))
    reserve=json.loads((research/'confirm9_corpus/manifest.json').read_text());assert not {r['episode_id'] for r in plan}&{r['episode_id'] for r in reserve['episodes']}
    grouped=defaultdict(list)
    for row in plan:grouped[row['source'],row['demo']].append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.plan.json').write_text(json.dumps(dict(rows=plan,scope='Every known acquire-bowl query in train108/dev24; optional examined confirmation8 regression',
        reserve9_manifest_sha256=hashlib.sha256((research/'confirm9_corpus/manifest.json').read_bytes()).hexdigest(),thresholds=dict(contact_m=.002,far_bowl_m=.05,motion_m=.005,motion_rad=.02),
        interpretation='Privileged current contact and past8 motion only; no intent truth, automatic relabel, source exclusion, model inputs or confirmation9 processing'),indent=2))
    import egl_probe
    egl_probe.get_available_devices=lambda:[a.gpu]
    import pybullet as p
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver
    env=None;current_source=None;records=[];counts=defaultdict(Counter)
    try:
        for (source,demo),rows in sorted(grouped.items()):
            with h5py.File(source,'r') as f:
                if source!=current_source:
                    if env is not None:env.env.simulator.disconnect();env=None
                    meta=json.loads(f['data'].attrs['env_args']);env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=meta['env_kwargs']['ig_config'],postprocess_visual_obs=False,
                        render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640);current_source=source
                    groups=collision_groups(descriptors(env.env.simulator.renderer));robot=env.env.robots[0].robot_ids[0]
                    hand=[j for j in range(p.getNumJoints(robot)) if p.getJointInfo(robot,j)[12].decode() in ['gripper_link','r_gripper_finger_link','l_gripper_finger_link']];assert len(hand)==3
                ep=f['data'][demo];states=np.asarray(ep['states']);saver=WorldSaver(exclude_body_ids=[]);assert saver.serialized_length==states.shape[1]
                needed=sorted({max(0,row['step']+delta) for row in rows for delta in [-8,0]});cache={}
                for step in needed:
                    saver.deserialize(states[step]);saver.restore();p.performCollisionDetection();cache[step]=snapshot(p,robot,hand,groups)
                for row in rows:
                    now=cache[row['step']];before=cache[max(0,row['step']-8)];values=[]
                    for current,previous in zip(now,before):
                        assert (current['body'],current['link'])==(previous['body'],previous['link'])
                        translation=float(np.linalg.norm(np.asarray(current['position'])-previous['position']))
                        rotation=float(2*np.arccos(np.clip(abs(np.dot(current['orientation'],previous['orientation'])),0,1)))
                        values.append(dict(current,translation_m=translation,rotation_rad=rotation))
                    facts=contact_facts(values,row['target']);records.append(dict(row,facts=facts,groups=values));count=counts[row['partition']]
                    count['queries']+=1
                    for key in ['target_far','target_touching','competing_contact','competing_moving_contact']:count[key]+=facts[key]
                    if row['future_motion_steps'] is not None and row['future_motion_steps']>0:count['future_motion_labels']+=1
                print(json.dumps(dict(source=Path(source).name,demo=demo,processed=len(records),total=len(plan),counts={k:dict(v) for k,v in counts.items()})),flush=True)
        a.output.write_text(json.dumps(dict(protocol='contact_priority_audit_v1',records=records,counts={k:dict(v) for k,v in counts.items()},
            plan_sha256=hashlib.sha256(a.output.with_suffix('.plan.json').read_bytes()).hexdigest(),
            caveat='Contact with shared drawer collision link cannot resolve visual handles. Audit flags do not prove intent or justify dropping/relabeling frames.'),indent=2))
    finally:
        if env is not None:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--include-confirm8',action='store_true');p.add_argument('--gpu',type=int,default=1);main(p.parse_args())
