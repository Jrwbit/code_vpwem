"""Audit weak holding labels against direct contacts and moving support bodies.

Audit-only privileged scene geometry. No labels are overwritten, no physics is
stepped, and no contact/object-state fields are added to model inputs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detector_comparison_20261030'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_stage_target_20261101'))
from annotate import descriptors
from phase_logic import holding_proxy


def snapshot(p,robot,bowl,hand,grasped):
    hand_pose=p.getLinkState(robot,hand['gripper_link'],computeForwardKinematics=True)[4:6];bowl_pose=p.getBasePositionAndOrientation(bowl)
    relative=p.multiplyTransforms(*p.invertTransform(*hand_pose),*bowl_pose)[0]
    distances={name:min([point[8] for point in p.getClosestPoints(robot,bowl,.15,linkIndexA=link,linkIndexB=-1)],default=.15) for name,link in hand.items()}
    hand_contacts=defaultdict(set)
    for point in p.getContactPoints(bodyA=robot):
        if point[3] in hand.values() and point[8]<=.002:hand_contacts[(point[2],point[4])].add(point[3])
    supports={}
    for point in p.getContactPoints(bodyA=bowl):
        body,link=point[2],point[4]
        if body==robot or point[8]>.002 or point[7][2]<.3:continue
        pose=p.getLinkState(body,link,computeForwardKinematics=True)[4:6] if link>=0 else p.getBasePositionAndOrientation(body)
        relative_support=p.multiplyTransforms(*p.invertTransform(*pose),*bowl_pose)[0]
        supports[f'{body}:{link}']=dict(position=list(pose[0]),orientation=list(pose[1]),bowl_relative=list(relative_support),hand_links=sorted(hand_contacts[(body,link)]))
    return dict(grasped=float(grasped),bowl_distance=float(min(distances.values())),bowl_relative=list(relative),finger_distances=distances,supports=supports)


def facts(current,previous):
    bilateral=all(current['finger_distances'][name]<=.005 for name in ['r_gripper_finger_link','l_gripper_finger_link'])
    coupled=[]
    for key,value in current['supports'].items():
        if key not in previous['supports']:continue
        before=previous['supports'][key]
        delta=float(np.linalg.norm(np.asarray(value['position'])-before['position']))
        dot=abs(float(np.dot(value['orientation'],before['orientation'])));angle=2*np.arccos(np.clip(dot,0,1))
        relative=float(np.linalg.norm(np.asarray(value['bowl_relative'])-before['bowl_relative']))
        if (delta>=.01 or angle>=.05) and relative<=.025 and len(value['hand_links'])>=2:coupled.append(key)
    return dict(bilateral_bowl_5mm=bilateral,supported=bool(current['supports']),moving_carrier_contacts=coupled,
        carrier_motion_with_unilateral_grip=bool(coupled and not bilateral))


def main(a):
    import egl_probe
    egl_probe.get_available_devices=lambda:[a.gpu]
    import pybullet as p
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver
    if a.output.exists():raise FileExistsError(a.output)
    assert len(a.corpora)==len(a.phases)
    frames={};groups=defaultdict(set)
    for corpus,phase_path in zip(a.corpora,a.phases):
        manifest=json.loads((corpus/'manifest.json').read_text());phase=json.loads(phase_path.read_text())
        assert phase['manifest_sha256']==hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()
        by_id={row['id']:row for row in phase['records']}
        for row in manifest['records']:
            key=(str(Path(row['source']).resolve()),row['demo'],row['step']);entry=frames.setdefault(key,dict(tags=[],old_held_flags=[]))
            entry['tags'].append(str(corpus.name));entry['old_held_flags'].append(bool(by_id[row['id']]['held_now']));groups[key[:2]].add(key[2])
    env=None;current_source=None;records=[];totals=defaultdict(int)
    try:
        for (source,demo),steps in sorted(groups.items()):
            with h5py.File(source,'r') as f:
                if current_source!=source:
                    if env is not None:env.env.simulator.disconnect();env=None
                    meta=json.loads(f['data'].attrs['env_args']);env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=meta['env_kwargs']['ig_config'],postprocess_visual_obs=False,render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640);current_source=source
                    bowl=next(z for z in descriptors(env.env.simulator.renderer) if z['cls']==0)['inst'].pybullet_uuid;robot=env.env.robots[0].robot_ids[0]
                    hand={p.getJointInfo(robot,j)[12].decode():j for j in range(p.getNumJoints(robot)) if p.getJointInfo(robot,j)[12].decode() in ['gripper_link','r_gripper_finger_link','l_gripper_finger_link']};assert len(hand)==3
                ep=f['data'][demo];states=np.asarray(ep['states']);proprio=np.asarray(ep['obs/proprio']);saver=WorldSaver(exclude_body_ids=[]);assert saver.serialized_length==states.shape[1]
                base={max(0,min(len(states)-1,t+delta)) for t in steps for delta in [-16,-8,0,8,16]};needed=sorted(base|{max(0,t-8) for t in base});cache={}
                for t in needed:
                    saver.deserialize(states[t]);saver.restore();p.performCollisionDetection();cache[t]=snapshot(p,robot,bowl,hand,proprio[t,2])
                def old_hold(t):
                    t=max(0,min(len(states)-1,t));return bool(holding_proxy(cache[t],cache[max(0,t-8)]))
                for t in sorted(steps):
                    old=old_hold(t);evidence=facts(cache[t],cache[max(0,t-8)]);entry=frames[(source,demo,t)]
                    reproduced=all(old==value for value in entry['old_held_flags'])
                    future=[max(0,min(len(states)-1,t+delta)) for delta in [8,16]]
                    old_future=[u for u in future if old_hold(u)]
                    ambiguous_soon=bool(old_future and all(facts(cache[u],cache[max(0,u-8)])['carrier_motion_with_unilateral_grip'] for u in old_future))
                    records.append(dict(source=source,demo=demo,step=t,tags=entry['tags'],old_held=old,old_holding_reproduced=reproduced,
                        evidence=evidence,old_held_with_carrier_conflict=old and evidence['carrier_motion_with_unilateral_grip'],all_old_future_holds_carrier_conflicted=ambiguous_soon,
                        geometry=cache[t]))
                    totals['frames']+=1;totals['old_held']+=old;totals['old_holding_reproduction_mismatches']+=not reproduced
                    totals['old_held_with_carrier_conflict']+=old and evidence['carrier_motion_with_unilateral_grip'];totals['ambiguous_future_holding_frames']+=ambiguous_soon
                    totals['old_held_supported_without_bilateral_grip']+=old and evidence['supported'] and not evidence['bilateral_bowl_5mm']
                print(json.dumps(dict(source=Path(source).name,demo=demo,frames=len(steps),cumulative=dict(totals))),flush=True)
        a.output.write_text(json.dumps(dict(protocol='contact_support_audit_only_v1',totals=dict(totals),records=records,
            thresholds=dict(hand_contact_m=.002,bilateral_bowl_m=.005,support_motion_m=.01,support_relative_stability_m=.025),
            caveat='Potential label conflicts, not human intent GT. One-finger grasps may be legitimate. No labels changed; future geometry only audits training/eval labels, never inference.'),indent=2))
    finally:
        if env is not None:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpora',type=Path,nargs='+',required=True);p.add_argument('--phases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);main(p.parse_args())
