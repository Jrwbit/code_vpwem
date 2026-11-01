"""Privileged stage proxies exported separately from the causal model features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detector_comparison_20261030'))
from proxy import target_at
from phase_logic import holding_proxy,stage_label,ROLES


def main(a):
    import egl_probe
    egl_probe.get_available_devices=lambda:[a.gpu]
    import pybullet as p
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver
    from annotate import descriptors
    if a.output.exists():raise FileExistsError(a.output)
    corpus=json.loads((a.corpus/'manifest.json').read_text());events=json.loads(a.events.read_text())['events']
    env=None;scene=None;records=[];traces=[]
    try:
        for ep in corpus['episodes']:
            rows=[r for r in corpus['records'] if r['episode_id']==ep['episode_id']]
            if not rows:continue
            with h5py.File(ep['source'],'r') as f:
                meta=json.loads(f['data'].attrs['env_args']);cfg=meta['env_kwargs']['ig_config']
                signature={k:cfg[k] for k in ('scene_id','scene_instance','robot','load_room_types')}
                signature['objects']=[(o['name'],o['filename'],o['class_id']) for o in cfg['objects']]
                if env is None:
                    env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=cfg,postprocess_visual_obs=False,
                        render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640);scene=signature
                elif scene!=signature:raise ValueError('Scene mismatch')
                bowl=next(part for part in descriptors(env.env.simulator.renderer) if part['cls']==0)
                body=bowl['inst'].pybullet_uuid;robot=env.env.robots[0];robot_id=robot.robot_ids[0]
                fingers=[j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id,j)[12].decode()
                         in ('gripper_link','r_gripper_finger_link','l_gripper_finger_link')]
                assert len(fingers)==3
                states=np.asarray(f['data'][ep['demo']]['states']);proprio=np.asarray(f['data'][ep['demo']]['obs/proprio'])
                saver=WorldSaver(exclude_body_ids=[]);assert saver.serialized_length==states.shape[1]
                # Every sample's holding test uses only current and previous geometry.
                base={max(0,min(len(states)-1,r['step']+delta)) for r in rows for delta in (-16,-8,0,8,16)}
                needed=sorted(base | {max(0,step-8) for step in base})
                evidence={}
                for step in needed:
                    saver.deserialize(states[step]);saver.restore()
                    link=robot.parts['gripper_link'].body_part_index
                    pose=p.getLinkState(robot_id,link,computeForwardKinematics=True);ee,orn=pose[4:6]
                    bowl_pos=p.getBasePositionAndOrientation(body)[0]
                    relative=p.multiplyTransforms(*p.invertTransform(ee,orn),bowl_pos,[0,0,0,1])[0]
                    distance=min((point[8] for finger in fingers for point in p.getClosestPoints(robot_id,body,.15,
                                        linkIndexA=finger,linkIndexB=-1)),default=.15)
                    evidence[step]=dict(grasped=float(proprio[step,2]),bowl_distance=float(distance),bowl_relative=list(relative))
                def held(step):
                    step=max(0,min(len(states)-1,step));previous=max(0,step-8)
                    return holding_proxy(evidence[step],evidence[previous])
                local=[e for e in events if e['episode_id']==ep['episode_id']]
                for r in rows:
                    step=r['step'];current=held(step);before=any(held(step-delta) for delta in (8,16))
                    soon=any(held(step+delta) for delta in (8,16))
                    old=target_at(local,step)
                    label=stage_label(old,current,before,soon,evidence[step]['bowl_distance']<=.10,bowl['identity'])
                    records.append(dict(id=r['id'],episode_id=ep['episode_id'],step=step,split=ep['split'],label=label,
                        old_target=old,held_now=bool(current),held_before=bool(before),held_soon=bool(soon),
                        evidence=evidence[step]))
                traces.append(dict(episode_id=ep['episode_id'],samples=len(rows),held_now=sum(r['held_now'] for r in records if r['episode_id']==ep['episode_id'])))
                print(json.dumps(traces[-1]),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(protocol='stage_role_proxy_v1',roles=ROLES,records=records,episodes=traces,
            manifest_sha256=hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest(),
            limitations='Weak geometric holding, not contact/intent GT; future holding informs training-only pre-grasp labels; carry/place means no labelled recipient region, not no manipulation intent'),indent=2))
    finally:
        if env is not None:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);main(p.parse_args())
