"""Restore recorded states without stepping; export motion + gripper proximity proxies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detector_comparison_20261030'))
from proxy import motion_intervals, target_at


def main(args):
    import egl_probe
    egl_probe.get_available_devices=lambda:[args.render_gpu]
    import pybullet as bullet
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver,get_movable_joints
    from annotate import descriptors
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    env=None;events=[];rejected=[];scene_signature=None;episode_rows=[]
    try:
        for ep in manifest['episodes']:
            with h5py.File(ep['source'],'r') as handle:
                meta=json.loads(handle['data'].attrs['env_args']);config=meta['env_kwargs']['ig_config']
                signature={key:config[key] for key in ('scene_id','scene_instance','robot','load_room_types')}
                signature['objects']=[(obj['name'],obj['filename'],obj['class_id']) for obj in config['objects']]
                if env is None:
                    env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=config,
                        postprocess_visual_obs=False,render=False,render_offscreen=True,
                        use_image_obs=True,image_height=480,image_width=640)
                    scene_signature=signature
                elif signature!=scene_signature:
                    raise ValueError('Different scene layout needs a separate simulation')
                parts=[part for part in descriptors(env.env.simulator.renderer) if part['cls'] in (0,2,3,5)]
                robot=env.env.robots[0].robot_ids[0]
                fingers=[link for link in range(bullet.getNumJoints(robot))
                         if bullet.getJointInfo(robot,link)[12].decode() in
                         ('gripper_link','r_gripper_finger_link','l_gripper_finger_link')]
                assert len(fingers)==3
                states=np.asarray(handle['data'][ep['demo']]['states'])
                saver=WorldSaver(exclude_body_ids=[]);offsets={};offset=0
                for body in saver.body_savers:
                    offsets[body.body]=offset;offset+=body.serialized_length
                assert offset==states.shape[1]
                candidates=[]
                for part in parts:
                    body=part['inst'].pybullet_uuid;start=offsets[body]
                    links=[-1] if part['indices'] is None else sorted(set(part['inst'].link_ids[i] for i in part['indices']))
                    if part['cls']==0:
                        values=states[:,start:start+3];threshold=.002;minimum=.015
                    else:
                        joints=list(get_movable_joints(body))
                        if links[0] not in joints:continue
                        values=states[:,start+13+joints.index(links[0])]
                        threshold=.002 if part['cls']==2 else .0005
                        minimum=.03 if part['cls']==2 else .01
                    for onset,end in motion_intervals(values,threshold,minimum):
                        probes=sorted(set(np.linspace(max(0,onset-4),min(len(states)-1,onset+8),7,dtype=int)))
                        distances=[]
                        for step in probes:
                            saver.deserialize(states[step]);saver.restore()
                            distances.append(min((point[8] for finger in fingers for link in links
                                for point in bullet.getClosestPoints(robot,body,.15,
                                    linkIndexA=finger,linkIndexB=link)),default=.15))
                        item=dict(episode_id=ep['episode_id'],instance_id=part['identity'],
                            class_id=part['cls'],onset=onset,end=end,min_surface_distance_m=float(min(distances)),
                            probe_steps=list(map(int,probes)),source='motion_and_gripper_proximity_proxy')
                        if min(distances)<=.05:candidates.append(item)
                        else:rejected.append(item)
                events.extend(candidates)
                episode_rows.append(dict(episode_id=ep['episode_id'],split=ep['split'],events=len(candidates),
                                         states_sha256=hashlib.sha256(states.tobytes()).hexdigest()))
                print(json.dumps(episode_rows[-1]),flush=True)
        records=[]
        for row in manifest['records']:
            target=target_at([event for event in events if event['episode_id']==row['episode_id']],row['step'])
            records.append(dict(id=row['id'],episode_id=row['episode_id'],step=row['step'],split=row['split'],target=target))
        args.output.write_text(json.dumps(dict(protocol='motion_proximity_target_proxy_v1',period_seconds=.05,
            lookahead_steps=40,maximum_surface_distance_m=.05,episodes=episode_rows,events=events,
            rejected_events=rejected,records=records,manifest_sha256=hashlib.sha256((args.corpus/'manifest.json').read_bytes()).hexdigest(),
            limitations='Privileged training/scoring proxy, not human intent or recorded contact forces; drawer contact cannot distinguish handle from its parent collision link'),indent=2))
    finally:
        if env is not None:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--render-gpu',type=int,default=0)
    main(p.parse_args())
