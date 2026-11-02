"""Offline contact-geometry audit of ambiguous rack/bowl transitions.

Privileged scene poses are audit/label evidence only, never selector inputs.
Restores recorded states and refreshes collision detection without stepping physics.
"""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detector_comparison_20261030'))
from annotate import descriptors


def main(a):
    import egl_probe
    egl_probe.get_available_devices=lambda:[a.gpu]
    import pybullet as p
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver
    if a.output.exists():raise FileExistsError(a.output)
    env=None;records=[]
    try:
        with h5py.File(a.source,'r') as f:
            meta=json.loads(f['data'].attrs['env_args']);cfg=meta['env_kwargs']['ig_config']
            env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=cfg,postprocess_visual_obs=False,render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640)
            parts=descriptors(env.env.simulator.renderer);bowl=next(z for z in parts if z['cls']==0);rack=next(z for z in parts if z['cls']==3 and z['identity'].endswith(':rack1'))
            bowl_body=bowl['inst'].pybullet_uuid;rack_body=rack['inst'].pybullet_uuid;rack_links=sorted({rack['inst'].link_ids[j] for j in rack['indices']})
            robot=env.env.robots[0];robot_id=robot.robot_ids[0]
            hand={p.getJointInfo(robot_id,j)[12].decode():j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id,j)[12].decode() in ['gripper_link','r_gripper_finger_link','l_gripper_finger_link']}
            assert len(hand)==3
            ep=f['data'][a.demo];states=np.asarray(ep['states']);proprio=np.asarray(ep['obs/proprio']);saver=WorldSaver(exclude_body_ids=[]);assert saver.serialized_length==states.shape[1]
            for step in range(a.start,a.stop+1,a.stride):
                saver.deserialize(states[step]);saver.restore();p.performCollisionDetection()
                hand_pose=p.getLinkState(robot_id,hand['gripper_link'],computeForwardKinematics=True)[4:6];bowl_pose=p.getBasePositionAndOrientation(bowl_body)
                rack_pose=p.getLinkState(rack_body,rack_links[0],computeForwardKinematics=True)[4:6] if rack_links[0]>=0 else p.getBasePositionAndOrientation(rack_body)
                bowl_hand=p.multiplyTransforms(*p.invertTransform(*hand_pose),*bowl_pose)[0]
                bowl_rack=p.multiplyTransforms(*p.invertTransform(*rack_pose),*bowl_pose)[0]
                distances={}
                for name,link in hand.items():
                    bp=p.getClosestPoints(robot_id,bowl_body,.2,linkIndexA=link,linkIndexB=-1)
                    rp=[point for target in rack_links for point in p.getClosestPoints(robot_id,rack_body,.2,linkIndexA=link,linkIndexB=target)]
                    distances[name]=dict(bowl_distance=min([point[8] for point in bp],default=.2),rack_distance=min([point[8] for point in rp],default=.2),
                        bowl_within_2mm=sum(point[8]<=.002 for point in bp),rack_within_2mm=sum(point[8]<=.002 for point in rp))
                contacts=[dict(hand_link=point[3],other_body=point[2],other_link=point[4],distance=point[8]) for point in p.getContactPoints(bodyA=robot_id) if point[3] in hand.values()]
                rack_joints={str(j)+':'+p.getJointInfo(rack_body,j)[1].decode():dict(position=p.getJointState(rack_body,j)[0],lower=p.getJointInfo(rack_body,j)[8],upper=p.getJointInfo(rack_body,j)[9]) for j in range(p.getNumJoints(rack_body)) if p.getJointInfo(rack_body,j)[2]!=p.JOINT_FIXED}
                support=[dict(body=point[2],link=point[4],distance=point[8],normal=list(point[7])) for point in p.getContactPoints(bodyA=bowl_body) if point[2]==rack_body]
                records.append(dict(step=step,rack_joints=rack_joints,bowl_rack_contacts=support,grasped=float(proprio[step,2]),hand_distances=distances,bowl_relative_hand=list(bowl_hand),bowl_relative_rack=list(bowl_rack),contacts=contacts))
                print(json.dumps(records[-1]),flush=True)
            a.output.write_text(json.dumps(dict(source=str(a.source),demo=a.demo,robot_id=robot_id,bowl_body=bowl_body,rack_body=rack_body,rack_links=rack_links,records=records,
                caveat='Geometric collision evidence after state restoration; no physics step or force-history reconstruction; distances beyond .2m capped. Scene/object GT confined to offline audit.'),indent=2))
    finally:
        if env is not None:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--demo',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=432);p.add_argument('--stop',type=int,default=520);p.add_argument('--stride',type=int,default=4);p.add_argument('--gpu',type=int,default=4);main(p.parse_args())
