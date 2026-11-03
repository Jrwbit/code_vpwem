"""Inspect collision coverage and visual ambiguity in one recorded source scene."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
from audit_contact_priority import collision_groups,descriptors


def shape_summary(p,body,link):
    shapes=p.getCollisionShapeData(body,link)
    return dict(count=len(shapes),types=[int(x[2]) for x in shapes],
        assets=[Path(x[4].decode() if isinstance(x[4],bytes) else x[4]).name for x in shapes])


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    with h5py.File(a.source,'r') as f:meta=json.loads(f['data'].attrs['env_args'])
    import egl_probe
    egl_probe.get_available_devices=lambda:[a.gpu]
    import pybullet as p
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=meta['env_kwargs']['ig_config'],postprocess_visual_obs=False,
        render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640)
    try:
        groups=collision_groups(descriptors(env.env.simulator.renderer));robot=env.env.robots[0].robot_ids[0]
        for group in groups:group['collision_shapes']=shape_summary(p,group['body'],group['link'])
        hand={p.getJointInfo(robot,j)[12].decode():dict(link=j,**shape_summary(p,robot,j)) for j in range(p.getNumJoints(robot))
            if p.getJointInfo(robot,j)[12].decode() in ['gripper_link','r_gripper_finger_link','l_gripper_finger_link']}
        assert len(hand)==3
        value=dict(source=str(a.source.resolve()),config_sha256=hashlib.sha256(json.dumps(meta['env_kwargs']['ig_config'],sort_keys=True).encode()).hexdigest(),
            groups=groups,hand=hand,caveat='Shape availability does not prove collision surface matches the visual handle; shared link cannot resolve an instance or human intent.')
        a.output.write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))
    finally:env.env.simulator.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1);main(p.parse_args())
