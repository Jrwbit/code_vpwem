"""Export complete held-out episodes; derive scoring-only articulation proxies."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def export(base, output):
    if output.exists(): raise FileExistsError(output)
    source=json.loads((base/'manifest.json').read_text())
    episodes=[e for e in source['episodes'] if e['split']=='test']
    (output/'images').mkdir(parents=True)
    records=[]
    for i,e in enumerate(episodes):
        with h5py.File(e['source'],'r') as f:
            for step,im in enumerate(f['data'][e['demo']]['obs/rgb']):
                identity=f'temporal_{i}_{step}';filename='images/'+identity+'.png'
                Image.fromarray(im).save(output/filename)
                records.append(dict(id=identity,episode_id=e['episode_id'],source=e['source'],
                    task=e['task'],demo=e['demo'],step=step,split='test',file=filename,
                    anchor=True,clip=True,sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    source.update(records=records,episodes=episodes,protocol='complete_heldout_temporal_v1')
    (output/'manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps(dict(frames=len(records),episodes=len(episodes))),flush=True)


def joint_events(parts, states, episode_id):
    """Offline GT proxy, never an input to detector/bank/policy."""
    from gibson2.external.pybullet_tools.utils import WorldSaver,get_movable_joints
    saver=WorldSaver(exclude_body_ids=[]);offsets={};offset=0
    for body in saver.body_savers:
        offsets[body.body]=offset;offset+=body.serialized_length
    assert states.shape[1]==offset
    events=[]
    for part in parts:
        if part['cls'] not in (2,5):continue
        body=part['inst'].pybullet_uuid
        joint=part['inst'].link_ids[part['indices'][0]]
        movable=list(get_movable_joints(body));column=offsets[body]+13+movable.index(joint)
        q=states[:,column];threshold=.0005 if part['cls']==5 else .002
        changes=np.flatnonzero(np.abs(np.diff(q))>=threshold)+1
        groups=[]
        for t in changes:
            if not groups or t-groups[-1][-1]>4:groups.append([int(t)])
            else:groups[-1].append(int(t))
        for group in groups:
            start,end=group[0],group[-1]
            if np.ptp(q[max(0,start-1):end+1]) < (.01 if part['cls']==5 else .03):continue
            events.append(dict(episode_id=episode_id,instance_id=part['identity'],class_id=part['cls'],
                onset=start,end=end,joint=joint,unit='m' if part['cls']==5 else 'rad',
                net_change=float(q[end]-q[max(0,start-1)]),source='articulation_motion_proxy'))
    return events


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.base,a.output)
