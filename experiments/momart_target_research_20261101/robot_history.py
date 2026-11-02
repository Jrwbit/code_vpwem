"""Causal measured finger positions and previously executed command features.

Only the two robot-owned finger columns are read from state archives. No object
pose, contact, future command, or label enters these public feature functions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def frame_key(row):
    return json.dumps([str(Path(row['source']).resolve()),row['demo'],int(row['step'])],separators=(',',':'))


def finger_features(current,past,elapsed_steps):
    current=np.asarray(current,dtype=np.float32);past=np.asarray(past,dtype=np.float32)
    if current.shape!=(2,) or past.shape!=(2,) or elapsed_steps<0:raise ValueError('Expected two measured finger positions and nonnegative elapsed steps')
    # Position change, NOT measured joint velocity. No division by assumed dt.
    return np.r_[current,current-past,np.mean(np.abs(current)),elapsed_steps/8.].astype(np.float32)


def command_features(executed_commands):
    commands=np.asarray(executed_commands,dtype=np.float32)
    if commands.ndim!=2 or commands.shape[1]!=10 or len(commands)>8:raise ValueError('Expected up to eight already executed 10D commands')
    if not len(commands):return np.zeros(31,np.float32)
    return np.r_[commands[-1],commands.mean(0),commands.std(0),len(commands)/8.].astype(np.float32)


def history_at(finger_positions,commands,step):
    if not 0<=step<len(commands) or len(finger_positions)!=len(commands):raise ValueError('History step/sequence length mismatch')
    past=max(0,step-8)
    return (finger_features(finger_positions[step],finger_positions[past],step-past),
            command_features(commands[past:step]))


def attach_history(data,path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['protocol']=='causal_robot_history_v1' and meta['actions_strictly_before_observation']
    values=np.load(path,allow_pickle=False);keys=meta['keys'];lookup={key:i for i,key in enumerate(keys)}
    assert len(lookup)==len(keys)==len(values['gripper'])==len(values['actions'])
    indices=[lookup[frame_key(row)] for row in data['rows']]
    data['history_gripper']=values['gripper'][indices];data['history_actions']=values['actions'][indices]
    return data


def history_view(data,view,signals):
    from data import feature_view
    c,x=feature_view(data,rich=view=='rich_geometry',geometry=True)
    names={'gripper':['history_gripper'],'actions':['history_actions'],'both':['history_gripper','history_actions']}
    return c,np.concatenate([x]+[data[key] for key in names[signals]],-1).astype(np.float32)


def main(a):
    from robot_geometry import RobotGeometry
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    layouts={};sources={}
    for path in a.geometry_metadata:
        for source,layout in json.loads(path.read_text())['layouts'].items():
            if source in layouts:assert layouts[source]['offset']==layout['offset']
            assert layout['eef_median']<.03 and layout['eef_p95']<.10
            layouts[source]=layout
    rows={}
    for corpus in a.corpora:
        manifest=json.loads((corpus/'manifest.json').read_text())
        for row in manifest['records']:rows.setdefault(frame_key(row),row)
        sources[str(corpus/'manifest.json')]=hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()
    groups={}
    for key,row in rows.items():groups.setdefault((row['source'],row['demo']),[]).append((key,row['step']))
    geometry=RobotGeometry(a.urdf);joints=['r_gripper_finger_joint','l_gripper_finger_joint'];columns={}
    finger={};commands={};head_errors={}
    try:
        for (source,demo),entries in sorted(groups.items()):
            offset=layouts[source]['offset'];columns[source]=[offset+13+geometry.movable.index(geometry.joints[name]) for name in joints]
            with h5py.File(source,'r') as f:
                ep=f['data'][demo];n=len(ep['actions']);assert ep['actions'].shape[1]==10
                # h5py requires increasing column indices; reorder into joint-name order.
                cols=columns[source];ordered=sorted(cols);q=np.asarray(ep['states'][:,ordered])[:,[ordered.index(c) for c in cols]]
                hcols=[offset+13+geometry.movable.index(geometry.joints[name]) for name in ['head_pan_joint','head_tilt_joint']]
                head=np.stack([np.asarray(ep['states'][:,col]) for col in hcols],-1)
                error=float(np.sqrt(np.mean((head-np.asarray(ep['obs/proprio'][:,:2]))**2)))
                if error>.02:raise ValueError('Robot state layout head mismatch: '+str((source,demo,error)))
                head_errors[source+'/'+demo]=error
                for key,t in entries:
                    if not 0<=t<n:raise ValueError('Observation step outside command sequence')
                    # Observation[t] is before command[t]; never include that command.
                    finger[key],commands[key]=history_at(q,ep['actions'],t)
            print(json.dumps(dict(source=Path(source).name,demo=demo,frames=len(entries))),flush=True)
    finally:geometry.close()
    keys=list(rows);a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,gripper=np.stack([finger[k] for k in keys]),actions=np.stack([commands[k] for k in keys]))
    a.output.with_suffix('.json').write_text(json.dumps(dict(protocol='causal_robot_history_v1',keys=keys,manifest_sha256=sources,
        actions_strictly_before_observation=True,observation_history_steps=8,command_window='actions[max(0,t-8):t]',
        finger_columns=columns,finger_joint_names=joints,head_layout_rmse=head_errors,
        gripper_features='current2, position_change2, mean_abs_width1, elapsed_steps/8; no measured velocities claimed',
        action_features='last10, mean10, std10, observed_count/8',
        inference_requirement='Robot measured two finger positions at t and t-8; last eight actually executed policy commands. Never demonstration future/current commands or simulator object states.',
        caveat='Historical demonstration commands may have a different distribution from deployed policy commands; task improvement requires closed-loop testing.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpora',type=Path,nargs='+',required=True);p.add_argument('--geometry-metadata',type=Path,nargs='+',required=True)
    p.add_argument('--urdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
