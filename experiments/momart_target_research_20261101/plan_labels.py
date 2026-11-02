"""Label-only future command export. Never import targets into inference features."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from robot_history import frame_key


def future_commands(commands,step,horizon=8):
    if commands.shape[1]!=10 or not 0<=step<len(commands):raise ValueError('Expected aligned 10D command sequence')
    value=np.asarray(commands[step:min(len(commands),step+horizon)],np.float32)
    target=np.zeros((horizon,10),np.float32);valid=np.zeros(horizon,bool)
    target[:len(value)]=value;valid[:len(value)]=True
    return target,valid


def load_plan_labels(data,path):
    path=Path(path);m=json.loads(path.with_suffix('.json').read_text())
    assert m['protocol']=='future_commands_labels_only_v1' and m['forbidden_as_inference_input']
    lookup={key:i for i,key in enumerate(m['keys'])};arrays=np.load(path,allow_pickle=False)
    assert len(lookup)==len(m['keys'])==len(arrays['target'])
    index=[lookup[frame_key(row)] for row in data['rows']]
    return arrays['target'][index],arrays['valid'][index]


def main(a):
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    metadata=json.loads(a.observation_metadata.read_text());keys=metadata['keys'];files={};targets=[];valid=[]
    try:
        for key in keys:
            source,demo,t=json.loads(key)
            if source not in files:files[source]=h5py.File(source,'r')
            target,mask=future_commands(files[source]['data'][demo]['actions'],t)
            targets.append(target);valid.append(mask)
    finally:
        for f in files.values():f.close()
    targets=np.stack(targets);valid=np.stack(valid)
    if not np.isfinite(targets).all() or np.abs(targets[valid]).max()>1.001:raise ValueError('Unexpected command range')
    np.savez_compressed(a.output,target=targets,valid=valid)
    a.output.with_suffix('.json').write_text(json.dumps(dict(protocol='future_commands_labels_only_v1',keys=keys,
        observation_metadata_sha256=hashlib.sha256(a.observation_metadata.read_bytes()).hexdigest(),horizon=8,
        forbidden_as_inference_input=True,command_range=[float(targets[valid].min()),float(targets[valid].max())],
        meaning='Demonstration actions[t:t+8] supervision only, end padding explicitly masked; no future observations extracted.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['observation-metadata','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
