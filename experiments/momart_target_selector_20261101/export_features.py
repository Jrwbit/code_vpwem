"""Causal RGB/proprio + detector inputs only. No target/event/GT inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.perception import FrozenPerception,file_sha256
from momart_object_memory.batch_perception import perceive_batch


def main(a):
    torch.set_num_threads(4)
    corpus=json.loads((a.corpus/'manifest.json').read_text());rows=[row for row in corpus['records'] if row['step'] % a.stride == 0]
    if a.output.exists():raise FileExistsError(a.output)
    tasks=json.loads(a.task_vocabulary.read_text())['tasks'] if a.task_vocabulary else sorted({row['task'] for row in rows});files={}
    if not {row['task'] for row in rows}.issubset(tasks):raise ValueError('Unknown task')
    perceive=FrozenPerception(OmegaConf.load(a.config).object_memory,a.device)
    features=[];valid=[];contexts=[];detections=[]
    try:
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size];images=[];past=[];states=[]
            for row in batch:
                image=np.asarray(Image.open(a.corpus/row['file']).convert('RGB'))
                assert hashlib.sha256(image.tobytes()).hexdigest()==row['sha256']
                if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
                obs=files[row['source']]['data'][row['demo']]['obs'];step=row['step'];previous=max(0,step-8)
                np.testing.assert_array_equal(image,np.asarray(obs['rgb'][step]))
                images.append(image);past.append(np.asarray(obs['rgb'][previous]))
                current=np.asarray(obs['proprio'][step],dtype=np.float32)
                task=np.eye(len(tasks),dtype=np.float32)[tasks.index(row['task'])]
                states.append(np.r_[current,current-np.asarray(obs['proprio'][previous]),task])
            objects=perceive_batch(perceive,np.stack(images))
            with torch.inference_mode():
                rgb=torch.from_numpy(np.stack(images+past)).to(a.device).permute(0,3,1,2).float()/127.5-1
                global_features=perceive.visual(rgb).mean((-1,-2)).cpu().numpy()
            for index,items in enumerate(objects):
                candidates=np.zeros((8,139),np.float32);mask=np.zeros(8,bool);public=[]
                for slot,obj in enumerate(items):
                    candidates[slot]=np.r_[obj.feature,obj.box,obj.score,np.eye(6)[obj.class_id]]
                    mask[slot]=True
                    public.append(dict(class_id=obj.class_id,score=obj.score,box=(obj.box*np.array([120,90,120,90])).tolist()))
                features.append(candidates);valid.append(mask)
                contexts.append(np.r_[global_features[index],global_features[index+len(batch)],states[index]])
                detections.append(dict(id=batch[index]['id'],objects=public))
            print(json.dumps(dict(processed=min(start+a.batch_size,len(rows)),total=len(rows))),flush=True)
    finally:
        for handle in files.values():handle.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,candidates=np.stack(features),valid=np.stack(valid),context=np.stack(contexts).astype(np.float32))
    a.output.with_suffix('.json').write_text(json.dumps(dict(ids=[row['id'] for row in rows],tasks=tasks,
        detections=detections,perception=perceive.metadata,batch_size=a.batch_size,
        features='128D ROI + xyxy + score + class one-hot; current/past global visual, current/delta proprio, task',
        past_offset_steps=8,ground_truth_input=False,future_input=False,
        manifest_sha256=file_sha256(a.corpus/'manifest.json')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--task-vocabulary',type=Path)
    main(p.parse_args())
