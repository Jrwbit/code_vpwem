"""Resumable frozen-perception cache, causal replay, then the authorized train job."""
import argparse
import gc
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from .cache import replay_signature, write_cache
from .perception import FrozenPerception, validate_perception_metadata
from .state import ObjectObservation


def encode(observations, feature_dim):
    buffer=io.BytesIO()
    np.savez(buffer, classes=np.array([o.class_id for o in observations],np.int64),
             boxes=np.array([o.box for o in observations],np.float32).reshape(-1,4),
             scores=np.array([o.score for o in observations],np.float32),
             features=np.array([o.feature for o in observations],np.float32).reshape(-1,feature_dim))
    return buffer.getvalue()


def decode(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as a:
        return [ObjectObservation(int(c),b.copy(),float(s),f.copy())
                for c,b,s,f in zip(a['classes'],a['boxes'],a['scores'],a['features'])]


def save_status(path, **value):
    value.update(updated_unix=time.time(), supervisor_pid=os.getpid())
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(value),flush=True)


def prepare_dataset(cfg,name,perceive,status_path):
    validate_perception_metadata(cfg.object_memory,perceive.metadata)
    source=Path(cfg.data_root)/(name+'.zarr')
    replay,names,mapping=training_replay(zarr.open_group(str(source),mode='r'))
    signature=replay_signature(replay,mapping,names)
    signature.update(source=str(source.resolve()),perception=perceive.metadata)
    root=Path(cfg.object_memory.cache_root);root.mkdir(parents=True,exist_ok=True)
    database=root/(name+'.perception.sqlite')
    con=sqlite3.connect(str(database))
    try:
        con.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, signature TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS observations (frame INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        old=con.execute('SELECT signature FROM metadata WHERE id=1').fetchone()
        if old is not None and json.loads(old[0])!=signature:
            raise ValueError('Perception cache provenance differs: '+name)
        if old is None:
            con.execute('INSERT INTO metadata VALUES (1,?)',(json.dumps(signature,sort_keys=True),));con.commit()
        count,maximum=con.execute('SELECT COUNT(*),MAX(frame) FROM observations').fetchone()
        if count != (maximum+1 if maximum is not None else 0):
            raise ValueError('Perception cache has missing frame indices')
        n=len(replay['rgb']);started=time.monotonic();initial=count
        save_status(status_path,stage='perception',dataset=name,completed_frames=count,total_frames=n)
        for frame in range(count,n):
            objects=perceive(np.asarray(replay['rgb'][frame]))
            con.execute('INSERT INTO observations VALUES (?,?)',
                        (frame,encode(objects,int(cfg.object_memory.feature_dim))))
            if (frame+1)%64==0 or frame+1==n:
                con.commit()
                rate=(frame+1-initial)/max(time.monotonic()-started,1e-6)
                save_status(status_path,stage='perception',dataset=name,
                            completed_frames=frame+1,total_frames=n,frames_per_second=rate,
                            dataset_remaining_seconds=(n-frame-1)/rate)
        def read(frame):
            row=con.execute('SELECT payload FROM observations WHERE frame=?',(int(frame),)).fetchone()
            if row is None: raise ValueError('Incomplete perception frame')
            return decode(row[0])
        save_status(status_path,stage='causal_replay',dataset=name,completed_frames=n,total_frames=n)
        write_cache(cfg,source,root/(name+'.objects.zarr'),perceive,observation_reader=read,resume=True)
    finally:
        con.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true',help='Validate and build caches without starting policy training')
    args=parser.parse_args();cfg=OmegaConf.load(args.config)
    work=Path(cfg.work_dir);work.mkdir(parents=True,exist_ok=True)
    status=work/'pipeline_status.json'
    try:
        torch.set_num_threads(4)
        perceive=FrozenPerception(cfg.object_memory,args.device)
        for spec in cfg.task_datasets:
            prepare_dataset(cfg,str(spec.name),perceive,status)
        if args.prepare_only:
            save_status(status,stage='cache_complete',policy_training_started=False)
            return
        del perceive;gc.collect();torch.cuda.empty_cache()
        command=[sys.executable,'-m','momart_object_memory.train','--config',args.config,
                 '--mode',str(cfg.object_memory.mode),'--device',args.device]
        child=subprocess.Popen(command)
        save_status(status,stage='training',training_pid=child.pid,command=command,
                    gradient_steps=int(cfg.gradient_steps),physical_batch_size=int(cfg.physical_batch_size))
        result=child.wait()
        if result:raise RuntimeError('Training exited with code '+str(result))
        save_status(status,stage='complete',gradient_steps=int(cfg.gradient_steps))
    except Exception as error:
        save_status(status,stage='failed',error=str(error))
        raise


if __name__=='__main__':main()
