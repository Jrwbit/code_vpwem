"""Audit the own-robot torso schema on each source before cache replay."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from omegaconf import OmegaConf
from .prepare_cache import dataset,RobotGeometry,save_json


def main(a):
    cfg=OmegaConf.load(a.config);records=[]
    for spec in cfg.task_datasets:
        name=str(spec.name);d=dataset(cfg,name);demo=next(iter(d._demo_id_to_start_indices))
        with h5py.File(a.hdf_root/(name+'.hdf5'),'r') as f:
            config=json.loads(f['data'].attrs['env_args'])['env_kwargs']['ig_config'];ep=f['data'][demo]
            geometry=RobotGeometry(a.urdf,float(config.get('robot_scale',1.)))
            try:
                value=geometry.infer_robot_layout(np.asarray(ep['states']),np.asarray(ep['obs/proprio']))
                save_json(a.cache_root/(name+'.robot_layout.json'),value)
                records.append(dict(source=name,demo=demo,layout=value))
            finally:geometry.close()
    save_json(a.cache_root/'robot_schema_audit.json',dict(sources=records));print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','hdf-root','cache-root','urdf']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
