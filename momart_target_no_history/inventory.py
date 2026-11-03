"""Audit all old policy sources and read-only reusable perception stores."""
import argparse
import json
from pathlib import Path
import sqlite3
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from momart_object_memory.cache import replay_signature
from momart_object_memory.perception import mask_perception_contract


def inventory(config,perception_root):
    expected=mask_perception_contract(config.object_memory);records=[]
    for spec in config.task_datasets:
        name=str(spec.name);source=Path(config.data_root)/(name+'.zarr')
        replay,names,mapping=training_replay(zarr.open_group(str(source),mode='r'))
        signature=replay_signature(replay,mapping,names)
        signature.update(source=str(source.resolve()),perception=expected)
        path=Path(perception_root)/(name+'.perception.sqlite')
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as con:
            old=json.loads(con.execute('SELECT signature FROM metadata WHERE id=1').fetchone()[0])
            count,minimum,maximum=con.execute('SELECT COUNT(*),MIN(frame),MAX(frame) FROM observations').fetchone()
        n=len(replay['rgb']);matches=signature==old
        if not matches or count!=n or minimum!=0 or maximum!=n-1:
            raise ValueError('Incomplete or incompatible perception: '+name+' '+str((matches,count,n,minimum,maximum)))
        records.append(dict(source=name,frames=n,episodes=len(names),perception_reusable=True))
    return dict(sources=records,total_frames=sum(r['frames'] for r in records),perception=expected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--perception-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=inventory(OmegaConf.load(a.config),a.perception_root);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='perception'},indent=2))


if __name__=='__main__':main()
