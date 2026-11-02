"""Freeze a winner across comparable dense-dev screens before fresh confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark import write_json


def main(a):
    candidates=[];sources=[]
    for screen in a.screens:
        raw=(screen/'registry.json').read_bytes();registry=json.loads(raw)
        sources.append(dict(path=str(screen),registry_sha256=hashlib.sha256(raw).hexdigest(),records=len(registry)))
        for name,value in registry.items():
            c=value['calibration'];assert c['acceptance_revision']==3
            candidates.append((screen,name,value))
    def merit(item):
        c=item[2]['calibration'];return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else c['metrics']['f1'],-item[2]['raw']['dev']['wrong_acquire'])
    screen,name,winner=max(candidates,key=merit);a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'selected.json',dict(name=name,result=winner,source_screen=str(screen),screens=sources,method_count=len(candidates),
        acceptance_revision=3,selection_partition='same dense old/dev only',criterion='V3 feasibility, then maximum correct-box recall, then fewer raw acquisition wrong-class errors',
        confirmation2_evaluated=False))
    print(json.dumps(dict(name=name,source=str(screen),dev=winner['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screens',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
