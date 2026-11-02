"""Verify copied bundle using an isolated interpreter and public-only frame pack."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
from benchmark_geometry import load_geometry_data
from export_runtime_bundle import sha
from runtime import FrameFeatures,SingleBoxSelector


CHILD='''import sys,json
from pathlib import Path
bundle=Path(sys.argv[1]);sys.path.insert(0,str(bundle/'api'))
import numpy as np,torch,runtime,models,availability_features,calibration
from runtime import FrameFeatures,SingleBoxSelector
torch.set_num_threads(2)
archive=np.load(sys.argv[2],allow_pickle=False)
selector=SingleBoxSelector(bundle/'selected.json',bundle);probability=[];slots=[]
for i in range(len(archive['step'])):
 frame=FrameFeatures(**{k:archive[k][i] for k in FrameFeatures.__dataclass_fields__ if k in archive})
 result=selector.select(frame,str(archive['episode'][i]),int(archive['step'][i]))
 probability.append(result['probability']);slots.append(result['slot'])
np.savez(sys.argv[3],probability=np.asarray(probability),slots=np.asarray(slots))
paths=[str(Path(m.__file__).resolve()) for m in [runtime,models,availability_features,calibration]]
assert all(Path(p).is_relative_to(bundle/'api') for p in paths)
print(json.dumps(dict(imported_modules=paths)))
'''


def main(a):
    torch.set_num_threads(2);manifest=json.loads((a.bundle/'manifest.json').read_text())
    assert all(sha(a.bundle/name)==digest for name,digest in manifest['files'].items())
    data=load_geometry_data(a.root);selector=SingleBoxSelector(a.selection,a.root)
    order=sorted(range(len(data['rows'])),key=lambda i:(data['rows'][i]['episode_id'],data['rows'][i]['step']))
    public={key:data[key][order] for key in FrameFeatures.__dataclass_fields__ if key in data}
    public['task']=np.array([data['rows'][i]['task'] for i in order]);public['episode']=np.array([data['rows'][i]['episode_id'] for i in order]);public['step']=np.array([data['rows'][i]['step'] for i in order])
    expected=[];slots=[]
    for j,i in enumerate(order):
        frame=FrameFeatures(**{key:value[j] for key,value in public.items() if key in FrameFeatures.__dataclass_fields__})
        result=selector.select(frame,public['episode'][j],int(public['step'][j]));expected.append(result['probability']);slots.append(result['slot'])
    with tempfile.TemporaryDirectory(prefix='momart_bundle_check_') as tmp:
        tmp=Path(tmp);relocated=tmp/'copied_bundle';shutil.copytree(a.bundle,relocated)
        np.savez(tmp/'public_frames.npz',**public)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        proc=subprocess.run([sys.executable,'-I','-c',CHILD,str(relocated),str(tmp/'public_frames.npz'),str(tmp/'actual.npz')],cwd=tmp,env=env,text=True,capture_output=True,check=True)
        actual=np.load(tmp/'actual.npz',allow_pickle=False);error=float(np.abs(actual['probability']-expected).max());disagreements=int((actual['slots']!=slots).sum())
        assert error<1e-6 and disagreements==0,(error,disagreements)
        imports=json.loads(proc.stdout.strip())
    report=dict(frames=len(order),episodes=len(set(public['episode'])),max_probability_difference=error,slot_disagreements=disagreements,
        bundle_manifest_sha256=sha(a.bundle/'manifest.json'),selection_sha256=sha(a.selection),public_frame_fields=sorted(public),
        subprocess='python -I, empty working directory, copied bundle API only',**imports,
        caveat='Temporary import paths removed after verification; benchmark labels never passed to subprocess; not raw perception or action-policy integration')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','selection','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
