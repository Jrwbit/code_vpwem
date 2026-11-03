"""Check streaming public-only API against the saved batch predictor, including unknown frames."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from benchmark_geometry import load_geometry_data
from mask_benchmark import attach_masks
from confirm import predict_component,apply_calibration
from runtime import FrameFeatures,SingleBoxSelector


def main(a):
    torch.set_num_threads(2);research=a.root/'momart_target_research_20261101'
    if a.growth:
        from growth_data import load_growth_data
        data=load_growth_data(a.root)
    else:data=load_geometry_data(a.root)
    if not a.growth and (research/'mask_old.npz').exists():data=attach_masks(data,[research/'mask_old.npz',research/'mask_new.npz'])
    selection=json.loads(a.selection.read_text());component=selection['result']['component'];calibration=selection['result']['calibration']
    batch=predict_component(component,data,a.root);expected=apply_calibration(batch,calibration,data['candidates'][:,:,133:].argmax(-1))
    runtime=SingleBoxSelector(a.selection,a.root);actual=np.zeros(batch.shape,np.float32);chosen=np.full(len(batch),8,int);elapsed=[]
    public_keys=set(FrameFeatures.__dataclass_fields__)-{'task'}
    for episode in sorted({r['episode_id'] for r in data['rows']}):
        for i in sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step']):
            row=data['rows'][i];kwargs={key:data[key][i] for key in public_keys if key in data};frame=FrameFeatures(task=row['task'],**kwargs)
            if a.growth:assert frame.extra_candidates is None and frame.extra_context is None and frame.mask_candidates is None
            started=time.perf_counter();result=runtime.select(frame,episode,row['step']);elapsed.append(time.perf_counter()-started)
            actual[i]=result['probability'];chosen[i]=result['slot']
            repeated=runtime.select(frame,episode,row['step']);np.testing.assert_array_equal(repeated['probability'],result['probability'])
    max_error=float(np.max(np.abs(actual-batch)));disagreement=int((chosen!=expected).sum())
    assert max_error<1e-5 and disagreement==0,(max_error,disagreement)
    result=dict(frames=len(batch),episodes=len({r['episode_id'] for r in data['rows']}),max_probability_error=max_error,
        accepted_slot_disagreements=disagreement,median_step_ms=float(np.median(elapsed)*1000),p95_step_ms=float(np.quantile(elapsed,.95)*1000),
        duplicate_steps='idempotent on all frames',inputs=sorted({key for key in public_keys if key in data}|{'task'}),
        dataclass_defaults_not_supplied=sorted(public_keys-set(data)),
        caveat='Selector timings exclude perception/geometry extraction; labels loaded by batch scoring harness never passed into runtime API')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','selection','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--growth',action='store_true',help='Replay all10592 enlarged-growth queries instead of the historical910')
    main(p.parse_args())
