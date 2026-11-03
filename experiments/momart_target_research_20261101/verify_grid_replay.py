"""Replay saved fixed-clock selectors against GPU caches and causal CPU streaming."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from availability_features import replace_availability
from calibration import apply_calibration
from data import feature_view
from growth_data import load_growth_data
from main_history import load_history,query_context
from models import RecurrentPointer
from recurrent_selection import make_batch,sequence_probability


def stream_probability(net,data,history,c,x,mask,normalizer):
    """One chronological public observation at a time; temporary queries discarded."""
    lookup={(r['episode_id'],r['step']):i for i,r in enumerate(data['rows'])}
    assert len(lookup)==len(data['rows'])
    h=torch.tensor((history['context']-normalizer['xm'])/normalizer['xs'])
    output=np.full((len(data['rows']),9),np.nan,np.float32);episode=None;state=None
    with torch.inference_mode():
        for j,row in enumerate(history['rows']):
            if row['episode_id']!=episode:episode=row['episode_id'];state=None
            index=lookup.get((episode,row['step']));grid=row['is_grid']
            if not grid and index is None:continue
            slot=0 if index is None else index
            gap=0. if row['step']==0 else np.log(2) if grid else np.log1p((row['step']%8)/8)
            logits,next_state=net.step(c[slot:slot+1],h[j:j+1],mask[slot:slot+1],torch.tensor([gap],dtype=x.dtype),state)
            if grid:state=next_state
            if index is not None:output[index]=logits.softmax(-1)[0].numpy()
    assert np.isfinite(output).all()
    return output


def main(a):
    torch.set_num_threads(2);data=load_growth_data(a.root);records={}
    for path in sorted(a.screen.glob('*.pt')):
        if a.names and path.stem not in a.names:continue
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['spec']['kind']!='grid_gru':continue
        history=load_history(ck['spec']['main_history_path']);raw,_=feature_view(data,geometry=True);norm=ck['normalizer']
        c=torch.tensor((raw-norm['cm'])/norm['cs']);x=torch.tensor((query_context(data,history)-norm['xm'])/norm['xs']);mask=torch.tensor(data['valid'])
        net=RecurrentPointer(c.shape[-1],x.shape[-1],ck['spec']['width'],ck['spec']['reset_each_frame'],ck['spec'].get('skip_current',False)).eval();net.load_state_dict(ck['state'])
        batch=sequence_probability(net,make_batch(data,history,c,x,mask,norm));stream=stream_probability(net,data,history,c,x,mask,norm)
        cached=np.load(a.screen/(path.stem+'_bias0.0_probabilities.npz'))['probability']
        checks={}
        for label,p,q in [('gpu_cache_to_cpu_batch',cached,batch),('cpu_batch_to_causal_stream',batch,stream)]:
            calibrated_mismatch=0
            for offset in [0.,4.]:
                pp=replace_availability(p,p[:,:8].sum(-1),0.,offset);qq=replace_availability(q,q[:,:8].sum(-1),0.,offset)
                for threshold in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
                    for margin in [0.,.1,.2]:
                        cal=dict(threshold=threshold,margin=margin)
                        calibrated_mismatch+=int(np.sum(apply_calibration(pp,cal)!=apply_calibration(qq,cal)))
            checks[label]=dict(max_probability_abs=float(np.max(np.abs(p-q))),raw_slot_mismatches=int(np.sum(p.argmax(-1)!=q.argmax(-1))),calibrated_slot_mismatches=calibrated_mismatch)
        records[path.stem]=dict(frames=len(batch),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=checks)
        a.output.write_text(json.dumps(dict(complete=False,models=records),indent=2));print(path.stem,checks,flush=True)
        assert all(v['max_probability_abs']<1e-4 and v['raw_slot_mismatches']==0 and v['calibrated_slot_mismatches']==0 for v in checks.values())
    assert records
    a.output.write_text(json.dumps(dict(complete=True,models=records),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','screen','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--names',nargs='*');main(p.parse_args())
