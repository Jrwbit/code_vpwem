"""Verify a complete serialized component tree against its development cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from calibration import apply_calibration
from confirm import predict_component
from growth_data import load_growth_data


def main(a):
    torch.set_num_threads(2);selected=json.loads((a.screen/'selected.json').read_text());data=load_growth_data(a.root)
    p=predict_component(selected['result']['component'],data,a.root)
    q=np.load(a.screen/(selected['name']+'_probabilities.npz'))['probability'];cal=selected['result']['calibration']
    result=dict(frames=len(p),max_probability_abs=float(np.max(np.abs(p-q))),raw_slot_mismatches=int(np.sum(p.argmax(-1)!=q.argmax(-1))),
        accepted_slot_mismatches=int(np.sum(apply_calibration(p,cal)!=apply_calibration(q,cal))),selection_sha256=hashlib.sha256((a.screen/'selected.json').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    assert result['max_probability_abs']<1e-5 and result['raw_slot_mismatches']==0 and result['accepted_slot_mismatches']==0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','screen','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
