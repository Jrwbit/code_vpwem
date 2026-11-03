"""Check actual serialized snapshots against archived GPU evaluation probabilities."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import write_json
from calibration import apply_calibration
from confirm import predict_component
from growth_data import load_growth_data


def main(a):
    torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError(a.output)
    if json.loads((a.study/'status.json').read_text())['stage']!='complete':raise ValueError('Replay study incomplete')
    data=load_growth_data(a.root);registry=json.loads((a.study/'registry.json').read_text());records={}
    names=[f'epoch{epoch}_bias{offset}' for epoch in a.epochs for offset in [0.,4.]]
    if a.include_selection:names.append(json.loads((a.study/'selected.json').read_text())['name'])
    for name in dict.fromkeys(names):
        v=registry[name]
        actual=predict_component(v['component'],data,a.root);expected=np.load(a.study/(name+'_probabilities.npz'))['probability']
        delta=float(np.abs(actual-expected).max());raw=int((actual.argmax(-1)!=expected.argmax(-1)).sum())
        accepted=int((apply_calibration(actual,v['calibration'])!=apply_calibration(expected,v['calibration'])).sum())
        records[name]=dict(max_probability_difference=delta,raw_slot_differences=raw,accepted_slot_differences=accepted,frames=len(actual))
    passed=all(v['max_probability_difference']<=1e-5 and v['raw_slot_differences']==0 and v['accepted_slot_differences']==0 for v in records.values())
    write_json(a.output,dict(passed=passed,records=records,tolerance=1e-5,scope='Saved snapshot weights/normalizers on existing public-feature arrays; not raw perception, independent intent or closed-loop success'))
    print(json.dumps(dict(passed=passed,records=records),indent=2))
    if not passed:raise ValueError('Serialized checkpoint parity failed; see saved report')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','study','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--epochs',type=int,nargs='+',default=[13,37]);p.add_argument('--include-selection',action='store_true');main(p.parse_args())
