"""Verify the frozen unified bundle against original weights on public features."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
RESEARCH=Path(__file__).resolve().parents[1]/'experiments/momart_target_research_20261101'
sys.path.insert(0,str(RESEARCH))
from runtime import FrameFeatures,SingleBoxSelector,view
from models import Pointer
from calibration import apply_calibration
from momart_object_memory.perception import file_sha256
from .train_unified import FAMILIES,family


def main(a):
    torch.set_num_threads(1)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if file_sha256(a.bundle/name)!=sha:raise ValueError('Frozen artifact hash differs: '+name)
    rt=SingleBoxSelector(a.bundle/'selected.json',a.bundle)
    checkpoint=torch.load(a.training/'unified_five_task_seed100.pt',map_location='cpu',weights_only=False)
    n=checkpoint['normalizer'];net=Pointer(155,450,width=64,kind='mlp').eval();net.load_state_dict(checkpoint['state'])
    features=np.load(a.research/'sink_extension_20261103_features.npz');geometry=np.load(a.research/'geometry_sink_extension_20261103.npz')
    rows=json.loads((a.research/'sink_extension_20261103_corpus/manifest.json').read_text())['records']
    contexts=np.c_[features['context'][:,:276],np.tile(np.eye(5)[4],(len(rows),1))].astype(np.float32)
    differences=[];slot_mismatch=0
    for i,row in enumerate(rows):
        frame=FrameFeatures(features['candidates'][i],contexts[i],features['valid'][i],FAMILIES[4],geometry_candidates=geometry['candidates'][i],geometry_context=geometry['context'][i])
        c,x=view(frame,'geometry')
        with torch.inference_mode():
            probability=net(torch.tensor(((c-n['cm'])/n['cs'])[None]),torch.tensor(((x-n['xm'])/n['xs'])[None]),torch.tensor(frame.valid[None]))[0].softmax(-1)[0].numpy()
        actual=rt.select(frame,row['episode_id'],row['step']);expected=apply_calibration(probability[None],rt.calibration)[0]
        differences.append(float(abs(probability-actual['probability']).max()));slot_mismatch+=actual['slot']!=expected
    result=dict(queries=len(rows),slot_disagreements=int(slot_mismatch),probability_max_abs=max(differences),
        model_weight_files=len(list((a.bundle/'weights').glob('*.pt'))),tasks=rt.tasks,bundle_manifest_sha256=file_sha256(a.bundle/'manifest.json'))
    if slot_mismatch or max(differences)>1e-5 or result['model_weight_files']!=1:raise ValueError(str(result))
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['bundle','training','research','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
