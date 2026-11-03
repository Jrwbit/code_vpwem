"""Offline main-camera timeline; future panels never enter model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from subject_labels import subject_label


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    research=a.research;manifest=json.loads((research/(a.prefix+'_corpus')/'manifest.json').read_text())
    center=next(r for r in manifest['records'] if r['id']==a.case_id);phase={x['id']:x for x in json.loads((research/(a.prefix+'_phases.json')).read_text())['records']}
    primary=np.load(a.evaluation/'primary_predictions.npz');reference=np.load(a.evaluation/'retention_predictions.npz');ids=primary['ids'].tolist()
    assert ids==reference['ids'].tolist()==[r['id'] for r in manifest['records']]
    lookup={r['step']:r for r in manifest['records'] if r['episode_id']==center['episode_id']};position={key:i for i,key in enumerate(ids)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,11));records=[]
    with h5py.File(center['source'],'r') as f:
        obs=f['data'][center['demo']]['obs'];length=len(obs['rgb'])
        for ax,delta in zip(axes.flat,range(-32,33,8)):
            step=center['step']+delta
            if not 0<=step<length:ax.axis('off');continue
            rgb=np.asarray(obs['rgb'][step]);row=lookup.get(step);ax.imshow(rgb,interpolation='nearest');ax.axis('off')
            texts=[f'step {step} | delta {delta:+d}'+(' | QUERY' if delta==0 else '')];item=dict(step=step,relative_step=delta,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),causal_at_center=delta<=0)
            if row is not None:
                assert row['sha256']==item['rgb_sha256'];saved=np.asarray(Image.open(research/(a.prefix+'_corpus')/row['file']).convert('RGB'));np.testing.assert_array_equal(saved,rgb)
                i=position[row['id']];label=subject_label(phase[row['id']]);target=label['target']['class_id'] if label and label['target'] else None
                texts.append(f'Weak subject class: {target}; grasped: {float(obs["proprio"][step,2]):.0f}')
                for name,z in [('old',reference),('new',primary)]:
                    slot=int(z['prediction'][i]);p=z['probability'][i];texts.append(f'{name}: '+('reject' if slot==8 else f'slot {slot}')+f', top p={p.max():.3f}')
                item['query_id']=row['id']
            else:texts.append('No aligned evaluation query; raw public main RGB only')
            ax.set_title('\n'.join(texts),fontsize=9);records.append(item)
    fig.suptitle(a.case_id+' | OFFLINE MAIN-VIEW REVIEW\nFuture panels are review material only; weak labels are not human intention truth.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=120);plt.close(fig)
    meta=dict(case=a.case_id,episode=center['episode_id'],panels=records,views=['main'],future_review_only=True,no_labels_changed=True,
        manifest_sha256=hashlib.sha256((research/(a.prefix+'_corpus')/'manifest.json').read_bytes()).hexdigest(),figure_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(dict(case=a.case_id,panels=len(records),figure_sha256=meta['figure_sha256'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['research','evaluation','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--case-id',required=True);main(p.parse_args())
