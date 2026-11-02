"""Temporal raw-image review of archived acquisition wrong-class predictions.

Future frames are used ONLY by this offline reviewer, never by selector inference.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data import load_group


def main(a):
    r=a.root/'momart_target_research_20261101';corpus=r/(a.prefix+'_corpus')
    data=load_group(corpus,r/(a.prefix+'_features.npz'),r/(a.prefix+'_phases.json'),'review')
    archived=np.load(a.predictions,allow_pickle=False);assert list(archived['ids'])==[row['id'] for row in data['rows']]
    pred=archived['prediction'];indices=[i for i,row in enumerate(data['rows']) if data['roles'][i]==0 and pred[i]<8 and row['objects'][pred[i]]['class_id']!=0]
    if not indices:raise ValueError('No acquisition wrong-class errors in archive')
    phases={row['id']:row for row in json.loads((r/(a.prefix+'_phases.json')).read_text())['records']}
    refs={row['id']:row['objects'] for row in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    fig,axes=plt.subplots(len(indices),5,figsize=(16,len(indices)*2.1),squeeze=False);files={};records=[];names=['bowl','door','door handle','rack','drawer','drawer handle']
    try:
        for ri,i in enumerate(indices):
            row=data['rows'][i];t=row['step'];slot=int(pred[i]);chosen=row['objects'][slot];phase=phases[row['id']]
            records.append(dict(id=row['id'],episode=row['episode_id'],step=t,predicted_class=chosen['class_id'],phase=phase))
            if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
            ep=files[row['source']]['data'][row['demo']];obs=ep['obs']
            for ci,offset in enumerate([-16,-8,0,8,16]):
                step=max(0,min(len(obs['rgb'])-1,t+offset));image=np.concatenate([np.asarray(obs['rgb'][step]),np.asarray(obs['rgb_wrist'][step])],axis=1)
                ax=axes[ri,ci];ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f"step {step} | grasp={float(obs['proprio'][step,2]):.1f}",fontsize=8)
                if offset==0:
                    x1,y1,x2,y2=chosen['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=1.5))
                    for obj in refs[row['id']]:
                        if obj['class_id']==0:
                            x1,y1,x2,y2=obj['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=1.))
                if ci==0:ax.set_ylabel(f"{row['task'].replace('_expert','').replace('_suboptimal','')}\n{row['demo']}@{t}\npred {names[chosen['class_id']]}",fontsize=8)
    finally:
        for f in files.values():f.close()
    fig.suptitle('Archived acquisition errors: red = selected box, green = visible CAD bowl; main / wrist\nTime context is for offline review only; proxy labels are not human intention GT',fontsize=11)
    fig.tight_layout(rect=[.02,0,1,.96]);fig.savefig(a.output,dpi=150);plt.close(fig)
    a.output.with_suffix('.json').write_text(json.dumps(records,indent=2));print(json.dumps(dict(errors=len(indices),output=str(a.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','predictions','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--prefix',default='confirm2');main(p.parse_args())
