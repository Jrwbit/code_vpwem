"""Main-camera training-crop evidence, including a detector-missing example."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from sensor_crop import apply_crop


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    plan=json.loads(a.archive.with_suffix('.plan.json').read_text());metadata=json.loads(a.archive.with_suffix('.json').read_text())
    lookup={x['base_index']:(x,y) for x,y in zip(plan['rows'],metadata['records'])};examples=[lookup[i] for i in a.base_indices]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(len(examples),2,figsize=(9,len(examples)*4),squeeze=False);evidence=[]
    for j,(row,record) in enumerate(examples):
        with h5py.File(row['source'],'r') as f:rgb=np.asarray(f['data'][row['demo']]['obs/rgb'][row['step']])
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256'];cropped=apply_crop(rgb,row['crop'])
        assert hashlib.sha256(cropped.tobytes()).hexdigest()==record['cropped_rgb_sha256']
        for ax,img,box in zip(axes[j],[rgb,cropped],[row['original_target_box'],row['cropped_target_box']]):
            ax.imshow(img,interpolation='nearest');ax.axis('off')
            if box is not None:
                x1,y1,x2,y2=box;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=1.5,color='#00e080'))
        for slot in record['positive_slots']:
            b=record['public_objects'][slot]['box'];axes[j,1].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,lw=1.5,color='#ff9d21'))
        axes[j,0].set_title(row['id']+' | original main RGB\nGreen: offline target visible-mask box',fontsize=10)
        axes[j,1].set_title('Crop '+str(row['crop'])+' | matching slots '+str(record['positive_slots'])+'\nOrange: re-detected matching proposal',fontsize=10)
        evidence.append(dict(base_index=row['base_index'],id=row['id'],crop=row['crop'],positive_slots=record['positive_slots'],original_rgb_sha256=row['source_rgb_sha256'],cropped_rgb_sha256=record['cropped_rgb_sha256']))
    fig.suptitle('TRAINING-ONLY MAIN-CAMERA CROP REVIEW\nReference masks choose crop/labels only; detector and model features are public.',fontsize=11,y=.95,va='top')
    fig.subplots_adjust(left=.03,right=.97,bottom=.03,top=.80,hspace=.35,wspace=.08);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=120);plt.close(fig)
    result=dict(examples=evidence,views=['main'],figure_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),caveat='Selected pipeline illustrations, not a representative success-rate estimate')
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(figure_sha256=result['figure_sha256'],examples=len(examples))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-indices',type=int,nargs='+',required=True);main(p.parse_args())
