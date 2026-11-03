"""Main-camera scientific evidence plot for an explicitly selected paired case."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    comparison=json.loads(a.comparison.read_text());prefix=comparison['prefix']
    case=next(x for x in comparison['examples'] if x['id']==a.case_id)
    corpus=a.research/(prefix+'_corpus');manifest=json.loads((corpus/'manifest.json').read_text())
    row=next(x for x in manifest['records'] if x['id']==a.case_id)
    rgb=np.asarray(Image.open(corpus/row['file']).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['sha256']
    features=json.loads((a.research/(prefix+'_features.json')).read_text())
    objects=next(x['objects'] for x in features['detections'] if x['id']==a.case_id)
    frozen=json.loads((a.evaluation/'frozen_evaluation_manifest.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax in axes:ax.imshow(rgb,interpolation='nearest');ax.axis('off')
    axes[0].set_title('Recorded main view; no side/wrist view')
    labels=[]
    for slot,obj in enumerate(objects):
        if obj['class_id']!=case['target']['class_id'] and slot!=case['primary']['slot']:continue
        color='#00c46b' if obj['class_id']==case['target']['class_id'] else '#ff9d21'
        x1,y1,x2,y2=obj['box'];axes[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,lw=2))
        labels.append(f"slot {slot}, detector class {obj['class_id']}: old {case['reference']['probability'][slot]:.3f}, new {case['primary']['probability'][slot]:.3f}")
    axes[1].set_title('Detector proposals; green=proxy target class')
    before=frozen['retention']['calibration'];after=frozen['primary']['calibration']
    actions={key:('reject' if case[key]['slot']==8 else f"class {case[key]['class_id']} accepted") for key in ['reference','primary']}
    note=f"Old threshold/gap={before['threshold']}/{before['margin']}: {actions['reference']}; new={after['threshold']}/{after['margin']}: {actions['primary']}"
    fig.suptitle(a.case_id+' | '+case['episode_id']+' | step '+str(case['step']),fontsize=10)
    fig.text(.5,.11,'\n'.join(labels)+'\n'+note+'\nExamined weak-proxy case only; intention not independently verified.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.27,1,.94));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=120);plt.close(fig)
    print(json.dumps(dict(case=a.case_id,views=['main'],source_rgb_sha256=row['sha256'],figure_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['research','comparison','evaluation','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--case-id',required=True);main(p.parse_args())
