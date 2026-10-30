"""Export auditable proxy examples and a physical-time detection timeline."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from temporal_metrics import match


def render(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((root/'manifest.json').read_text());rows=manifest['records']
    labels={a['id']:a['objects'] for a in json.loads((root/'annotations_all.json').read_text())['annotations']}
    events=json.loads((root/'operation_proxies.json').read_text())['events']
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(events),1,figsize=(12,2.1*len(events)),squeeze=False)
    for name,y,color in [('yolo',0,'#d97706'),('mask',1,'#2563eb')]:
        raw=json.loads((root/'results'/(name+'.json')).read_text())
        hits={r['id']:set(match(r['objects'],labels[r['id']]).values()) for r in raw['predictions'] if r['id'] in labels}
        for i,e in enumerate(events):
            ax=axes[i,0];seq=[r for r in rows if r['episode_id']==e['episode_id'] and abs(r['step']-e['onset'])<=16]
            for r in seq:
                visible=any(o['instance_id']==e['instance_id'] for o in labels[r['id']])
                if not visible:continue
                good=e['instance_id'] in hits[r['id']];x=(r['step']-e['onset'])*.05
                ax.scatter(x,y,c=color if good else '#d1d5db',s=25,marker='o')
                if r['step']%8 in (0,7):ax.scatter(x,y,facecolors='none',edgecolors='black',s=70,marker='s',linewidths=.8)
            ax.set_title(f"{e['episode_id'].split('/')[-1]} | {e['instance_id']} | onset={e['onset']}",fontsize=10,loc='left')
            ax.set_yticks([0,1],['YOLO640','Mask']);ax.set_ylim(-.5,1.5);ax.set_xlim(-.85,.85)
            ax.axvline(0,color='#111827',linestyle='--',lw=.8)
            ax.grid(axis='x',alpha=.2)
            if name=='yolo':ax.axvspan(-.4,.4,alpha=.08,color='blue')
    axes[-1,0].set_xlabel('Seconds relative to articulation-motion proxy onset')
    fig.suptitle('Colored = correct target; gray = visible target missed; square = policy-observed frame\nBlue band = proxy critical window; these are not human subtask labels',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(output/'temporal_timeline.png',dpi=130);plt.close(fig)
    sheet=Image.new('RGB',(1500,250*len(events)),'white');draw=ImageDraw.Draw(sheet)
    for i,e in enumerate(events):
        for j,delta in enumerate([-16,-8,0,8,16]):
            r=next(r for r in rows if r['episode_id']==e['episode_id'] and r['step']==e['onset']+delta)
            im=Image.open(root/r['file']).resize((288,216),Image.Resampling.NEAREST);pen=ImageDraw.Draw(im)
            for obj in labels[r['id']]:
                if obj['instance_id']==e['instance_id']:pen.rectangle(tuple(np.asarray(obj['box'])*2.4),outline='yellow',width=2)
            x=j*300;y=i*250;sheet.paste(im,(x,y));draw.text((x,y+217),f"{e['instance_id']} t={r['step']}",fill='black')
    sheet.save(output/'proxy_examples.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.corpus,a.output)
