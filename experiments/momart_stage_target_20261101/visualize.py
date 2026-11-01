"""Display improvements, added errors and coverage losses in the new evaluation set."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from phase_logic import ROLES

NAMES=['bowl','door','door handle','rack','drawer','drawer handle']


def main(a):
    d=json.loads(a.result.read_text());m=json.loads((a.corpus/'manifest.json').read_text())
    source={r['id']:r for r in m['records']}
    gt={r['id']:r['objects'] for r in json.loads((a.corpus/'annotations_all.json').read_text())['annotations']}
    rows=[r for r in d['predictions'] if r['tag']=='new' and r['label'] is not None]
    def cls(row,method):
        slot=row['predictions'][method]
        return -1 if slot==8 else row['objects'][slot]['class_id']
    groups=[('old wrong role',lambda r:r['label']['role']==0 and cls(r,'previous') not in (0,-1)),
            ('hard gate wrong role',lambda r:r['label']['role']==0 and cls(r,'stage') not in (0,-1)),
            ('new abstention',lambda r:r['label']['role']==0 and cls(r,'previous')==0 and cls(r,'flat')==-1),
            ('recipient unavailable',lambda r:r['label']['role']==4 and cls(r,'previous')!=-1 and cls(r,'flat')==-1)]
    selected=[];used=set()
    for title,condition in groups:
        candidates=[r for r in rows if condition(r) and r['id'] not in used][:2]
        for row in candidates:selected.append((title,row));used.add(row['id'])
    canvas=Image.new('RGB',(960,320*len(selected)),'#161b22');draw=ImageDraw.Draw(canvas)
    methods=['previous','flat','stage']
    for i,(title,row) in enumerate(selected):
        original=source[row['id']];image=Image.open(a.corpus/original['file']).convert('RGB').resize((320,240),Image.Resampling.NEAREST)
        truth=next((o for o in gt[row['id']] if row['label']['target'] and o['instance_id']==row['label']['target']['instance_id']),None)
        for j,method in enumerate(methods):
            x=j*320;y=i*320;canvas.paste(image,(x,y))
            def rectangle(box,color):draw.rectangle((np.asarray(box)*[320/120,240/90,320/120,240/90]+[x,y,x,y]).tolist(),outline=color,width=2)
            if truth:rectangle(truth['box'],'yellow')
            slot=row['predictions'][method]
            if slot<8:rectangle(row['objects'][slot]['box'],'cyan')
            target='NONE' if slot==8 else NAMES[row['objects'][slot]['class_id']]
            draw.text((x+4,y+244),f'{method}: {target}',fill='cyan')
            draw.text((x+4,y+260),'proxy: '+ROLES[row['label']['role']],fill='yellow')
            draw.text((x+4,y+276),title,fill='white')
            draw.text((x+4,y+292),f"{original['demo']} step {row['step']}",fill='white')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(dict(examples=[dict(category=t,id=r['id']) for t,r in selected],
        selection='first two per fixed error/coverage category; labels are proxies'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
