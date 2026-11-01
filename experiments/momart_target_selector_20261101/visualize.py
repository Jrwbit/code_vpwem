"""Show every labelled test failure plus class-diverse successes, never filtered metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

NAMES=['bowl','door','door handle','rack','drawer','drawer handle']


def main(a):
    result=json.loads(a.result.read_text());manifest=json.loads((a.corpus/'manifest.json').read_text())
    source={row['id']:row for row in manifest['records']}
    truth={row['id']:row['objects'] for row in json.loads((a.corpus/'annotations_all.json').read_text())['annotations']}
    test=[row for row in result['predictions'] if row['split']=='test' and row['labelled']]
    failures=[row for row in test if not row['correct']]
    selected=list(failures)
    for cls in range(6):
        candidates=[row for row in test if row['correct'] and row['target']['class_id']==cls]
        selected.extend(candidates[:2])
    width=640;height=520
    sheet=Image.new('RGB',(width*2,height*((len(selected)+1)//2)),'#161b22')
    review=[]
    for i,row in enumerate(selected):
        original=source[row['id']]
        frame=Image.new('RGB',(width,height),'#161b22')
        frame.paste(Image.open(a.corpus/original['file']).convert('RGB').resize((640,480),Image.Resampling.NEAREST),(0,0))
        draw=ImageDraw.Draw(frame)
        gt=next((o for o in truth[row['id']] if o['instance_id']==row['target']['instance_id']),None)
        def box(coords,color):draw.rectangle((np.array(coords)*[640/120,480/90,640/120,480/90]).tolist(),outline=color,width=3)
        if gt:box(gt['box'],'yellow')
        if row['selected_object']:box(row['selected_object']['box'],'cyan')
        draw.text((5,482),f"{original['task']} / {original['demo']} / t={row['step']}",fill='white')
        actual=NAMES[row['selected_object']['class_id']] if row['selected_object'] else 'NONE'
        draw.text((5,500),f"proxy={NAMES[row['target']['class_id']]} selected={actual} {'MATCH' if row['correct'] else 'FAIL'}",fill='white')
        sheet.paste(frame,((i%2)*width,(i//2)*height))
        review.append(dict(id=row['id'],correct=row['correct'],proxy_class=row['target']['class_id'],selected=row['selected_object']))
    a.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(dict(selection='all labelled test failures then first two correct per target class',examples=review),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
