"""Prepare reviewed pilot references for detector training and visual audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import yaml

COLORS=['red','cyan','orange','lime','magenta','yellow']


def prepare(corpus):
    manifest=json.loads((corpus/'manifest.json').read_text())
    labels=json.loads((corpus/'annotations_all.json').read_text())
    lookup={r['id']:r for r in labels['annotations']}
    assert set(lookup)=={r['id'] for r in manifest['records']}
    sets={s:{r['episode_id'] for r in manifest['records'] if r['split']==s}
          for s in ['train','dev','test']}
    assert not sets['train'] & sets['dev'] and not sets['train'] & sets['test'] and not sets['dev'] & sets['test']
    folder=corpus/'labels';folder.mkdir(exist_ok=True)
    for r in manifest['records']:
        lines=[]
        for o in lookup[r['id']]['objects']:
            x1,y1,x2,y2=o['box']
            assert 0<=x1<x2<=120 and 0<=y1<y2<=90 and o['pixels']>0
            lines.append('%s %.9f %.9f %.9f %.9f'%(o['class_id'],(x1+x2)/240,(y1+y2)/180,(x2-x1)/120,(y2-y1)/90))
        (folder/(r['id']+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
    for split in ['train','dev','test']:
        (corpus/(split+'.txt')).write_text(''.join(str((corpus/r['file']).resolve())+'\n'
                                                 for r in manifest['records'] if r['split']==split))
    # Test is not exposed to the trainer or model checkpoint selection.
    (corpus/'data.yaml').write_text(yaml.safe_dump(dict(path=str(corpus.resolve()),train='train.txt',val='dev.txt',
                                                     names=dict(enumerate(manifest['classes'])))))
    for split in ['train','dev','test']:
        rows=[r for r in manifest['records'] if r['split']==split]
        for page in range((len(rows)+15)//16):
            subset=rows[page*16:(page+1)*16]
            sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(subset):
                im=Image.open(corpus/r['file']).resize((288,216),Image.Resampling.NEAREST);pen=ImageDraw.Draw(im)
                for obj in lookup[r['id']]['objects']:
                    b=np.asarray(obj['box'])*2.4
                    pen.rectangle(tuple(b),outline=COLORS[obj['class_id']],width=1)
                    pen.text((b[0],b[1]),str(obj['class_id']),fill=COLORS[obj['class_id']])
                x=(i%4)*300;y=(i//4)*250;sheet.paste(im,(x,y));draw.text((x,y+217),r['id'],fill='black')
            sheet.save(corpus/f'annotated_{split}_{page}.png')
    # Class-specific masks are easier to verify than overlapping multi-class boxes.
    for cls in range(6):
        positives=[(r,o) for r in manifest['records'] for o in lookup[r['id']]['objects'] if o['class_id']==cls]
        indices=np.linspace(0,max(0,len(positives)-1),min(16,len(positives)),dtype=int)
        sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
        for i,index in enumerate(indices):
            r,o=positives[index];x=(i%4)*300;y=(i//4)*250
            im=Image.open(corpus/r['file']).resize((288,216),Image.Resampling.NEAREST);pen=ImageDraw.Draw(im)
            b=np.asarray(o['box'])*2.4;pen.rectangle(tuple(b),outline=COLORS[cls],width=2)
            sheet.paste(im,(x,y));draw.text((x,y+217),f"{r['id']} pixels={o['pixels']}",fill='black')
        sheet.save(corpus/f'annotated_class{cls}.png')
    print(json.dumps({s:[sum(o['class_id']==c for r in manifest['records'] if r['split']==s for o in lookup[r['id']]['objects']) for c in range(6)] for s in sets}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);a=p.parse_args();prepare(a.corpus)
