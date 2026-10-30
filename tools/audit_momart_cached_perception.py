#!/usr/bin/env python3
"""Read-only, reproducible visual sampling and conservative precision bounds.

Review labels are supplied separately by a visual reviewer, never inferred from
confidence. They are not independent ground truth and do not produce mAP/recall.
The original cache is opened read-only and is never rewritten.
"""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np


def read_predictions(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        return [dict(class_id=int(c),box=b.tolist(),score=float(s))
                for c,b,s in zip(data['classes'],data['boxes'],data['scores'])]


def sample(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import zarr
    from PIL import Image,ImageDraw,ImageFont
    from momart_no_history_baseline.cache import training_replay
    from momart_object_memory.cache import replay_signature
    con=sqlite3.connect('file:'+str(args.database.resolve())+'?mode=ro',uri=True)
    try:
        con.execute('BEGIN')
        count=con.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        population=min(count,args.population)
        if population < args.samples:raise ValueError('Not enough completed frames')
        indices=np.sort(np.random.default_rng(args.seed).choice(population,args.samples,replace=False))
        payloads=[(int(i),con.execute('SELECT payload FROM observations WHERE frame=?',(int(i),)).fetchone()[0]) for i in indices]
        metadata=json.loads(con.execute('SELECT signature FROM metadata WHERE id=1').fetchone()[0])
    finally:
        con.close()
    replay,names,mapping=training_replay(zarr.open_group(str(args.source),mode='r'))
    expected=replay_signature(replay,mapping,names)
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Source frame mapping or episode provenance differs')
    rows=[]
    for frame,blob in payloads:
        episode=int(np.searchsorted(replay.episode_ends,frame,side='right'))
        offset=0 if episode==0 else int(replay.episode_ends[episode-1])
        rows.append(dict(frame=frame,demo=names[episode],episode_step=frame-offset,
                         detections=read_predictions(blob)))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        Image.fromarray(np.asarray(replay['rgb'][row['frame']])).save(out/f"frame_{row['frame']}.png")
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    colors=['red','lime','cyan','yellow','magenta','orange']
    labels=['bowl','door','handle','rack','drawer','drawer_handle']
    for cls in range(6):
        predictions=[(r,k,o) for r in rows for k,o in enumerate(r['detections']) if o['class_id']==cls]
        sheet=Image.new('RGB',(1920,max(1,(len(predictions)+3)//4)*235),(25,25,25));pen=ImageDraw.Draw(sheet)
        for j,(r,k,o) in enumerate(predictions):
            raw=Image.fromarray(np.asarray(replay['rgb'][r['frame']])).resize((240,180),Image.Resampling.NEAREST)
            boxed=raw.copy();ImageDraw.Draw(boxed).rectangle(tuple(np.array(o['box'])*[240,180,240,180]),outline=colors[cls],width=2)
            x=(j%4)*480;y=(j//4)*235;sheet.paste(raw,(x,y));sheet.paste(boxed,(x+240,y))
            pen.text((x+2,y+182),f"f{r['frame']} pred{k} {labels[cls]} {o['score']:.2f}",fill='white',font=font)
        sheet.save(out/f'class_{cls}.png')
    record=dict(seed=args.seed,population=population,n=len(rows),classes=labels,samples=rows)
    (out/'sample.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(sampled_frames=len(rows),predictions=sum(len(r['detections']) for r in rows))))


def score(args):
    sample=json.loads(args.sample.read_text());review=json.loads(args.review.read_text())
    predictions={(r['frame'],i):o for r in sample['samples'] for i,o in enumerate(r['detections'])}
    seen=set();wrong=[0]*6;reviewed=[0]*6
    for r in review['reviews']:
        key=(r['frame'],r['prediction_index'])
        if key not in predictions or key in seen:raise ValueError('Invalid/duplicate review reference')
        seen.add(key);cls=predictions[key]['class_id'];reviewed[cls]+=1
        if r['judgment']=='clearly_wrong':wrong[cls]+=1
        elif r['judgment'] not in ('visually_plausible','uncertain'):raise ValueError('Unknown visual judgment')
    counts=[sum(p['class_id']==i for p in predictions.values()) for i in range(6)]
    result=dict(frames=sample['n'],predictions=len(predictions),per_class_counts=counts,
                per_class_reviewed=reviewed,per_class_clearly_wrong=wrong,
                clear_errors=sum(wrong),optimistic_sample_precision_upper_bound=(len(predictions)-sum(wrong))/len(predictions),
                interpretation='Conditional on visual error judgments; all unreviewed/uncertain boxes counted as correct for this upper bound. Not GT precision, recall, AP, or a population confidence bound.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('sample');p.add_argument('--database',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--population',type=int,required=True)
    p.add_argument('--samples',type=int,default=36);p.add_argument('--seed',type=int,default=20261030);p.set_defaults(run=sample)
    p=sub.add_parser('score');p.add_argument('--sample',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.set_defaults(run=score)
    args=parser.parse_args();args.run(args)


if __name__=='__main__':main()
