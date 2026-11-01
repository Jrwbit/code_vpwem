"""Causal pointer rollout on recorded observations; proxies are scoring-only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
import cv2
from proxy import target_at,target_candidate
from selector import TargetSelector,TargetLock

CLASSES=['bowl','dishwasher door','door handle','rack','drawer','drawer handle']


def main(a):
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    # Prediction first. No GT or events are read in this block.
    meta=json.loads(a.features.with_suffix('.json').read_text());features=np.load(a.features,allow_pickle=False)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['tasks']==meta['tasks']
    net=TargetSelector(ckpt['candidate_dim'],ckpt['context_dim']).eval();net.load_state_dict(ckpt['model'])
    with torch.inference_mode():
        probabilities=net(torch.tensor((features['candidates']-ckpt['candidate_mean'])/ckpt['candidate_std']),
            torch.tensor((features['context']-ckpt['context_mean'])/ckpt['context_std']),
            torch.tensor(features['valid'])).softmax(-1).numpy()
    corpus=json.loads((a.corpus/'manifest.json').read_text());source={row['id']:row for row in corpus['records']}
    detections={row['id']:row['objects'] for row in meta['detections']}
    predictions=[];locks={}
    for i,identity in enumerate(meta['ids']):
        row=source[identity];objects=detections[identity];raw=int(probabilities[i].argmax())
        lock=locks.setdefault(row['episode_id'],TargetLock())
        chosen=lock.choose(objects,probabilities[i],row['step'])
        predictions.append(dict(id=identity,episode_id=row['episode_id'],step=row['step'],
            raw=raw if raw<len(objects) else None,locked=chosen,probabilities=probabilities[i].tolist(),objects=objects))
    (a.output/'predictions.json').write_text(json.dumps(dict(ground_truth_input=False,future_input=False,predictions=predictions),indent=2))
    # Labels may only enter the evaluation/visualization below.
    events=json.loads(a.targets.read_text())['events']
    truth={row['id']:row['objects'] for row in json.loads((a.corpus/'annotations_all.json').read_text())['annotations']}
    results={};writers={};sheets=[]
    for name in ('raw','locked','confidence'):
        results[name]=dict(labelled=0,visible=0,available=0,correct=0,selected=0)
    for i,pred in enumerate(predictions):
        row=source[pred['id']];objects=pred['objects'];target=target_at([e for e in events if e['episode_id']==row['episode_id']],row['step'])
        reference=next((obj for obj in truth[row['id']] if target and obj['instance_id']==target['instance_id']),None)
        matched=target_candidate(objects,truth[row['id']],target)
        if target:
            for name in results:
                index=pred[name] if name!='confidence' else (int(np.argmax([o['score'] for o in objects])) if objects else None)
                hit=index is not None and target_candidate([objects[index]],truth[row['id']],target)==0
                value=results[name];value['labelled']+=1;value['visible']+=reference is not None
                value['available']+=matched is not None and matched>=0;value['correct']+=hit;value['selected']+=index is not None
        frame=Image.open(a.corpus/row['file']).convert('RGB').resize((640,480),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(640,600),'#161b22');canvas.paste(frame,(0,0));draw=ImageDraw.Draw(canvas)
        def rectangle(box,color):
            coords=(np.asarray(box)*np.array([640/120,480/90,640/120,480/90])).tolist()
            draw.rectangle(coords,outline=color,width=3)
        if reference:rectangle(reference['box'],'yellow')
        index=pred['raw'];selected=objects[index] if index is not None else None
        if selected:rectangle(selected['box'],'cyan')
        label=CLASSES[selected['class_id']] if selected else 'NONE'
        draw.text((8,490),row['episode_id'],fill='white')
        draw.text((8,510),f"step={row['step']}  causal selection={label}",fill='cyan')
        draw.text((8,530),'Yellow: offline motion/proximity proxy, not human intent GT',fill='yellow')
        draw.text((8,550),'Cyan: predicted target box; no GT/future input to selector',fill='white')
        draw.text((8,570),'Playback: physical time, observations every 8 steps (0.4s)',fill='white')
        episode=row['episode_id']
        if episode not in writers:
            path=a.output/(episode.replace('/','_')+'.mp4')
            writers[episode]=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2.5,(640,600))
            if not writers[episode].isOpened():raise RuntimeError('Video encoder unavailable')
        writers[episode].write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
        if target and reference and len(sheets)<12 and i%3==0:sheets.append(canvas)
    for writer in writers.values():writer.release()
    if sheets:
        sheet=Image.new('RGB',(1280,600*((len(sheets)+1)//2)),'white')
        for i,frame in enumerate(sheets):sheet.paste(frame,((i%2)*640,(i//2)*600))
        sheet.save(a.output/'examples.png')
    for value in results.values():
        value['conditional_accuracy']=value['correct']/value['available'] if value['available'] else None
        value['visible_recall']=value['correct']/value['visible'] if value['visible'] else None
    (a.output/'summary.json').write_text(json.dumps(dict(episodes=len(writers),observations=len(predictions),results=results,
        caveat='Two previously used detector-test episodes, weak proxy labels; continuous observation replay, not robot control rollout'),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
