"""Main-camera continuous replay beside exact archived selector observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from data import load_group
from growth_data import load_growth_data,growth_partitions
from calibration import apply_calibration
from sequence_metrics import sequence_metrics
from error_impact import analyze_impact


CLASSES=['bowl','door','door handle','rack','drawer','drawer handle']
SOURCES=[('confirm6','confirmation6_incumbent_blend','primary'),('confirm7','confirmation7_epoch_selection','retention'),
         ('confirm8','confirm8_recurrent_confirmation','retention'),('confirm9','confirm9_frozen_confirmation','retention')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    import imageio_ffmpeg
    a.output.mkdir(parents=True);r=a.root/'momart_target_research_20261101';groups={};inputs={}
    data=load_growth_data(a.root);part=growth_partitions(data)
    study=r/'positive_set_training_v4';reg=json.loads((study/'registry.json').read_text());v=reg['blend_frozen_configuration']
    prob=np.load(study/'blend_frozen_configuration_probabilities.npz')['probability'];pred=apply_calibration(prob,v['calibration'])
    groups['dev']=(data,pred,prob,part['dev']);inputs['dev_probability']=sha(study/'blend_frozen_configuration_probabilities.npz')
    expected_selection=sha(r/'frozen_for_confirm6_incumbent_blend/selected.json')
    for prefix,evaluation,member in SOURCES:
        directory=r/evaluation;frozen=json.loads((directory/'frozen_evaluation_manifest.json').read_text())
        assert frozen['selection_sha256' if member=='primary' else 'retention_selection_sha256']==expected_selection
        data=load_group(r/(prefix+'_corpus'),r/(prefix+'_features.npz'),r/(prefix+'_phases.json'),'examined')
        z=np.load(directory/(member+'_predictions.npz'));assert z['ids'].tolist()==[row['id'] for row in data['rows']]
        groups[prefix]=(data,z['prediction'],z['probability'],np.flatnonzero(data['y']>=0))
        inputs[prefix]=dict(predictions_sha256=sha(directory/(member+'_predictions.npz')),manifest_sha256=sha(r/(prefix+'_corpus')/'manifest.json'))
    audits={};seen=set();clips=[]
    for name,(data,pred,prob,index) in groups.items():
        eps={data['rows'][i]['episode_id'] for i in index};assert not seen&eps;seen|=eps
        audit=analyze_impact(data,pred,index);seq=sequence_metrics(pred,data,index)
        assert audit['summary']['subject_changes']==seq['gold_subject_changes']
        assert audit['summary']['failed_at_change']==seq['gold_subject_changes']-seq['correct_box_at_change']
        assert audit['summary']['stale_previous_class']==seq['stale_previous_class_at_change']
        assert audit['summary']['stable_class_jumps']==seq['stable_class_switches']
        assert audit['summary']['stable_both_selected_pairs']==seq['stable_both_selected']
        audits[name]=audit
        for event in audit['acquire_events']:
            clips.append(dict(group=name,kind='acquire_wrong_class',center=event['indices'][0],last=event['indices'][-1],episode_id=event['episode_id'],details=event))
        for event in audit['subject_changes']:
            if not event['correct']:clips.append(dict(group=name,kind='switch_not_correct',center=event['index'],last=event['index'],episode_id=event['episode_id'],details=event))
    totals={key:sum(audits[name]['summary'][key] for name,_,_ in SOURCES) for key in audits['dev']['summary'] if not key.startswith('max_')}
    (a.output/'impact.json').write_text(json.dumps(dict(groups=audits,confirm6_to9_totals=totals,inputs=inputs,selection_sha256=expected_selection),indent=2))
    print('COUNTS',json.dumps({k:v['summary'] for k,v in audits.items()}),flush=True)
    if a.audit_only:return
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(fontpath),21);small=ImageFont.truetype(str(fontpath),18)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();width,height=1280,720
    def writer(path):return imageio_ffmpeg.write_frames(str(path),(width,height),fps=a.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,ffmpeg_log_level='error',output_params=['-movflags','+faststart'])
    reel=writer(a.output/'all_error_events.mp4');reel.send(None);manifest=[];files={};references={}
    try:
        for n,clip in enumerate(clips,1):
            name=clip['group'];data,pred,prob,index=groups[name];row=data['rows'][clip['center']];episode=row['episode_id'];prefix=None
            if name!='dev':prefix=name
            # Reference masks/boxes are review-only; they never enter inference.
            if prefix is not None and prefix not in references:
                references[prefix]={x['id']:x['objects'] for x in json.loads((r/(prefix+'_corpus')/'annotations_all.json').read_text())['annotations']}
            if name=='dev':
                for p in ['dense','confirmation','confirm2','growth']:
                    if p in references:continue
                    m=json.loads((r/(p+'_corpus')/'manifest.json').read_text());ann={x['id']:x['objects'] for x in json.loads((r/(p+'_corpus')/'annotations_all.json').read_text())['annotations']}
                    references[p]={ (str(Path(q['source']).resolve()),q['demo'],q['step']):ann[q['id']] for q in m['records'] }
            source=row['source']
            if source not in files:files[source]=h5py.File(source,'r')
            obs=files[source]['data'][row['demo']]['obs'];length=len(obs['rgb'])
            start=max(0,row['step']-a.context_steps);end=min(length-1,data['rows'][clip['last']]['step']+a.context_steps)
            ids=sorted([i for i,x in enumerate(data['rows']) if x['episode_id']==episode and start<=x['step']<=end],key=lambda i:data['rows'][i]['step'])
            bystep={data['rows'][i]['step']:i for i in ids};last_query=None;checks=[]
            path=a.output/f'{n:02d}_{name}_{clip["kind"]}.mp4';stream=writer(path);stream.send(None)
            try:
                for step in range(start,end+1):
                    rgb=np.asarray(obs['rgb'][step]);canvas=Image.new('RGB',(width,height),'#10151e');draw=ImageDraw.Draw(canvas)
                    draw.text((20,10),f'{n:02d}/{len(clips)} {name} | {clip["kind"]} | {row["task"]}',font=font,fill='white')
                    draw.text((20,42),f'{row["demo"]} | source step {step} | event step {row["step"]}',font=font,fill='white')
                    draw.text((20,80),'MAIN CAMERA: continuous recorded replay',font=small,fill='white')
                    canvas.paste(Image.fromarray(rgb).resize((600,450),Image.Resampling.NEAREST),(20,110))
                    if step in bystep:
                        last_query=bystep[step];q=data['rows'][last_query];digest=hashlib.sha256(rgb.tobytes()).hexdigest();assert digest==q['sha256'];checks.append(dict(id=q['id'],step=step,rgb_sha256=digest))
                    if last_query is not None:
                        i=last_query;q=data['rows'][i];qrgb=np.asarray(obs['rgb'][q['step']]);canvas.paste(Image.fromarray(qrgb).resize((600,450),Image.Resampling.NEAREST),(660,110))
                        draw.text((660,80),f'EXACT SELECTOR QUERY step {q["step"]} (held)',font=small,fill='white')
                        label=q['label'];target=label['target'] if label else None;truth=[]
                        if name!='dev':truth=references[name][q['id']]
                        else:
                            key=(str(Path(source).resolve()),q['demo'],q['step'])
                            matches=[ref[key] for p,ref in references.items() if p in ['dense','confirmation','confirm2','growth'] and key in ref];assert len(matches)==1;truth=matches[0]
                        targetobj=next((x for x in truth if target and x['instance_id']==target['instance_id']),None)
                        def box(b,color,pad=0):
                            coords=[660+b[0]*5-pad,110+b[1]*5-pad,660+b[2]*5+pad,110+b[3]*5+pad];draw.rectangle(coords,outline=color,width=4)
                        if targetobj:box(targetobj['box'],'#50ed85',3)
                        slot=int(pred[i]);selected=q['objects'][slot] if slot<8 else None
                        if selected:box(selected['box'],'#ff5959')
                        targettext=CLASSES[target['class_id']] if target else 'unknown intent'
                        selecttext=CLASSES[selected['class_id']] if selected else 'REJECT / no box'
                        outcome='UNKNOWN TARGET' if not label else ('CORRECT BOX' if slot<8 and data['positive'][i,slot] else 'REJECT' if slot==8 else 'WRONG BOX')
                        draw.text((20,580),f'RED = model selected: {selecttext} | {outcome} | max p={prob[i].max():.3f}',font=font,fill='#ff9090')
                        draw.text((20,614),f'GREEN = weak reference subject: {targettext} | not human intention truth',font=font,fill='#7fffac')
                        draw.text((20,648),'Right panel holds the exact query RGB/boxes; left continues. No interpolated predictions.',font=small,fill='white')
                    else:draw.text((660,80),'No archived query yet',font=small,fill='white')
                    draw.text((20,682),f'Offline recorded demonstration, NOT policy rollout | display {a.fps} source steps/s | main camera only',font=small,fill='#b7c7e0')
                    frame=np.asarray(canvas);stream.send(frame);reel.send(frame)
            finally:stream.close()
            manifest.append(dict(number=n,file=path.name,sha256=sha(path),source=source,demo=row['demo'],start_step=start,end_step=end,event=clip,query_rgb_checks=checks))
            print('CLIP',n,len(clips),path.name,flush=True)
    finally:
        reel.close()
        for handle in files.values():handle.close()
    (a.output/'video_manifest.json').write_text(json.dumps(dict(clips=manifest,reel='all_error_events.mp4',reel_sha256=sha(a.output/'all_error_events.mp4'),
        source_steps_per_display_second=a.fps,real_control_frequency_claimed=False,views=['main'],no_new_model_inference=True,all_error_events_included=True,
        note='Recorded source demonstrations and fixed archived selector outputs; errors are weak-proxy-defined, not task failures. Right exact-query panel is held between evaluated observations.'),indent=2))
    print('COMPLETE',len(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--context-steps',type=int,default=40);p.add_argument('--fps',type=int,default=20)
    p.add_argument('--audit-only',action='store_true',help='Write exact frame/event/recovery and stable-jump counts without rendering')
    main(p.parse_args())
