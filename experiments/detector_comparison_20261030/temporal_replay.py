"""Replay real frozen features through the production bank and causal cache.

GT IDs score outputs after the bank update; they never influence association.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import numpy as np
from PIL import Image
import torch
import zarr
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from temporal_metrics import match,miss_runs,event_scores,stats,wilson,identity_continuity,mixed_identity_history
from momart_object_memory.state import ObservationStream,make_bank
from momart_object_memory.association import association_contract
from momart_object_memory.perception import bounded_detections,file_sha256
from momart_object_memory.prepare_train import decode
from momart_object_memory.cache import write_cache,select_mode
from momart_object_memory.model import OBJECT_KEYS,ObjectMemoryEncoder


def replay(a):
    torch.set_num_threads(2);torch.manual_seed(100)
    encoder=ObjectMemoryEncoder(128,6).eval()
    manifest=json.loads((a.corpus/'candidate_manifest.json' if (a.corpus/'candidate_manifest.json').exists() else a.corpus/'manifest.json').read_text())
    rows=manifest['records'];labels=json.loads((a.corpus/'annotations_all.json').read_text())
    truth={r['id']:r['objects'] for r in labels['annotations']}
    raw=json.loads(a.predictions.read_text());pred={r['id']:r for r in raw['predictions']}
    assert set(pred)=={r['id'] for r in rows}
    scored=[r for r in rows if r['id'] in truth]
    hits={r['id']:set(match(pred[r['id']]['objects'],truth[r['id']]).values()) for r in scored}
    proxies=json.loads((a.corpus/'operation_proxies.json').read_text())
    observed={r['id'] for r in rows if r['step']%8 in (0,7)}
    result=dict(events=event_scores(proxies['events'],scored,truth,hits,observed),
        scope=dict(episodes=len(manifest['episodes']),raw_frames=len(rows),gt_frames=len(scored),
                   period_seconds=.05,action_steps=8,first_hit_lookback_steps=16,
                   critical_before_after_steps=[8,8],critical_definition='articulation_motion_proxy_not_human_stages'),
        prediction_provenance={k:v for k,v in raw.items() if k!='predictions'},
        association=association_contract({'method':a.association}),
        input_sha256={str(p):file_sha256(p) for p in (a.predictions,a.predictions.with_suffix('.sqlite'),
            a.corpus/'annotations_all.json',a.corpus/'operation_proxies.json')},banks={})
    result['miss_streaks']={}
    for e in manifest['episodes']:
        seq=[r for r in scored if r['episode_id']==e['episode_id']]
        result['miss_streaks'][e['episode_id']]=dict(dense=miss_runs(seq,truth,hits),policy=miss_runs([r for r in seq if r['id'] in observed],truth,hits,True),by_class={})
        for cls in (2,5):
            subset={k:[o for o in v if o['class_id']==cls] for k,v in truth.items()}
            result['miss_streaks'][e['episode_id']]['by_class'][str(cls)]=dict(dense=miss_runs(seq,subset,hits),policy=miss_runs([r for r in seq if r['id'] in observed],subset,hits,True))
    con=sqlite3.connect(str(a.predictions.with_suffix('.sqlite')))
    features=[decode(payload) for _,payload in con.execute('SELECT frame,payload FROM observations ORDER BY frame')];con.close()
    assert len(features)==len(rows)
    indices={r['id']:i for i,r in enumerate(rows)}
    for slots in a.slots:
        selected=[[d[3] for d in bounded_detections([(o.class_id,o.box,o.score,o) for o in obs],slots)] for obs in features]
        cfg=OmegaConf.create(dict(obs_steps=2,action_steps=8,object_memory=dict(slots=slots,history=3,feature_dim=128,
            match_iou=.3,period_seconds=.05,overflow='evict_oldest_invisible',
            association={'method':a.association})))
        evidence={};records=[];kept_refs=[];serial=0;last_gt={};gt_last_track={}
        switches=fragmentations=evictions=critical_evictions=0;residual_jumps=[];last_residual={};query_times=[]
        history_labels={};critical_identity=[];contaminated_reads=[]
        snapshots={};episode=None;seen_checks=missing_checks=0;update_seconds={}
        bank_hits={r['id']:set() for r in scored}
        def observe(index):return selected[int(index)]
        stream=ObservationStream(make_bank(cfg),observe)
        for i,r in enumerate(rows):
            if r['episode_id']!=episode:
                stream.reset();episode=r['episode_id'];evidence={};last_gt={};gt_last_track={};last_residual={};history_labels={}
            if r['id'] not in observed:continue
            before={id(t):t for t in stream.bank.tracks};t0=time.perf_counter()
            stream.observe([i],[r['step']],[r['step']*.05]);bank=stream.bank
            update_seconds[i]=time.perf_counter()-t0
            for t in bank.tracks:
                if id(t) not in before:kept_refs.append(t);serial+=1
            removed=set(before)-{id(t) for t in bank.tracks};evictions+=len(removed)
            upcoming={e['instance_id'] for e in proxies['events'] if e['episode_id']==episode and r['step']<=e['onset']<=r['step']+16}
            critical_evictions+=sum(evidence.get(k,(None,0))[0] in upcoming for k in removed)
            objects=[dict(class_id=o.class_id,box=(o.box*[120,90,120,90]).tolist(),score=o.score) for o in selected[i]]
            assignment=match(objects,truth.get(r['id'],[]))
            bank_hits[r['id']]=set(assignment.values())
            for track in bank.tracks:
                key=id(track)
                if not track['visible']:continue
                j=next(j for j,o in enumerate(selected[i]) if o.class_id==track['class_id'] and np.array_equal(o.box,track['box']))
                target=assignment.get(j);evidence[key]=(target,r['step'])
                history_labels[key]=(history_labels.get(key,[])+[target])[-3:]
                if target is not None:
                    if key in last_gt and last_gt[key]!=target:switches+=1
                    if target in gt_last_track and gt_last_track[target]!=key:fragmentations+=1
                    last_gt[key]=target;gt_last_track[target]=key
                    g=next(g for g in truth[r['id']] if g['instance_id']==target)
                    residual=(np.asarray(objects[j]['box'][:2])+objects[j]['box'][2:]-np.asarray(g['box'][:2])-g['box'][2:])/2
                    if target in last_residual and last_residual[target][0]==r['step']-1:
                        residual_jumps.append(float(np.linalg.norm(residual-last_residual[target][1])))
                    last_residual[target]=(r['step'],residual)
            # GT labels remain in this scorer; production bank only sees RGB/ROI.
            for eidx,e in enumerate(proxies['events']):
                if e['episode_id']!=episode or not e['onset']-8<=r['step']<=e['onset']+8:
                    continue
                target=e['instance_id']
                if any(g['instance_id']==target for g in truth.get(r['id'],[])):
                    keys=[id(t) for t in bank.tracks if t['visible'] and evidence.get(id(t),(None,))[0]==target]
                    critical_identity.append(dict(event=eidx,step=r['step'],track=keys[0] if keys else None))
            snapshot=bank.snapshot('a3')
            missing_checks+=int(np.sum(snapshot['valid'] & ~snapshot['visible']))
            assert not snapshot['boxes'][~snapshot['visible']].any()
            for slot,track in enumerate(bank.tracks):
                assert snapshot['valid'][slot] and (snapshot['visible'][slot]==track['visible'])
                seen_checks+=1
            if r['step']%8==0:
                snapshots[i]={k:v.copy() for k,v in snapshot.items()}
                t0=time.perf_counter()
                with torch.inference_mode():encoder({k:torch.as_tensor(v)[None] for k,v in snapshot.items()})
                seconds=(time.perf_counter()-t0)+update_seconds[i]+sum(pred[rows[j]['id']]['detect_seconds']+pred[rows[j]['id']]['roi_seconds'] for j in [i] if j>=0)
                # Query uses [t-1,t], except reset [0,0] is evaluated once.
                if r['step']>0:seconds+=update_seconds[i-1]+pred[rows[i-1]['id']]['detect_seconds']+pred[rows[i-1]['id']]['roi_seconds']
                query_times.append(seconds*1000)
                for e in proxies['events']:
                    if e['episode_id']==episode and e['onset']-8<=r['step']<=e['onset']+8:
                        entries=[(target,step) for key,(target,step) in evidence.items() if key in {id(t) for t in bank.tracks} and target==e['instance_id']]
                        ages=[(r['step']-step)*.05 for _,step in entries]
                        live={id(t) for t in bank.tracks}
                        contaminated=[key for key in live if mixed_identity_history(history_labels.get(key,[]),e['instance_id'])]
                        # This is input contamination, not evidence of a policy action effect.
                        contaminated_reads.append(dict(episode_id=episode,event_onset=e['onset'],
                            step=r['step'],mixed_identity_slots=len(contaminated),
                            unknown_history_slots=sum(None in history_labels.get(key,[]) for key in live)))
                        records.append(dict(event_onset=e['onset'],episode_id=episode,step=r['step'],target=e['instance_id'],retained_correct_latest=bool(ages),
                            current_correct=any(age==0 for age in ages),fresh={str(ttl):any(age<=ttl for age in ages) for ttl in (.4,.8,1.6)},youngest_correct_age=min(ages) if ages else None))
        # Production offline writer vs actual phase-0 online snapshots, with real ROI features.
        with tempfile.TemporaryDirectory(prefix='momart_temporal_') as temp:
            source=Path(temp)/'source.zarr';out=Path(temp)/'objects.zarr';root=zarr.open_group(str(source),mode='w')
            ends=[];count=0
            for e in manifest['episodes']:count+=sum(r['episode_id']==e['episode_id'] for r in rows);ends.append(count)
            root.attrs.update(vpwem_cache_complete=True,vpwem_cache_kind='momart',demo_names=['demo_'+str(i) for i in range(len(ends))])
            root.create_dataset('meta/episode_ends',data=np.asarray(ends))
            root.create_dataset('data/rgb',data=np.stack([np.asarray(Image.open(a.corpus/r['file'])) for r in rows]),chunks=(32,90,120,3))
            root.create_dataset('data/action',data=np.zeros((len(rows),10),np.float32))
            class Perception:
                metadata={'diagnostic':True,'prediction_sha256':file_sha256(a.predictions)}
            write_cache(cfg,source,out,Perception(),observation_reader=observe)
            cached=zarr.open_group(str(out),mode='r');comparisons=0;token_comparisons=0
            for i,online in snapshots.items():
                offline={k:cached[k][i] for k in OBJECT_KEYS}
                for mode in ('a1','a2','a3'):
                    left=select_mode(online,mode);right=select_mode(offline,mode)
                    for key in OBJECT_KEYS:np.testing.assert_array_equal(left[key],right[key]);comparisons+=1
                    with torch.inference_mode():
                        l,_=encoder({k:torch.as_tensor(v)[None] for k,v in left.items()});rr,_=encoder({k:torch.as_tensor(v)[None] for k,v in right.items()})
                    torch.testing.assert_close(l,rr,atol=0,rtol=0);token_comparisons+=1
        result['banks'][str(slots)]=dict(
            critical_identity=identity_continuity(critical_identity,len(proxies['events'])),
            critical_history_contamination=dict(definition='A3 reads a retained slot containing both target and a different known GT identity within last 3 observations; unknown labels are unassessed; no policy-effect claim',
                contaminated_query_events=sum(r['mixed_identity_slots']>0 for r in contaminated_reads),
                query_events=len(contaminated_reads),records=contaminated_reads),
            memory_first_hit=event_scores(proxies['events'],scored,truth,bank_hits,observed)['summary']['policy'],
            identity_switches=switches,fragmentations=fragmentations,evictions=evictions,
            upcoming_proxy_target_evictions=critical_evictions,residual_center_jump_pixels=stats(residual_jumps),
            offline_online_array_equalities=comparisons,offline_online_token_equalities=token_comparisons,
            token_encoder='untrained_shared_seed100_encoder_CPU; equivalence_only_not_policy_quality',
            seen_visible_checks=seen_checks,missing_current_boxes_zero_checks=missing_checks,
            critical_memory=records,object_branch_query_ms=stats(query_times),
            latency_scope='sum of sampled per-frame detector+ROI and CPU bank+token timing; excludes old main boxer, DiT, camera/robot; not full DP latency')
    result['wilson_example_32tp_1fp']=wilson(32,1)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['banks','miss_streaks','prediction_provenance']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--association',choices=('iou_greedy','iou_hungarian','appearance_motion'),default='iou_greedy')
    p.add_argument('--slots',type=int,nargs='+',default=[8,16])
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();replay(a)
