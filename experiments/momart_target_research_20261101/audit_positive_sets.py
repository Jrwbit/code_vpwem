"""Inspect weak equivalent-box ambiguity without changing any label or score."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from growth_data import load_growth_data,growth_partitions
from perception_data import collect_detector_queries
from detector_metrics import iou
from proxy import target_candidate
from confirm import predict_component
from calibration import apply_calibration
from benchmark import write_json


def matching_instances(obj,truth):
    return [other['instance_id'] for other in truth if target_candidate([obj],truth,other)==0]


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    r=a.root/'momart_target_research_20261101';data=load_growth_data(a.root);splits=growth_partitions(data)
    queries,hashes=collect_detector_queries(data,r)
    refs={q['base_index']:q['source']['objects'] for part in queries.values() for q in part}
    selection=a.selection or r/'frozen_for_confirm6_incumbent_blend/selected.json'
    selected=json.loads(selection.read_text())['result']
    probability=predict_component(selected['component'],data,a.root);pred=apply_calibration(probability,selected['calibration'])
    summary={};details=[]
    for split in ['train','dev','old_dev','new_dev']:
        multiple=[];ambiguous=[];wrong_acquire=[];all_ambiguous=[];correct_ambiguous=[];correct_unique=0
        for i in splits[split]:
            row=data['rows'][i];target=row['label']['target'];slots=np.flatnonzero(data['positive'][i])
            if data['roles'][i]==0 and pred[i]<8 and row['objects'][pred[i]]['class_id']!=0:
                wrong_acquire.append(dict(id=row['id'],step=row['step'],positive_slots=slots.tolist(),selected=int(pred[i])))
            truth=refs[i]
            other_matches={int(slot):[instance for instance in matching_instances(row['objects'][slot],truth) if instance!=target['instance_id']] for slot in slots}
            if any(other_matches.values()):all_ambiguous.append(row['id'])
            if pred[i]<8 and data['positive'][i,pred[i]]:
                if other_matches[int(pred[i])]:correct_ambiguous.append(row['id'])
                else:correct_unique+=1
            if len(slots)<2:continue
            gold=next(obj for obj in truth if obj['instance_id']==target['instance_id'])
            objects=[]
            for slot in slots:
                obj=row['objects'][slot]
                others=other_matches[int(slot)]
                objects.append(dict(slot=int(slot),target_iou=iou(obj['box'],gold['box']),also_matches_other_instances=others))
            entry=dict(split=split,id=row['id'],episode_id=row['episode_id'],step=row['step'],role=int(data['roles'][i]),target_instance=target['instance_id'],
                canonical=int(data['y'][i]),current_selected=int(pred[i]),current_correct=bool(pred[i]<8 and data['positive'][i,pred[i]]),
                positive_objects=objects,pair_iou=iou(row['objects'][slots[0]]['box'],row['objects'][slots[1]]['box']))
            multiple.append(entry)
            if any(obj['also_matches_other_instances'] for obj in objects):ambiguous.append(entry['id'])
        summary[split]=dict(samples=len(splits[split]),multiple=len(multiple),also_matches_other_instance=len(ambiguous),ambiguous_ids=ambiguous,
            all_queries_with_ambiguous_positive=len(all_ambiguous),all_ambiguous_ids=all_ambiguous,
            current_weak_correct_unambiguous=correct_unique,current_weak_correct_ambiguous=len(correct_ambiguous),current_ambiguous_correct_ids=correct_ambiguous,
            current_correct_multiple=sum(x['current_correct'] for x in multiple),current_wrong_acquire=wrong_acquire,
            multiple_pair_iou_range=[min(x['pair_iou'] for x in multiple),max(x['pair_iou'] for x in multiple)] if multiple else None)
        if split in ['train','dev']:details.extend(multiple)
    write_json(a.output,dict(protocol='equivalent_positive_boxes_audit_v1',summary=summary,details=details,source_hashes=hashes,
        selection=str(selection),selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        caveat='Box-IoU weak references only. No instance label correction, denominator change or independent intention truth. Frozen current model used for diagnostic only.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection',type=Path,help='Saved public selector configuration; defaults to the fixed current candidate')
    main(p.parse_args())
