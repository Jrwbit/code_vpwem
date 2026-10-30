"""Score fixed dev thresholds separately on original and new test episodes.

The capacity diagnostic applies the memory branch's class-balanced selection
*after* confidence thresholds, using recorded RGB-only detections. No threshold
or checkpoint is reselected on either test subset.
"""
import argparse
import json
from pathlib import Path
from metrics import measure


def bounded(records, thresholds, capacity):
    out=[]
    for row in records:
        candidates=[o for o in row['objects'] if o['score']>=thresholds[o['class_id']]]
        candidates.sort(key=lambda o:(-o['score'],o['class_id'],*o['box']))
        chosen=[];classes=set()
        for i,o in enumerate(candidates):
            if o['class_id'] not in classes:
                chosen.append(i);classes.add(o['class_id'])
        chosen=chosen[:capacity]
        for i in range(len(candidates)):
            if len(chosen)>=capacity:break
            if i not in chosen:chosen.append(i)
        out.append(dict(row,objects=[candidates[i] for i in chosen]))
    return out


def summarize(corpus, results, output):
    manifest=json.loads((corpus/'manifest.json').read_text())
    labels={r['id']:r for r in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    groups={'original_test':{r['id'] for r in manifest['records'] if r['split']=='test' and not r['id'].startswith('supp_')},
            'confirmation_test':{r['id'] for r in manifest['records'] if r['split']=='test' and r['id'].startswith('supp_')},
            'all_test':{r['id'] for r in manifest['records'] if r['split']=='test'}}
    result={}
    for name in ('yolo_before','yolo_after','mask_before','mask_after'):
        raw=json.loads((results/(name+'.json')).read_text()); thresholds=raw['selection']['thresholds']
        predictions=[r for r in raw['predictions'] if r['split']=='test']
        result[name]=dict(weights_sha256=raw['weights_sha256'],thresholds=thresholds,
                         latency_ms=raw['latency_ms'],physical_gpu=raw['physical_gpu'])
        for group, ids in groups.items():
            rows=[r for r in predictions if r['id'] in ids];assert len(rows)==len(ids)
            result[name][group]=dict(frames=len(rows),full=measure(rows,labels,thresholds),
                capacity8=measure(bounded(rows,thresholds,8),labels,thresholds),
                capacity16=measure(bounded(rows,thresholds,16),labels,thresholds))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for name,data in result.items():
        print(name)
        for group in groups:
            full=data[group]['full'];print(group,full['micro'])
            print('drawer',full['per_class'][4], 'handle',full['per_class'][5])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.corpus,a.results,a.output)
