"""Produce reviewable tables and plots from frozen detector outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ORDER=['gdino_six','gdino_strict','gdino_joint','world_320','world_640',
       'yolo_n_320','yolo_n_640','yolo_native640','mask_ft','bowl_combined','sam_detector','sam_oracle']
COLORS=['red','cyan','orange','lime','magenta','yellow']


def summarize(output):
    results={name:json.loads((output/'results'/(name+'.json')).read_text()) for name in ORDER}
    corpus=output/'corpus_v3';manifest=json.loads((corpus/'manifest.json').read_text())
    labels={a['id']:a for a in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    folder=output/'summary';folder.mkdir(exist_ok=True)
    percent=lambda x:'—' if x is None else f'{100*x:.2f}'
    lines=['| Method | Precision % | Recall % | Macro F1 % | Median ms | P95 ms |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ORDER:
        r=results[name];m=r['test']['micro'];lat=r['latency_ms']
        if name=='bowl_combined':m=r['test']['per_class'][0]
        lines.append(f"| {name} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(r['test']['macro_f1']) if name!='bowl_combined' else 'bowl only'} | {lat['median']:.2f} | {lat['p95']:.2f} |")
    (folder/'table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    # Compare native 640 and Mask R-CNN with actual reference counts.
    lines=['| Class | GT | YOLO640 TP/FP/FN | YOLO640 P/R % | Mask TP/FP/FN | Mask P/R % |',
           '|---|---:|---|---|---|---|']
    for cls,name in enumerate(manifest['classes']):
        a=results['yolo_native640']['test']['per_class'][cls];b=results['mask_ft']['test']['per_class'][cls]
        counts=lambda r:f"{r['tp']}/{r['fp']}/{r['fn']}"
        lines.append(f"| {name} | {a['tp']+a['fn']} | {counts(a)} | {percent(a['precision'])}/{percent(a['recall'])} | {counts(b)} | {percent(b['precision'])}/{percent(b['recall'])} |")
    (folder/'classes.md').write_text('\n'.join(lines)+'\n')
    # Display selected hard cases, using saved outputs, not rerunning inference.
    ids=['0_demo_42_388','1_demo_42_228','2_demo_50_180','3_demo_24_966']
    columns=['reference','gdino_six','world_320','yolo_native640','mask_ft']
    sheet=Image.new('RGB',(1200,4*210+30),'white');draw=ImageDraw.Draw(sheet)
    for col,name in enumerate(columns):draw.text((col*240+4,8),name,fill='black')
    index={r['id']:r for r in manifest['records']}
    lookup={name:{r['id']:r for r in results[name]['predictions']} for name in columns if name!='reference'}
    for row,identity in enumerate(ids):
        record=index[identity]
        for col,name in enumerate(columns):
            im=Image.open(corpus/record['file']).convert('RGB').resize((240,180),Image.Resampling.NEAREST)
            pen=ImageDraw.Draw(im)
            if name=='reference':objects=labels[identity]['objects']
            else:
                th=results[name]['selection']['thresholds']
                objects=[o for o in lookup[name][identity]['objects'] if o['score']>=th[o['class_id']]]
            for obj in objects:
                box=np.asarray(obj['box'])*2;cls=obj['class_id']
                pen.rectangle(tuple(box),outline=COLORS[cls],width=1)
                pen.text((box[0],box[1]),str(cls),fill=COLORS[cls])
            sheet.paste(im,(col*240,30+row*210));draw.text((col*240+2,211+row*210),identity,fill='black')
    sheet.save(folder/'comparison_examples.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    aliases={'gdino_six':'GDINO six prompts','gdino_strict':'GDINO phrase check',
             'gdino_joint':'GDINO joint','world_320':'World 320','world_640':'World 640',
             'yolo_n_320':'YOLO 320/320','yolo_n_640':'YOLO 320/640 control',
             'yolo_native640':'YOLO 640/640','mask_ft':'Mask R-CNN'}
    offsets={'yolo_native640':(4,12),'mask_ft':(4,-17),
             'gdino_six':(-100,12),'gdino_strict':(-105,-17),
             'world_320':(-10,10),'world_640':(6,6)}
    for name in ORDER[:9]:
        r=results[name];x=r['latency_ms']['median'];y=r['test']['macro_f1']*100
        ax.scatter([x],[y],s=50)
        ax.annotate(aliases[name],(x,y),xytext=offsets.get(name,(4,4)),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set_xlabel('Detector latency median, ms (log scale)')
    ax.set_ylabel('Six-class macro F1, % (IoU >= 0.5)');ax.set_ylim(0,100)
    ax.set_title('Episode-disjoint pilot: 63 test frames, 305 reference objects')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(folder/'quality_latency.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);summarize(p.parse_args().output)
