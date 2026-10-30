"""Export a small, reviewable annotation bundle; never includes model weights."""
import argparse
import json
from pathlib import Path
import zipfile


def export(corpus,destination):
    manifest=json.loads((corpus/'manifest.json').read_text())
    labels={a['id']:a for a in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    coco=dict(images=[],annotations=[],categories=[dict(id=i+1,name=c) for i,c in enumerate(manifest['classes'])])
    for image_id,r in enumerate(manifest['records'],1):
        coco['images'].append(dict(id=image_id,file_name=r['file'],width=120,height=90,
                                   split=r['split'],source_id=r['id'],episode_id=r['episode_id']))
        for o in labels[r['id']]['objects']:
            x1,y1,x2,y2=o['box']
            coco['annotations'].append(dict(id=len(coco['annotations'])+1,image_id=image_id,
                category_id=o['class_id']+1,bbox=[x1,y1,x2-x1,y2-y1],area=o['pixels'],iscrowd=0,
                instance_id=o['instance_id'],tiny=o['tiny']))
    (corpus/'annotations.coco.json').write_text(json.dumps(coco,indent=2))
    readme=('MoMaRT detector pilot, visible-mask-assisted reference.\n'
        'This is assistant-curated simulation-assisted annotation, not independent human GT.\n'
        'COCO category IDs: 1..6; YOLO class IDs: 0..5. RGB: original 120x90.\n'
        'boxes use [x1,y1,x2,y2), with exclusive maximum pixel bounds.\n'
        'masks/*.npz arrays follow each frame objects order in annotations_all.json.\n'
        'Use manifest splits by episode; do not randomly split neighbouring frames.\n'
        'See review.json for source hashes and limitations; 9 candidates rejected by alignment.\n')
    (corpus/'README.txt').write_text(readme)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['README.txt','manifest.json','annotations_all.json','annotations.coco.json','review.json']:
            z.write(corpus/name,name)
        for r in manifest['records']:
            for name in [r['file'],'labels/'+r['id']+'.txt','masks/'+r['id']+'.npz']:
                z.write(corpus/name,name)
    print(json.dumps(dict(bundle=str(destination),bytes=destination.stat().st_size,
                          images=len(coco['images']),objects=len(coco['annotations']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.corpus,a.output)
