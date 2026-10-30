"""VIMA-style in-domain adaptation using actual Mask R-CNN, not its policy.

CAD-visible masks supervise the mask branch as well as the six box classes.
Checkpoint selection uses development F1; the test split is never opened here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from metrics import calibrate


def make_model():
    backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=None,trainable_layers=3,
                                norm_layer=FrozenBatchNorm2d)
    model=MaskRCNN(backbone,num_classes=7,min_size=320,max_size=448,
                  box_score_thresh=.01,box_nms_thresh=.5,box_detections_per_img=100)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,7)
    model.roi_heads.mask_predictor=MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels,256,7)
    return model


class Corpus(Dataset):
    def __init__(self,path,split):
        self.path=path
        manifest=json.loads((path/'manifest.json').read_text())
        self.rows=[r for r in manifest['records'] if r['split']==split]
        self.labels={a['id']:a for a in json.loads((path/'annotations_all.json').read_text())['annotations']}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];objects=self.labels[row['id']]['objects']
        image=torch.from_numpy(np.asarray(Image.open(self.path/row['file'])).copy()).permute(2,0,1).float()/255
        masks=np.load(self.path/'masks'/(row['id']+'.npz'))['masks']
        assert len(masks)==len(objects)
        target=dict(boxes=torch.tensor([o['box'] for o in objects],dtype=torch.float32).reshape(-1,4),
                    labels=torch.tensor([o['class_id']+1 for o in objects],dtype=torch.int64),
                    masks=torch.tensor(masks,dtype=torch.uint8),image_id=torch.tensor(index))
        return image,target


def train(args):
    torch.set_num_threads(4);torch.manual_seed(100);np.random.seed(100);random.seed(100)
    review=json.loads((args.corpus/'review.json').read_text())
    assert review['accepted_for']=='pilot_comparison'
    assert review['annotation_sha256']==hashlib.sha256((args.corpus/'annotations_all.json').read_bytes()).hexdigest()
    train=Corpus(args.corpus,'train');dev=Corpus(args.corpus,'dev')
    loader=DataLoader(train,batch_size=4,shuffle=True,num_workers=2,collate_fn=lambda x:tuple(zip(*x)))
    model=make_model()
    pretrained=torch.load(args.weights,map_location='cpu',weights_only=True)
    pretrained={k:v for k,v in pretrained.items() if not k.startswith(('roi_heads.box_predictor.','roi_heads.mask_predictor.'))}
    result=model.load_state_dict(pretrained,strict=False)
    assert all(k.startswith(('roi_heads.box_predictor.','roi_heads.mask_predictor.')) for k in result.missing_keys)
    assert not result.unexpected_keys
    model=model.cuda()
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,23],gamma=.1)
    args.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();best=-1
    for epoch in range(args.epochs):
        model.train();losses=[]
        for images,targets in loader:
            images=[im.cuda() for im in images]
            targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
            loss=sum(model(images,targets).values())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step();model.eval();records=[]
        with torch.inference_mode():
            for index,row in enumerate(dev.rows):
                image,_=dev[index];prediction=model([image.cuda()])[0]
                objects=[dict(class_id=int(c)-1,box=b.tolist(),score=float(s))
                         for b,s,c in zip(prediction['boxes'].cpu(),prediction['scores'].cpu(),prediction['labels'].cpu())]
                # Match benchmark clipping, NMS and shared bowl color prior.
                from benchmark import sanitize
                rgb=(image.permute(1,2,0).numpy()*255).round().astype(np.uint8)
                records.append(dict(id=row['id'],objects=sanitize(rgb,objects)))
        selection=calibrate(records,dev.labels);score=selection['dev']['macro_f1']
        if score>best:
            best=score;torch.save(dict(model=model.state_dict(),epoch=epoch+1,dev=selection,
                seed=100,reference='curated_CAD_visible_mesh',test_used=False),args.output/'best.pt')
        log=dict(epoch=epoch+1,loss=float(np.mean(losses)),dev_macro_f1=score,best=best,seconds=time.perf_counter()-start)
        with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        print(json.dumps(log),flush=True)
    (args.output/'training_elapsed.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,epochs=args.epochs,seed=100)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=25);train(p.parse_args())
