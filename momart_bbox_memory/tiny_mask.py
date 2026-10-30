"""Small RGB segmentation detector/refiner trained only on fit episodes."""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .interaction import mask_to_box


class TinyMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,12,3,padding=1),nn.ReLU())
        self.coarse=nn.Sequential(nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Conv2d(36,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,1,1))
    def forward(self,x):
        fine=self.fine(x)
        coarse=self.coarse(F.avg_pool2d(fine,2))
        coarse=F.interpolate(coarse,size=fine.shape[-2:],mode="bilinear",align_corners=False)
        return self.head(torch.cat([fine,coarse],dim=1))


def mask_box(prob,threshold=.5,region=None):
    mask=prob>=threshold
    if region is not None:
        x1,y1,x2,y2=region
        limited=np.zeros_like(mask);limited[y1:y2,x1:x2]=mask[y1:y2,x1:x2];mask=limited
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n<=1:return None,0.
    # The task has one target bowl; choose the component with strongest evidence.
    scores=[float(prob[labels==i].sum()) for i in range(1,n)]
    best=int(np.argmax(scores))+1
    if stats[best,cv2.CC_STAT_AREA]<3:return None,0.
    selected=labels==best
    box=mask_to_box(selected,minimum_pixels=3)
    return np.asarray(box,np.float32),float(prob[selected].mean())


def load_fit(root):
    images,masks=[],[]
    for path in sorted(Path(root).glob("*/manifest.json")):
        meta=json.loads(path.read_text())
        for item in meta["streams"]:
            if item["split"]!="fit":continue
            with np.load(path.parent/item["file"]) as data:
                images.extend(data["rgb"]);masks.extend(data["mask"])
    return np.stack(images),np.stack(masks)


def train(args):
    torch.set_num_threads(2);torch.manual_seed(100);np.random.seed(100);random.seed(100)
    images,masks=load_fit(args.root)
    model=TinyMask().to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x=torch.from_numpy(images).permute(0,3,1,2).float()/255
    y=torch.from_numpy(masks[:,None]).float()
    history=[]
    for epoch in range(args.epochs):
        indices=torch.randperm(len(x));losses=[]
        for start in range(0,len(x),32):
            ids=indices[start:start+32]
            batch=x[ids].to(args.device);truth=y[ids].to(args.device)
            # Photometric and mirror augmentation preserve the target mask.
            batch=(batch*(.85+.3*torch.rand(len(batch),1,1,1,device=args.device))).clamp(0,1)
            if random.random()<.5:batch=batch.flip(-1);truth=truth.flip(-1)
            logits=model(batch)
            bce=F.binary_cross_entropy_with_logits(logits,truth,pos_weight=torch.tensor(20.,device=args.device))
            prob=logits.sigmoid()
            dice=1-(2*(prob*truth).sum((1,2,3))+1)/(prob.sum((1,2,3))+truth.sum((1,2,3))+1)
            loss=bce+dice.mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)))
        history.append(record);print(json.dumps(record),flush=True)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.cpu().state_dict(),parameters=sum(p.numel() for p in model.parameters()),
                    fit_frames=len(x),epochs=args.epochs),output/"tiny_mask.pt")
    (output/"training.json").write_text(json.dumps(dict(history=history,fit_frames=len(x),
        parameters=sum(p.numel() for p in model.parameters())),indent=2)+"\n")


def cache(args):
    torch.set_num_threads(2)
    saved=torch.load(Path(args.output)/"tiny_mask.pt",map_location="cpu",weights_only=False)
    model=TinyMask().to(args.device).eval();model.load_state_dict(saved["model"])
    for path in sorted(Path(args.root).glob("*/manifest.json")):
        meta=json.loads(path.read_text())
        for item in meta["streams"]:
            if item["split"]=="fit":continue
            source=path.parent/item["file"]
            with np.load(source) as data:images=data["rgb"]
            with torch.inference_mode():
                batch=torch.from_numpy(images).permute(0,3,1,2).float().to(args.device)/255
                probs=model(batch).sigmoid().cpu().numpy()[:,0]
            np.savez_compressed(source.with_suffix(".tiny.npz"),prob=probs.astype(np.float32))
        print(json.dumps(dict(task=path.parent.name,tiny_cached=True)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","cache"])
    p.add_argument("--root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--epochs",type=int,default=20)
    args=p.parse_args();(train if args.mode=="train" else cache)(args)


if __name__=="__main__":main()
