#!/usr/bin/env python3
"""Offline arm-stow and ordered-waist entry search; never controls hardware."""
import sys,json,copy,math,time,itertools,argparse
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser(description=__doc__)
for name in ('snapshot','bundle','scene','shared_solution','output'):
 ap.add_argument(name,type=Path)
ap.add_argument('--seconds',type=float,default=90)
ap.add_argument('--samples',type=int,default=15000)
args=ap.parse_args()
s=json.loads(args.snapshot.read_text())
data=json.loads(args.bundle.read_text())
root,control=Path(data["sim_root"]),Path(data["control_root"])
sys.path[:0]=[str(control),str(root),str(args.bundle.parents[3]/"app")]
from sim.helios_insert import corridor as cr,fixtures as fx,scene as sc,swap as sw
from helios_hmi.controller_kinematics import apply_collision_model,attach_models,signature
from hk_pointcloud_registration.robot.corridor_worker import configure_geometry
from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model
assert signature(data["site"])==signature(attach_models({},s,required=True))
scene=json.loads(args.scene.read_text())
j=s["real"]["joints"]
scene["static_pose"]={"head":[j[f"{i}-Joint"] for i in (5,6)],"right_arm":[j[f"AR5-5_07R_joint_{i}"] for i in range(1,8)]}
w=np.array([j[f"{i}-Joint"] for i in range(1,5)])
a=np.array([j[f"AR5-5_07L_joint_{i}"] for i in range(1,8)])
t=np.array(json.loads(args.shared_solution.read_text())["common"]["waist_model_rad"])
sc.VENDOR=control/"helios_hmi/model/Wheel_Robot"
original=cr.build
cr.build=lambda cfg:selected_terminal_model(apply_collision_model(original(cfg),data["site"]),"JM160")
out=args.output;out.mkdir(parents=True,exist_ok=True)
configure_geometry(scene,fx,data["site"],root,out)
ctx=cr.Ctx(copy.deepcopy(scene),viz=False)
robot=ctx.robot
for name,(lo,hi) in s["limits"].items():
 if name in robot.lim:
  x,y=robot.lim[name];robot.lim[name]=(max(x,lo),min(y,hi))
lo,hi=cr._bounds(robot,cr.LEFT,math.radians(5))
rng=np.random.default_rng(202610221453)
started=time.monotonic();counts={"sampled":0,"endpoint_ok":0,"arm_line_ok":0,"sweep_ok":0}
solution=None
for i in range(args.samples):
 if time.monotonic()-started>args.seconds:break
 q=np.clip(a+rng.normal(0,(.25,.5,.8,1.2)[i%4],7),lo,hi) if i%5 else rng.uniform(lo,hi)
 counts["sampled"]+=1
 if not cr.config_ok(ctx,t,q,clear=.015,adjust=True) or not cr.config_ok(ctx,w,q,clear=.015,adjust=True):continue
 counts["endpoint_ok"]+=1
 try: f1=cr.check_line(ctx,np.r_[w,a],np.r_[w,q],adjust=True)
 except sw.PathError:continue
 counts["arm_line_ok"]+=1
 route=None
 try:f2=cr.check_line(ctx,np.r_[w,q],np.r_[t,q],adjust=True)
 except sw.PathError:
  f2=None
  for order in itertools.permutations(range(4)):
   previous=w.copy(); frames=[];route=[w.tolist()]
   try:
    for axis in order:
     following=previous.copy();following[axis]=t[axis]
     frames+=cr.check_line(ctx,np.r_[previous,q],np.r_[following,q],adjust=True)
     route.append(following.tolist());previous=following
   except sw.PathError:continue
   f2=frames;break
  if f2 is None:continue
 counts["sweep_ok"]+=1
 solution={"waist_route_if_sequential":route,"left_stow_rad":q.tolist(),"minimum_clearance_mm":min(r["clear"] for r in f1+f2)*1000,"arm_frames":len(f1),"waist_frames":len(f2)}
 break
result={"controller_writes":False,"start_waist_rad":w.tolist(),"start_left_rad":a.tolist(),"target_waist_rad":t.tolist(),"static_pose":scene["static_pose"],"counts":counts,"elapsed_s":time.monotonic()-started,"solution":solution,"scope":"current arm to stow, then waist only; no detection or insertion route"}
(out/"stow_search.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
