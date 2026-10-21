#!/usr/bin/env python3
"""Field recovery execution record for the exact 2026-10-21 interruption.

This performs robot motion through the existing HMI owner. It is deliberately
bound to the recorded failed operation, site, calibration and native progress.
It is not a general retry command; running it again must reject changed state.
"""
import copy,hashlib,json,math,sys,tempfile,time,urllib.request,urllib.error,uuid
from pathlib import Path
import numpy as np
out=Path('/home/bit-bhr/helios_hmi_sessions/diagnostics/fault-resume-20261021')
root=Path('/home/bit-bhr/r435-release/r435-vision-1080ti-v1')
source=root/'outputs/rokae_trajectory/6ef0a844c8e341c29b2c5af82683390e'
data=json.loads((source/'input.json').read_text());previous=json.loads((out/'return_execution_plan.json').read_text())
sys.path[:0]=[str(out),data['control_root'],data['sim_root'],str(root/'app')]
from check_interrupted_return import completed_prefix
from sim.helios_insert import corridor as cr,fixtures as fx,scene as sc
from helios_hmi.controller_kinematics import attach_models,signature,apply_collision_model
from hk_pointcloud_registration.robot.corridor_worker import configure_geometry,planning_limits
from hk_pointcloud_registration.robot.rokae_site import site_snapshot
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def req(path,body=None,timeout=5):
 headers={'Content-Type':'application/json'}
 if body is not None:headers['X-CSRF-Token']=req('/inspection/state')['motion']['csrf_token']
 r=urllib.request.Request('http://127.0.0.1:50168'+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
 try:
  with opener.open(r,timeout=timeout) as f:r=json.load(f)
 except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode())
 if r.get('ok') is False:raise RuntimeError(r.get('error','HMI failed'))
 return r
s=req('/inspection/state');m=s['motion'];last=m['last_result']
assert not any(m.get(k) for k in ['busy','latched','capturing']) and not s['plan'].get('active')
assert last['safe_exit_confirmed'] and last['error']=='route waypoint timeout'
assert last['native_log'].endswith('20261021_105903_a2c53f315aea.jsonl')
for part in s['real']['components'].values():assert part['ok'] and 0<=part['age_s']<=1 and (part['power'],part['mode'],part['op'])==('off','manual','idle') and abs(part['max_vel'])<=.01
site=attach_models(site_snapshot()[0],s,required=True)
assert site==data['site'],'Tool/site changed'
assert hashlib.sha256((root/'calibration/rokae_current.json').read_bytes()).hexdigest()==data['calibration']['sha256']
events=[json.loads(line) for line in Path(last['native_log']).read_text().splitlines() if line.strip()]
completed=completed_prefix(previous,events);assert completed==5
j=s['real']['joints'];names=[f'AR5-5_07L_joint_{i}' for i in range(1,8)]
w=np.array([j[f'{i}-Joint'] for i in range(1,5)]);a=np.array([j[n] for n in names]);start=a.copy()
assert max(abs(w-np.array(data['waist'])))<np.radians(1)
static={'head':[j[f'{i}-Joint'] for i in (5,6)],'right_arm':[j[f'AR5-5_07R_joint_{i}'] for i in range(1,8)]}
for k,v in static.items():assert max(abs(np.array(v)-data['static_pose'][k]))<np.radians(1)
sc.VENDOR=Path(data['control_root'])/'helios_hmi/model/Wheel_Robot'
build=cr.build;cr.build=lambda cfg:apply_collision_model(build(cfg),site)
scene=json.loads((source/'scene_used.json').read_text());scene['static_pose']=static
limits=planning_limits(s['limits'],site);scene['left_arm_soft_limits_rad']={'lower':[limits[n][0] for n in sc.LEFT],'upper':[limits[n][1] for n in sc.LEFT]}
steps=[];minimum=float('inf')
with tempfile.TemporaryDirectory(prefix='finish-return-') as tmp:
 configure_geometry(scene,fx,site,Path(data['sim_root']),Path(tmp));ctx=cr.Ctx(copy.deepcopy(scene),viz=False)
 for step in previous['segments'][0]['steps'][completed:]:
  target=np.array(step['target']);count=max(1,math.ceil(max(abs(target-a))/np.radians(10)));initial=a.copy()
  for n in range(1,count+1):
   b=initial+(target-initial)*n/count
   for key,value in zip(sc.WAIST+sc.LEFT,np.r_[w,b]):assert limits[key][0]+np.radians(2)<=value<=limits[key][1]-np.radians(2)
   frames=cr.check_line(ctx,np.r_[w,a],np.r_[w,b],adjust=True)
   minimum=min(minimum,*(f['clear']*1000 for f in frames));steps.append({'target':b.tolist()});a=b
seg=copy.deepcopy(previous['segments'][0]);seg.update(start=start.tolist(),hold={'waist':w.tolist()},steps=steps,max_step_deg=10,name='中断返回剩余原关节线（每点不超过10度）')
doc=dict(previous,segments=[seg]);name='interrupted_return_split_'+uuid.uuid4().hex+'.json'
(out/'return_split_plan.json').write_text(json.dumps(doc,ensure_ascii=False));(Path.home()/'helios_hmi_sessions/plans'/name).write_text(json.dumps(doc,ensure_ascii=False))
validation={'collision_checked':True,'steps':len(steps),'minimum_model_clearance_mm':minimum,'completed_previous_return_points':completed,'max_step_deg':10,'speed_mm_s':10}
(out/'return_split_validation.json').write_text(json.dumps(validation));print(json.dumps(validation),flush=True)
now=req('/inspection/state');assert not now['motion']['busy'] and not now['motion']['latched']
for key,value in j.items():assert abs(now['real']['joints'][key]-value)<np.radians(.05)
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'config/rokae_site.json',root/'calibration/rokae_current.json']}
req('/plan/load',{'name':name});pre=req('/plan/preview',{'segment':0,'whole_task':True,'max_step_deg':160,'speed':10},15)
active=False
try:
 r=req('/plan/start',{'token':pre['token'],'confirmation':True,'field_confirmed':True,'faults_checked':True},15);op=r['operation_id'];active=True;cursor=None;deadline=time.monotonic()+120*len(steps)+60
 print(json.dumps({'event':'started','operation_id':op,'plan':name}),flush=True)
 while time.monotonic()<deadline:
  s=req('/inspection/state',timeout=3);p=s['plan']
  assert p['operation_id']==op
  for path,sha in bindings.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
  record={k:p.get(k) for k in ['step','active','phase','error']}
  if record!=cursor:print(json.dumps(record,ensure_ascii=False),flush=True);cursor=record
  if not p['active']:
   active=False;s['motion'].pop('csrf_token',None);(out/'return_split_final.json').write_text(json.dumps(s,ensure_ascii=False))
   print(json.dumps({'event':'final','plan':record,'result':s['motion'].get('last_result'),'components':s['real']['components']},ensure_ascii=False),flush=True);break
  req('/motion/heartbeat',{'operation_id':op},3);time.sleep(.2)
 else:raise TimeoutError('Return timed out')
finally:
 if active:req('/plan/stop',{})
