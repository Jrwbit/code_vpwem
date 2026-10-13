import os,json,time,hashlib,shutil,subprocess,urllib.request
from pathlib import Path
V=Path('/home/bit-bhr/r435-release/r435-vision-1080ti-v1');H=Path('/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop');P=Path('/tmp/taizhou-production-ui-20261013')
os.environ['PATH']='/home/bit-bhr/.local/bin:'+os.environ['PATH']
def run(args,**kw):return subprocess.check_output(args,**kw).decode().strip()
def get(port,path):
 with urllib.request.urlopen('http://127.0.0.1:'+str(port)+path,timeout=5) as r:return json.load(r)
def idle():
 s=get(50168,'/inspection/state');m=s['motion'];assert not any(m.get(k) for k in ('busy','latched','capturing')),('device_active',m.get('phase'))
 for c in s['real']['components'].values():assert c.get('ok') and c.get('age_s',9)<1.5 and c['power']=='off' and c['mode']=='manual' and c['op']=='idle' and abs(c['max_vel'])<=.01,c
 q=get(8765,'/api/pipeline/status');assert q.get('state') not in ('queued','running','cancelling'),q
 return s
idle();run(['sha256sum','-c','SHA256SUMS'],cwd=P)
before={}
for name,root,expected in [('vision',V,'4781197610879f7bb491699cd2b81b4b532e26ba'),('hmi',H,'b93a31a1a05f198bdf39101d5c617022bb02c451')]:
 before[name]=run(['git','-C',str(root),'rev-parse','HEAD']);assert before[name]==expected
 assert not run(['git','-C',str(root),'diff','--name-only'])
 assert not run(['git','-C',str(root),'diff','--cached','--name-only'])
B=V.parent/'backups'/('production-ui-'+time.strftime('%Y%m%d-%H%M%S'));B.mkdir(mode=0o700)
for name,root in [('vision',V),('hmi',H)]:
 run(['git','-C',str(root),'bundle','create',str(B/(name+'-before.bundle')),'HEAD'])
 with (B/(name+'-before.tar')).open('wb') as f:subprocess.run(['git','-C',str(root),'archive','HEAD'],stdout=f,check=True)
 (B/(name+'-status.txt')).write_text(run(['git','-C',str(root),'status','--short']))
assets=[V/'config/rokae_site.json',V/'calibration/rokae_current.json',H/'helios_hmi/native/xcore_motion',H/'helios_hmi/native/xcore_state_stream',H/'open_real.sh',H/'open_simulation.sh',V/'open_inspection.sh',V/'start.sh']
assets+=list((V/'config').glob('*.json'));assets+=list(Path('/home/bit-bhr/Desktop').glob('*.desktop'))
sha={}
for f in dict.fromkeys(assets):
 if not f.is_file():continue
 dest=B/'field'/f.relative_to('/home/bit-bhr');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);sha[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(B/'before.json').write_text(json.dumps({'git':before,'files':sha},indent=2));shutil.copytree(P,B/'package')
print('backup',B,flush=True)
idle();run(['bash',str(V/'start.sh'),'stop'],cwd=V)
s=get(50168,'/inspection/state');assert not any(s['motion'].get(k) for k in ('busy','latched','capturing'))
pid=int(Path('/home/bit-bhr/.local/state/jinlong-hmi/real.pid').read_text())
assert Path('/proc/'+str(pid)+'/cwd').resolve()==H
cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes();assert b'helios_hmi' in cmd and b'--enable-write' in cmd
os.kill(pid,15)
for _ in range(100):
 try:
  state=Path('/proc/'+str(pid)+'/stat').read_text().split()[2]
  if state=='Z':break
 except FileNotFoundError:break
 time.sleep(.2)
else:raise RuntimeError('HMI owner did not exit; refusing deployment')
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:c=(d/'cmdline').read_bytes()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if c.startswith(str(H/'helios_hmi/native/xcore_').encode()):raise RuntimeError('native owner remains '+d.name)
manifest=json.loads((P/'manifest.json').read_text())
for name,root in [('vision',V),('hmi',H)]:
 run(['git','-C',str(root),'fetch',str(P/(name+'.bundle')),'HEAD'])
 run(['git','-C',str(root),'merge','--ff-only','FETCH_HEAD'])
 assert run(['git','-C',str(root),'rev-parse','HEAD'])==manifest['modules'][name]['revision']
new=H/'helios_hmi/native/xcore_motion.new';shutil.copy2(P/'runtime/xcore_motion',new);new.chmod(0o755)
check=run([str(new),'--check-request','--move','left_arm','--target','0,0,0,0,0,0,0','--expected','0,0,0,0,0,0,0','--ack','HELIOS_WEB_ACK::left_arm','--max-delta-deg','5','--timeout','120','--parent-pid',str(os.getpid()),'--start-gate','--power-only']);assert json.loads(check)['controller_writes'] is False
os.replace(new,H/'helios_hmi/native/xcore_motion')
log=Path('/home/bit-bhr/.local/state/jinlong-hmi/real.log')
with log.open('ab') as f:
 process=subprocess.Popen(['bash',str(H/'helios_hmi/run_ipc.sh'),'--enable-write','--sessions','/home/bit-bhr/helios_hmi_sessions/handeye_ipc'],cwd=H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
Path('/home/bit-bhr/.local/state/jinlong-hmi/real.pid').write_text(str(process.pid)+'\n')
for _ in range(100):
 try:
  s=get(50168,'/inspection/state')
  if all(c.get('ok') and c.get('age_s',9)<1.5 for c in s['real']['components'].values()):break
 except Exception:pass
 time.sleep(.2)
else:raise RuntimeError('HMI read feedback startup failed')
run(['bash',str(V/'start.sh'),'start','--enable-robot-motion','--defer-camera-start'],cwd=V)
for f,old in sha.items():
 if f in (str(H/'helios_hmi/native/xcore_motion'),str(V/'open_inspection.sh')):continue
 assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==old,('field asset changed',f)
record={'status':'deployed_pending_verification','backup':str(B),'before':before,'after':{k:x['revision'] for k,x in manifest['modules'].items()},'native_sha256':hashlib.sha256((H/'helios_hmi/native/xcore_motion').read_bytes()).hexdigest(),'robot_actions_started':False,'vision_flags':['--enable-robot-motion','--defer-camera-start']}
(B/'deployment_record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
