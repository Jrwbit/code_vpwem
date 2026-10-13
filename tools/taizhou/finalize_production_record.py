from pathlib import Path
import json,hashlib,shutil
P=Path('/home/wjr/code/code_vpwem/outputs/deployment/production-ui-final-20261013-r1')
M=json.loads((P/'manifest.json').read_text());probe=json.loads((P/'evidence/taizhou-production-after-probe.json').read_text())
for name,item in M['modules'].items():
 for runtime in [probe['repositories'][name],probe[name+'_runtime']]:
  assert runtime['revision']==item['revision']
  assert runtime['source_sha256']==item['source_sha256']
 assert not probe[name+'_runtime']['dirty']
assert all(c['ok'] and c['age_s']<1.5 and c['power']=='off' and c['mode']=='manual' and c['op']=='idle' for c in probe['components'].values())
assert not any(probe['motion'].get(k) for k in ['busy','latched','capturing'])
assert probe['vision']['motion_requested'] and not probe['vision']['motion_enabled']
before=json.loads(Path('/tmp/taizhou-production-before-probe.json').read_text())
for key in ['site','calibration','native_state','desktop_launcher']:assert before['field_files'][key]==probe['field_files'][key]
assert probe['field_files']['native_motion']=='8b247e46389d2bfc335a19ef1dc2aa2be5d8b4be9d8990bd6f5738a9b3c2d10c'
assert 'Ran 276 tests' in (P/'evidence/taizhou-production-ipc-tests-20261013.log').read_text()
assert (P/'evidence/taizhou-production-ipc-tests-20261013.log').read_text().rstrip().endswith('OK')
browser=json.loads((P/'evidence/taizhou-production-browser-20261013.json').read_text());assert not browser['errors'] and len(browser['evidence'])==13
shutil.copytree('/home/wjr/code/code_vpwem/outputs/deployment/production-ui-20261013/runtime',P/'runtime')
record={'status':'software_deployed_and_verified_field_acceptance_pending','verified_at':probe['observed_at_utc'],'backup':'/home/bit-bhr/r435-release/backups/production-ui-20261013-212137','revisions':{k:v['revision'] for k,v in M['modules'].items()},'runtime':{k:probe[k] for k in ['vision_runtime','hmi_runtime','processes']},'field_files':probe['field_files'],'tests':{'vision_local':276,'vision_ipc':276,'hmi_local':180,'hmi_ipc':180,'native_offline':76,'browser_fenced_groups':13},'robot_actions_started':False,'field_gates_preserved':True,'camera_current_status':'OpenDevice 0x8006000e; field recovery pending','physical_acceptance':False}
(P/'deployment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
with (P/'SHA256SUMS').open('w') as out:
 for file in sorted(P.rglob('*')):
  if file.is_file() and file.name!='SHA256SUMS':out.write(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(P))+'\n')
print(json.dumps({'status':record['status'],'revisions':record['revisions'],'source_and_runtime_match':True,'field_files_preserved':True}))
