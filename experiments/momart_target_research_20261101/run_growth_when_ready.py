"""Wait on the identified preparation process, audit labels, then run paired fits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def process_state(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
        return fields[0],fields[19]  # kernel state and starttime (field 22)
    except FileNotFoundError:return None


def main(a):
    status_path=a.research/'growth_prepare_status.json';watch=a.research/'growth_followup_status.json'
    audit=a.research/'growth_support_audit.json';output=a.research/'growth_v4_seed100_300'
    if watch.exists() or audit.exists() or output.exists():raise FileExistsError('Follow-up already started or output exists')
    state=process_state(a.prepare_pid)
    if state is None:raise RuntimeError('Expected live preparation process; inspect terminal state before starting follow-up')
    command=Path(f'/proc/{a.prepare_pid}/cmdline').read_bytes()
    if b'prepare_corpus.sh' not in command:raise ValueError('PID does not identify corpus preparation')
    identity=state[1]
    def update(stage):
        record=dict(stage=stage,pid=os.getpid(),prepare_pid=a.prepare_pid,prepare_starttime=identity,gpu=a.gpu,updated_unix=time.time())
        temp=watch.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(watch)
        print(json.dumps(record),flush=True)
    update('waiting_for_identified_preparation')
    while True:
        status=json.loads(status_path.read_text())
        if status['stage']=='ready_public_features':break
        if status['stage']=='failed':raise RuntimeError('Preparation reported failure; no automatic restart')
        current=process_state(a.prepare_pid)
        if current is None or current[0]=='Z' or current[1]!=identity:raise RuntimeError('Preparation stopped without successful ready state')
        time.sleep(10)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(a.research/'deps')+':'+str(a.project))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    commands=[('support_audit',[str(a.python),'-u','experiments/momart_target_research_20261101/audit_support_corpus.py',
        '--corpora',str(a.research/'growth_corpus'),'--phases',str(a.research/'growth_phases.json'),'--output',str(audit),'--gpu',str(a.gpu)]),
        ('training',[str(a.python),'-u','experiments/momart_target_research_20261101/benchmark_growth.py',
        '--root',str(a.research.parent),'--output',str(output),'--device','cuda:0'])]
    for stage,cmd in commands:
        update(stage);child_env=dict(env)
        if stage=='training':child_env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        result=subprocess.run(cmd,cwd=a.project,env=child_env)
        if result.returncode:
            update('failed_'+stage);raise RuntimeError('Follow-up stopped with exit '+str(result.returncode))
    update('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['project','research','python']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prepare-pid',type=int,required=True);p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
