#!/usr/bin/env python3
"""Explicitly authorized three-cycle replay through the existing HMI owner.

Runs an unchanged, hash-bound route. Revalidates collision geometry each round,
uses native per-segment preview/start/watchdog, and stops on any failure. Never
clears alarms, unlocks a latch, connects to an SDK or retries a failed motion.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from record_inspection_trace import sample


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_delta(a, b):
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in [*a, *b]):
        raise RuntimeError('关节反馈无效')
    return max(abs(x-y) for x, y in zip(a, b))


def completed_step(state, operation):
    p = state.get('plan') or {}
    return (p.get('operation_id') == operation and p.get('active') is False
            and p.get('phase') == 'segment_done' and not p.get('error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--planner-python', type=Path, required=True)
    parser.add_argument('--confirm-three-rounds', action='store_true')
    args = parser.parse_args()
    if not args.confirm_three_rounds:
        parser.error('Explicit three-round authorization flag required')
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    release = Path.home()/'r435-release/r435-vision-1080ti-v1'
    source = json.loads((args.source/'input.json').read_text())
    plan = json.loads((args.source/'plan.json').read_text())
    plan_sha = digest(args.source/'plan.json')
    if len(plan['segments']) != 8 or sum(len(s['steps']) for s in plan['segments']) != 26:
        raise RuntimeError('Expected archived 8-stage/26-step inspection-return cycle')
    if any(s['group'] != 'left_arm' for s in plan['segments']):
        raise RuntimeError('Only left arm may move in this experiment')
    config_path, cal_path = release/'config/rokae_site.json', release/'calibration/rokae_current.json'
    if json.loads(config_path.read_text()) != source['site'] or digest(cal_path) != source['calibration']['sha256']:
        raise RuntimeError('工位、工具或标定与原路线不一致')
    config_sha, cal_sha = digest(config_path), digest(cal_path)
    opener = build_opener(ProxyHandler({}))
    records = []
    armed = False
    def request(path, body=None, timeout=5, port=50168):
        headers = {'Content-Type': 'application/json'}
        if body is not None:
            headers['X-CSRF-Token'] = request('/inspection/state')['motion']['csrf_token']
        req = Request(f'http://127.0.0.1:{port}{path}', headers=headers,
                      data=None if body is None else json.dumps(body).encode())
        try:
            with opener.open(req, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as e:
            raise RuntimeError(str(e.code)+': '+e.read().decode()) from e
        if result.get('ok') is False:
            raise RuntimeError(result.get('error', 'HMI request failed'))
        return result
    def check(state, idle=False):
        if digest(config_path) != config_sha or digest(cal_path) != cal_sha or digest(args.source/'plan.json') != plan_sha:
            raise RuntimeError('实验期间配置、标定或原路线改变')
        if state['motion'].get('latched'):
            raise RuntimeError('安全退出锁定')
        for name in ('left_arm', 'right_arm', 'body'):
            p = state['real']['components'][name]
            if not p.get('ok') or not 0 <= p['age_s'] < 1:
                raise RuntimeError('反馈过期：'+name)
            if idle and (p['power']!='off' or p['mode']!='manual' or p['op']!='idle' or abs(p['max_vel'])>.01):
                raise RuntimeError('不是停稳下电空闲：'+name)
        joints = state['real']['joints']
        for names, values in (([f'{i}-Joint' for i in (5,6)], source['static_pose']['head']),
                              ([f'AR5-5_07R_joint_{i}' for i in range(1,8)], source['static_pose']['right_arm'])):
            if maximum_delta([joints[n] for n in names], values) > math.radians(1):
                raise RuntimeError('头部／右臂改变，场景失效')
        held = plan['segments'][0]['hold']['waist']
        waist = [joints[f'{i}-Joint'] for i in range(1,5)]
        if maximum_delta(waist, held) > math.radians(1):
            raise RuntimeError('腰部偏离原保持姿态')
    def log(kind, **values):
        item = {'time': time.time(), 'local_time': datetime.now().astimezone().isoformat(), 'event': kind, **values}
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
        print(json.dumps(item, ensure_ascii=False), flush=True)
    try:
        for trial in range(1, 4):
            state = request('/inspection/state'); check(state, idle=True)
            if state['motion']['busy'] or state['plan']['active'] or (out/'STOP').exists():
                raise RuntimeError('活动任务或停止请求')
            vision = request('/api/robot/status', port=8765)
            if vision['busy'] or vision.get('trajectory', {}).get('state') in ('planning', 'running'):
                raise RuntimeError('检测侧仍有活动任务')
            snapshot = out/f'trial{trial}_start.json'
            safe = {'real': state['real'], 'raw': sample(state)['raw'], 'limits': state['limits']}
            snapshot.write_text(json.dumps(safe))
            validation = out/f'trial{trial}_validation.json'
            subprocess.run([str(args.planner_python), str(args.validator), str(args.source), str(snapshot), str(validation)], check=True, timeout=120)
            verified = json.loads(validation.read_text())
            if verified['source_plan_sha256'] != plan_sha or not verified['collision_checked']:
                raise RuntimeError('碰撞复核绑定不一致')
            check(request('/inspection/state'), idle=True)
            identifier = uuid.uuid4().hex
            folder = release/'outputs/rokae_trajectory'/identifier; folder.mkdir()
            for name in ('plan.json', 'receipt.json', 'scene_used.json'):
                shutil.copy2(args.source/name, folder/name)
            data = dict(source, repeatability={'trial': trial, 'original_source': args.source.name,
                                             'original_plan_sha256': plan_sha, 'visual_task_id': '20261019_230620_d9d3cd'})
            (folder/'input.json').write_text(json.dumps(data, ensure_ascii=False))
            name = 'inspection_'+identifier+'.json'
            target = Path.home()/'helios_hmi_sessions/plans'/name
            shutil.copy2(args.source/'plan.json', target)
            loaded = request('/plan/load', {'name': name})
            loaded_sha = loaded['sha256']
            log('round_start', trial=trial, job=identifier, source_plan_sha256=plan_sha)
            for index, segment in enumerate(plan['segments']):
                state = request('/inspection/state'); check(state, idle=True)
                if (out/'STOP').exists() or state['plan']['sha256'] != loaded_sha:
                    raise RuntimeError('停止请求或已加载计划被替换')
                if maximum_delta([state['real']['joints'][f'{i}-Joint'] for i in range(1,5)],
                                 verified['current_waist_model_rad']) > math.radians(.1):
                    raise RuntimeError('腰部相对本轮碰撞复核姿态发生变化')
                preview = request('/plan/preview', {'segment': index, 'max_step_deg': 120,
                                  'speed': segment.get('speed_mm_s', 21), 'waist_speed': 1}, timeout=15)
                armed = True
                result = request('/plan/start', {'token': preview['token'], 'confirmation': True,
                                 'field_confirmed': True, 'faults_checked': True}, timeout=15)
                operation = result['operation_id']; deadline=time.monotonic()+600; cursor=None
                while time.monotonic()<deadline:
                    state = request('/inspection/state', timeout=2); check(state)
                    if (out/'STOP').exists() or state['plan']['sha256'] != loaded_sha:
                        raise RuntimeError('停止请求或计划改变')
                    if completed_step(state, operation):
                        break
                    p = state['plan']
                    if p.get('operation_id') != operation or not p.get('active'):
                        raise RuntimeError(p.get('error') or '阶段退出异常／外部停止')
                    if (p.get('step'),state['motion'].get('phase')) != cursor:
                        cursor=(p.get('step'),state['motion'].get('phase'))
                        log('progress',trial=trial,segment=index,step=cursor[0],phase=cursor[1])
                    try:
                        request('/motion/heartbeat', {'operation_id': operation}, timeout=2)
                    except RuntimeError:
                        if not completed_step(request('/inspection/state'), operation):
                            raise
                    time.sleep(.35)
                else:
                    raise TimeoutError('阶段超过600秒')
                log('segment_done', trial=trial,segment=index)
                if segment.get('pause_at_target') or index==len(plan['segments'])-1:
                    label='target' if segment.get('pause_at_target') else 'home'
                    for n in range(20):
                        state=request('/inspection/state');check(state,idle=True)
                        if (out/'STOP').exists() or state['plan']['sha256'] != loaded_sha:
                            raise RuntimeError('停止请求或计划改变')
                        (out/f'trial{trial}_{label}_{n:02d}.json').write_text(json.dumps(sample(state)))
                        time.sleep(.25)
                    log('settled_sample',trial=trial,position=label)
            armed=False
            records.append({'trial':trial,'job':identifier,'state':'completed'})
            (out/'results.json').write_text(json.dumps(records,indent=2))
            log('round_completed',trial=trial,job=identifier)
        log('all_completed',rounds=3)
    except BaseException as exc:
        log('failed',error=str(exc),completed_rounds=len(records))
        if armed:
            try:request('/stop',{},timeout=10);log('stop_requested')
            except Exception as e:log('stop_unconfirmed',error=str(e))
        raise


if __name__ == '__main__':
    main()
