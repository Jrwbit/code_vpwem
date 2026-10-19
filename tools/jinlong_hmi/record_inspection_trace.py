#!/usr/bin/env python3
"""Passive HTTP-only inspection recorder. Never connects to an SDK or sends POST.

Run on the IPC. Runtime traces, visual artifacts and configuration snapshots must
stay outside Git. A stop file or duration limit ends recording without stopping
or otherwise changing the robot.
"""
import argparse
import hashlib
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


RAW_KEYS = ('ok', 'component', 't', 'rx', 'reader_epoch', 'type', 'controller_ip',
            'controller_id', 'power', 'mode', 'op', 'q', 'qd', 'q_raw', 'qd_raw',
            'q_trunk', 'tau', 'flange_pose6', 'T_base_flange_rowmajor')


def pick(data, keys):
    return {k: data[k] for k in keys if k in data}


def sample(state):
    return {'source_t': state.get('t'),
            'components': state.get('real', {}).get('components'),
            'joints_model_rad': state.get('real', {}).get('joints'),
            'raw': {name: pick(data, RAW_KEYS) for name, data in (state.get('raw') or {}).items()},
            'motion': pick(state.get('motion') or {}, ('busy', 'latched', 'phase', 'operation_id',
                                                       'operation_kind', 'last_result')),
            'plan': pick(state.get('plan') or {}, ('name', 'sha256', 'active', 'phase', 'operation_id',
                                                 'segment', 'step', 'error', 'segments'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path, default=Path.home()/'r435-release/r435-vision-1080ti-v1')
    parser.add_argument('--period', type=float, default=.1)
    parser.add_argument('--duration', type=float, default=3600)
    args = parser.parse_args()
    if not .05 <= args.period <= 5 or args.duration <= 0:
        parser.error('period must be 0.05–5 seconds and duration must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    started = time.monotonic()
    stats = {'started_at': datetime.now().astimezone().isoformat(), 'samples': 0,
             'errors': 0, 'max_request_s': 0, 'max_sample_gap_s': 0,
             'period_s': args.period, 'http_only': True, 'controller_writes': False}

    def get(opener, port, path):
        with opener.open(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
            return json.load(response)

    def archive(source, relative):
        if not source.is_file():
            return
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dest = out/'artifacts'/relative
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            return
        # Preserve changed versions, rather than replacing evidence.
        if dest.exists():
            dest = dest.with_name(dest.stem+'_'+digest[:12]+dest.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

    def metadata():
        opener = build_opener(ProxyHandler({}))
        with (out/'events.jsonl').open('w') as log:
            while not stop.is_set():
                event = {'time': time.time(), 'local_time': datetime.now().astimezone().isoformat()}
                try:
                    d = get(opener, 8765, '/api/robot/status')
                    event['trajectory'] = d.get('trajectory')
                    event['references'] = d.get('waist_recovery_references')
                    event['robot'] = pick(d.get('robot') or {}, ('busy', 'latched', 'last_result',
                                                               'waist_model_rad', 'joints_rad', 'T_base_flange'))
                    v = get(opener, 8765, '/api/pipeline/status')
                    event['vision'] = pick(v, ('id', 'state', 'error', 'message', 'started_at', 'finished_at', 'result'))
                    for name in ('config/rokae_site.json', 'calibration/rokae_current.json'):
                        archive(args.release/name, Path(name))
                    task = v.get('id')
                    if isinstance(task, str) and re.fullmatch(r'[0-9]{8}_[0-9]{6}_[0-9a-f]+', task):
                        for name in ('web_task.json', 'result.json', 'latest.json', 'camera_intrinsics.json', 'latest.jpg'):
                            rel = Path('outputs/hik_web_pipeline')/task/name
                            archive(args.release/rel, rel)
                    job = (d.get('trajectory') or {}).get('id')
                    if isinstance(job, str) and re.fullmatch(r'[0-9a-f]{32}', job):
                        for name in ('input.json', 'plan.json', 'receipt.json', 'scene_used.json', 'planner.log'):
                            rel = Path('outputs/rokae_trajectory')/job/name
                            archive(args.release/rel, rel)
                    native = (d.get('robot') or {}).get('last_result', {}).get('native_log')
                    if isinstance(native, str):
                        source = Path(native)
                        if source.parent == Path.home()/'helios_hmi_sessions/motion':
                            archive(source, Path('motion')/source.name)
                except Exception as exc:
                    event['error'] = str(exc)
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                stop.wait(1)

    thread = threading.Thread(target=metadata, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    last = None
    try:
        with (out/'joints.jsonl').open('w') as log:
            while time.monotonic()-started < args.duration and not (out/'STOP').exists():
                before = time.monotonic()
                row = {'time': time.time(), 'local_time': datetime.now().astimezone().isoformat()}
                try:
                    row.update(sample(get(opener, 50168, '/inspection/state')))
                    stats['samples'] += 1
                except Exception as exc:
                    row['error'] = str(exc)
                    stats['errors'] += 1
                elapsed = time.monotonic()-before
                row['request_s'] = elapsed
                stats['max_request_s'] = max(stats['max_request_s'], elapsed)
                if last is not None:
                    stats['max_sample_gap_s'] = max(stats['max_sample_gap_s'], before-last)
                last = before
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                stats['last_sample_at'] = row['local_time']
                tmp = out/'status.tmp'
                tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
                tmp.replace(out/'status.json')
                stop.wait(max(0, args.period-(time.monotonic()-before)))
    finally:
        stop.set()
        thread.join(timeout=6)
        stats['finished_at'] = datetime.now().astimezone().isoformat()
        (out/'status.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
