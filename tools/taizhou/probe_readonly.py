#!/usr/bin/env python3
"""Read existing IPC files and GET endpoints. Never start an SDK or send a POST."""
import hashlib
import json
import subprocess
import os
import time
from pathlib import Path
from urllib.request import build_opener, ProxyHandler

ROOTS = {'vision': Path('/home/bit-bhr/r435-release/r435-vision-1080ti-v1'),
         'hmi': Path('/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop')}
opener = build_opener(ProxyHandler({}))


def get(port, path):
    try:
        with opener.open(f'http://127.0.0.1:{port}{path}', timeout=3) as response:
            return json.load(response)
    except Exception as error:
        return {'probe_error': type(error).__name__}


def repository(root):
    if not root.is_dir():
        return {'exists': False}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], timeout=3, stderr=subprocess.DEVNULL)
    try:
        digest = hashlib.sha256()
        names = sorted(name for name in git('ls-files', '-z').decode().split('\0')
                       if name and name not in ('SOURCE_MANIFEST.json', 'SHA256SUMS'))
        for name in names:
            digest.update(name.encode() + b'\0' + hashlib.sha256((root / name).read_bytes()).digest())
        return {'exists': True, 'revision': git('rev-parse', 'HEAD').decode().strip(),
                'status': git('status', '--porcelain').decode().splitlines(), 'source_sha256': digest.hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {'exists': True, 'probe_error': 'git_or_source_unavailable'}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def process(pid_file):
    try:
        pid = int(pid_file.read_text().strip())
        if pid <= 1:
            raise ValueError('Invalid service PID')
        proc = Path('/proc') / str(pid)
        arguments = (proc / 'cmdline').read_bytes().split(b'\0')
        flags = [value.decode() for value in arguments if value in
                 (b'--enable-write', b'--enable-robot-motion', b'--defer-camera-start', b'--fake', b'--no-camera')]
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        uptime = float(Path('/proc/uptime').read_text().split()[0])
        started = time.time() - uptime + int(stat[19]) / os.sysconf('SC_CLK_TCK')
        return {'pid': pid, 'cwd': str((proc / 'cwd').resolve()), 'flags': flags, 'started_at': started}
    except (OSError, ValueError, IndexError):
        return {'probe_error': 'pid_or_process_unavailable'}


def main():
    state = get(50168, '/inspection/state')
    motion = state.get('motion', {})
    vision = get(8765, '/api/robot/status')
    pipeline = get(8765, '/api/pipeline/status')
    selected_motion = {key: motion.get(key) for key in ('enabled', 'busy', 'latched', 'capturing', 'phase', 'generation', 'resume_in_s')}
    raw = state.get('raw', {})
    identities = {name: {key: part.get(key) for key in ('controller_id', 'controller_ip', 'type')}
                  for name, part in raw.items() if isinstance(part, dict)}
    result = {'repositories': {name: repository(root) for name, root in ROOTS.items()},
              'processes': {'vision': process(ROOTS['vision'] / 'outputs/hik_rgb_fast/service/service.pid'),
                            'hmi': process(Path('/home/bit-bhr/.local/state/jinlong-hmi/real.pid'))},
              'vision_runtime': get(8765, '/api/version'), 'hmi_runtime': state.get('runtime_version'),
              'hmi_probe_error': state.get('probe_error'), 'hmi_revision_label': state.get('revision'),
              'components': state.get('real', {}).get('components'), 'identities': identities, 'motion': selected_motion,
              'pipeline': {key: pipeline.get(key) for key in ('id', 'state', 'node', 'scheme', 'active_task_type')},
              'vision': {key: vision.get(key) for key in ('backend', 'motion_enabled', 'motion_requested', 'busy', 'phase', 'blockers', 'probe_error')},
              'field_files': {name: checksum(path) for name, path in {
                  'calibration': ROOTS['vision'] / 'calibration/rokae_current.json',
                  'site': ROOTS['vision'] / 'config/rokae_site.json',
                  'native_motion': ROOTS['hmi'] / 'helios_hmi/native/xcore_motion',
                  'native_state': ROOTS['hmi'] / 'helios_hmi/native/xcore_state_stream',
                  'desktop_launcher': ROOTS['hmi'] / 'open_real.sh'}.items()}}
    # CSRF tokens, process environments, credentials and camera frames never enter this report.
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
