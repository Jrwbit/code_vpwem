#!/usr/bin/env python3
"""Capture a fresh IPC vision task and export a non-executable simulation input.

Only the existing local vision/HMI HTTP services are used. No robot motion,
power, alarm, drag, or HMI plan execution endpoint is called.
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path


REMOTE = r'''
import json, pathlib, sys, time, urllib.request, urllib.error
options = json.load(sys.stdin)
root = pathlib.Path.home() / 'r435-release/r435-vision-1080ti-v1'
def request(path, payload=None, port=8765, timeout=40):
    req = urllib.request.Request('http://127.0.0.1:'+str(port)+path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from None

if options.get('import_session'):
    config = request('/api/robot/config')
    token = request('/api/robot/session')['session_token']
    request('/api/robot/calibration/import', {
        'session': options['import_session'], 'motion_token': token,
        'expected_calibration_sha256': config['calibration']['sha256']})

task = request('/api/pipeline/status')
if options['capture']:
    if task.get('state') in ('queued', 'running', 'cancelling'):
        raise RuntimeError('An existing vision job is active; it was left untouched')
    params = options.get('params') or task.get('params') or {}
    if options.get('model'):
        params['product_type'] = options['model']
    if not params.get('product_type'):
        raise RuntimeError('Specify --model for the first capture after service startup')
    scheme = task.get('scheme') or 'rokae_depth_fast_recovery'
    if options.get('ensure_resident'):
        request('/api/camera/mode', {'mode':'working_mode_4_resident',
            'task_type':'junction_box', 'params':params})
    request('/api/pipeline/run', {'node':'full', 'scheme':scheme,
        'task_type':'junction_box', 'params':params})
    deadline = time.monotonic()+100
    while time.monotonic() < deadline:
        task = request('/api/pipeline/status', timeout=10)
        if task.get('state') not in ('queued', 'running', 'cancelling'):
            break
        time.sleep(.5)
if task.get('state') != 'completed' or task.get('task_type') != 'junction_box':
    raise RuntimeError('A completed current junction-box observation is required')
preview = request('/api/robot/hover/preview', {'task_id':task['id'], 'hover_mm':options['hover_mm']})
folder = pathlib.Path(task['output_dir']).resolve()
if not folder.is_relative_to((root/'outputs').resolve()):
    raise RuntimeError('Unexpected result path')
state = request('/state', port=50168)
config = request('/api/robot/config')
print(json.dumps({'preview':preview,
    'state':{key:state[key] for key in ('t','real','motion','plan','replay','arm_roles')},
    'calibration':json.loads((root/'calibration/rokae_current.json').read_text()),
    'site':config['config'], 'result':json.loads((folder/'result.json').read_text())}))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ssh-host", default="jinlong-ipc")
    ap.add_argument("--host-key-alias", help="Use an existing SSH known-host identity for another IPC address")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capture", action="store_true", help="Acquire a new camera observation")
    ap.add_argument("--model", choices=("100", "132", "160"))
    ap.add_argument("--params-json", type=Path, help="Existing camera/detection settings to preserve after a service restart")
    ap.add_argument("--ensure-resident", action="store_true", help="Start the existing RGB-D resident camera mode before capture")
    ap.add_argument("--hover-mm", type=float, default=150)
    ap.add_argument("--import-session", help="Explicitly apply a solved HMI session first; does not grant independent acceptance")
    args = ap.parse_args()
    if not 60 <= args.hover_mm <= 300 or args.ssh_host.startswith("-"):
        ap.error("invalid hover distance or SSH host")
    if args.out.exists():
        ap.error("output exists; use a new file for each observation")
    payload = {"capture": args.capture, "model": args.model, "hover_mm": args.hover_mm,
               "import_session": args.import_session, "ensure_resident": args.ensure_resident,
               "params": json.loads(args.params_json.read_text()) if args.params_json else None}
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    if args.host_key_alias:
        ssh += ["-o", "HostKeyAlias="+args.host_key_alias]
    output = subprocess.check_output(ssh + [args.ssh_host,
                                      "python3 -c " + shlex.quote(REMOTE)],
                                     input=json.dumps(payload).encode(), timeout=150)
    bundle = json.loads(output)
    bundle["state"]["motion"].pop("csrf_token", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"task_id": bundle["preview"]["task_id"], "fixture": bundle["preview"]["fixture"],
                      "hover_mm": bundle["preview"]["hover_mm"], "bundle": str(args.out),
                      "controller_writes": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
