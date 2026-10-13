#!/usr/bin/env python3
"""Package committed software and Git history; preserve all field assets on the IPC."""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT)


def package(root, name, target, baseline, output):
    if git(root, 'status', '--porcelain', '--untracked-files=normal').strip():
        raise RuntimeError(f'{name}: commit or preserve the working tree before packaging')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    git(root, 'merge-base', '--is-ancestor', baseline, head)
    archive = git(root, 'archive', '--format=tar', head)
    digest = hashlib.sha256()
    checksums = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        for entry in sorted(contents.getmembers(), key=lambda item: item.name):
            if entry.isdir():
                continue
            path = Path(entry.name)
            if not entry.isfile() or path.is_absolute() or '..' in path.parts:
                raise RuntimeError(f'{name}: non-regular source entry: {entry.name}')
            if path.suffix.lower() in {'.pt', '.pth', '.npy', '.npz', '.png', '.jpg', '.jpeg', '.pyc', '.log', '.pem', '.key'}:
                raise RuntimeError(f'{name}: runtime or private asset in source: {entry.name}')
            if any(part in {'.venv', 'outputs', 'models', 'sdk', '__pycache__', 'captures'} for part in path.parts):
                raise RuntimeError(f'{name}: runtime directory in source: {entry.name}')
            data = contents.extractfile(entry).read()
            value = hashlib.sha256(data)
            checksums[entry.name] = value.hexdigest()
            digest.update(entry.name.encode() + b'\0' + value.digest())
    (output / f'{name}.tar').write_bytes(archive)
    bundle = output / f'{name}.bundle'
    git(root, 'bundle', 'create', str(bundle), 'HEAD')
    git(root, 'bundle', 'verify', str(bundle))
    return {'target': target, 'baseline': baseline, 'revision': head,
            'source_sha256': digest.hexdigest(), 'source_files': checksums,
            'changes': git(root, 'log', '--format=%h %s', f'{baseline}..{head}').decode().splitlines()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision', type=Path, required=True)
    parser.add_argument('--hmi', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'deployment_status': 'prepared_only',
                'ipc': 'bit-bhr@192.168.10.66', 'field_assets_included': False, 'modules': {}}
    for name, root, target, baseline in (
        ('vision', args.vision, '/home/bit-bhr/r435-release/r435-vision-1080ti-v1', 'e8bfa04'),
        ('hmi', args.hmi, '/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop', '3163665')):
        manifest['modules'][name] = package(root.resolve(), name, target, baseline, args.output.resolve())
    if args.evidence:
        evidence = args.output / 'evidence'
        evidence.mkdir()
        for source in args.evidence:
            shutil.copy2(source, evidence / source.name)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (args.output / 'SHA256SUMS').open('w') as stream:
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(args.output)) + '\n')
    print(json.dumps({'output': str(args.output), 'modules': {name: data['revision'] for name, data in manifest['modules'].items()},
                      'deployment_status': manifest['deployment_status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
